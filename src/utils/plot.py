"""Config reading, run matching and plotting helpers for scripts that plot curves.

Two rules everything here rests on.

NULLS. A null config value is written as `null` by sweep.format_override, comes
back from yaml as None, and -- once a column of them goes through a DataFrame and
parquet -- reads back as NaN, because a key only some configurations record is
missing everywhere else and pandas types the all-empty column as float64. An
absent key and a recorded null are therefore the same thing, and `is_null` knows
the spellings a null picks up along that path -- and only those, so a config
value that merely reads like one ("none") stays a value.

EQUALITY. Two config values are the same when `value_str` renders them the same.
That is the only equality in this module: the same rule decides whether a run
matches a config YAML, whether a row belongs in a cell, and what a curve is
labelled. A second pairwise comparison would eventually disagree with it.

`resolve_yaml` reimplements Hydra's `defaults:` chain rather than composing it,
so reading a config group needs no Hydra run. `load_config_group` does import
src.utils.id for the keys that do not define a configuration, which pulls hydra
in; nothing else here does.
"""

import colorsys
import io
import itertools
import json
import os
import re
import shutil
import urllib.error
import urllib.request
import warnings
import zipfile
import ast
from ast import literal_eval
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import matplotlib.colors as mc
from matplotlib import font_manager
from matplotlib import pyplot as plt
from pathvalidate import sanitize_filename


# -----------------------------------------------------------------------------
# ----- Values ----------------------------------------------------------------
# -----------------------------------------------------------------------------

# The spellings a null ACQUIRES on its way through this pipeline, exactly as it
# acquires them: `None` and `nan` are what str() gives the Python and numpy
# objects, `<NA>` and `NaT` what pandas gives its own, `null` how
# sweep.format_override writes one.
#
# Matched case-sensitively, and `none` and `na` are deliberately absent. In YAML
# a bare `none` is the string "none", not a null, and a config is free to use it
# as a value -- an encoder or a normalization named "none" is an ordinary
# setting, and reading it as a missing key would match it against every run that
# never recorded the key. A null WRITTEN in a config reaches yaml as None
# whichever of `null`, `Null`, `NULL` or `~` it was spelled with, so nothing here
# has to recognise those.
_NULL_SPELLINGS = frozenset({"None", "nan", "NaN", "null", "<NA>", "NaT"})

# What value_str renders every null as, and so the only null a value_str'd
# Series can hold.
NULL_STR = "None"


def is_null(v) -> bool:
    """True for a null: None, NaN, pd.NA, pd.NaT, and the exact strings those
    become once stringified (see _NULL_SPELLINGS). Containers are never null --
    an empty list is a value, not a missing one."""

    if isinstance(v, (list, tuple, set, dict, np.ndarray)):
        return False
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() in _NULL_SPELLINGS
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def value_str(v) -> str:
    """A config value as the string that stands for it: what to compare by, group
    by, key a dict on, and print in a label.

    Close to `str(v)`, except: integral numbers normalize ("64.0" -> "64"), since
    a key only some configurations record forces its column to float; containers
    render element-wise, since a list hyperparameter comes back from parquet as
    an unhashable numpy array repr'd without commas; every null spelling
    collapses to `NULL_STR`, so an absent key and a recorded null read alike."""

    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(value_str(x) for x in v) + "]"
    if is_null(v):
        return NULL_STR
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f.is_integer() else repr(f)


def native(v):
    """Strip numpy types recursively: np.int64 -> int, np.ndarray -> list. Used to
    repr hyperparameters into the generated missing-runs script without it needing
    numpy."""

    if isinstance(v, np.ndarray):
        return [native(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [native(x) for x in v]
    if isinstance(v, dict):
        return {k: native(x) for k, x in v.items()}
    return v


# -----------------------------------------------------------------------------
# ----- Config files ----------------------------------------------------------
# -----------------------------------------------------------------------------

def deep_merge(
    base: dict,
    override: dict,
    replace_keys: frozenset = frozenset(),
) -> dict:
    """Recursively merge override into base; child wins on leaves and non-dict
    values. Keys in replace_keys are replaced wholesale instead of recursing."""

    out = dict(base)
    for k, v in override.items():
        if k in replace_keys:
            out[k] = v
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v, replace_keys)
        else:
            out[k] = v
    return out


def resolve_yaml(
    yaml_files: dict,
    name: str,
    seen: set = None,
    replace_keys: frozenset = frozenset(),
) -> dict:
    """Resolve a YAML file's Hydra-style `defaults:` chain by hand (deep merge,
    child wins). Follows `defaults:` only; `${...}` interpolations are NOT
    evaluated, which is why `flatten_cfg` drops the keys still holding one."""

    if seen is None:
        seen = set()
    if name in seen or name not in yaml_files:
        return {}
    seen.add(name)
    cfg = yaml.safe_load(Path(yaml_files[name]).read_text(encoding="utf-8")) or {}
    defaults = cfg.pop("defaults", [])
    merged: dict = {}
    for dep in defaults:
        dep_name = dep if isinstance(dep, str) else next(iter(dep.values()))
        merged = deep_merge(
            merged,
            resolve_yaml(yaml_files, dep_name, seen.copy(), replace_keys),
            replace_keys,
        )
    return deep_merge(merged, cfg, replace_keys)


def flatten_cfg(nested: dict, drop_interpolations=True) -> dict:
    """Nested config -> {dotted_key: value}. The one place deciding what a flat
    config looks like: process_data.py and `load_config_group` must agree on it or
    a row never matches the YAML it came from.

    `drop_interpolations` leaves out the values still holding "${", which is the
    default. What defines a run is decided once, by src.utils.id.DROPPED_CFG_KEYS,
    before cfg.yaml is written."""

    records = pd.json_normalize(nested, sep=".").to_dict(orient="records")
    if not records:
        return {}
    flat = records[0]
    if drop_interpolations:
        flat = {k: v for k, v in flat.items() if not (isinstance(v, str) and "${" in v)}
    return flat


def load_config_group(config_dir: str) -> dict:
    """Every YAML in a Hydra config group directory -- configs/environment,
    configs/algorithm -- with `defaults:` resolved, as {stem: {flat_key: value}}.
    Both groups are matched against a run identically; only the directory differs.

    Does NOT compose with agent/default, whose keys are CLI-swept per experiment
    and are not part of a config's identity.

    The keys that do not define a configuration are dropped, as they are from a
    run's cfg.yaml: a YAML declaring one would be compared against a run that
    cannot record it, and would match nothing."""

    # Imported here rather than at module level: src.utils.id pulls in hydra.
    from src.utils.id import DROPPED_CFG_KEYS

    def defining(flat):
        return compared_keys(flat, DROPPED_CFG_KEYS)

    yaml_files = {p.stem: p for p in Path(config_dir).glob("*.yaml")}
    return {
        name: defining(flatten_cfg(resolve_yaml(yaml_files, name)))
        for name in yaml_files
    }


def ignored_cfg_keys(plot_cfg: dict) -> frozenset:
    """The keys a plot config declares as not defining a configuration, as a
    frozenset ready to pass to `assign_configs`.

        ignored_cfg_keys:
          - experiment.testing_episodes

    They are exempt from the comparison against the config YAMLs, for keys a
    launch sets for every run alike: the YAML describes a configuration none of
    them were run at, and the key says nothing about which of them a run is.
    A bare scalar is one key, and a sub-key of a listed one is exempt too."""

    raw = plot_cfg.get("ignored_cfg_keys") or []
    if isinstance(raw, str):
        raw = [raw]
    return frozenset(str(key) for key in raw)


# -----------------------------------------------------------------------------
# ----- Matching --------------------------------------------------------------
# -----------------------------------------------------------------------------

def exempt_key(key: str, ignore) -> bool:
    """Whether `ignore` holds `key`, by exact name or as the dotted prefix of
    it: `a.b` is exempted by `a.b` and by `a`."""

    return key in ignore or any(key.startswith(f"{i}.") for i in ignore)


def compared_keys(declared: dict, ignore) -> dict:
    """Return the keys of `declared` that take part in a comparison: everything
    but the ones `ignore` exempts."""

    return {
        key: value for key, value in declared.items()
        if not exempt_key(key, ignore)
    }


def assign_configs(
    df: pd.DataFrame,
    configs: dict,
    ignore: frozenset = frozenset(),
    overridden_by: tuple = None,
) -> pd.Series:
    """Which config YAML each row was launched from, as a Series of stems.

    THE RULE: a config matches a run when EVERY key it declares equals what the
    run recorded -- an absent key counts as null, so a declared value the run
    never recorded is a mismatch. A run matching no config is "unknown": it is
    not the configuration any YAML names, and calling it the nearest one would
    draw it as a configuration it is not. Where several match, the one declaring
    the most keys wins, as the more specific description of the same run.

    `ignore` exempts swept keys: a run differs from its YAML in those by
    construction, since sweeping a key is what overriding it means. Exempting a
    key two configs differ in ONLY on makes them indistinguishable, and which of
    them a run is assigned to is then arbitrary.

    `overridden_by` is `(labels, configs)` for a group Hydra composes after this
    one, with `labels` each row's assignment to it (as returned by this function).
    Keys that row's config declares are skipped when matching the row, since the
    later group decided them."""

    col_of = {c[1]: c for c in df.columns if c[0] == "hyperparameter"}

    candidates = {
        name: compared_keys(configs[name], ignore) for name in sorted(configs)
    }
    candidates = {name: declared for name, declared in candidates.items() if declared}

    # Ties reported once each rather than once per row: the set of configs a row
    # could be is decided by its signature, so every row of a signature ties the
    # same way and one message says all of it.
    reported_ties = set()

    def label(row, skip):
        def recorded(key):
            return value_str(row[col_of[key]] if key in col_of else None)

        best_name, best_len, tied = None, None, []
        for name, declared in candidates.items():
            declared = {
                key: value for key, value in declared.items() if key not in skip
            }
            if not declared:
                continue
            if any(
                recorded(key) != value_str(value)
                for key, value in declared.items()
            ):
                continue
            if best_name is None or len(declared) > best_len:
                best_name, best_len, tied = name, len(declared), [name]
            elif len(declared) == best_len:
                tied.append(name)
        # Two configs matching and declaring as many keys as each other describe
        # the run equally well, and the one that wins is whichever sorted first.
        # That is arbitrary, so it is said out loud: it means either the two are
        # indistinguishable in the data, or `ignore` exempted the only key they
        # differ in.
        if len(tied) > 1:
            key = tuple(tied)
            if key not in reported_ties:
                reported_ties.add(key)
                print(
                    f"WARNING: {list(tied)} all match a run and declare "
                    f"{best_len} key(s) each, so which one it is assigned to is "
                    f"arbitrary; '{tied[0]}' is used."
                )
        return best_name or "unknown"

    # The keys to skip depend only on which config of the later group a row was
    # assigned, so they are resolved once per stem.
    if overridden_by is None:
        over_stem = pd.Series("", index=df.index)
        skip_of = {"": frozenset()}
    else:
        over_labels, over_configs = overridden_by
        over_stem = over_labels.map(lambda v: split_config_label(str(v))[0])
        skip_of = {
            stem: frozenset((over_configs or {}).get(stem, {}))
            for stem in over_stem.unique()
        }

    # Labelling a row only depends on the columns some config declares and on
    # which keys it skips, so rows agreeing on those get the same label: build a
    # signature from them, label one row per distinct signature, map the rest.
    # value_str first -- list-valued hyperparameters arrive from parquet as numpy
    # arrays, which are unhashable, and it also keeps 64 and 64.0 in one group.
    keys = dict.fromkeys(key for declared in candidates.values() for key in declared)
    sig = over_stem.astype(str)
    for key in keys:
        if key in col_of:
            sig = sig + "\x00" + df[col_of[key]].map(value_str)

    first = ~sig.duplicated()
    resolved = {
        signature: label(row, skip_of[stem])
        for signature, stem, (_, row) in zip(
            sig[first],
            over_stem[first],
            df[first].iterrows(),
        )
    }
    return sig.map(resolved).astype(str)


def split_config_label(label: str):
    """The inverse of the label `assign_configs` writes:
    `lunar_lander (environment.train_from_pixels=True)` ->
    ("lunar_lander", {"environment.train_from_pixels": True}).

    Splits on a space that starts a `key=`, not on every space, since a value can
    hold one ("[mean, max]"). Values come back through literal_eval where they
    parse, so 100000 is an int and True a bool; anything else stays the string it
    was printed as."""

    stem, bracket, rest = label.partition(" (")
    if not bracket:
        return label, {}
    # One closing paren, the one that matches the opening this split on: a value
    # can end in a paren of its own, and rstrip would take that too.
    if rest.endswith(")"):
        rest = rest[:-1]
    overrides = {}
    for pair in re.split(r" (?=[\w.]+=)", rest):
        key, _, text = pair.partition("=")
        try:
            overrides[key] = literal_eval(text)
        except (ValueError, SyntaxError):
            overrides[key] = text
    return stem, overrides


def null_mask(df: pd.DataFrame, col: tuple) -> pd.Series:
    """Rows whose `col` is null, its subtree included.

    Two rules on top of equality. A declared null matches a row that has no such
    key at all: a column only some runs record is empty for all the others, so
    the data cannot tell an absent key from a recorded null.

    And a declared null on a key that other rows expand into a subtree means "no
    subtree at all", so every `key.*` column must be unset as well. For example,
    for `foo: null` where other rows set `foo.a` and `foo.b`: those runs record
    the children and never the parent, so the bare `foo` column is empty for them
    too, and without the child check a declared null would select every row
    rather than the runs that have no foo."""

    if col in df.columns:
        mask = df[col].map(value_str) == NULL_STR
    else:
        mask = pd.Series(True, index=df.index)

    section, key = col
    for child in df.columns:
        if child[0] == section and child[1].startswith(f"{key}."):
            mask = mask & (df[child].map(value_str) == NULL_STR)
    return mask


def _value_mask(df: pd.DataFrame, col: tuple, values: list) -> pd.Series:
    """Rows whose `col` holds one of `values` -- one key of a sweep entry, ORed
    over the values it may take. For example, values `[1, 2]` selects every row
    recording either.

    A null among the values is matched by `null_mask`, whatever else is declared
    beside it: the subtree rule holds for `foo: [null, bar]` exactly as it does
    for `foo: null`."""

    others = {value_str(v) for v in values if not is_null(v)}

    if others and col in df.columns:
        mask = df[col].map(value_str).isin(others)
    else:
        mask = pd.Series(False, index=df.index)

    if any(is_null(v) for v in values):
        mask = mask | null_mask(df, col)
    return mask


def select_rows(df: pd.DataFrame, declared: dict, section: str = "hyperparameter"):
    """The rows matching a sweep entry, and the keys that failed to constrain
    them: `(mask, unconstrained)`.

    `declared` is one sweep entry, `{key: [values that key may take]}`. A row is
    selected when EVERY key holds one of its values. For example

        select_rows(df, {"a": [1, 2], "b": ["x"]})

    selects the rows whose `a` is 1 or 2 AND whose `b` is "x".

    The values are always a list, even when there is one. A hyperparameter's own
    value can be a list, so a bare one would be read as several alternatives:
    `{"a": [1, 2]}` means "1 or 2", and selecting the single list value `[1, 2]`
    takes `{"a": [[1, 2]]}`. Anything else raises rather than guess.

    `unconstrained` holds the keys `df` has no column for. Such a key cannot
    select anything, so it is skipped and named instead of silently selecting
    everything -- the entry then covers more rows than it declares, which the
    caller is expected to report. For example, if `df` has no `b` column:

        select_rows(df, {"a": [1], "b": ["x"]}) -> (rows whose a is 1, ["b"])

    A key declared null is the exception: an unrecorded key IS null, so it
    selects exactly the rows without that key and is not reported."""

    mask = pd.Series(True, index=df.index)
    unconstrained = []
    for key, values in declared.items():
        if not isinstance(values, (list, tuple)):
            raise TypeError(
                f"select_rows: {key} must map to a list of values, got {values!r}"
            )
        if (section, key) not in df.columns and not any(is_null(v) for v in values):
            unconstrained.append(key)
            continue
        mask &= _value_mask(df, (section, key), list(values))
    return mask, unconstrained


# -----------------------------------------------------------------------------
# ----- Plot config -----------------------------------------------------------
# -----------------------------------------------------------------------------

def apply_filters(
    frame: pd.DataFrame,
    pairs: list[str],
    flag: str,
    keep: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Keep (or drop) the rows matching `KEY=VALUE` pairs. The same key twice is
    an OR, different keys are an AND. Raises ValueError on a malformed pair or a
    key the frame does not have, for the caller to turn into whatever its command
    line does with a bad argument.

    Returns (frame, [messages]) -- one message per key, saying how many rows it
    took out, which the caller prints if it is being verbose."""

    filters = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"{flag} value '{pair}' must be in KEY=VALUE form")
        key, value = pair.split("=", 1)
        filters.setdefault(key, set()).add(value)

    messages = []
    for key, values in filters.items():
        col = ("hyperparameter", key)
        if col not in frame.columns:
            raise ValueError(
                f"{flag} key '{key}' not found among hyperparameter columns"
            )
        # The values are compared as value_str renders them, on both sides: what
        # the data holds as 0.0 is written 0, and `KEY=0.0` should still match it.
        mask = frame[col].map(value_str).isin({value_str(v) for v in values})
        before = len(frame)
        frame = frame[mask] if keep else frame[~mask]
        messages.append(f"{flag} {key}={sorted(values)}: {before} -> {len(frame)} rows")
    return frame, messages


PLOT_CONFIG_DIR = "configs/plots"


def load_plot_configs(name: str):
    """Every plot config `-p name` asks for, as [(name, path, contents)].

    A directory is a set of figures that belong together -- one sweep's, say --
    with a file per set inside it, and every one of them is drawn: `-p example`
    reads configs/plots/example/*.yaml. A single file is one such set.

    The name each comes back with is its path under PLOT_CONFIG_DIR without the
    extension, e.g. "example/default", which is what a caller writing one
    directory of figures per config should name it by -- otherwise every
    `default.yaml` in the repo writes over the same place."""

    for candidate in (
        name,
        os.path.join(PLOT_CONFIG_DIR, str(name)),
        os.path.join(PLOT_CONFIG_DIR, f"{name}.yaml"),
    ):
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            files = [path]
            break
        if path.is_dir():
            files = sorted(p for p in path.glob("*.yaml"))
            if not files:
                raise SystemExit(f"No .yaml in {path}, so there is nothing to plot.")
            break
    else:
        raise SystemExit(
            f"No plot config '{name}', and none in {PLOT_CONFIG_DIR} by that name."
        )

    out = []
    for path in files:
        try:
            stem = os.path.splitext(os.path.relpath(path, PLOT_CONFIG_DIR))[0]
        except ValueError:  # another drive: no path relative to that one
            stem = path.stem
        if stem.startswith(".."):
            stem = path.stem
        out.append((stem.replace(os.sep, "/"), str(path), read_plot_config(path)))
    return out


def read_plot_config(path, seen=()):
    """One plot config, with whatever it inherits from already under it.

        default: [main]      # or `defaults:`, or a bare name

    names configs in the same directory to start from, later ones winning over
    earlier ones and this file winning over all of them. The merge is at the top
    level only: a block this file declares REPLACES the inherited one rather than
    adding to it, so an ablation that lists its own `configs` draws those and not
    also the ones it inherited -- and it still gets the environments, statistics
    and layout of the config it started from."""

    path = Path(path)
    if str(path) in seen:
        raise SystemExit(f"Plot config {path} inherits from itself.")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # Both spellings are popped before either is read: `or` short-circuits, so
    # popping them inside one expression leaves the second key in `cfg` whenever
    # the first is present, and it merges in as a block of its own.
    declared = cfg.pop("default", None)
    declared_plural = cfg.pop("defaults", None)
    parents = declared or declared_plural or []
    if isinstance(parents, (str, os.PathLike)):
        parents = [parents]

    merged = {}
    for parent in parents:
        for candidate in (
            path.parent / f"{parent}.yaml",
            path.parent / str(parent),
            Path(PLOT_CONFIG_DIR) / f"{parent}.yaml",
        ):
            if candidate.is_file():
                merged.update(read_plot_config(candidate, (*seen, str(path))))
                break
        else:
            raise SystemExit(
                f"{path} inherits from '{parent}', which is not a config in "
                f"{path.parent}."
            )
    merged.update(cfg)
    return merged


# The key an entry names itself with; everything else in it selects runs. Of
# those, `algorithm` and `environment` hold a label assign_configs wrote rather
# than a recorded value, and are matched on their stem.
LABEL_KEY = "plot.label"
COLOR_KEY = "plot.color"
# Keys under this prefix are directives about how to draw an entry, not
# hyperparameters selecting its runs. A run records none of them, so one left in
# a selector would select nothing.
PLOT_PREFIX = "plot."
LABEL_KEYS = ("algorithm", "environment")


def plot_entries(plot_cfg: dict, with_directives: bool = False):
    """The configurations a plot config draws, in the order it writes them, as
    [(group name, label, selector)], or [(group name, label, selector,
    directives)] under `with_directives`.

        configs:
          <group>:
            - <hyperparameter>: <value>     # what selects the runs
              plot.label: "..."             # what to call them
              plot.color: "purple"          # what to draw it in

    A group is a family drawn in one hue; what a group is called does not reach
    the figure. An entry that names no hyperparameter would select every run, so
    it is dropped and reported rather than drawn.

    Every key under `plot.` is a directive rather than a hyperparameter, so it is
    taken out of the selector and returned separately, under the name it was
    written with. A run records none of them, and one left in the selector would
    select nothing."""

    entries = []
    for name, group in (plot_cfg.get("configs") or {}).items():
        for i, entry in enumerate(group or []):
            selector = dict(entry)
            directives = {
                key: selector.pop(key)
                for key in list(selector) if key.startswith(PLOT_PREFIX)
            }
            label = directives.get(LABEL_KEY)
            if not selector:
                print(
                    f"WARNING: config '{name}' entry {i + 1} names no "
                    f"hyperparameter, so it selects every run; skipping."
                )
                continue
            found = (
                name,
                label or ", ".join(f"{k}={value_str(v)}" for k, v in selector.items()),
                selector,
            )
            entries.append(found + (directives,) if with_directives else found)
    return entries


# Within a colour group: the first entry is a darkened palette colour, the last
# lands at MAX_BRIGHTEN, and the ones between are spaced evenly, so a group of two
# reads as clearly as a group of four.
DARKEN = -0.25
MAX_BRIGHTEN = 0.72
LINESTYLES = ["-", "--", ":", "-."]
# Bars carry the same distinction the linestyles carry in the curves, so a
# printout in grey still tells a group's entries apart.
HATCHES = ["", "///", "...", "xxx", "\\\\"]


def shade(color, amount):
    """Lighten (amount > 0) or darken (amount < 0) a colour and return it, by
    moving its lightness that fraction of the way to white or to black."""

    h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(color))
    l = min(1.0, l + amount * (1.0 - l)) if amount >= 0 else max(0.0, l + amount * l)
    return colorsys.hls_to_rgb(h, l, s)


def style_entries(plot_cfg: dict):
    """The entries a plot config draws, in the order it writes them, each with
    what to draw it in, as a list of dicts with:

        group      the colour group it belongs to
        selector   what the runs must hold, as {hyperparameter: value}
        label      what to call them
        color      the group's hue, shaded by the entry's position in it
        linestyle  varied within a group as well, so a printout in grey still reads
        hatch      the same distinction, for bars

    A group's entries share a hue; a group of one keeps the palette colour
    undarkened, since it has no siblings to reserve brightness for.

    Which hue a group gets is fixed by `color_scheme`, a list of group names in
    palette order -- so a group is the same colour whichever figure draws it, and
    however many others it is drawn beside. A group the list leaves out takes the
    next free index, in the order the config writes it.

    An entry's `plot.color` overrides the shade it would have been given,
    whatever its group and its position in it, for the entry that is not one of
    the family the rest of the group is. Anything matplotlib reads as a colour
    does: a name, a hex string, an (r, g, b) triple. Its linestyle and hatch are
    left alone, so it stays distinguishable in grey.

    Every script that puts several configurations on one axis reads its entries
    here, so a configuration is the same colour in all of them."""

    by_group = {}
    for name, label, selector, directives in plot_entries(
        plot_cfg, with_directives=True
    ):
        by_group.setdefault(name, []).append((label, selector, directives))

    hue = {name: i for i, name in enumerate(plot_cfg.get("color_scheme") or [])}
    for name in by_group:
        if name not in hue:
            hue[name] = len(hue)

    entries = []
    for name, group in by_group.items():
        base = PALETTE[hue[name] % len(PALETTE)]
        for i, (label, selector, directives) in enumerate(group):
            if len(group) == 1:
                color = base
            elif i == 0:
                color = shade(base, DARKEN)
            else:
                color = shade(shade(base, DARKEN), MAX_BRIGHTEN * i / (len(group) - 1))
            if directives.get(COLOR_KEY) is not None:
                try:
                    color = mc.to_rgb(directives[COLOR_KEY])
                except ValueError:
                    raise SystemExit(
                        f"'{label}' asks for {COLOR_KEY}="
                        f"{directives[COLOR_KEY]!r}, which matplotlib does not "
                        f"read as a colour."
                    )
            entries.append({
                "group": name,
                "selector": selector,
                "label": label,
                "color": color,
                "linestyle": LINESTYLES[i % len(LINESTYLES)],
                "hatch": HATCHES[i % len(HATCHES)],
            })
    return entries


def composed_mask(
    frame,
    flat,
    assigned: tuple = LABEL_KEYS,
    ignore: frozenset = frozenset(),
):
    """The rows of `frame` whose whole recorded configuration is `flat`, as a
    boolean mask.

    `flat` is a configuration as {dotted key: value}, composed from what an entry
    or a sweep names (see src.utils.id.composed_config). The comparison runs in
    both directions over the hyperparameter columns: a key `flat` does not hold
    is one the row must not record either, since a configuration is the whole of
    what it sets. `value_str` renders a null the same as an unrecorded key, so
    absent and null compare equal.

    `assigned` names the columns assign_configs wrote. Those are labels rather
    than recorded keys, and a composed configuration holds a config group as its
    contents rather than under the group's own name, so they take no part.

    `ignore` is the same exemption `assign_configs` takes, for keys a launch set
    for every run alike: composing gives them the value the YAMLs declare, which
    is not the value any run holds. A key exempt from one comparison and not the
    other would let a run be an environment and not be any configuration of it."""

    recorded = {c[1] for c in frame.columns if c[0] == "hyperparameter"}
    if any(
        k not in recorded and not is_null(v) and not exempt_key(k, ignore)
        for k, v in flat.items()
    ):
        # A key the configuration sets and no run recorded: no row is it.
        return pd.Series(False, index=frame.index)

    mask = pd.Series(True, index=frame.index)
    for col in frame.columns:
        if (
            col[0] != "hyperparameter"
            or col[1] in assigned
            or exempt_key(col[1], ignore)
        ):
            continue
        mask &= frame[col].map(value_str) == value_str(flat.get(col[1]))
    return mask


def entry_config(env: str, selector: dict, label: str) -> dict | None:
    """The whole configuration an entry composes to in `env`, as
    {dotted key: value}, or None when the entry pins another environment.

    An entry names part of a configuration; the rest comes from the config files
    it selects and from configs/default.yaml under them. The environment is
    composed in from the group being drawn, since one entry stands for one
    configuration per environment of its group."""

    # Imported here rather than at module level: src.utils.id imports this
    # module, so a module-level import would be a cycle, and src.utils.sweep
    # scans CONFIG_DIR as it is imported, which need not be readable for the rest
    # of this module to work.
    from src.utils.id import composed_config
    from src.utils.sweep import CONFIG_GROUPS, format_override

    overrides = dict(selector)
    named = overrides.get("environment")
    if named is not None and split_config_label(value_str(named))[0] != env:
        return None
    overrides["environment"] = env

    for key, value in overrides.items():
        if key in CONFIG_GROUPS and split_config_label(value_str(value))[1]:
            raise SystemExit(
                f"Entry '{label}' selects {key}={value_str(value)}, which names "
                f"a config file and the overrides applied on top of it. That is "
                f"a label a figure prints, not something Hydra can compose: name "
                f"the file and the overridden keys separately."
            )

    return composed_config(
        tuple(format_override(k, v) for k, v in sorted(overrides.items()))
    )


def composed_sweep_mask(
    df: pd.DataFrame,
    sweep: list,
    global_sweep: dict,
    assigned: tuple = LABEL_KEYS,
) -> tuple[pd.Series, str]:
    """The rows whose whole recorded configuration is one a sweep COMPOSES, as
    (mask, message).

    An entry declares part of a configuration; the rest comes from the config
    files it selects and from configs/default.yaml under them, so a run launched
    before any of those changed no longer belongs to the sweep even where every
    key the entry declares still agrees.

    Compared over every key the data records, in both directions: a key the
    composed config does not hold is one the run must not record either, since a
    configuration is the whole of what it sets. `value_str` renders a null the
    same as an unrecorded key, so absent and null compare equal, while a recorded
    value against an absent key does not.

    `assigned` names the columns assign_configs wrote. Those name the group a run
    was launched from, and a composed config holds a group as its contents rather
    than under the group's own name, so comparing them would fail every row. The
    contents are compared, which is what decides the config.

    The message says how the comparison was split, for a caller being verbose."""

    # Imported here rather than at module level, as in entry_config: the one is
    # an import cycle, the other scans CONFIG_DIR at import time.
    from src.utils.id import composed_config
    from src.utils.sweep import format_override

    recorded = {
        c[1]: c for c in df.columns
        if c[0] == "hyperparameter" and c[1] not in assigned
    }
    # value_str once per column, not once per configuration: the comparison below
    # runs for every combination the sweep declares.
    as_str = {key: df[col].map(value_str) for key, col in recorded.items()}
    # A column holding one value throughout cannot tell two runs apart, so it is
    # checked once per configuration as a scalar instead of as a mask over every
    # row. A configuration that disagrees on one of them matches no run, and the
    # check runs before any mask is built.
    constant = {
        key: series.iat[0]
        for key, series in as_str.items() if series.nunique() == 1
    }
    varying = {
        key: series for key, series in as_str.items() if key not in constant
    }

    composed = pd.Series(False, index=df.index)
    for entry in sweep:
        keys = list(global_sweep) + [k for k in entry if k not in global_sweep]
        declared = [entry.get(k, global_sweep.get(k)) for k in keys]
        for combo in itertools.product(*declared):
            hp = dict(zip(keys, combo))
            flat = composed_config(
                tuple(format_override(k, v) for k, v in hp.items())
            )
            # A key the composed config sets and the data has no column for is a
            # key no run recorded, so no run has this configuration. A null is
            # the exception: an absent column already means null everywhere.
            if any(k not in as_str and not is_null(v) for k, v in flat.items()):
                continue
            if any(value_str(flat.get(k)) != v for k, v in constant.items()):
                continue
            mask = pd.Series(True, index=df.index)
            for key, series in varying.items():
                mask &= series == value_str(flat.get(key))
            composed |= mask

    message = (
        f"Matching on {len(varying)} varying key(s), "
        f"{len(constant)} constant one(s) checked per configuration"
    )
    return composed, message


def env_groups(plot_cfg: dict):
    """Read a plot config's `environments` and return {group name: [environment]},
    which is what splits the figures: environments in one group share a figure,
    and each group gets its own.

    A group's environments are written either as a list, or as the mapping that
    also gives each of them a label and its y-axis limits (see plot_labels) --
    what an environment is called and which figure it lands in are the same
    question asked twice, so they are answered in one place."""

    raw = plot_cfg.get("environments") or {}
    if not raw:
        return {}
    groups = {}
    for name, entry in raw.items():
        if isinstance(entry, (list, tuple)):
            groups[name] = list(entry)
        elif isinstance(entry, dict) and not {"label", "ylim"} & set(entry):
            groups[name] = list(entry)
        else:
            # Not a group at all: one environment, at the top level, so the whole
            # block is a single unnamed group.
            return {"": list(raw)}
    return groups


def plot_labels(plot_cfg: dict):
    """Display labels and y-axis limits from a plot config.

        statistics:  {name: label}
        environments: {name: label} or {name: {label:, ylim: {statistic: [lo, hi]}}},
                      optionally nested one level under a group name

    Returns (statistics, environments, ylims), the first two mapping a name to its
    label and the last mapping (environment, statistic) to a pair of bounds. A
    config declaring neither gives three empty dicts: labelling is optional, and a
    name with no entry is shown as it is written in the data.

    A figure is labelled by the config that draws it. The same statistic can be
    "Test Return" in one figure and "return" in another, and a name that has to be
    the same everywhere is the one already in the data -- which is what the
    interactive pages show, having no plot config to read.

    Algorithms are not labelled here. What names a configuration is the plot
    config's `plot.label`, the sweep entry it came from, or the hyperparameters
    that tell it apart from its neighbours -- none of which a name-to-label map
    can reach."""

    cfg = plot_cfg or {}
    environments, ylims = {}, {}

    def add_env(name, entry):
        if isinstance(entry, dict):
            environments[name] = entry.get("label", name)
            for stat, bounds in (entry.get("ylim") or {}).items():
                ylims[(name, stat)] = tuple(bounds)
        else:
            environments[name] = entry

    for name, entry in (cfg.get("environments") or {}).items():
        # A mapping is a group of environments unless it describes one.
        if isinstance(entry, dict) and not {"label", "ylim"} & set(entry):
            for env, env_entry in entry.items():
                add_env(env, env_entry)
        else:
            add_env(name, entry)

    return dict(cfg.get("statistics") or {}), environments, ylims


# -----------------------------------------------------------------------------
# ----- Missing runs ----------------------------------------------------------
# -----------------------------------------------------------------------------

def write_missing_runs_script(
    folder: str,
    missing_runs_data: dict,
    submitit_module: str = "submit_jobs_slurm",
    filename: str = "submit_missing_runs.py",
) -> tuple[str, int]:
    """Write a script relaunching every missing run through
    `{submitit_module}.submit_jobs(...)` -- all cluster settings stay in that
    module, picked up on import. Returns (path, n_jobs). With nothing missing the
    script is still written, with an empty `missing_jobs`, so the file always
    reflects the run that produced it rather than a stale earlier one.

    `missing_runs_data` is `{combo_key: (hp_overrides, [seeds])}` and submit_jobs
    takes `(hp_overrides, seeds)` pairs, so the overrides are passed through as
    they are -- `environment` and `algorithm` are ordinary keys among them.

    The command line of the written script is read off submit_jobs' own
    signature: every parameter becomes an option passed straight back, keeping
    its default, or required when it has none. Change that signature and the next
    script picks the change up. Raises if the module or the function is not
    there, rather than write a script that cannot pass on the options."""

    jobs = [
        (
            {k: native(v) for k, v in hp_overrides.items()},
            sorted({native(s) for s in seeds}),
        )
        for hp_overrides, seeds in missing_runs_data.values()
        if seeds
    ]

    source = Path(submitit_module.replace(".", "/") + ".py")
    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found: the script's command line is read off submit_jobs()"
        )
    func = next(
        (
            node
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef) and node.name == "submit_jobs"
        ),
        None,
    )
    if func is None:
        raise AttributeError(f"no submit_jobs() in {source}")

    options, passed = [], []
    spec = func.args
    # Every parameter but the jobs themselves, paired with its default or None
    # where it has none.
    padding = [None] * (len(spec.args) - len(spec.defaults))
    named = list(zip(spec.args, padding + list(spec.defaults)))[1:]
    named += list(zip(spec.kwonlyargs, spec.kw_defaults))

    for arg, node in named:
        if node is None:
            options.append(f'parser.add_argument("--{arg.arg}", required=True)')
            passed.append(f"{arg.arg}=args.{arg.arg}")
            continue
        try:
            default = ast.literal_eval(node)
        except ValueError:
            continue
        if isinstance(default, bool):
            action = "store_false" if default else "store_true"
            options.append(f'parser.add_argument("--{arg.arg}", action="{action}")')
        elif isinstance(default, (int, float, str)):
            options.append(
                f'parser.add_argument("--{arg.arg}", type={type(default).__name__}, '
                f"default={default!r})"
            )
        else:
            options.append(f'parser.add_argument("--{arg.arg}", default={default!r})')
        passed.append(f"{arg.arg}=args.{arg.arg}")

    job_lines = "[]"
    if jobs:
        job_lines = "[\n"
        for hp_overrides, seeds in jobs:
            job_lines += f"    ({hp_overrides!r}, {seeds!r}),\n"
        job_lines += "]"

    call = ", ".join(["missing_jobs"] + passed)
    script = (
        "# Auto-generated -- relaunch missing runs.\n"
        f"# Delegates to {submitit_module}.submit_jobs(...) so all cluster\n"
        "# settings stay in one place.\n"
        "import argparse\n"
        f"from {submitit_module} import submit_jobs\n"
        "\n"
        "# (hp_overrides_dict, [missing_seeds]) -- the shape submit_jobs expects.\n"
        f"missing_jobs = {job_lines}\n"
        "\n"
        "parser = argparse.ArgumentParser()\n"
        + "".join(f"{line}\n" for line in options)
        + "args = parser.parse_args()\n"
        "\n"
        f"submit_jobs({call})\n"
    )
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(script)
    return path, len(jobs)


def launch_overrides(groups: dict, hp: dict) -> dict:
    """What to relaunch a configuration with: its hyperparameters plus the config
    groups it came from.

    `groups` maps a Hydra group name to the label assign_configs produced for it,
    e.g. {"environment": "lunar_lander (train_from_pixels=True)"}. A label carries
    the keys that did not match the group's YAML, and submit_jobs wants the group
    name it can pass as `environment=` with those keys as ordinary overrides
    beside it, so the labels are split here. A group whose value is None is left
    out rather than guessed at. `hp` wins over anything a label contributed."""

    overrides = {}
    for group, label in groups.items():
        if label is None:
            continue
        stem, label_overrides = split_config_label(str(label))
        overrides[group] = stem
        overrides.update(label_overrides)
    return {**overrides, **hp}


def record_missing(
    missing_runs: dict,
    overrides: dict,
    found: list[int],
    expected: int,
) -> bool:
    """Note the seeds a configuration is short of in `missing_runs`, the dict
    write_missing_runs_script writes out. True when something was missing.

    `overrides` is every key that identifies the configuration, `environment` and
    `algorithm` among them if they are known -- submit_jobs takes them as ordinary
    overrides, so nothing here treats them apart."""

    missing = [s for s in range(expected) if s not in found]
    if not missing:
        return False
    overrides = {k: native(v) for k, v in overrides.items()}
    key = tuple(sorted((k, value_str(v)) for k, v in overrides.items()))
    _, seeds = missing_runs.setdefault(key, (overrides, []))
    seeds.extend(s for s in missing if s not in seeds)
    return True


# -----------------------------------------------------------------------------
# ----- Style -----------------------------------------------------------------
# -----------------------------------------------------------------------------

PALETTE = [
    *[c.upper() for c in sns.color_palette("colorblind").as_hex()],
    "#000000",
]

_FONT_NAME = "Libertinus Serif"
_FONT_URL = "https://github.com/alerque/libertinus/releases/download/v7.051/Libertinus-7.051.zip"


def ensure_font():
    """Download the serif font if not on disk, register its OTFs with matplotlib,
    set rcParams."""

    if os.name == "nt":
        font_dir = Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "Windows" / "Fonts"
    else:
        font_dir = Path.home() / ".local" / "share" / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)

    existing = list(font_dir.glob("LibertinusSerif-*.otf"))
    if not existing:
        print(f"Downloading {_FONT_NAME}...")
        try:
            with urllib.request.urlopen(_FONT_URL, timeout=5) as resp:
                data = resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"Skipping {_FONT_NAME} download (no connection): {e}")
            return
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if name.endswith(".otf") and "static/OTF/" in name and "Serif" in name:
                    target = font_dir / Path(name).name
                    target.write_bytes(z.read(name))
                    existing.append(target)
        print(f"Installed to {font_dir}")

    # addfont does not persist across sessions.
    for path in existing:
        font_manager.fontManager.addfont(str(path))

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [_FONT_NAME, "DejaVu Serif"]


def detect_precision(values, min_prec=2, max_prec=6, sig_figs=2):
    """Decimal places for a set of numbers: distinct values must format to
    distinct strings, small numbers keep enough significant digits, integers use
    fewer decimals."""

    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not valid:
        return min_prec

    arr = np.array(valid)
    abs_vals = np.abs(arr[arr != 0])

    unique_vals = np.sort(np.unique(arr))
    if len(unique_vals) > 1:
        diffs = np.diff(unique_vals)
        min_diff = np.min(diffs[diffs > 0]) if np.any(diffs > 0) else None
    else:
        min_diff = None

    prec = min_prec

    if min_diff is not None and min_diff > 0 and min_diff < 1.0:
        lz = int(np.floor(-np.log10(min_diff)))
        prec = max(prec, lz + sig_figs)

    if len(abs_vals) > 0:
        min_val = np.min(abs_vals)
        if min_val < 1.0:
            lz = int(np.floor(-np.log10(min_val)))
            prec = max(prec, lz + sig_figs)

    return max(min_prec, min(max_prec, prec))


def set_3_ticks(ax, which="both"):
    """Enforce three ticks on an axis."""

    if which in ("y", "both"):
        ylo, yhi = ax.get_ylim()
        ticks = [ylo, (ylo + yhi) / 2, yhi]
        ax.set_yticks(ticks)
        ax.set_ylim(ylo, yhi)
        prec = detect_precision(ticks, min_prec=2, max_prec=6)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(f"%.{prec}f"))
        ax.tick_params(axis="y", pad=0)
        yticks = ax.yaxis.get_major_ticks()
        if len(yticks) >= 1:
            yticks[0].label1.set_verticalalignment("bottom")
        if len(yticks) >= 3:
            yticks[-1].label1.set_verticalalignment("top")
    if which in ("x", "both"):
        xlo, xhi = ax.get_xlim()
        ticks = [xlo, (xlo + xhi) / 2, xhi]
        ax.set_xticks(ticks)
        prec_x = detect_precision(ticks, min_prec=0, max_prec=4)
        if prec_x > 0:
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter(f"%.{prec_x}f"))
        xticks = ax.xaxis.get_major_ticks()
        if len(xticks) >= 1:
            xticks[0].label1.set_horizontalalignment("left")
        if len(xticks) >= 3:
            xticks[-1].label1.set_horizontalalignment("right")


# -----------------------------------------------------------------------------
# ----- Naming ----------------------------------------------------------------
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _latex_available():
    """Whether a real LaTeX installation is on PATH. Some labels use LaTeX-only
    macros (e.g. `\\displaystyle`) that mathtext cannot parse, so machines without
    LaTeX need them stripped rather than usetex just disabled."""

    return shutil.which("latex") is not None


def tex_kwargs(label):
    """matplotlib kwargs enabling LaTeX only when the label contains `$...$` math
    and LaTeX is installed. Without it, mathtext renders the math instead."""

    return (
        {"usetex": True}
        if isinstance(label, str) and "$" in label and _latex_available()
        else {}
    )


def save_figure(fig, folder, name, dpi=300):
    """Save `fig` as `folder/name.png` and return the path to it.

    The name is sanitized with pathvalidate, which strips the characters a
    filename cannot hold."""

    name = sanitize_filename(name, platform="auto")
    path = os.path.join(folder, name + ".png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
    return path


def hyperparameter_to_label(keys) -> dict:
    """Give every hyperparameter the shortest name that still tells it apart, and
    return them as {key: label}.

    A hyperparameter is named by its last dotted segment, and by as many trailing
    segments as it takes when that is not unique -- only the keys that clash grow,
    and they grow until nothing clashes. For example, `a.b.lr` and `a.c.lr` become
    "b.lr" and "c.lr" while a lone `a.b.gamma` stays "gamma". Accepts dotted
    strings or (section, key) column tuples and keys the result on what it was
    given."""

    dotted = {k: (k if isinstance(k, str) else k[1]) for k in keys}
    depth = {k: 1 for k in keys}
    while True:
        labels = {k: ".".join(dotted[k].split(".")[-depth[k]:]) for k in keys}
        taken = list(labels.values())
        grown = False
        for k, label in labels.items():
            if taken.count(label) > 1 and depth[k] < len(dotted[k].split(".")):
                depth[k] += 1
                grown = True
        if not grown:
            return labels


# -----------------------------------------------------------------------------
# ----- Memory ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def memory_fractions(step: float) -> list:
    """The fractions of the replay memory the visits are counted up to, one
    checkpoint per `step` and the whole memory last.

    Plain floats: a checkpoint keys the maps computed for it, prints as itself,
    and reaches a Vega-Lite spec through json.dumps, which has no encoder for the
    numpy scalars np.arange yields.

    Raises ValueError on a step outside (0, 1], for the caller to report against
    the option it read the step from: which flag that is belongs to the script,
    not here."""

    if not 0 < step <= 1:
        raise ValueError(f"must be in (0, 1], got {step}")
    fractions = [
        round(float(f), 10)
        for f in np.arange(step, 1.0 + 1e-9, step)
    ]
    if fractions[-1] < 1.0:
        fractions.append(1.0)
    return fractions


# -----------------------------------------------------------------------------
# ----- Curves ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def smooth(arr, span):
    """Average `arr` over a window of `span` points either side and return it, the
    same length as it came in.

    The ends have no window to average over, so each of the first and last `span`
    points averages what there is -- a shrinking window rather than a padded one,
    which would pull the curve towards zero exactly where it starts and ends. A
    span of 0 leaves the array as it is, and one as wide as the array leaves every
    point at the mean of the whole of it."""

    arr = np.asarray(arr, dtype=float)
    return np.array([
        arr[max(0, i - span):i + span + 1].mean()
        for i in range(len(arr))
    ])


# Sample std (ddof=1), as a confidence interval over a sample of seeds requires.
# A constant because every band in every script goes through ci_bounds, which
# keeps them all at one definition.
CI_DDOF = 1
# The normal quantile, which is the t quantile in the limit. `ci_bounds` takes it
# when a caller passes it as `z`; the default multiplier is the t quantile
# `_t_quantile` returns, which holds at its last tabulated entry.
CI_Z = 1.96

# Two-sided 95% Student-t quantiles by degrees of freedom (seeds - 1). A band
# around a mean of a HANDFUL of seeds is a t interval, not a normal one: the
# standard deviation is estimated from the same few runs as the mean, and the
# normal quantile ignores that. At five seeds the difference is 2.776 against
# 1.96, i.e. a band reported 40% too narrow.
#
# Tabulated rather than computed: the inverse of the t distribution needs an
# incomplete beta, which is more numerics than a plotting module should carry,
# and the table is exact where it has an entry. Between the sparse entries at
# the end, and beyond the last of them, see _t_quantile.
_T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    40: 2.021, 60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980,
}
_T_95_DF = sorted(_T_95)

# The table as np.interp wants it, built once: x ascending in 1/dof, y the
# quantile at each. 1/dof increases as dof decreases, so both are reversed.
# ci_bounds runs once per cell per statistic, and these are constants.
_T_95_X = (1.0 / np.array(_T_95_DF, dtype=float))[::-1]
_T_95_Y = np.array([_T_95[d] for d in _T_95_DF], dtype=float)[::-1]


def _t_quantile(dof):
    """The two-sided 95% Student-t quantile at `dof` degrees of freedom, as an
    array the same shape as `dof`.

    Interpolated in 1/dof between the tabulated entries, which is the variable
    the quantile is nearly linear in. Beyond the last of them it is held at that
    entry rather than dropped to CI_Z: the quantile approaches CI_Z from above,
    so one seed past the table would otherwise narrow the band by a step of a
    hundredth. A dof of 0 (one seed, so no spread to estimate) takes the first
    entry; the half-width there is 0 whatever it is multiplied by, since the
    standard deviation is NaN.

    Both ends clamp, which is what np.interp does with no `left`/`right`: below
    dof=1 the first entry, above the last the last one. `1/max(dof, 1)` already
    caps x at the table's top end, so only the low end is ever extrapolated."""

    return np.interp(
        1.0 / np.maximum(np.asarray(dof, dtype=float), 1.0), _T_95_X, _T_95_Y
    )


def ci_bounds(data, ddof=CI_DDOF, z=None):
    """Reduce seeds to a mean and a confidence band, returning (mean, half-width,
    n_alive) along axis 0 and ignoring NaN.

    `n_alive` is the number of seeds with a value at each x, and it is what the
    half-width divides by. Using the total seed count NARROWS the band exactly
    where the data thins out: a seed that died early leaves NaN in its tail, so
    the tail averages fewer seeds but was reported as if it had them all. Where
    one seed survives the std is undefined and the half-width is 0, so callers
    should show n_alive alongside the band.

    The multiplier is the two-sided 95% t quantile at `n_alive - 1` degrees of
    freedom, taken per x for the same reason the denominator is: the tail of a
    run that lost seeds is an interval over fewer of them. Pass `z` to fix one
    multiplier instead, e.g. CI_Z for the normal interval."""

    data = np.asarray(data, dtype=float)
    alive = np.sum(~np.isnan(data), axis=0)
    multiplier = _t_quantile(alive - 1) if z is None else z
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0, ddof=ddof)
        half = multiplier * std / np.sqrt(np.maximum(alive, 1))
    return mean, np.nan_to_num(half, nan=0.0, posinf=0.0), alive


def summarize(values):
    """Reduce the seeds of one configuration to (mean, half-width of its 95%
    interval), or None when no seed produced a number."""

    values = [v for v in values if np.isfinite(v)]
    if not values:
        return None
    mean, half, _ = ci_bounds(np.asarray(values, dtype=float).reshape(-1, 1))
    return float(mean[0]), float(half[0])


# Extra room between two adjacent bars from different colour groups, so a group of
# entries reads as one block, and the gap between two of the same group. The slot
# is widened with the bars so the group gap survives a wider bar.
GROUP_GAP_BAR = 0.18
BASE_BAR_GAP = 0.03


def bar_positions(drawn, bar_width):
    """Return the x of each bar in `drawn`, left to right.

    A wider gap goes wherever two neighbours belong to different colour groups, so
    that a group of entries reads as one block."""

    slot = bar_width + BASE_BAR_GAP
    xs = [0.0]
    for prev, cur in zip(drawn, drawn[1:]):
        gap = slot + (GROUP_GAP_BAR if cur["group"] != prev["group"] else 0.0)
        xs.append(xs[-1] + gap)
    return xs


def draw_bars(
    ax,
    items,
    ylabel,
    bar_width,
    font_size=12,
    invert_y=False,
    annotate=False,
):
    """Draw one bar per (entry, summary) of `items`, in that order.

    A summary of None keeps its slot empty, so the same configuration is in the
    same place in every subplot.

    `invert_y` turns the axis over, and the bars keep their lengths: a bar stays
    proportional to its value, so on an inverted axis the shorter bar is still
    the smaller value."""

    xs = bar_positions([entry for entry, _ in items], bar_width)
    # Every bar stands from one floor, so a negative value stands beside its
    # neighbours and the comparison stays between the bars. The floor sits below
    # the lowest interval, so the shortest bar still has a foot to be seen by.
    lows = [m - e for _, v in items if v is not None for m, e in [v] if np.isfinite(m)]
    floor = min(lows) - 0.1 * abs(min(lows)) if lows else 0.0

    means, tops = [], []
    for x, (entry, value) in zip(xs, items):
        if value is None:
            continue
        mean, err = value
        ax.bar(
            x,
            mean - floor,
            bottom=floor,
            width=bar_width,
            yerr=err,
            color=entry["color"],
            hatch=entry["hatch"],
            edgecolor="black",
            linewidth=0.6,
            capsize=2,
            error_kw={"linewidth": 0.8, "ecolor": "black"},
        )
        means.append(mean)
        if annotate and np.isfinite(mean):
            tops.append((x, mean + (err if np.isfinite(err) else 0.0), mean))

    if xs:
        slot = bar_width + BASE_BAR_GAP
        ax.set_xlim(xs[0] - slot / 2, xs[-1] + slot / 2)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=font_size - 2, pad=1)
    ax.set_ylabel(ylabel, fontsize=font_size, **tex_kwargs(ylabel))
    if means:
        ax.set_ylim(bottom=floor)
    if annotate and tops:
        # Headroom above the tallest bar, or its label is clipped by the frame.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.08 * (hi - lo))
        offset = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        prec = detect_precision([m for _, _, m in tops], min_prec=2, max_prec=6)
        for x, top, mean in tops:
            ax.text(
                x,
                top + offset,
                f"{mean:.{prec}f}",
                ha="center",
                va="bottom",
                fontsize=font_size - 4,
                clip_on=False,
            )
    if invert_y:
        ax.invert_yaxis()


def error_shade_plot(ax, data, stepsize, smoothing_window=0, x_start=0, **kwargs):
    """Plot the mean across seeds with a shaded 95% confidence interval. Takes
    standard pyplot.plot arguments (color, linestyle, linewidth, label, ...)."""

    y, error, _alive = ci_bounds(data)
    x = [x_start + stepsize * step for step in range(len(y))]
    if smoothing_window > 0:
        y = smooth(y, smoothing_window)
        error = smooth(error, smoothing_window)

    (line,) = ax.plot(x, y, **kwargs)
    ax.fill_between(
        x,
        y - error,
        y + error,
        alpha=0.2,
        linewidth=0.0,
        color=line.get_color(),
    )
    return (line,)


# -----------------------------------------------------------------------------
# ----- Grid ------------------------------------------------------------------
# -----------------------------------------------------------------------------
#
# A figure is a stack of blocks; a block is a grid of cells; a cell holds the runs
# that share one combination of the hyperparameters its block puts on rows and
# columns. What goes in a cell is the caller's business -- curves in
# interactive_curves.py, one heatmap in interactive_heatmaps.py -- and so is the
# policy that decides which hyperparameters become axes, since a curve can carry
# a hyperparameter that a heatmap cannot. What is here is the vocabulary both
# need: the block, the values an axis takes, the mask that finds a cell's rows,
# and the text that names it.
#
# Values are compared through `value_str` throughout, so 64 and 64.0 are one
# value and a list-valued hyperparameter can be compared at all. `series_of`
# closes the section: it is the step from the rows a cell holds to the curves
# they decode into, which every caller needs before it can draw anything.


@dataclass
class Block:
    """One stack of subplots: one sweep entry's runs.

    `col_params` and `row_params` are the hyperparameter columns the grid is laid
    out by, `col_combos` and `row_combos` the values each takes, in the order they
    are drawn. `index` names the block for a caller that builds one figure per
    environment and lays them over each other afterwards: a sweep entry with no
    runs in one environment leaves no block there, so a block's position in the
    list is not the same thing in two figures and cannot be what they are matched
    by."""

    label: str
    df: pd.DataFrame
    col_params: list
    row_params: list
    col_combos: list
    row_combos: list
    index: int = 0


def make_block(label, frame, axes, index=0):
    """Assign axes alternately to nested column and row levels, outermost first,
    and return the Block they make.

    `axes` is [(hyperparameter column, its values)] in the order they should
    nest."""

    cols, rows = axes[0::2], axes[1::2]
    return Block(
        label=label,
        df=frame,
        col_params=[c for c, _ in cols],
        row_params=[c for c, _ in rows],
        col_combos=list(itertools.product(*[v for _, v in cols])),
        row_combos=list(itertools.product(*[v for _, v in rows])),
        index=index,
    )


def iter_cells(block):
    """Walk the grid, yielding (row index, row values, column index, column
    values, rows) for every cell -- including the cells no run landed in, which
    are what a declared configuration that never ran looks like.

    One mask per row and one per column, intersected per cell: a column's mask is
    the same under every row, and building it again for each of them would cost a
    pass over the frame -- a whole column scan where the value is null, see
    null_mask -- for a result already in hand."""

    col_masks = [
        mask_of(block.df, block.col_params, col_vals)
        for col_vals in block.col_combos
    ]
    for r, row_vals in enumerate(block.row_combos):
        row_mask = mask_of(block.df, block.row_params, row_vals)
        for c, col_vals in enumerate(block.col_combos):
            cell = block.df[row_mask & col_masks[c]]
            yield r, row_vals, c, col_vals, cell


def mask_of(frame, params, values):
    """Return the mask of the rows whose `params` hold `values`.

    A column the data never recorded holds null for every row, so it matches a
    declared null and nothing else -- which is what makes a declared value with no
    runs an empty cell instead of a crash. A null goes through `null_mask`, so a
    value and a cell of the grid read a null the same way, subtree and all."""

    mask = pd.Series(True, index=frame.index)
    for p, v in zip(params, values):
        if is_null(v):
            mask &= null_mask(frame, p)
        elif p in frame.columns:
            mask &= frame[p].map(value_str) == value_str(v)
        else:
            mask &= pd.Series(False, index=frame.index)
    return mask


def values_of(frame, col, declared=None):
    """Return the values an axis takes.

    With a sweep the axis is what the sweep DECLARES, in the order it declares
    them: a declared value no run produced keeps its row or column and draws an
    empty cell. A hole is a result -- it says that configuration has no data, and
    dropping it silently renumbers the grid instead. Values found in the data but
    not declared are appended after. Without a sweep the axis is whatever the data
    holds, sorted. Nulls are config values either way, not missing data."""

    seen = {}
    if col in frame.columns:
        for v in frame[col]:
            seen.setdefault(value_str(v), v)
    values = list(seen.values())
    # Numeric order where every value is a scalar, textual order otherwise. A
    # list-valued hyperparameter must not reach float(): a one-element array
    # converts (deprecated, and it would order lists by their first element),
    # a longer one raises.
    scalar = not any(isinstance(v, (list, tuple, np.ndarray)) for v in values)

    def numeric(v):
        """Sort key placing the nulls after the numbers.

        A null never reaches float(): NaN compares false against every value, so
        a list holding one comes back in no order at all rather than in a wrong
        one, and the string spellings of a null convert or raise by spelling."""

        return (1, 0.0) if is_null(v) else (0, float(v))

    try:
        present = (
            sorted(values, key=numeric) if scalar
            else sorted(values, key=value_str)
        )
    except (TypeError, ValueError):
        present = sorted(values, key=value_str)
    if not declared:
        return present
    rest = {value_str(v): v for v in present}
    for d in declared:
        rest.pop(value_str(d), None)
    return list(declared) + list(rest.values())


def varies(frame, col):
    """Return whether `col` holds more than one distinct value in `frame`.

    Compared through value_str, so 64 and 64.0 count once and list-valued
    hyperparameters can be counted at all -- the numpy arrays they arrive as are
    unhashable. A column the frame does not have is null for every row, which is
    one value, not several."""

    return col in frame.columns and frame[col].map(value_str).nunique() > 1


def label_of(labels, params, values):
    """Return one point on the grid as text, for example
    `learning_rate=0.001, optimizer=adam`.

    `labels` names the hyperparameters (see plot.hyperparameter_to_label),
    `params` are their columns and `values` the ones a cell or a curve holds for
    them, paired in order. Empty for a block with no axis of that kind, which the
    caller may need to turn into something non-empty."""

    return ", ".join(f"{labels[p[1]]}={value_str(v)}" for p, v in zip(params, values))


def vega_field_name(labels, key):
    """Give a hyperparameter a name a Vega-Lite expression can read it under, and
    return it.

    That is its label with the dots taken out: in a Vega-Lite field string a dot
    is a nested-object accessor, and the same name is interpolated into both
    `datum.<field>` and a `p_<field>` param, where a dot would be a syntax error.
    `labels` is the map hyperparameter_to_label built."""

    return labels[key].replace(".", "_")


# The columns a cell is decoded from: where a run's seed is recorded, and the two
# hyperparameters that turn a point index into a training step.
SEED = ("seed", "experiment.rng_seed")
STEPS = ("hyperparameter", "experiment.training_steps")
POINTS = ("hyperparameter", "experiment.testing_points")

# The columns every script that reads a run needs by name. `environment` and
# `algorithm` hold the label assign_configs wrote rather than anything a run
# recorded; `environment.id` is what a run records when there are no YAMLs to
# match it against. `time` is one number per run rather than a curve, which is
# why a caller reads it off the column instead of going through series_of.
ENV = ("hyperparameter", "environment")
ENV_ID = ("hyperparameter", "environment.id")
ALGO = ("hyperparameter", "algorithm")
TIME = ("statistic", "time")


def assign_groups(
    df,
    env_configs,
    algo_configs,
    ignored,
    env_id_fallback=True,
    algo_optional=True,
    algo_decides_usable=True,
    cache=None,
):
    """Put the environment and algorithm each run was launched from on `df` under
    `ignored`, and return (environment, algorithm, environment stems, usable rows).

    A key a launch set for every run alike says nothing about which configuration
    a run is, and the YAML describes one none of them were run at, so `ignored`
    exempts it. Everything else must match exactly; a run matching no YAML of a
    group is "unknown" there.

    `ignored` decides the whole assignment, so one set of exemptions produces one
    answer and a caller holding several must ask once per set: exempting a key
    changes which YAML a run matches. Pass a dict as `cache` to keep the answers
    by `ignored`, since asking again with the same set compares the frame against
    every YAML a second time to reach the same result.

    Two flags decide what a group with no YAMLs at all means, which is a question
    only the caller can answer -- whether a group it was given nothing for is a
    group it does not care about, or one it cannot proceed without.

    `env_id_fallback` lets the id a run recorded stand in for an environment
    label, since it still tells two environments apart. False labels every run
    "unknown", so none of them is usable.

    `algo_optional` drops the algorithm from the question: the returned algorithm
    is None, `df[ALGO]` is left alone, and the usable rows are decided by the
    environment alone. False labels every run "unknown" there instead, so again
    none is usable.

    The usable rows are the runs placed in a config of every group that had YAMLs,
    as a boolean mask. Returned rather than left on the frame because which rows
    those are depends on `ignored`, so it belongs with the exemptions it came
    from: a mask built under one set says nothing about a comparison made under
    another. A run outside it is not drawn -- its recorded keys would still let an
    entry select it, and it would then be drawn as a configuration it is not.

    `algo_decides_usable` False keeps the algorithm label -- it is still assigned
    and still put on the frame -- but stops it deciding the usable rows, which are
    then the runs placed in an environment alone. For a caller selecting runs by a
    whole composed configuration rather than by a filter over recorded keys: the
    comparison it makes already holds every key an algorithm YAML declares, and
    holds it at the value the configuration composes to. A YAML declaring a
    default a sweep overrode agrees with no run it launched, which makes every one
    of them "unknown" and drawable by nothing -- a configuration cannot be both
    the one an entry composes and no configuration at all."""

    # The flags that decide what goes into the cache are part of its key.
    # `algo_decides_usable` is applied below, to whatever comes back, so it stays
    # out of it.
    key = (ignored, bool(env_id_fallback), bool(algo_optional))
    if cache is None or key not in cache:
        if env_configs:
            env = assign_configs(df, env_configs, ignored)
        elif env_id_fallback and ENV_ID in df.columns:
            env = df[ENV_ID].map(value_str)
        else:
            env = pd.Series("unknown", index=df.index)
        if algo_configs:
            algo = assign_configs(
                df,
                algo_configs,
                ignored,
                overridden_by=(env, env_configs),
            )
        elif algo_optional:
            algo = None
        else:
            algo = pd.Series("unknown", index=df.index)
        if cache is None:
            cache = {}
        cache[key] = (env, algo)

    env, algo = cache[key]
    df[ENV] = env
    matched = env != "unknown"
    if algo is not None:
        df[ALGO] = algo
        if algo_decides_usable:
            matched &= algo != "unknown"
    env_stem = env.map(lambda v: split_config_label(str(v))[0])
    return env, algo, env_stem, matched


def rows_cols(spec, n):
    """Turn a config's `[rows, cols]` into a real arrangement for `n` items, where
    -1 means "as many as it takes". Defaults to one row.

    Resolution only: whether the arrangement has room for the items is the
    caller's question, since only some of them fill cells by index. See
    fitted_rows_cols."""

    rows, cols = (spec or [1, -1])[:2]
    if rows == -1 and cols == -1:
        rows, cols = 1, n
    elif rows == -1:
        rows = -(-n // cols)
    elif cols == -1:
        cols = -(-n // rows)
    return int(rows), int(cols)


def fitted_rows_cols(config_name, key, n, spec):
    """`rows_cols`, refusing an arrangement with fewer cells than items.

    For the grids whose cells are filled by index: the surplus items have nowhere
    to go, and the figure would fail partway through with an index error naming
    neither the config nor the key.

    `config_name` names the config in the message. `key` names the option and
    `spec` is its value, passed separately because a key nested under a block is
    not one a config declares at the top level, so no single lookup would reach
    every one of them.

    A legend takes plain `rows_cols` instead: matplotlib's `ncol` wraps the
    surplus handles into further rows by itself, so there is no cell to overflow
    and an arrangement too small for the entries is not an error."""

    rows, cols = rows_cols(spec, n)
    if rows * cols < n:
        raise SystemExit(
            f"{config_name} asks for {key}: [{rows}, {cols}], which is "
            f"{rows * cols} cell(s) for {n} item(s). Give it enough cells, or "
            f"-1 for a dimension to be filled in."
        )
    return rows, cols


# The cell disagreements series_of has already reported, so that a sweep where
# every cell holds runs of two training lengths says so once rather than once per
# cell per statistic.
_MIXED_AXIS_WARNED: set = set()


def series_of(frame, stat):
    """Decode the runs of one cell into a curve per seed and return
    (values, stepsize, training steps, seed ids), or None when there is nothing
    plottable.

    `values` is a DataFrame with one column per seed and one row per recorded
    point, `stepsize` the training steps between two points, and `seed ids` names
    the columns in order. A statistic is stored as a JSON array per run, so this
    is the step between what a cell holds and what can be drawn from it: whatever
    goes in a cell -- a mean and its band, one seed's line, an area -- is computed
    from these columns.

    Returns None rather than raising when the statistic is not a curve at all: a
    device name, a git hash, a job id and a single number are all recorded the
    same way. Seeds are deduplicated first, since a (configuration, seed)
    recorded twice would add a column that is not independent and narrow the
    confidence band for it; a seed whose curve holds no finite value is dropped
    for the same reason it cannot be drawn."""

    stat_col = ("statistic", stat)
    if any(c not in frame.columns for c in (stat_col, STEPS, POINTS)):
        return None
    # Read before the deduplication below, which reorders the frame: these say
    # what the cell was run at, and taking them from whichever row ends up first
    # afterwards would scale the x axis by another run's training length.
    recorded_steps = frame[STEPS].dropna()
    recorded_points = frame[POINTS].dropna()
    if SEED in frame.columns:
        # Only among the rows that record one: drop_duplicates counts two NaNs as
        # the same value, so runs recording no seed would collapse into one --
        # which is a run dropped from the band rather than a duplicate removed
        # from it.
        seeded = frame[SEED].notna()
        frame = pd.concat([
            frame[seeded].drop_duplicates(subset=[SEED]),
            frame[~seeded],
        ])
    valid = frame[stat_col].notna()
    raw = frame.loc[valid, stat_col]
    seeds = (
        frame.loc[valid, SEED].to_list()
        if SEED in frame.columns
        else [None] * len(raw)
    )
    try:
        arrays = [json.loads(a) for a in raw]
    except (ValueError, TypeError):
        return None
    # Not every statistic is a curve: a device name, a git hash, a job id or a
    # single number is recorded the same way and is not plottable.
    try:
        arrays = [np.asarray(a, dtype=float) for a in arrays]
    except (ValueError, TypeError):
        return None
    if any(a.ndim != 1 for a in arrays):
        return None
    keep = [bool(np.any(np.isfinite(a))) for a in arrays]
    arrays = [a for a, k in zip(arrays, keep) if k]
    seeds = [s for s, k in zip(seeds, keep) if k]
    if not arrays:
        return None
    # A column that is present but null for every row of the cell, and a testing
    # points of zero, are both a cell whose points cannot be placed on a step
    # axis. Returning None puts them where every other undecodable cell goes,
    # rather than raising out of the middle of a figure.
    if recorded_steps.empty or recorded_points.empty:
        return None
    steps = float(recorded_steps.iloc[0])
    points = float(recorded_points.iloc[0])
    if not points:
        return None
    # Runs of one cell that trained for different lengths, or were tested a
    # different number of times. The first is taken, so every seed of the cell is
    # placed on that one's step axis and the rest are drawn at the wrong x. It is
    # the cell a caller has already warned about when it holds runs differing in
    # a key it does not lay out -- but the step axis is the consequence that
    # reaches the figure, so it is said here, where the value is chosen.
    #
    # Said once per disagreement, not once per cell: series_of runs for every
    # cell and every statistic, and a sweep where this happens at all has it in
    # all of them.
    for column, taken in ((recorded_steps, steps), (recorded_points, points)):
        if column.nunique() > 1:
            message = (
                f"WARNING: the runs of one cell disagree on {column.name[1]} "
                f"({sorted(set(column))}); {taken:g} is the one every seed of "
                f"it is drawn against."
            )
            if message not in _MIXED_AXIS_WARNED:
                _MIXED_AXIS_WARNED.add(message)
                print(message)
    return (
        pd.concat([pd.Series(a) for a in arrays], axis=1),
        steps / points,
        steps,
        seeds,
    )


# -----------------------------------------------------------------------------
# ----- Sweep layout ----------------------------------------------------------
# -----------------------------------------------------------------------------
#
# What the two interactive pages share: a sweep file decides the grid, and both
# read it the same way. One block per config_sweep entry, subplots inside a block
# for the keys that entry declares, and the keys global_sweep declares spent on
# whatever the page draws several of in one cell -- a curve per combination in
# interactive_curves.py, a dropdown in interactive_heatmaps.py, which draws one
# map per cell and has nothing to overlay.


def swept_keys(global_sweep: dict, sweep: list) -> frozenset:
    """The keys a sweep overrides, as the `ignore` set assign_configs takes.

    A key the sweep overrides carries a stale value in every config YAML, so it
    cannot say which file a run came from. Bare group names are left out: they
    select a file rather than override anything in it, which is what the dot
    tests for."""

    declared = {**global_sweep, **{k: v for entry in sweep for k, v in entry.items()}}
    return frozenset(k for k in declared if "." in k)


class SweepLayout:
    """The grid one sweep file lays out, over one data frame.

    `curves` says what a cell holds. With it, the keys `global_sweep` declares
    become the curve axes -- a line per combination, drawn over each other in one
    cell -- and are returned beside the blocks. Without it a cell holds one thing
    and there is nothing to overlay, so they become no axis at all and the caller
    spends them on a dropdown instead.

    `labels` is the map hyperparameter_to_label built, for the titles."""

    def __init__(self, df, sweep, global_sweep, labels, curves, vprint=None):
        self.df = df
        self.sweep = sweep
        self.global_sweep = global_sweep
        self.labels = labels
        self.curves = curves
        self.vprint = vprint or (lambda *a: None)
        self._entry_rows = {}
        self._curve_axes = None

    def entry_rows(self, i, entry):
        """The rows `entry` selects across every environment, not only the one
        being laid out.

        The values an axis takes are read from here, so the same axis has the
        same rows and columns in every grid. They are read from this entry's rows
        only, though: a value another entry declares belongs to another block,
        and would show up as a row or column that can never be filled.

        Filled on first use and kept: `blocks` runs once per environment, and
        this slice depends on the entry alone."""

        if i not in self._entry_rows:
            self._entry_rows[i] = self.df[select_rows(self.df, entry)[0]]
        return self._entry_rows[i]

    def blocks(self, df_env):
        """Lay one environment's runs out and return (blocks, curve axes).

        A block is a stack of subplots and its axes are what the grid is split
        by; the curve axes are what a subplot draws a line per, and they are the
        same for every block, so a combination keeps its colour throughout the
        figure. They are empty under `curves=False`.

        An axis is what the sweep DECLARES, not what the data happens to hold: a
        declared value with no runs still gets its row, column or curve."""

        # The environment separates figures and the seed is the confidence band,
        # so neither can be a row, a column or a curve within one figure.
        hp_cols = [
            c for c in df_env.columns
            if c[0] == "hyperparameter"
            and c[1] not in (SEED[1], ENV[1])
        ]

        # The curve axes are read from the WHOLE frame, not this environment's
        # slice. `values_of` appends values it finds in the data after the
        # declared ones, so an environment that ran an extra value would number
        # its curves differently from its neighbours -- and a curve's index is
        # what its colour, its bar's position and the average-AUC tab all match
        # on across environments.
        #
        # Computed once for that reason: nothing in it comes from `df_env`, since
        # a non-empty slice has the frame's own columns.
        if self._curve_axes is None:
            self._curve_axes = []
            if self.curves:
                for key, declared in self.global_sweep.items():
                    col = ("hyperparameter", key)
                    if len(declared) > 1 and col in hp_cols:
                        self._curve_axes.append(
                            (col, values_of(self.df, col, declared))
                        )
        curve_axes = self._curve_axes

        blocks = []
        matched = pd.Series(False, index=df_env.index)
        for i, entry in enumerate(self.sweep):
            mask, unconstrained = select_rows(df_env, entry)
            if unconstrained:
                print(
                    f"WARNING: sweep entry {i + 1} keys not in the data, "
                    f"not used to select its rows: {unconstrained}"
                )
            sub = df_env[mask]
            matched |= mask
            if sub.empty:
                self.vprint(f"sweep entry {i + 1} matched no rows, skipping.")
                continue

            axes = []
            for key, declared in entry.items():
                col = ("hyperparameter", key)
                # Not a key global_sweep also declares, even where this entry
                # redeclares it: with curves those are drawn over each other in
                # one cell, and without them they have a dropdown of their own --
                # and a key that is both an axis and a dropdown blanks every row
                # of its own axis but the one the dropdown is on.
                if (
                    key not in self.global_sweep
                    and len(declared) > 1
                    and col in hp_cols
                ):
                    axes.append((
                        col,
                        values_of(self.entry_rows(i, entry), col, declared),
                    ))
            # The sweep file decides the layout. A hyperparameter it does not
            # declare never becomes an axis, however much it varies in the rows
            # the entry selects: an entry loose enough to admit runs from other
            # sweeps would otherwise produce a grid of every other key in the
            # gzip.
            taken = (
                [c for c, _ in axes]
                + [("hyperparameter", k) for k in self.global_sweep]
            )
            merged = [c[1] for c in hp_cols if c not in taken and varies(sub, c)]
            if merged:
                becomes = (
                    "those runs are averaged into one curve" if self.curves
                    else "only one run per cell is drawn"
                )
                print(
                    f"WARNING: sweep entry {i + 1} does not declare {merged}, "
                    f"which vary in the rows it selects: {becomes}. "
                    f"Use --include/--exclude to pick a value."
                )

            # An entry that pins nothing gets no title. Blocks are identified by
            # their index, not by this.
            pinned = ", ".join(
                f"{self.labels[k]}={value_str(v[0])}"
                for k, v in entry.items() if len(v) == 1
            )
            # `i`, not the block's position in the list: an entry that matched no
            # run here leaves no block, and a caller laying the environments over
            # each other matches them by this number. A position would put one
            # entry's cells under another entry's label.
            blocks.append(make_block(pinned, sub, axes, index=i))

        unmatched = int((~matched).sum())
        if unmatched:
            print(
                f"WARNING: {unmatched} row(s) match no sweep entry and are not "
                f"plotted."
            )
        return blocks, curve_axes


def declared_environments(sweep: list, global_sweep: dict):
    """The environments a sweep declares, as a set of stems, or None when it
    declares none.

    None is "declared nothing", which is not the same as an empty set: an entry
    that names no environment of its own and finds none in `global_sweep`
    constrains none of them, so there is nothing to filter by and the data
    decides. A caller has to be able to tell the two apart."""

    per_entry = [
        entry.get("environment", global_sweep.get("environment")) for entry in sweep
    ]
    if any(values is None for values in per_entry):
        return None
    return {value_str(v) for values in per_entry for v in values}


def keep_declared_environments(df, declared_envs, verb: str):
    """Drop the runs whose environment the sweep does not declare, and return the
    frame. `declared_envs` of None leaves it alone.

    The gzip holds whatever ran in the folder, and a sweep that names two
    environments is a figure about those two. Matched on the stem, as the sweep
    names an environment by its config file while an overridden run is labelled
    `<stem> (key=value)`.

    `verb` is what the caller does with a run -- "plotted", "drawn" -- for the
    messages."""

    if declared_envs is None or df.empty:
        return df
    stems = df[ENV].map(lambda e: split_config_label(str(e))[0])
    keep_env = stems.isin(declared_envs)
    dropped = sorted({str(e) for e in df.loc[~keep_env, ENV].unique()}, key=str)
    if dropped:
        print(
            f"{len(dropped)} environment(s) in the data are not declared by the "
            f"sweep and are not {verb}: {dropped}"
        )
    df = df[keep_env]
    if df.empty:
        raise SystemExit(
            f"No run in the data belongs to an environment the sweep declares: "
            f"{sorted(declared_envs)}"
        )
    return df


def keep_composed_runs(df, sweep, global_sweep, verb: str, vprint=lambda *a: None):
    """Drop the runs whose whole configuration is not one the sweep composes, and
    return the frame (see composed_sweep_mask).

    An entry declares part of a configuration; the rest comes from the config
    files it selects and from configs/default.yaml under them, so a run launched
    before any of those changed no longer belongs to the sweep even where every
    key the entry declares still agrees."""

    if df.empty:
        return df
    composed, message = composed_sweep_mask(df, sweep, global_sweep)
    vprint(f"\n{message}")
    outside = int((~composed).sum())
    if outside:
        print(
            f"{outside} run(s) do not match any configuration the sweep composes "
            f"and are not {verb}."
        )
    df = df[composed]
    if df.empty:
        raise SystemExit(
            "No run in the data matches a configuration the sweep composes."
        )
    return df


# -----------------------------------------------------------------------------
# ----- Interactive pages -----------------------------------------------------
# -----------------------------------------------------------------------------

# The shell both interactive pages are written into: the vega scripts, the tab
# strip, and the machinery that shows one panel at a time. What differs between
# them is how a panel is drawn, which each passes in as `render`, a function
# `render(i, carry)` the shell calls to fill a panel and re-fill it afterwards.
#
# Substituted rather than formatted: the body is a page of JavaScript, and under
# str.format every brace in it has to be doubled -- so one brace typed singly is
# a KeyError raised a long way from the line that caused it.
_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
<style>
  body { font-family: sans-serif; margin: 1.5em; }
  #tabs { display: flex; flex-wrap: wrap; gap: 0.4em; margin-bottom: 1.2em; }
  #tabs button { font: inherit; padding: 0.35em 0.9em; cursor: pointer;
                 border: 1px solid #ccc; border-radius: 4px; background: #f6f6f6; }
  #tabs button.active { background: #333; color: #fff; border-color: #333; }
  /* Own block, not vega-embed's container: its stylesheet takes the dropdowns
     out of flow and draws them over the legend on every resize. */
  .controls { display: flex; flex-wrap: wrap; align-items: center;
              gap: 0.4em 1.5em; margin-bottom: 1.2em; }
  .controls .vega-bindings { display: contents; }
  .controls .vega-bind { margin: 0; }
  .controls .toggle { white-space: nowrap; }
  .vis { overflow-x: auto; }
</style>
</head>
<body>
<div id="tabs"></div>
<div id="panels"></div>
<script type="text/javascript">
  const panels = __PANELS_JSON__;
  const tabsEl = document.getElementById('tabs');
  const panelsEl = document.getElementById('panels');
  const views = [];
  const drawn = [];

  // Where a rejected spec is reported. Vega-Lite leaves an empty page and says
  // why in the console, which is nowhere anyone looks; put it where the plots go.
  function showError(i, err) {
    document.getElementById('vis' + i).innerHTML =
      '<pre style="color:#a00;white-space:pre-wrap">' + err + '</pre>';
  }

  // Clear a panel's controls and embed `spec` into it, handing the view back.
  function embedInto(i, spec) {
    if (views[i]) { views[i].finalize(); }
    document.getElementById('controls' + i).innerHTML = '';
    return vegaEmbed('#vis' + i, spec,
              {actions: {export: true, source: false, compiled: false, editor: false},
               bind: '#controls' + i})
      .then(result => (views[i] = result.view))
      .catch(err => { showError(i, err); });
  }

__RENDER__

  function show(i) {
    Array.from(panelsEl.children).forEach((p, j) => p.hidden = j !== i);
    Array.from(tabsEl.children).forEach((b, j) => b.className = j === i ? 'active' : '');
    // Embed on first view: vega sizes a chart from its container, and a hidden
    // one has no width, so drawing every panel up front lays them all out wrong.
    if (!drawn[i]) {
      drawn[i] = true;
      render(i, null);
    } else {
      redraw(i);
    }
  }

  panels.forEach((panel, i) => {
    const button = document.createElement('button');
    button.textContent = panel.title;
    button.onclick = () => show(i);
    tabsEl.appendChild(button);

    const div = document.createElement('div');
    div.hidden = true;
    div.innerHTML = '<div class="controls" id="controls' + i + '"></div>'
                  + '<div class="vis" id="vis' + i + '"></div>';
    panelsEl.appendChild(div);
  });

  show(0);
</script>
</body>
</html>
"""


def html_page(title: str, panels: list, render_js: str) -> str:
    """One interactive page, and return it as text.

    `render_js` is the JavaScript that draws a panel: it must define
    `render(i, carry)`, which embeds panel `i` (see embedInto) carrying the
    values in `carry`, and `redraw(i)`, which the shell calls when a panel
    already drawn is shown again -- for a page with nothing to redo, a function
    that returns."""

    return (
        _PAGE_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__RENDER__", render_js)
        .replace("__PANELS_JSON__", json.dumps(panels))
    )
