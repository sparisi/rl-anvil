"""Config values, config-group reading and run matching, without plotting
dependencies.

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

Config groups are read by composing each of their YAMLs with Hydra (see
`load_config_group`), so a `defaults:` chain means what it means to Hydra.
Importing this module imports src.utils.id, which pulls hydra in.
"""

import os
import re
from ast import literal_eval
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

from omegaconf import OmegaConf

from src.utils.id import (
    DROPPED_CFG_KEYS,
    compose_cfg,
    flatten_cfg,
    load_yaml,
)
from src.utils.paths import CONFIG_DIR


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
    collapses to `NULL_STR`, so an absent key and a recorded null read alike.
    A string that parses as a number normalizes as that number, so a value a
    PyYAML reader returned as a string, such as `1e-3`, matches the number."""

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
    """Return `v` with numpy arrays and tuples converted to lists and numpy
    scalars to Python scalars, recursively through lists, tuples and dicts."""

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

def load_config_group(config_dir: str) -> dict:
    """Every YAML in a Hydra config group directory -- configs/environment,
    configs/algorithm -- composed by Hydra with its `defaults:` list, as
    {name: {flat_key: value}}. Both groups are matched against a run identically;
    only the directory differs.

    Subdirectories are searched too. A YAML's name is its path relative to
    `config_dir` without the suffix (e.g., gymnasium/lunar_lander), which is the
    name Hydra selects it by. `config_dir` must be under CONFIG_DIR, where Hydra
    composes from.

    Does NOT compose with configs/default.yaml, whose keys are CLI-swept per
    experiment and are not part of a config's identity. Interpolations are left
    unresolved, and `flatten_cfg` drops the keys still holding one.

    A YAML is composed from an empty primary config as `+<group>=<name>`, after
    `+<group>=<option>` for every group its `defaults:` list overrides
    (`override /agent: svf`), so the override has a group to replace. A key
    holding the value it has in the option configs/default.yaml selects for an
    overridden group is dropped, as the keys of configs/default.yaml are: what is
    kept is what the YAML and its options change. A key the YAML itself sets to
    that same value is dropped with them.

    The keys that do not define a configuration are dropped, as they are from a
    run's cfg.yaml: a YAML declaring one would be compared against a run that
    cannot record it, and would match nothing."""

    try:
        group = Path(config_dir).resolve().relative_to(Path(CONFIG_DIR).resolve())
    except ValueError:
        raise ValueError(
            f"{config_dir} is not under {CONFIG_DIR}, so Hydra cannot compose "
            f"its files."
        ) from None
    names = [
        p.relative_to(config_dir).with_suffix("").as_posix()
        for p in Path(config_dir).rglob("*.yaml")
    ]
    configs = {}
    for name in names:
        overridden = overridden_groups(Path(config_dir) / f"{name}.yaml")
        flat = flatten_cfg(
            OmegaConf.to_container(
                compose_cfg(
                    (
                        *(f"+{g}={option}" for g, option in overridden),
                        f"+{group.as_posix()}={name}",
                    ),
                    config_name=None,
                ),
                resolve=False,
            )
        )
        for g, _option in overridden:
            base = default_group_config(g)
            flat = {k: v for k, v in flat.items() if k not in base or base[k] != v}
        configs[name] = compared_keys(flat, DROPPED_CFG_KEYS)
    return configs


def overridden_groups(path) -> list:
    """Return (group, option) for every `override /<group>: <option>` entry in
    the `defaults:` list of the YAML `path`, in the order the list holds them."""

    found = []
    for entry in (load_yaml(str(path)) or {}).get("defaults") or []:
        if not isinstance(entry, dict):
            continue
        for key, option in entry.items():
            if key.startswith("override "):
                found.append((key.split(None, 1)[1].lstrip("/"), option))
    return found


@lru_cache(maxsize=None)
def default_group_config(group: str) -> dict:
    """Return the flat config of the option configs/default.yaml selects for the
    Hydra group `group`, composed on its own, or {} when default.yaml selects
    none. The result is cached: every call with the same `group` returns the
    same object."""

    defaults = (load_yaml(os.path.join(CONFIG_DIR, "default.yaml")) or {}).get("defaults")
    for entry in defaults or []:
        if isinstance(entry, dict) and group in entry:
            return flatten_cfg(
                OmegaConf.to_container(
                    compose_cfg((f"+{group}={entry[group]}",), config_name=None),
                    resolve=False,
                )
            )
    return {}


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

    # Whether a row is null for a declared key, subtree included: the one rule
    # for a declared null, null_mask's, read once per key over the whole frame.
    keys = dict.fromkeys(key for declared in candidates.values() for key in declared)
    is_null = {key: null_mask(df, ("hyperparameter", key)) for key in keys}

    def label(row, skip):
        def matches(key, value):
            if value_str(value) == NULL_STR:
                return bool(is_null[key].loc[row.name])
            return key in col_of and value_str(row[col_of[key]]) == value_str(value)

        best_name, best_len, tied = None, None, []
        for name, declared in candidates.items():
            declared = {
                key: value for key, value in declared.items() if key not in skip
            }
            if not declared:
                continue
            if any(not matches(key, value) for key, value in declared.items()):
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

    # Labelling a row only depends on the columns some config declares, on its
    # nullness for them and on which keys it skips, so rows agreeing on those get
    # the same label: build a signature from them, label one row per distinct
    # signature, map the rest. value_str first -- list-valued hyperparameters
    # arrive from parquet as numpy arrays, which are unhashable, and it also
    # keeps 64 and 64.0 in one group.
    sig = over_stem.astype(str)
    for key in keys:
        if key in col_of:
            sig = sig + "\x00" + df[col_of[key]].map(value_str)
        sig = sig + "\x00" + is_null[key].astype(str)

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
    `gymnasium/lunar_lander (environment.train_from_pixels=True)` ->
    ("gymnasium/lunar_lander", {"environment.train_from_pixels": True}).

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
