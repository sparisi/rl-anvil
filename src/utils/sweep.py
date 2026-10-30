"""A sweep file has two top-level keys, both optional:

    global_sweep: a mapping of {hydra_override_key: [values]} whose
        cross-product applies to every entry of config_sweep.
    config_sweep: a list of such mappings, each one fixing hyperparameters that
        only make sense together (e.g. an encoder and the hidden_size it needs).

Example:

    global_sweep:
      agent.critic.approximator.dropout_p: [0.0, 0.1]
      environment: [gymnasium/lunar_lander, gym_gridworlds/travel_field_small]
      algorithm: [eps_greedy]

    config_sweep:
      - agent.critic.approximator.encoder.id: [RawGridEncoder]
        agent.critic.approximator.encoder.num_centers: [20]
      - agent.critic.approximator.encoder.id: [MaxoutGridEncoder]
        agent.critic.approximator.encoder.hidden_size: [16, 64]

    Total configurations:
    - global_sweep: 2 x 2 x 1 = 4
    - first config_sweep: 1 -> 1 x 4 = 4
    - second config_sweep: 2 -> 2 x 4 = 8
    - total: 4 + 8 = 12

A value listed twice for one key, or equal to an earlier one (1 and 1.0), is
run once.

Every configuration the sweep expands to must set `environment` and `algorithm`,
in `global_sweep` or in its `config_sweep` entry.
"""

import itertools
import os
import re
import string
from functools import lru_cache

from hydra.core.override_parser.overrides_parser import OverridesParser

from src.utils.id import escape_interpolations, load_yaml
from src.utils.paths import CONFIG_DIR, SWEEP_DIR


def _tag_bools(v):
    """Return `v` with every bool, also inside lists, replaced by the pair
    (bool, v), so that a bool compares equal only to a bool."""

    if isinstance(v, bool):
        return (bool, v)
    if isinstance(v, list):
        return [_tag_bools(x) for x in v]
    return v


def unique_values(path, key, values):
    """Return `values` without the ones equal to an earlier value, printing a
    warning for each one dropped. A bool equals only a bool, also inside lists,
    so True and 1 are both kept, and so are [True] and [1]."""

    kept = []
    for v in values:
        if any(_tag_bools(v) == _tag_bools(k) for k in kept):
            print(f"WARNING: {path}: {key}={v!r} equals an earlier value and is run once.")
        else:
            kept.append(v)
    return kept


def load_sweep(name):
    """Read global_sweep / config_sweep from SWEEP_DIR/<name>.yaml.

    `name` can also be a path to a YAML file anywhere. Returns
    (global_sweep, config_sweep), with missing keys defaulting to "no extra
    hyperparameters" (i.e., a single configuration with nothing swept). The file
    is read as Hydra reads config files (src.utils.id.load_yaml), and repeated
    values of a key are dropped (unique_values). A `config_sweep` entry that,
    together with `global_sweep`, sets no `environment` or no `algorithm` stops
    the program.
    """
    path = name if os.path.isfile(name) else os.path.join(SWEEP_DIR, f"{name}.yaml")
    if not os.path.isfile(path):
        raise SystemExit(f"Sweep file not found: {path}")
    sweep = load_yaml(path) or {}
    unknown = set(sweep) - {"global_sweep", "config_sweep"}
    if unknown:
        raise SystemExit(f"Unknown keys in {path}: {sorted(unknown)}")
    global_sweep = sweep.get("global_sweep") or {}
    config_sweep = sweep.get("config_sweep") or [{}]
    # A bare scalar is a one-value sweep; accepting it keeps the YAML readable
    # for keys that are pinned rather than swept.
    global_sweep = {
        k: unique_values(path, k, v if isinstance(v, list) else [v])
        for k, v in global_sweep.items()
    }
    config_sweep = [
        {
            k: unique_values(path, k, v if isinstance(v, list) else [v])
            for k, v in (entry or {}).items()
        }
        for entry in config_sweep
    ]
    # An empty list would make the product of the sweep empty, dropping every
    # other key of its block.
    empty = [k for k, v in global_sweep.items() if not v]
    empty += [k for entry in config_sweep for k, v in entry.items() if not v]
    if empty:
        raise SystemExit(f"Empty sweep values in {path} for: {empty}")
    # Every configuration of an entry sets the same keys, so checking each entry
    # with global_sweep checks every configuration.
    for i, entry in enumerate(config_sweep):
        for key in ("environment", "algorithm"):
            if key not in global_sweep and key not in entry:
                raise SystemExit(
                    f"{path}: config_sweep entry {i} sets no `{key}`, and neither does "
                    f"global_sweep."
                )
    return global_sweep, config_sweep


def expand_sweep(global_sweep, config_sweep):
    """Yield every configuration a sweep declares as a {key: value} dict: for
    each entry of `config_sweep`, the cross-product of the values of the
    `global_sweep` keys and of the entry's own keys, the entry's values taking
    precedence for a key both declare. The keys come in `global_sweep` order,
    followed by the entry's remaining keys."""

    for entry in config_sweep:
        keys = list(global_sweep) + [k for k in entry if k not in global_sweep]
        declared = [entry.get(k, global_sweep.get(k)) for k in keys]
        for combo in itertools.product(*declared):
            yield dict(zip(keys, combo))


@lru_cache(maxsize=None)
def config_groups():
    """Keys that select a Hydra config GROUP (configs/<key>/<value>.yaml) rather
    than a config value, read from the directories under CONFIG_DIR so that a
    new group never has to be registered here by hand. CONFIG_DIR is scanned on
    the first call, and later calls return the cached tuple."""
    try:
        return tuple(
            sorted(entry.name for entry in os.scandir(CONFIG_DIR) if entry.is_dir())
        )
    except OSError:
        raise SystemExit(f"Cannot list config groups in {CONFIG_DIR}")

# Characters Hydra's override grammar accepts unquoted.
_UNQUOTED_CHARS = frozenset(string.ascii_letters + string.digits + "_-./+@:")

_PARSE_OVERRIDE = OverridesParser.create().parse_override


def _reads_back(text):
    """True if Hydra reads the unquoted override value `text` as that string."""

    try:
        return _PARSE_OVERRIDE(f"k={text}").value() == text
    except Exception:
        return False


def _quote(text):
    """`text` as a single-quoted override value that Hydra and OmegaConf read
    back as `text`. "${" is escaped for OmegaConf (escape_interpolations), and
    the backslashes Hydra unescapes, those right before a quote or at the end,
    are doubled."""

    text = escape_interpolations(text)

    def escape(m):
        return m.group(1) * 2 + ("\\'" if m.group(2) else "")

    return "'" + re.sub(r"(\\*)('|$)", escape, text) + "'"


def format_override(key, value):
    """Render one Hydra override as a single argv token.

    Keys in config_groups() are rendered bare (`environment=gymnasium/lunar_lander`),
    everything else as a forced override (`++agent.critic.lr=0.1`).

    Hydra's override grammar reads a bare `None` as the string "None", not
    OmegaConf 'null' (that is then parsed to None). List values are rendered
    without spaces, e.g., `[mean,max]`. A string is rendered bare only if Hydra
    reads it back as the same string, so "10", "null" or "true" stay strings,
    and is single-quoted otherwise (_quote). A dict value raises a ValueError.
    A sweep value cannot be an interpolation: '${' is escaped, so a value such
    as "${agent.critic.lr}" reaches the config as that literal string.

    A null renders as `null` and reaches cfg.yaml as None. By the time the
    analysis side reads that back it is indistinguishable from a key the run
    never recorded, which is why src.utils.matching.is_null treats the two as one
    value -- see that module's docstring. Anything comparing a sweep-declared
    value against recorded data must go through matching.select_rows.

    What this renders feeds compose_cfg, whose output is hashed into the
    config_id. Rendering only has to preserve the VALUE Hydra parses: quoting
    and int/float spelling do not change the config, so they do not change ids.
    """
    def fmt(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, (list, tuple)):
            return "[" + ",".join(fmt(x) for x in v) + "]"
        if isinstance(v, dict):
            raise ValueError(f"{key}: a mapping cannot be rendered as an override ({v!r}).")
        text = str(v)
        if text and all(c in _UNQUOTED_CHARS for c in text):
            if not isinstance(v, str) or _reads_back(text):
                return text
        return _quote(text)
    return f"{'' if key in config_groups() else '++'}{key}={fmt(value)}"
