"""A sweep file has two top-level keys, both optional:

    global_sweep: a mapping of {hydra_override_key: [values]} whose
        cross-product applies to every entry of config_sweep.
    config_sweep: a list of such mappings, each one fixing hyperparameters that
        only make sense together (e.g. an encoder and the hidden_size it needs).

Example:

    global_sweep:
      agent.critic.approximator.dropout_p: [0.0, 0.1]
      environment: [lunar_lander, travel_field_small]
      algorithm: [eps_greedy]

    config_sweep:
      - agent.critic.approximator.encoder.id: [RawGridEncoder]
        agent.critic.approximator.encoder.num_centers: [20]
      - agent.critic.approximator.encoder.id: [MaxoutGridEncoder]
        agent.critic.approximator.encoder.hidden_size: [16, 64]

    Total configurations:
    - global_sweep: 2
    - first config_sweep: 1 -> 1 x 2 = 2
    - second config_sweep: 2 -> 2 x 2 = 4
    - total: 2 + 4 = 6

Note: either section MUST define `environment` and `algorithm`.
"""

import os
import string
import yaml

from src.utils.paths import CONFIG_DIR, SWEEP_DIR


def load_sweep(name):
    """Read global_sweep / config_sweep from SWEEP_DIR/<name>.yaml.

    `name` can also be a path to a YAML file anywhere. Returns
    (global_sweep, config_sweep), with missing keys defaulting to "no extra
    hyperparameters" (i.e., a single configuration with nothing swept).
    """
    path = name if os.path.isfile(name) else os.path.join(SWEEP_DIR, f"{name}.yaml")
    if not os.path.isfile(path):
        raise SystemExit(f"Sweep file not found: {path}")
    with open(path) as f:
        sweep = yaml.safe_load(f) or {}
    unknown = set(sweep) - {"global_sweep", "config_sweep"}
    if unknown:
        raise SystemExit(f"Unknown keys in {path}: {sorted(unknown)}")
    global_sweep = sweep.get("global_sweep") or {}
    config_sweep = sweep.get("config_sweep") or [{}]
    # A bare scalar is a one-value sweep; accepting it keeps the YAML readable
    # for keys that are pinned rather than swept.
    global_sweep = {k: v if isinstance(v, list) else [v] for k, v in global_sweep.items()}
    config_sweep = [
        {k: v if isinstance(v, list) else [v] for k, v in (entry or {}).items()}
        for entry in config_sweep
    ]
    return global_sweep, config_sweep


def config_groups():
    """Keys that select a Hydra config GROUP (configs/<key>/<value>.yaml) rather
    than a config value, read from the directories under CONFIG_DIR so that a
    new group never has to be registered here by hand."""
    try:
        return tuple(
            sorted(entry.name for entry in os.scandir(CONFIG_DIR) if entry.is_dir())
        )
    except OSError:
        raise SystemExit(f"Cannot list config groups in {CONFIG_DIR}")


CONFIG_GROUPS = config_groups()

# Characters Hydra's override grammar accepts unquoted.
_UNQUOTED_CHARS = frozenset(string.ascii_letters + string.digits + "_-./+@:")


def format_override(key, value):
    """Render one Hydra override as a single argv token.

    Keys in CONFIG_GROUPS are rendered bare (`environment=lunar_lander`),
    everything else as a forced override (`++agent.critic.lr=0.1`).

    Hydra's override grammar reads a bare `None` as the string "None", not
    OmegaConf 'null' (that is then parsed to None). List values are rendered
    without spaces, e.g., `[mean,max]`. Values holding characters the grammar
    treats specially are single-quoted.

    A null renders as `null` and reaches cfg.yaml as None. By the time the
    analysis side reads that back it is indistinguishable from a key the run
    never recorded, which is why src.utils.config.is_null treats the two as one
    value -- see its module docstring. Anything comparing a sweep-declared value
    against recorded data must go through config.match_declared.

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
        text = str(v)
        if text and all(c in _UNQUOTED_CHARS for c in text):
            return text
        return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"
    return f"{'' if key in CONFIG_GROUPS else '++'}{key}={fmt(value)}"
