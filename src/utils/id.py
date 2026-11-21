import copy
import hashlib
import json
import os
import re
import yaml
from functools import lru_cache
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.errors import OmegaConfBaseException
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from rich import print

from src.utils.misc import atomic_open
from src.utils.paths import CONFIG_DIR


# Where configurations with tuned hyperparameters per (algorithm, environment)
# pairs are stored.
TUNED_PATH = os.path.join(CONFIG_DIR, "tunings", "{}.yaml")

# Keys that do not define a configuration. See configs/full_config.yaml for details.
# Every other key must resolve to the same value wherever it is composed: a value
# read from the environment or the clock (${oc.env:...}, ${now:...}) gives a job
# a different config ID at run time than at submission, and ${hydra:...} does not
# resolve outside a Hydra run.
DROPPED_CFG_KEYS = (
    "hydra",
    "wandb",
    "extra",
    "meta",
    "results",
    "tuned",
    "experiment.rng_seed",
    # "agent.critic.approximator.device",
)

# The columns of the results table that hold a run's seed and its config ID.
SEED = ("seed", "experiment.rng_seed")
CONFIG_ID = ("config", "config_id")

# Characters of the md5 digest a config ID keeps.
CONFIG_ID_LEN = 8

# A config ID is the first CONFIG_ID_LEN of the 32 hex characters of the MD5 of
# the config_identity() YAML: `cfg` stripped of DROPPED_CFG_KEYS, with sorted
# keys and ints converted to floats.
# This is much better than Hydra default directory naming that concatenates all
# keys=value in the DictConfig.


def strip_config(cfg: DictConfig, dropped: tuple = DROPPED_CFG_KEYS) -> dict:
    """Return `cfg` as a plain resolved dict, without the keys in `dropped`.

    Top-level dropped keys are never resolved, so a missing value or an
    interpolation that cannot resolve under one of them raises nothing. Each
    kept top-level node is resolved with its parent, so interpolations into
    other kept keys resolve as they do in `cfg`."""

    top = {d for d in dropped if "." not in d}
    container = {}
    for key in cfg:
        if key in top:
            continue
        node = cfg[key]
        container[key] = (
            OmegaConf.to_container(node, resolve=True, throw_on_missing=True)
            if OmegaConf.is_config(node)
            else node
        )
    return drop_keys(container, dropped)


def drop_keys(d: dict, drop: tuple) -> dict:
    """Remove each dotted key in `drop`, with its whole subtree, from the nested
    dict `d` in place, and return `d`. Keys absent from `d` are ignored."""

    for dotted in drop:
        *parents, leaf = dotted.split(".")
        node = d
        for k in parents:
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(leaf, None)
    return d


def numbers_to_float(x):
    """Return `x` with every int converted to float, recursively through dicts
    and lists, so that `lrate=1` and `lrate=1.0` (same run, two YAML spellings)
    do not hash to two different config IDs. Bools are kept."""

    if isinstance(x, dict):
        return {k: numbers_to_float(v) for k, v in x.items()}
    if isinstance(x, list):
        return [numbers_to_float(v) for v in x]
    if isinstance(x, bool):  # bool subclasses int, so it goes first
        return x
    if isinstance(x, int):
        return float(x)
    return x


def escape_interpolations(x):
    """Return `x` with "${" in every string escaped, recursively through dicts
    and lists, so that YAML written from it loads back to the same strings.
    Backslashes right before "${" are doubled, as OmegaConf reads "\\\\${" as a
    literal backslash followed by an interpolation."""

    if isinstance(x, dict):
        return {k: escape_interpolations(v) for k, v in x.items()}
    if isinstance(x, list):
        return [escape_interpolations(v) for v in x]
    if isinstance(x, str):
        return re.sub(r"(\\*)\$\{", lambda m: m.group(1) * 2 + "\\${", x)
    return x


def config_for_save(cfg: DictConfig, dropped: tuple = DROPPED_CFG_KEYS) -> str:
    """Return the YAML written to cfg.yaml: `cfg` resolved, with sorted keys,
    without the keys in `dropped`, and with its original types. A resolved
    string holding "${" is written escaped, so cfg.yaml loads back to the same
    string and hashes to the same config ID."""

    return OmegaConf.to_yaml(
        OmegaConf.create(
            escape_interpolations(strip_config(cfg, dropped))
        ),
        sort_keys=True,
    )


def config_identity(cfg: DictConfig, dropped: tuple = DROPPED_CFG_KEYS) -> str:
    """Return the YAML that the config ID hashes: the same YAML as
    config_for_save(), with every int converted to float."""

    # The conversion applies to the hashed YAML only: the config written to disk
    # and the config the run is built from keep their original types.
    return OmegaConf.to_yaml(
        OmegaConf.create(
            escape_interpolations(numbers_to_float(strip_config(cfg, dropped)))
        ),
        sort_keys=True,
    )


def make_id(
    cfg: DictConfig | None = None,
    overrides=None,
    n_chars: int = CONFIG_ID_LEN,
) -> str:
    """Return the config ID of `cfg`, or of the config that `overrides`
    compose. Exactly one of the two must be given."""

    if (cfg is None) == (overrides is None):
        raise ValueError("Pass either cfg or overrides, and not both.")
    if overrides is not None:
        cfg = compose_cfg(overrides)
    return id_of(config_identity(cfg), n_chars)


def id_of(identity: str, n_chars: int = CONFIG_ID_LEN) -> str:
    """Return the config ID of a config_identity() YAML: the first `n_chars`
    hex characters of its MD5 digest."""

    digest = hashlib.md5(
        identity.encode(),
        usedforsecurity=False,
    ).hexdigest()
    return digest[:n_chars]


@lru_cache(maxsize=None)
def _composed_config(overrides: tuple) -> dict:
    """Return the flat config that `overrides` compose, without the keys in
    DROPPED_CFG_KEYS. The result is cached: every call with the same
    `overrides` returns the same object."""

    return flatten_cfg(strip_config(compose_cfg(overrides)), drop_interpolations=False)


def composed_config(overrides: tuple) -> dict:
    """Return the flat config that `overrides` compose, as {dotted_key: value},
    without the keys in DROPPED_CFG_KEYS.

    The keys are the dotted paths that flatten_cfg() produces.

    Values holding "${" are kept (drop_interpolations=False).

    `overrides` is a tuple (rather than a list) so the composition is cached.
    Each call returns its own deep copy of the cached result.
    """

    return copy.deepcopy(_composed_config(overrides))


def compose_cfg(overrides=(), config_name: str = "default") -> DictConfig:
    """Return the DictConfig that Hydra composes from the config file
    `config_name` (a path relative to CONFIG_DIR, without the suffix) and the
    `overrides`.

    For example
        compose_cfg(("environment=gymnasium/lunar_lander", "++agent.critic.lr=0.001"))

    returns the DictConfig resulting from running
        python main.py environment=gymnasium/lunar_lander ++agent.critic.lr=0.001

    and compose_cfg(config_name="environment/gymnasium/lunar_lander") composes
    that file and its `defaults:` list alone, without configs/default.yaml.

    If an override sets `tuned` to a tuning name or a list of names, e.g.
    `++tuned=rho` or `++tuned=[beta,rho]`, the values in those TUNED_PATH
    files override some parameters (apply_tuned).
    The key `tuned` is in DROPPED_CFG_KEYS, so it is not part of the config ID.
    """

    tuned = any(o.split("=", 1)[0].lstrip("+~") == "tuned" for o in overrides)
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name=config_name, overrides=list(overrides), return_hydra_config=tuned)
    if tuned:
        if cfg.get("tuned"):
            apply_tuned(cfg, cfg.hydra)
        with open_dict(cfg):
            del cfg["hydra"]
    return cfg


@lru_cache(maxsize=None)
def tuned_table(name: str) -> dict:
    """Return the file of the tuning `name`, TUNED_PATH.format(name), as
    {algorithm: {environment: {dotted_key: value}}}. The result is cached:
    every call with the same `name` returns the same object.

    Raise ValueError if the file does not exist, or if it names an algorithm or
    an environment that has no YAML in configs/algorithm or configs/environment."""

    path = TUNED_PATH.format(name)
    if not os.path.isfile(path):
        raise ValueError(f"The tuning {name} has no file {path}.")
    table = load_yaml(path) or {}
    for algorithm, environments in table.items():
        groups = (
            ("algorithm", algorithm),
            *(("environment", e) for e in environments),
        )
        for group, group_name in groups:
            if not os.path.isfile(os.path.join(CONFIG_DIR, group, f"{group_name}.yaml")):
                raise ValueError(
                    f"{path} names the {group} {group_name}, which has no YAML "
                    f"in {os.path.join(CONFIG_DIR, group)}."
                )
    return table


def apply_tuned(cfg: DictConfig, hydra_cfg: DictConfig) -> None:
    """Overwrite in `cfg` the keys that the tunings named in `cfg.tuned` (a
    name or a list of names) list for the algorithm and environment chosen in
    `hydra_cfg`, the `hydra` node of the composition. Keys set by an override
    are skipped.

    Raise ValueError if `cfg.tuned` is neither a string nor a list, if one of
    the tunings has no entry for that algorithm, or if two of the tunings set
    the same key for that algorithm and environment. A tuning with an entry
    for the algorithm and none for the environment sets nothing."""

    names = cfg.tuned
    if isinstance(names, str):
        names = [names]
    elif OmegaConf.is_list(names) or isinstance(names, list):
        names = [str(n) for n in names]
    else:
        raise ValueError(f"tuned must be a tuning name or a list of names, got {names!r}.")

    choices = hydra_cfg.runtime.choices
    algorithm = choices.get("algorithm")
    environment = choices.get("environment")
    entry = {}
    for name in names:
        table = tuned_table(name)
        if algorithm not in table:
            raise ValueError(
                f"The tuning {name} has no entry for the algorithm {algorithm} "
                f"({TUNED_PATH.format(name)})."
            )
        values = table[algorithm].get(environment, {})
        shared = sorted(entry.keys() & values.keys())
        if shared:
            raise ValueError(
                f"The tunings {names} all set {shared} for {algorithm} in "
                f"{environment}."
            )
        entry.update(values)

    keys = [o.split("=", 1)[0].lstrip("+~") for o in hydra_cfg.overrides.task]
    overridden = [k for k in keys if k not in choices]
    for key, value in entry.items():
        if any(key == o or key.startswith(f"{o}.") for o in overridden):
            continue
        OmegaConf.update(cfg, key, value)


def prepare_run(cfg: DictConfig, verbose: bool = True) -> str | None:
    """Return the config ID of `cfg`, or None if the run should be skipped.

    The run is skipped if a data.npz for the same run exists and
    results.force_run is not set. The presence of videos, heatmaps, ... does
    not skip the run.

    Otherwise, make the seed directory (if data is saved) and save the cfg.yaml
    that identifies this run. No other folder is created. On return
    results.data_dir points at the seed folder, so everything the run writes
    goes into <data_dir>/<config_id>/<rng_seed>/
    Call it once per `cfg`: a second call appends the folders again.

    With results.force_run, the files already in the seed folder are not
    removed. Files the new run writes under the same names are overwritten,
    and the others stay next to them.

    If `cfg.tuned` names one or more tunings, their values are first set in
    `cfg` (apply_tuned), with the `hydra` node of the running Hydra app.
    """

    if cfg.get("tuned"):
        apply_tuned(cfg, HydraConfig.get())

    cfg_identity = config_identity(cfg)
    config_id = id_of(cfg_identity)

    need_data_dir = (
        cfg.results.save_videos or
        cfg.results.save_heatmaps or
        cfg.results.save_memory
    )
    if need_data_dir and cfg.results.data_dir is None:
        if verbose:
            print(
                ":warning:  The user requested to save data but "
                "[bold italic]results.data_dir[/bold italic] is not defined. "
                "Using default directory: [italic]data[/italic]"
            )
        cfg.results.data_dir = "data"

    # Nothing to save, the run can start
    if cfg.results.data_dir is None:
        return config_id

    config_dir = os.path.join(cfg.results.data_dir, config_id)
    run_dir = os.path.join(config_dir, str(cfg.experiment.rng_seed))
    yaml_path = os.path.join(
        config_dir,
        "cfg.yaml",
    )

    # Checked before the data of this seed, so a collision or an edited cfg.yaml
    # raises for every seed, including the ones that already have data.
    if os.path.exists(yaml_path):
        # Compare identities, not file bytes: cfg.yaml keeps the types it was
        # written with, so two seeds passing 1 and 1.0 for the same key write
        # different bytes for the same configuration.
        try:
            existing_identity = config_identity(OmegaConf.load(yaml_path))
        except OmegaConfBaseException as e:
            raise RuntimeError(
                f"{yaml_path} cannot be resolved ({e}). It may predate the escaping "
                f"of '${{' in cfg.yaml; rewrite it or remove the folder."
            ) from e
        if existing_identity != cfg_identity:
            raise RuntimeError(
                f"{yaml_path} already exists but describes a different "
                f"configuration. Either two distinct configurations collided on "
                f"the same {len(config_id)}-character ID (see src.utils.id), or "
                f"cfg.yaml has been edited."
            )

    data_path = os.path.join(run_dir, "data.npz")
    if os.path.exists(data_path):
        if verbose:
            print(f":warning:  Data for this seed already exists: {data_path}.")
        if not cfg.results.force_run:
            if verbose:
                print(
                    ":x: Stopping. If you want to force this run, "
                    "pass [italic]results.force_run=True[/italic]."
                )
            return None
        elif verbose:
            print(
                ":white_check_mark: Overriding. To prevent it, interrupt "
                "before the end of the run. Next time, run with "
                "[italic]results.force_run=False[/italic]."
            )

    os.makedirs(config_dir, exist_ok=True)

    if not os.path.exists(yaml_path):
        # Seeds of one config_id share this folder and may start within
        # milliseconds of each other, if launched in parallel (e.g., with SLURM jobs).
        # Thus, more than one can find the file missing and write at the same time.
        # To prevent that, write to a temp file and rename (atomic_open):
        # os.replace is atomic on POSIX, so a reader never sees a half-written
        # cfg.yaml. Every seed writes the same configuration, so which one lands
        # last does not change the config ID. The int or float types in cfg.yaml
        # are those of the seed that lands last.
        with atomic_open(yaml_path, "w", encoding="utf-8") as f:
            f.write(config_for_save(cfg))

    # The seed folder is made only once cfg.yaml matches this configuration
    os.makedirs(run_dir, exist_ok=True)

    # Update the directory (append /<config_id>/<rng_seed>)
    cfg.results.data_dir = run_dir

    if verbose:
        print(f":floppy_disk: Data will be saved at {run_dir}")
    return config_id


def decode_stat(raw):
    """
    Decode one cell of a statistics table and return it as a Python value: a
    string holding JSON is parsed (a list for a curve, a number or a string
    for a scalar), a string that is not JSON is returned as it is, and
    anything else is returned unchanged.
    """

    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


def load_yaml(path: str):
    """Return the YAML file `path` as plain Python values, read the way Hydra
    reads config files: with OmegaConf's loader, which reads an exponent float
    such as 1e-3 as a float where PyYAML reads it as a string. Interpolations
    are left unresolved, and an empty file returns {}."""

    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


@lru_cache(maxsize=None)
def read_run_cfg(path) -> tuple:
    """Read the cfg.yaml at `path` and return (cfg, resolved dict, None), with
    the dict being strip_config(cfg). When the file cannot be read, is empty or
    not a mapping, or cannot be resolved, return (None, None, reason), with the
    reason written to follow the file's name. The result is cached by `path`, so
    each file is read once per process and repeated calls return the same
    objects."""

    try:
        cfg = OmegaConf.load(path)
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as e:
        return None, None, f"could not be read ({e})"
    if not isinstance(cfg, DictConfig) or not cfg:
        return None, None, f"is empty or not a mapping (parsed as {type(cfg).__name__})"
    try:
        return cfg, strip_config(cfg), None
    except OmegaConfBaseException as e:
        return None, None, f"could not be resolved ({e})"


def flatten_cfg(nested: dict, drop_interpolations=True) -> dict:
    """Return `nested` flattened to {dotted_key: value}.

    Nested mappings are flattened recursively, with keys joined by ".". An empty
    mapping is kept as the value "{}", so a key holding one stays distinct from
    an absent key. With `drop_interpolations`, string values holding "${" are
    left out.
    """

    flat = {}
    for key, value in nested.items():
        if isinstance(value, dict) and value:
            for sub_key, sub_value in flatten_cfg(value, drop_interpolations=False).items():
                flat[f"{key}.{sub_key}"] = sub_value
        else:
            flat[str(key)] = "{}" if isinstance(value, dict) else value
    if drop_interpolations:
        flat = {k: v for k, v in flat.items() if not (isinstance(v, str) and "${" in v)}
    return flat
