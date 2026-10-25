import copy
import hashlib
import os
import tempfile
from functools import lru_cache
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from rich import print

from src.utils.paths import CONFIG_DIR


# Keys that do not define a configuration. See configs/full_config.yaml for details.
DROPPED_CFG_KEYS = (
    "hydra",
    "wandb",
    "extra",
    "meta",
    "results",
    "experiment.rng_seed",
)

# Characters of the md5 digest a config ID keeps.
CONFIG_ID_LEN = 8

# A config ID is the first CONFIG_ID_LEN of the 32 hex characters of the MD5 of
# `cfg` stripped of DROPPED_CFG_KEYS.
# This is much better than Hydra default directory naming that concatenates all
# keys=value in the DictConfig.


def strip_config(cfg: DictConfig, dropped: tuple = DROPPED_CFG_KEYS) -> dict:
    """Return `cfg` as a plain resolved dict, without the keys in `dropped`."""

    container = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
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


def config_for_save(cfg: DictConfig, dropped: tuple = DROPPED_CFG_KEYS) -> str:
    """Return the YAML written to cfg.yaml: `cfg` resolved, with sorted keys,
    without the keys in `dropped`, and with its original types."""

    return OmegaConf.to_yaml(
        OmegaConf.create(strip_config(cfg, dropped)),
        sort_keys=True,
    )


def config_identity(cfg: DictConfig, dropped: tuple = DROPPED_CFG_KEYS) -> str:
    """Return the YAML that the config ID hashes: the same YAML as
    config_for_save(), with every int converted to float."""

    # With ints as floats, `lrate=1` and `lrate=1.0` hash to the same config ID.
    # The conversion applies to the hashed YAML only: the config written to disk
    # and the config the run is built from keep their original types.
    return OmegaConf.to_yaml(
        OmegaConf.create(numbers_to_float(strip_config(cfg, dropped))),
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
    digest = hashlib.md5(
        config_identity(cfg).encode(),
        usedforsecurity=False,
    ).hexdigest()
    return digest[:n_chars]


@lru_cache(maxsize=None)
def _composed_config(overrides: tuple) -> dict:
    """Return the flat config that `overrides` compose, without the keys in
    DROPPED_CFG_KEYS. The result is cached: every call with the same
    `overrides` returns the same object."""

    # Imported here rather than at module level: src.utils.plot pulls in
    # matplotlib and seaborn, which the training path has no use for.
    from src.utils.plot import flatten_cfg

    return flatten_cfg(strip_config(compose_cfg(overrides)), drop_interpolations=False)


def composed_config(overrides: tuple) -> dict:
    """Return the flat config that `overrides` compose, as {dotted_key: value},
    without the keys in DROPPED_CFG_KEYS.

    The keys have the same shape as the 'hyperparameter' columns of a
    process_data parquet, so the two compare key by key.

    Interpolations are kept: the config is resolved, so a value still holding
    "${" is one the config contains, and the parquet holds it too.

    `overrides` is a tuple (rather than a list) so the composition is cached.
    Each call returns its own deep copy of the cached result.
    """

    return copy.deepcopy(_composed_config(overrides))


def compose_cfg(overrides) -> DictConfig:
    """Return the DictConfig that main.py builds from the Hydra `overrides`,
    without running main.py.

    For example
        compose_cfg(("environment=gymnasium/lunar_lander", "++agent.critic.lr=0.001"))

    returns the DictConfig resulting from running
        python main.py environment=gymnasium/lunar_lander ++agent.critic.lr=0.001
    """

    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="default", overrides=list(overrides))


def prepare_run(cfg: DictConfig, verbose: bool = True) -> str | None:
    """Return the config ID of `cfg`, or None if the run should be skipped.

    The run is skipped if a data.npz for the same run exists and
    results.force_run is not set. The presence of videos, heatmaps, ... does
    not skip the run.

    Otherwise, make the seed directory (if data is saved) and save the cfg.yaml
    that identifies this run. No other folder is created. On return
    results.data_dir points at the seed folder, so everything the run writes
    goes into <data_dir>/<config_id>/<rng_seed>/
    """

    cfg_identity = config_identity(cfg)
    config_id = make_id(cfg)

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

    os.makedirs(run_dir, exist_ok=True)

    yaml_path = os.path.join(
        config_dir,
        "cfg.yaml",
    )

    if os.path.exists(yaml_path):
        # Compare identities, not file bytes: cfg.yaml keeps the types it was
        # written with, so two seeds passing 1 and 1.0 for the same key write
        # different bytes for the same configuration.
        if config_identity(OmegaConf.load(yaml_path)) != cfg_identity:
            raise RuntimeError(
                f"{yaml_path} already exists but describes a different "
                f"configuration. Either two distinct configurations collided on "
                f"the same {len(config_id)}-character ID (see src.utils.id), or "
                f"cfg.yaml has been edited."
            )
    else:
        # Seeds of one config_id share this folder and may start within
        # milliseconds of each other, if launched in parallel (e.g., with SLURM jobs).
        # Thus, more than one can find the file missing and write at the same time.
        # To prevent that, write to a temp file and rename: os.replace is atomic
        # on POSIX, so a reader never sees a half-written cfg.yaml. Every seed
        # writes identical content, so which one lands last does not matter.
        fd, tmp_path = tempfile.mkstemp(
            dir=config_dir,
            prefix=".cfg-",
            suffix=".yaml",
        )
        os.close(fd)
        try:
            with open(tmp_path, "w") as f:
                f.write(config_for_save(cfg))
            os.chmod(tmp_path, 0o644)  # mkstemp creates with 0600
            os.replace(tmp_path, yaml_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    # Update the directory (append /<config_id>/<rng_seed>)
    cfg.results.data_dir = run_dir

    if verbose:
        print(f":floppy_disk: Data will be saved at {run_dir}")
    return config_id
