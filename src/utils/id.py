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
    """`cfg` as a plain resolved dict, without the keys that do not define a
    configuration."""
    container = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
    )
    return drop_keys(container, dropped)


def drop_keys(d: dict, drop: tuple) -> dict:
    for dotted in drop:
        *parents, leaf = dotted.split(".")
        node = d
        for k in parents:
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(leaf, None)
    return d


def numbers_to_float(x):
    """Ints to float, so that `lrate=1` and `lrate=1.0` -- same run, two
    YAML spellings -- do not hash to two different config IDs.

    Only ever fed to make the ID. The config written to disk and the config the
    run is built from keep their original types, so ints stay ints.
    """
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
    """The YAML written to cfg.yaml: resolved, sorted, stripped of the keys
    that do not define the configuration, types untouched."""
    return OmegaConf.to_yaml(
        OmegaConf.create(strip_config(cfg, dropped)),
        sort_keys=True,
    )


def config_identity(cfg: DictConfig, dropped: tuple = DROPPED_CFG_KEYS) -> str:
    """What make_id() hashes: config_for_save() with ints as floats."""
    return OmegaConf.to_yaml(
        OmegaConf.create(numbers_to_float(strip_config(cfg, dropped))),
        sort_keys=True,
    )


def make_id(
    cfg: DictConfig = None,
    overrides=None,
    n_chars: int = CONFIG_ID_LEN,
) -> str:
    """The config ID of `cfg`, or of the config `overrides` would compose."""

    assert (cfg is None) != (overrides is None), (
        "must pass either cfg or overrides, and not both"
    )
    if overrides is not None:
        cfg = compose_cfg(overrides)
    digest = hashlib.md5(
        config_identity(cfg).encode(),
        usedforsecurity=False,
    ).hexdigest()
    return digest[:n_chars]


@lru_cache(maxsize=None)
def composed_config(overrides: tuple) -> dict:
    """The flat config `overrides` compose, as {dotted_key: value}, without the
    keys that do not define a configuration.

    The same shape process_data records for a run, so the two compare key by key:
    a sweep entry declares part of a configuration, and the rest comes from the
    config files it selects and from configs/default.yaml under them.

    Interpolations are kept, unlike `flatten_cfg`'s default: this config is
    resolved, so a value still holding "${" is one the config really contains,
    and process_data keeps it too. Dropping it here would leave a key on one
    side of the comparison and not the other.

    `overrides` is a tuple (rather than a list) so the composition is cached.
    """

    # Imported here rather than at module level: src.utils.plot pulls in
    # matplotlib and seaborn, which the training path has no use for.
    from src.utils.plot import flatten_cfg

    return flatten_cfg(strip_config(compose_cfg(overrides)), drop_interpolations=False)


def compose_cfg(overrides) -> DictConfig:
    """The config main.py would build from `overrides`, without running main.py.

    For example
        compose_cfg(("environment=lunar_lander", "++agent.critic.lr=0.001"))

    returns the DictConfig resulting from running
        python main.py environment=lunar_lander ++agent.critic.lr=0.001
    """
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="default", overrides=list(overrides))


def prepare_run(cfg: DictConfig, verbose: bool = True) -> str | None:
    """Check for existing data.npz for the same run. If so, and the user has not
    forced this run, stop. The presence of videos, heatmaps, ... does not stop
    the run.

    Then, make the root directory to save data (if needed) and save the
    cfg.yaml that identifies this run.
    No other folder is created at this time. During an experiment, the code always
    runs os.makedirs(..., exist_ok=True) to guard against deleted folders.

    Returns the config ID, or None if the run should be skipped. On return
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
            print(":warning:  The user requested to save data but "
                  "[bold italic]results.data_dir[/bold italic] is not defined. "
                  "Using default directory: [italic]data[/italic]")
        cfg.results.data_dir = "data"

    # Nothing to save, the run can start
    if cfg.results.data_dir is None:
        return config_id

    # Make data root folder
    config_dir = os.path.join(cfg.results.data_dir, config_id)
    run_dir = os.path.join(config_dir, str(cfg.experiment.rng_seed))

    data_path = os.path.join(run_dir, "data.npz")
    if os.path.exists(data_path):
        if verbose:
            print(f":warning:  Data for this seed already exists: {data_path}.")
        if not cfg.results.force_run:
            if verbose:
                print(":x: Stopping. If you want to force this run, "
                      "pass [italic]results.force_run=True[/italic].")
            return None
        elif verbose:
            print(":white_check_mark: Overriding. To prevent it, interrupt "
                  "before the end of the run. Next time, run with "
                  "[italic]results.force_run=False[/italic].")

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
