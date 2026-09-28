"""Repo paths, resolved once and shared.

The root is found by walking up until the marker file appears, NOT by counting
directories with `parents[n]` or chained `os.path.dirname`. A counted depth is
silently wrong the moment a module moves (src/sweep_utils.py -> src/utils/sweep.py
already broke it twice), and the failure is a config that cannot be found rather
than an error pointing at the cause.
"""

from pathlib import Path

ROOT_MARKER = "pyproject.toml"


def repo_dir() -> Path:
    """The repository root: the first ancestor holding ROOT_MARKER."""
    here = Path(__file__).resolve()
    for folder in here.parents:
        if (folder / ROOT_MARKER).is_file():
            return folder
    raise RuntimeError(
        f"No {ROOT_MARKER} in any parent of {here}. This file must live inside "
        "the repository for the config paths to resolve."
    )


REPO_DIR = repo_dir()

# Strings, not Paths: they are passed to Hydra (initialize_config_dir) and joined
# with os.path elsewhere.
CONFIG_DIR = str(REPO_DIR / "configs")
SWEEP_DIR = str(REPO_DIR / "configs" / "sweeps")
