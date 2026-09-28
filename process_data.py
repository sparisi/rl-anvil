import os
import argparse
import numpy as np
import pandas as pd
import yaml
import json
from pathlib import Path
from tqdm import tqdm

from src.utils.id import DROPPED_CFG_KEYS


def process_data_folders(
    data_dir: str,
    verbose: bool = False,
    log_configs: bool = False,
) -> pd.DataFrame:
    """
    Loops over a specified folder structure (<data_dir>/<config_id>/<rng_seed>/),
    loads the 'data.npz' file from each seed subdirectory, extracts its NumPy
    array values, and consolidates them into a pandas DataFrame.
    It also reads the 'cfg.yaml' file in each '<config_id>' directory and
    includes its key-value pairs in the DataFrame.
    Additionally, it takes the 'rng_seed' from the name of the seed
    subdirectory (e.g., 33 from <config_id>/33/) and adds it as a column.
    Time-series entries (np.array from .npz files) are converted to strings for
    compatibility.

    Additionally, DataFrame columns are MultiIndexed and grouped according to
    'hyperparameter', 'statistic', and 'seed'.

    Columns with constant values across all rows are then dropped.

    The function expects the following folder structure:

    data_dir
    ├── config_A
    │   ├── cfg.yaml
    │   ├── 0
    │   │   └── data.npz
    │   └── 1
    │       └── data.npz
    ├── config_B
    │   ├── cfg.yaml
    │   ├── 0
    │   │   └── data.npz
    │   └── ...
    └── ...

    Any other content of a seed directory ('memory.npz', 'heatmaps/',
    'videos/', ...) is ignored.

    Args:
        data_dir (str): The path to the data directory, which contains
                        one or more '<config_id>' subdirectories.
        verbose (bool): If True, print the recap header and summary
                        (without the per-configuration details).
        log_configs (bool): If True, write the full recap (including every
                        configuration's hyperparameters) to a
                        'configs_log.txt' file inside data_dir.

    Returns:
        pandas.DataFrame: A DataFrame containing the combined data from all .npz files,
                          with configuration keys, and with constant columns dropped.
    """

    all_df_rows = []
    found_any_npz = False  # Flag to track if any NPZ files were processed
    skipped_folders: list[tuple[str, str]] = []  # (dirpath, reason)
    missing_npz: dict[str, list[int]] = {}  # cfg dirpath -> seeds lacking 'data.npz'

    if not os.path.isdir(data_dir):
        raise ValueError(
            f"Error: The provided data_dir '{data_dir}' is not a valid directory or does not exist."
        )

    print(f"Starting directory walk from: {data_dir}")

    # Collect the config directories first: any directory holding a 'cfg.yaml'.
    # Their subtrees are pruned so that seed directories -- and the potentially
    # large 'heatmaps'/'videos' folders inside them -- are never walked.
    cfg_dirs = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        if "cfg.yaml" in filenames:
            dirnames[:] = []
            cfg_dirs.append(dirpath)

    for dirpath in tqdm(cfg_dirs):
        # Make DataFrame from cfg.yaml
        cfg_file_path = os.path.join(dirpath, "cfg.yaml")
        cfg_dict = yaml.safe_load(Path(cfg_file_path).read_text())
        if not isinstance(cfg_dict, dict):
            skipped_folders.append(
                (
                    dirpath,
                    f"'cfg.yaml' is empty or malformed (parsed as {type(cfg_dict).__name__}).",
                )
            )
            continue
        df_cfg = pd.json_normalize(cfg_dict, sep=".")

        # Make a flat dict, dropping keys that are irrelevant for analysis
        dict_cfg_flat = df_cfg.to_dict(orient="records")[0]
        dict_cfg_flat = {
            ("hyperparameter", k): v
            for k, v in dict_cfg_flat.items()
            if k not in DROPPED_CFG_KEYS and not k.startswith("wandb")
        }
        dict_cfg_flat[("_meta", "cfg_dir")] = os.path.relpath(dirpath, data_dir)

        seed_dirs = sorted(
            (
                entry
                for entry in os.listdir(dirpath)
                if entry.isdigit() and os.path.isdir(os.path.join(dirpath, entry))
            ),
            key=int,
        )
        if not seed_dirs:
            skipped_folders.append((dirpath, "no seed subdirectories found."))
            continue

        for seed in seed_dirs:
            npz_file_path = os.path.join(dirpath, seed, "data.npz")
            if not os.path.isfile(npz_file_path):
                missing_npz.setdefault(dirpath, []).append(int(seed))
                continue

            row_dict = dict(dict_cfg_flat)
            # The seed always comes from the directory name, never from cfg.yaml's own
            # experiment.rng_seed field -- cfg.yaml is shared by every seed directory
            # under this config, so trusting its single recorded value would mislabel
            # every run with whichever seed happened to be in the yaml.
            row_dict[("seed", "experiment.rng_seed")] = int(seed)

            try:
                with np.load(npz_file_path) as data:
                    dict_data = {
                        ("statistic", k): json.dumps(data[k].tolist()) for k in data
                    }
                    all_df_rows.append(row_dict | dict_data)
                    found_any_npz = True
            except Exception as e:
                print(f"    Error loading {npz_file_path}: {e}")
                continue

    n_skipped = len(skipped_folders) + sum(len(seeds) for seeds in missing_npz.values())
    if verbose and n_skipped:
        print(f"Skipped {n_skipped} folders:")
        for path, reason in skipped_folders:
            print(f"  {path}: {reason}")
        for path, seeds in missing_npz.items():
            seed_str = ", ".join(str(s) for s in sorted(seeds))
            print(f"  {path}{os.sep} {{{seed_str}}}: 'data.npz' is missing.")

    if not found_any_npz:
        raise FileNotFoundError(
            f"\nNo .npz data found in the specified structure under '{data_dir}'."
        )

    # Create a pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(all_df_rows)

    # --- Recap ---
    seed_col = ("seed", "experiment.rng_seed")
    cfg_dir_col = ("_meta", "cfg_dir")
    hyp_cols = [c for c in df.columns if c[0] == "hyperparameter"]

    varying_hyp = sorted(
        [c for c in hyp_cols if df[c].astype(str).nunique() > 1],
        key=lambda c: c[1],
    )

    need_recap = verbose or log_configs
    header_lines: list[str] = []
    summary_line: str | None = None
    stat_line: str | None = None
    config_blocks: list[list[str]] = []

    if need_recap:
        header_lines = ["=" * 60, "RECAP", "=" * 60]

        stat_cols = sorted({c[1] for c in df.columns if c[0] == "statistic"})
        if stat_cols:
            stat_line = f"Statistics: {stat_cols}"

        if not hyp_cols or seed_col not in df.columns:
            summary_line = f"Total rows: {len(df)}"
        elif not varying_hyp:
            seeds = sorted(df[seed_col].dropna().unique().tolist())
            folders = sorted(df[cfg_dir_col].unique().tolist()) if cfg_dir_col in df.columns else []
            folder_str = f"  [{', '.join(folders)}]" if folders else ""
            summary_line = f"1 configuration{folder_str} — {len(seeds)} run(s), seeds: {seeds}"
        else:
            groupby_keys = [df[c].astype(str).rename(c[1]) for c in varying_hyp]
            groups = df.groupby(groupby_keys)
            summary_line = (
                f"{len(groups)} unique configuration(s). "
                f"Varying keys: {[c[1] for c in varying_hyp]}"
            )
            for i, (name, group) in enumerate(groups):
                if not isinstance(name, tuple):
                    name = (name,)
                seeds = sorted(group[seed_col].dropna().unique().tolist())
                folders = (
                    sorted(group[cfg_dir_col].unique().tolist())
                    if cfg_dir_col in group.columns
                    else []
                )
                folder_str = f"  [{', '.join(folders)}]" if folders else ""
                block = [f"Config {i + 1}{folder_str} — {len(seeds)} run(s), seeds {seeds}:"]
                for col, val in zip(varying_hyp, name):
                    block.append(f"  {col[1]} = {val}")
                config_blocks.append(block)

    if verbose:
        print()
        for line in header_lines:
            print(line)
        if summary_line is not None:
            print(summary_line)
        if stat_line is not None:
            print()
            print(stat_line)

    if log_configs:
        log_configs_path = os.path.join(data_dir, "configs_log.txt")
        with open(log_configs_path, "w", encoding="utf-8") as f:
            for line in header_lines:
                f.write(line + "\n")
            if summary_line is not None:
                f.write(summary_line + "\n")
            if stat_line is not None:
                f.write("\n")
                f.write(stat_line + "\n")
            f.write("\n")
            for block in config_blocks:
                for line in block:
                    f.write(line + "\n")
                f.write("\n")
        print(f"\nConfigs log saved to {log_configs_path}")

    # Drop internal metadata columns before returning
    meta_cols = [c for c in df.columns if c[0] == "_meta"]
    df = df.drop(columns=meta_cols)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--log_configs",
        action="store_true",
        help="Save configs_log.txt (with all configurations) in the parquet's directory.",
    )
    args = parser.parse_args()

    df = process_data_folders(args.folder, verbose=args.verbose, log_configs=args.log_configs)

    # print(f"header: {df.columns.values}")
    # print(df)

    gzip_dir = os.path.join(args.folder, "results.gzip")
    try:
        df.to_parquet(gzip_dir, index=False)
        print(f"\nDataFrame successfully saved to {gzip_dir}")
    except Exception as e:
        print(f"\nError saving DataFrame to {gzip_dir}: {e}")
