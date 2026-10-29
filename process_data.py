import os
import re
import shutil
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

from src.utils.id import CONFIG_ID, SEED, flatten_cfg, make_id, read_run_cfg

# Minimum number of rows in each row group of the merged parquet file.
MIN_ROW_GROUP_ROWS = 10_000

# Largest number of config directories a worker loads and writes to one part
# file. A worker holds the rows of one batch at a time, so this bounds its
# memory. When there are fewer than this many directories per worker, a batch
# holds the number of directories divided by the number of workers, rounded
# down and at least 1.
CONFIG_DIRS_PER_PART = 8

# ProcessPoolExecutor on Windows accepts at most this many workers.
MAX_WINDOWS_WORKERS = 61

CFG_DIR_COL = ("_meta", "cfg_dir")


def default_workers() -> int:
    """Return the number of CPUs this process may run on: the CPU affinity where
    the platform reports it (a SLURM allocation sets it), os.cpu_count()
    otherwise.
    """

    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def check_column_types(flat_cfgs) -> None:
    """Raise a TypeError naming every hyperparameter whose values pyarrow cannot
    store in one parquet column, with the types of its values.

    `flat_cfgs` are flat configs keyed by (group, key) tuples. Only the
    ("hyperparameter", key) columns are checked, and the error names each by
    its key. NaN values, which stand for keys absent from a config, are left
    out of the listed types.
    """

    df = pd.DataFrame(list(flat_cfgs))
    bad = []
    for col in df.columns:
        if col[0] != "hyperparameter":
            continue
        try:
            pa.array(df[col], from_pandas=True)
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
            types = sorted(
                {
                    type(v).__name__
                    for v in df[col]
                    if not (isinstance(v, float) and np.isnan(v))
                }
            )
            bad.append(f"  {col[1]}: {', '.join(types)}")
    if bad:
        raise TypeError(
            "These hyperparameters hold values of types that cannot share a "
            "parquet column:\n" + "\n".join(bad)
        )


def report(header: str, lines: list[str], verbose: bool) -> None:
    """Print `header` followed by `lines` when `verbose` is true, and `header`
    with a pointer to -v otherwise. Nothing is printed when `lines` is empty.
    """

    if not lines:
        return
    if verbose:
        print(f"{header}:")
        for line in lines:
            print(f"  {line}")
    else:
        print(f"{header}. Run with -v to list them.")


def read_cfg(dirpath: str, data_dir: str) -> tuple:
    """Read the 'cfg.yaml' of the config directory `dirpath` and return
    (flat config, skip reason, expected config ID).

    The flat config holds the key-value pairs of 'cfg.yaml' without the keys in
    DROPPED_CFG_KEYS, keyed by ("hyperparameter", key), the config ID taken
    from the name of `dirpath`, and the path of `dirpath` relative to
    `data_dir`. It is None when 'cfg.yaml' cannot be read, is empty, is not a
    mapping, or cannot be resolved (src.utils.id.read_run_cfg), and the skip
    reason says why; the skip reason is None otherwise.
    The expected config ID is the one 'cfg.yaml' hashes to
    (src.utils.id.make_id) when it differs from the name of `dirpath`, and
    None when they match.
    """

    cfg, cfg_dict, reason = read_run_cfg(os.path.join(dirpath, "cfg.yaml"))
    if reason is not None:
        return None, f"'cfg.yaml' {reason}.", None

    # Check the folder name against the config ID that cfg.yaml hashes to
    config_id = os.path.basename(os.path.abspath(dirpath))
    expected_id = make_id(cfg)
    if config_id == expected_id:
        expected_id = None

    flat = flatten_cfg(cfg_dict, drop_interpolations=False)
    dict_cfg_flat = {("hyperparameter", k): v for k, v in flat.items()}
    dict_cfg_flat[CONFIG_ID] = config_id
    dict_cfg_flat[CFG_DIR_COL] = os.path.relpath(dirpath, data_dir)
    return dict_cfg_flat, None, expected_id


def load_seeds(dirpath: str, dict_cfg_flat: dict) -> tuple:
    """Load the 'data.npz' of every seed directory of the config directory
    `dirpath` and return (rows, skipped folders, seeds lacking 'data.npz').

    Each row is `dict_cfg_flat` with the seed and every entry of 'data.npz' as
    a JSON string. Each skipped folder is a (path, reason) pair.
    """

    seed_dirs = sorted(
        (
            entry
            for entry in os.listdir(dirpath)
            if re.fullmatch(r"-?\d+", entry)
            and os.path.isdir(os.path.join(dirpath, entry))
        ),
        key=int,
    )
    if not seed_dirs:
        return [], [(dirpath, "no seed subdirectories found.")], []

    rows, skipped, missing = [], [], []
    for seed in seed_dirs:
        npz_file_path = os.path.join(dirpath, seed, "data.npz")
        if not os.path.isfile(npz_file_path):
            missing.append(int(seed))
            continue

        row_dict = dict(dict_cfg_flat)
        # The seed is the name of the seed directory, since cfg.yaml is shared
        # by all seeds of a config.
        row_dict[SEED] = int(seed)

        try:
            with np.load(npz_file_path) as data:
                dict_data = {
                    ("statistic", k): json.dumps(data[k].tolist()) for k in data
                }
                rows.append(row_dict | dict_data)
        except Exception as e:
            skipped.append(
                (
                    os.path.join(dirpath, seed),
                    f"'data.npz' could not be loaded ({e}).",
                )
            )
    return rows, skipped, missing


def load_batch(index: int, items: list, parts_dir: str) -> tuple:
    """Load the rows of a batch of config directories (load_seeds) and write
    them to the parquet file part_<index>.parquet in `parts_dir`.

    `items` are (config directory, flat config) pairs. Return (part path,
    columns, seeds per directory, skipped folders, seeds lacking 'data.npz' per
    directory). The part path is None when the batch has no row, and the
    columns are those of the part, in order of first appearance in the rows.
    The "_meta" columns are left out of the part.
    """

    rows, seeds_of, skipped, missing_of = [], {}, [], {}
    for dirpath, dict_cfg_flat in items:
        dir_rows, dir_skipped, dir_missing = load_seeds(dirpath, dict_cfg_flat)
        rows.extend(dir_rows)
        skipped.extend(dir_skipped)
        if dir_rows:
            seeds_of[dirpath] = [row[SEED] for row in dir_rows]
        if dir_missing:
            missing_of[dirpath] = dir_missing
    if not rows:
        return None, [], seeds_of, skipped, missing_of

    # Object columns keep the ints of a key that some rows lack as ints, as in
    # the schema write_parquet infers.
    part = pd.DataFrame(rows, dtype=object)
    part = part.drop(columns=[c for c in part.columns if c[0] == "_meta"])
    part_path = os.path.join(parts_dir, f"part_{index:06d}.parquet")
    pq.write_table(pa.Table.from_pandas(part, preserve_index=False), part_path)
    return part_path, list(part.columns), seeds_of, skipped, missing_of


def write_parquet(part_paths: list, columns: list, cfg_df: pd.DataFrame, path: str) -> None:
    """Merge the parquet files `part_paths` into one parquet file at `path`, with
    the columns `columns` in that order.

    The schema is the one pyarrow infers for the whole table: the
    hyperparameter and config columns are inferred from `cfg_df`, which holds
    the flat config of every config directory with a row, the seed column is
    int64, and every statistic column holds JSON strings. Each part is cast to
    that schema, with a null column for every column it lacks and for every
    column of null type (a key that is null or absent in every config), and
    the parts are written in row groups of at least MIN_ROW_GROUP_ROWS rows.
    """

    proxy = cfg_df.drop(columns=[c for c in cfg_df.columns if c[0] == "_meta"])
    proxy[SEED] = 0
    for col in columns:
        if col[0] == "statistic":
            proxy[col] = ""
    schema = pa.Schema.from_pandas(proxy[columns], preserve_index=False)

    pending, n_pending = [], 0
    # Snappy compresses and decompresses faster than gzip and writes larger files.
    with pq.ParquetWriter(path, schema, compression="snappy") as writer:
        for part_path in tqdm(part_paths, desc="Writing parquet"):
            part = pq.read_table(part_path)
            pending.append(
                pa.Table.from_arrays(
                    [
                        part[f.name].cast(f.type)
                        if f.name in part.column_names and not pa.types.is_null(f.type)
                        else pa.chunked_array([pa.nulls(part.num_rows, f.type)])
                        for f in schema
                    ],
                    schema=schema,
                )
            )
            n_pending += part.num_rows
            if n_pending >= MIN_ROW_GROUP_ROWS:
                writer.write_table(pa.concat_tables(pending))
                pending, n_pending = [], 0
        if pending:
            writer.write_table(pa.concat_tables(pending))


def process_data_folders(
    data_dir: str,
    parquet_path: str,
    verbose: bool = False,
    log_configs: bool = False,
    workers: int | None = None,
) -> None:
    """Write a parquet file at `parquet_path` with one row per seed directory
    (<config_id>/<rng_seed>/) under `data_dir` that holds a readable 'data.npz'
    file. A '<config_id>' directory is any directory holding a 'cfg.yaml', at
    any depth under `data_dir`.

    Each row holds the key-value pairs of the 'cfg.yaml' file in its
    '<config_id>' directory, without the keys in DROPPED_CFG_KEYS; every entry
    of its 'data.npz' file as a JSON string; the seed taken from the name of
    the seed directory (e.g., 33 from <config_id>/33/); and the config ID taken
    from the name of the '<config_id>' directory. Each column is named by a
    (group, key) tuple, with the groups 'hyperparameter', 'statistic', 'seed',
    and 'config'. The parquet file stores each name as the string of its
    tuple, "('<group>', '<key>')".

    A '<config_id>' directory whose name differs from the config ID of its
    'cfg.yaml' (src.utils.id.make_id) is reported. Its rows are kept. Rows that
    share their config ID and seed are reported and kept.

    Every 'cfg.yaml' is read before any 'data.npz' is loaded. A hyperparameter
    whose values cannot share a parquet column raises a TypeError that names it.
    The config directories are read in parallel by a pool of `workers`
    processes. Each worker writes the rows of a batch of config directories to
    a part file in '<parquet_path>.parts', and the parts are merged into
    `parquet_path` in the order of the directory walk (write_parquet), so no
    process holds every row at once. The merged file is written to
    '<parquet_path>.tmp' and moved onto `parquet_path` once complete. The part
    files and the temporary file are removed whether or not the write succeeds.

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
                        one or more '<config_id>' directories at any depth.
        parquet_path (str): The path of the parquet file to write.
        verbose (bool): If True, list the skipped folders, the folders whose
                        name differs from their config ID, and the rows sharing
                        their config ID and seed, and print the recap
                        header and summary (without the per-configuration
                        details).
        log_configs (bool): If True, write the full recap (including every
                        configuration's hyperparameters) to a
                        'configs_log.txt' file inside data_dir.
        workers (int | None): The number of worker processes, at most
                        MAX_WINDOWS_WORKERS on Windows. None uses
                        default_workers().
    """

    skipped_folders: list[tuple[str, str]] = []  # (dirpath, reason)
    missing_npz: dict[str, list[int]] = {}  # cfg dirpath -> seeds lacking 'data.npz'
    id_mismatches: list[tuple[str, str]] = []  # (dirpath, config ID of its cfg.yaml)
    seeds_of: dict[str, list[int]] = {}  # cfg dirpath -> seeds with a row
    columns: dict = {}  # row columns in order of first appearance (values unused)
    part_paths: list[str] = []

    data_dir = os.path.normpath(data_dir)
    if not os.path.isdir(data_dir):
        raise ValueError(
            f"The provided data_dir '{data_dir}' is not a valid directory or does not exist."
        )

    print(f"Starting directory walk from: {data_dir}")

    # Collect the config directories first: any directory holding a 'cfg.yaml'.
    # Their subtrees are pruned so that seed directories -- and the potentially
    # large 'heatmaps'/'videos' folders inside them -- are never walked. The
    # other subdirectories are walked in sorted order, so the rows come in the
    # same order on every filesystem.
    cfg_dirs = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        if "cfg.yaml" in filenames:
            dirnames[:] = []
            cfg_dirs.append(dirpath)
        dirnames.sort()

    workers = workers or default_workers()
    if os.name == "nt":
        workers = min(workers, MAX_WINDOWS_WORKERS)

    parts_dir = parquet_path + ".parts"
    tmp_path = parquet_path + ".tmp"
    pool = ProcessPoolExecutor(max_workers=workers)
    shutil.rmtree(parts_dir, ignore_errors=True)
    os.makedirs(parts_dir)
    try:
        # Every cfg.yaml is read, and its hyperparameter columns are checked,
        # before any data.npz is loaded. Each task reads a batch of cfg.yaml
        # files, so a read does not pay an inter-process round trip of its own.
        cfg_flats = {}  # cfg dirpath -> flat dict of its cfg.yaml
        results = pool.map(
            partial(read_cfg, data_dir=data_dir),
            cfg_dirs,
            chunksize=max(1, len(cfg_dirs) // (4 * workers)),
        )
        for dirpath, (dict_cfg_flat, reason, expected_id) in zip(
            cfg_dirs,
            tqdm(results, total=len(cfg_dirs), desc="Reading cfg.yaml"),
        ):
            if dict_cfg_flat is None:
                skipped_folders.append((dirpath, reason))
                continue
            if expected_id is not None:
                id_mismatches.append((dirpath, expected_id))
            cfg_flats[dirpath] = dict_cfg_flat

        check_column_types(cfg_flats.values())

        items = list(cfg_flats.items())
        size = max(1, min(CONFIG_DIRS_PER_PART, len(items) // workers))
        batches = [
            items[i:i + size]
            for i in range(0, len(items), size)
        ]

        # The bar advances by the directories of each batch as the batch
        # finishes. The results are stored by batch index, so the parts are
        # merged in the order of the directory walk.
        futures = {
            pool.submit(load_batch, i, batch, parts_dir): i
            for i, batch in enumerate(batches)
        }
        results = [None] * len(batches)
        with tqdm(total=len(items), desc="Loading data.npz") as bar:
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
                bar.update(len(batches[i]))
        for part_path, part_columns, batch_seeds, skipped, batch_missing in results:
            if part_path is not None:
                part_paths.append(part_path)
            columns.update(dict.fromkeys(part_columns))
            seeds_of.update(batch_seeds)
            skipped_folders.extend(skipped)
            missing_npz.update(batch_missing)
        pool.shutdown()

        n_skipped = len(skipped_folders) + sum(len(seeds) for seeds in missing_npz.values())
        skipped_lines = [f"{path}: {reason}" for path, reason in skipped_folders]
        for path, seeds in missing_npz.items():
            seed_str = ", ".join(str(s) for s in sorted(seeds))
            skipped_lines.append(f"{path}{os.sep} {{{seed_str}}}: 'data.npz' is missing.")
        report(f"Skipped {n_skipped} folders", skipped_lines, verbose)

        report(
            f"{len(id_mismatches)} config folder(s) not named after the config ID "
            f"of their cfg.yaml (rows kept)",
            [f"{path}: expected {expected_id}" for path, expected_id in id_mismatches],
            verbose,
        )

        if not part_paths:
            raise FileNotFoundError(
                f"No .npz data found in the specified structure under '{data_dir}'."
            )

        # One row per written row, with the columns the checks below need
        index = pd.DataFrame(
            [
                (cfg_flats[d][CONFIG_ID], seed, cfg_flats[d][CFG_DIR_COL])
                for d, seeds in seeds_of.items()
                for seed in seeds
            ],
            columns=["config_id", "seed", "cfg_dir"],
        )
        dups = index[index.duplicated(["config_id", "seed"], keep=False)]
        report(
            f"{len(dups)} row(s) share their config ID and seed with another row (rows kept)",
            [
                f"{config_id}, seed {seed}: {sorted(group['cfg_dir'])}"
                for (config_id, seed), group in dups.groupby(["config_id", "seed"])
            ],
            verbose,
        )

        # The flat config of every config directory with a row. Object columns
        # keep the ints of a key that some configs lack as ints, so pyarrow
        # infers int64 with nulls for that key, where pandas would make it float64.
        cfg_df = pd.DataFrame([cfg_flats[d] for d in seeds_of], dtype=object)

        # --- Recap ---

        need_recap = verbose or log_configs
        header_lines: list[str] = []
        summary_line: str | None = None
        stat_line: str | None = None
        config_blocks: list[list[str]] = []

        if need_recap:
            header_lines = ["=" * 60, "RECAP", "=" * 60]

            stat_cols = sorted({c[1] for c in columns if c[0] == "statistic"})
            if stat_cols:
                stat_line = f"Statistics: {stat_cols}"

            # Imported here, since src.utils.plot loads matplotlib and seaborn and
            # only the recap uses it. value_str renders every null spelling alike
            # and integral floats as ints, as the plotting scripts compare values.
            from src.utils.plot import value_str

            seeds_of_folder = {cfg_flats[d][CFG_DIR_COL]: set(s) for d, s in seeds_of.items()}
            hyp_cols = [c for c in cfg_df.columns if c[0] == "hyperparameter"]
            hyp_strs = {c: cfg_df[c].map(value_str) for c in hyp_cols}
            varying_hyp = sorted(
                [c for c in hyp_cols if hyp_strs[c].nunique() > 1],
                key=lambda c: c[1],
            )

            if not hyp_cols:
                summary_line = f"Total rows: {len(index)}"
            elif not varying_hyp:
                seeds = sorted(set(index["seed"]))
                folders = sorted(cfg_df[CFG_DIR_COL].tolist())
                folder_str = f"  [{', '.join(folders)}]"
                summary_line = (
                    f"1 configuration{folder_str} — {len(seeds)} run(s), seeds: {seeds}"
                )
            else:
                groupby_keys = [hyp_strs[c].rename(c[1]) for c in varying_hyp]
                groups = cfg_df.groupby(groupby_keys)
                summary_line = (
                    f"{len(groups)} unique configuration(s). "
                    f"Varying keys: {[c[1] for c in varying_hyp]}"
                )
                for i, (name, group) in enumerate(groups):
                    if not isinstance(name, tuple):
                        name = (name,)
                    folders = sorted(group[CFG_DIR_COL].tolist())
                    seeds = sorted(set().union(*(seeds_of_folder[f] for f in folders)))
                    folder_str = f"  [{', '.join(folders)}]"
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

        write_parquet(part_paths, list(columns), cfg_df, tmp_path)
        os.replace(tmp_path, parquet_path)
    except BaseException:
        # The tasks not yet started are dropped, so the error is raised once the
        # running ones finish, not once every task has run.
        pool.shutdown(cancel_futures=True)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--log_configs",
        action="store_true",
        help="Save configs_log.txt (with all configurations) in the parquet's directory.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of processes reading the data folders. Defaults to the number of CPUs.",
    )
    args = parser.parse_args()

    parquet_path = os.path.join(args.folder, "results.parquet")
    process_data_folders(
        args.folder,
        parquet_path,
        verbose=args.verbose,
        log_configs=args.log_configs,
        workers=args.workers,
    )
    print(f"\nDataFrame successfully saved to {parquet_path}")
