"""Upload the runs stored in a process_data parquet file to Weights & Biases.

Each row of the parquet (one seed of one configuration) becomes one W&B run:

- The config is the row's 'hyperparameter' columns, unflattened into the
  nested structure main.py passes to wandb.init(), plus experiment.rng_seed.
  It holds the keys of cfg.yaml, without the keys whose value is null. A key
  whose column has missing values and only integral values is uploaded as an
  integer, also where cfg.yaml held a float such as 1.0.
- The run is named '<config_id>_<seed>', as in main.py, and grouped by
  '<config_id>'. The config ID is the name of the run's config folder, which
  process_data stores in the ('config', 'config_id') column.
- Every 'statistic' column holding a list is logged as a time series, one
  wandb.log() call per checkpoint at step=steps[i]. Non-finite entries are not
  logged. Every statistic that is not a list goes to the run summary. A row
  fails if its 'steps' are not strictly increasing integers, or if one of its
  lists is not a list of numbers as long as 'steps'.

Finished runs already in the W&B project are skipped, matched by run name, so
re-running the script only uploads the rows added to the parquet since. A run
that exists but did not finish (e.g., an interrupted upload) is resumed: the
checkpoints after the last step it holds are logged, and the summary is set.
A run still in the 'running' state is skipped and reported.
A row whose run name was already skipped or submitted for an earlier row is
skipped and reported. Run names are matched across the whole project, so two
configurations from different data directories that share a config ID and a
seed count as the same run.

A finished run is skipped even if its data changed after the upload, as it
does when a seed is trained again with results.force_run=True: the config ID
and the seed, hence the run name, stay the same. To upload the new data,
delete the run from W&B and run the script again.

In offline mode, the script makes no network calls: the W&B project is not
read, every row is written as a new offline run, and none is skipped or
resumed. Running it twice before 'wandb sync' produces duplicate runs.

Runs are uploaded in parallel by --n_jobs worker processes. A row that cannot
be read, or whose upload raises an error, is reported at the end and does not
stop the other uploads. The script exits with status 1 if any row failed.

Example:
    python upload_wandb.py -f data_gcrl --project my_project --entity my_entity
"""

import os
import sys
import argparse
import multiprocessing
from ast import literal_eval
from itertools import chain
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

from src.utils.id import CONFIG_ID, SEED, decode_stat


def to_python(x):
    """Return `x` with numpy arrays converted to lists and numpy scalars to
    Python scalars, recursively through lists and dicts."""

    if isinstance(x, np.ndarray):
        return [to_python(v) for v in x.tolist()]
    if isinstance(x, list):
        return [to_python(v) for v in x]
    if isinstance(x, dict):
        return {k: to_python(v) for k, v in x.items()}
    if isinstance(x, np.generic):
        return x.item()
    return x


def is_missing(x) -> bool:
    """Return True if `x` is a scalar null: None, NaN of any float type, or
    pd.NA. Lists and arrays are never null."""

    if isinstance(x, (list, np.ndarray)):
        return False
    return bool(pd.isna(x))


def unflatten(flat: dict) -> dict:
    """Return the nested dict built from a dict keyed by dotted paths, one
    level per dot-separated part of the key."""

    nested = {}
    for dotted, value in flat.items():
        *parents, leaf = dotted.split(".")
        node = nested
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = value
    return nested


def decode(raw):
    """Return `raw` decoded by decode_stat if it is a string, and passed through
    to_python if it is not."""

    if isinstance(raw, str):
        return decode_stat(raw)
    return to_python(raw)


def is_number(x) -> bool:
    """Return True if `x` is an int or a float, and not a bool."""

    return isinstance(x, (int, float)) and not isinstance(x, bool)


def run_config(row: pd.Series, hyp_cols: list) -> tuple[str, str, dict]:
    """Return the run name, config ID and nested config of a parquet row. A
    value "{}", which flatten_cfg stores for an empty mapping, becomes {}."""

    flat_cfg = {
        c[1]: {} if isinstance(row[c], str) and row[c] == "{}" else to_python(row[c])
        for c in hyp_cols
        if not is_missing(row[c])
    }
    seed = int(row[SEED])
    config_id = row[CONFIG_ID]
    if not isinstance(config_id, str) or not config_id:
        raise ValueError(f"Invalid config ID {config_id!r}.")
    cfg = unflatten(flat_cfg | {SEED[1]: seed})
    return f"{config_id}_{seed}", config_id, cfg


def run_stats(row: pd.Series, stat_cols: list) -> tuple[list, dict, dict]:
    """Return the steps, time series and summary statistics of a parquet row.

    Raise an error if 'steps' is missing, or is not a non-empty list of strictly
    increasing integers, or if another statistic is a list that is not a list
    of numbers as long as 'steps'.
    """

    stats = {c[1]: decode(row[c]) for c in stat_cols if not is_missing(row[c])}
    steps = stats.pop("steps", None)
    if not isinstance(steps, list) or not steps:
        raise KeyError("'steps' is missing, empty, or not a list.")
    if not all(is_number(s) and np.isfinite(s) and float(s).is_integer() for s in steps):
        raise ValueError("'steps' holds non-integer entries.")
    steps = [int(s) for s in steps]
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("'steps' is not strictly increasing.")
    series = {k: v for k, v in stats.items() if isinstance(v, list)}
    for k, v in series.items():
        if len(v) != len(steps) or not all(is_number(x) for x in v):
            raise ValueError(
                f"Statistic '{k}' is a list but not a list of numbers as long as "
                f"'steps'."
            )
    summary = {k: v for k, v in stats.items() if k not in series}
    return steps, series, summary


def upload(
    name: str,
    config_id: str,
    cfg: dict,
    steps: list,
    series: dict,
    summary: dict,
    tags: list,
    project: str,
    entity: str,
    mode: str,
    run_id: str | None = None,
    last_step: int = -1,
) -> None:
    """Create one W&B run and log its time series and summary.

    If `run_id` is given, that run is resumed instead, and only the checkpoints
    after `last_step` (the last step it already holds) are logged.
    """

    with wandb.init(
        id=run_id,
        resume="must" if run_id is not None else None,
        config=cfg,
        name=name,
        group=config_id,
        tags=tags,
        project=project,
        entity=entity,
        mode=mode,
        settings=wandb.Settings(x_disable_stats=True, silent=True),
    ) as run:
        for i, step in enumerate(steps):
            if step <= last_step:
                continue
            values = {
                k: v[i]
                for k, v in series.items()
                if np.isfinite(v[i])
            }
            run.log(values, step=step)
        run.summary.update(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--folder",
        required=True,
        metavar="DIR",
        help="Data directory holding the parquet file.",
    )
    parser.add_argument(
        "--parquet",
        default="results.parquet",
        help="Name of the parquet file inside the data directory.",
    )
    parser.add_argument("--project", required=True, help="W&B project.")
    parser.add_argument("--entity", default=None, help="W&B entity.")
    parser.add_argument(
        "--mode",
        default="online",
        choices=["online", "offline"],
        help="W&B mode. Offline runs can be uploaded later with 'wandb sync'.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Tags added to every run.",
    )
    parser.add_argument(
        "-j",
        "--n_jobs",
        type=int,
        default=4,
        help="Number of runs uploaded in parallel.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the checkpoints, series and summary keys of every run.",
    )
    args = parser.parse_args()

    parquet_path = os.path.join(args.folder, args.parquet)
    df = pd.read_parquet(parquet_path)
    print(f"Loaded parquet file from: {parquet_path}")
    if all(isinstance(c, str) and c.startswith("(") for c in df.columns):
        df.columns = pd.MultiIndex.from_tuples([literal_eval(c) for c in df.columns])

    hyp_cols = [c for c in df.columns if c[0] == "hyperparameter"]
    stat_cols = [c for c in df.columns if c[0] == "statistic"]
    if ("statistic", "steps") not in stat_cols:
        raise KeyError(f"No 'steps' statistic in {parquet_path}.")
    if CONFIG_ID not in df.columns:
        raise KeyError(
            f"No {CONFIG_ID} column in {parquet_path}. Run process_data.py again "
            f"to write it."
        )

    # A key absent from some rows turns an integer column into float64. Such a
    # column, with missing values and only integral present values, is read
    # back as integers.
    for c in hyp_cols:
        column = df[c]
        if (
            pd.api.types.is_float_dtype(column)
            and column.isna().any()
            and (column.dropna() % 1 == 0).all()
        ):
            df[c] = column.astype("Int64")

    # Runs already in the project, by name. If several runs share a name, a
    # finished one takes precedence.
    existing = {}
    entity = args.entity
    if args.mode == "online":
        api = wandb.Api()
        entity = entity or api.default_entity
        runs = iter(api.runs(f"{entity}/{args.project}", per_page=1000))
        # Loading the first page raises ValueError if the project does not
        # exist yet. wandb.init() creates it. Errors on later pages propagate.
        found = True
        try:
            first = [next(runs)]
        except StopIteration:
            first = []
        except ValueError:
            found = False
        if found:
            n_existing = 0
            for run in chain(first, runs):
                n_existing += 1
                if run.name not in existing or run.state == "finished":
                    existing[run.name] = run
            print(
                f"Found {n_existing} run(s) with {len(existing)} distinct name(s) "
                f"in {entity}/{args.project}."
            )
        else:
            print(
                f"Project {entity}/{args.project} not found. The first upload "
                f"creates it."
            )

    skipped = 0
    seen = set()
    duplicates, running, resumed, uploaded, failed = [], [], {}, set(), []
    # Once a worker dies, every later upload fails with this error, and the
    # remaining rows are marked failed without their statistics being decoded
    broken = None

    # Tracks every row: skipped and unreadable rows count at once, uploads
    # when they end
    pbar = tqdm(total=len(df), desc="Runs")

    # Workers are spawned: the main process already runs threads (pyarrow,
    # tqdm), and forking a multi-threaded process can deadlock
    with ProcessPoolExecutor(
        max_workers=args.n_jobs,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        pending = {}

        def collect(return_when):
            """Record the outcome of the pending uploads that are done."""

            nonlocal broken
            done, _ = wait(pending, return_when=return_when)
            for future in done:
                name = pending.pop(future)
                try:
                    future.result()
                    uploaded.add(name)
                except BrokenProcessPool as e:
                    broken = e
                    failed.append((name, e))
                except Exception as e:
                    failed.append((name, e))
                pbar.update()

        for idx, row in df.iterrows():
            try:
                name, config_id, cfg = run_config(row, hyp_cols)
            except Exception as e:
                failed.append((f"row {idx}", e))
                pbar.update()
                continue

            # A name is seen once one of its rows is skipped or submitted. A
            # later row is tried only if every earlier row with its name failed
            # before submission. A failed upload makes later rows duplicates.
            if name in seen:
                duplicates.append(name)
                pbar.update()
                continue

            run_id, last_step = None, -1
            if name in existing:
                run = existing[name]
                if run.state == "finished":
                    seen.add(name)
                    skipped += 1
                    pbar.update()
                    continue
                if run.state == "running":
                    seen.add(name)
                    running.append(name)
                    pbar.update()
                    continue
                run_id = run.id
                if run.lastHistoryStep is not None:
                    last_step = run.lastHistoryStep

            if broken is not None:
                failed.append((name, broken))
                pbar.update()
                continue

            try:
                steps, series, summary = run_stats(row, stat_cols)
            except Exception as e:
                failed.append((name, e))
                pbar.update()
                continue

            if run_id is not None:
                resumed[name] = (run.state, last_step)

            if args.verbose:
                pbar.write(
                    f"{name}: {len(steps)} checkpoint(s), "
                    f"series {sorted(series)}, summary {sorted(summary)}"
                )

            # Bound the queued uploads, so that only a few rows' statistics
            # are held in memory at once
            if len(pending) >= 2 * args.n_jobs:
                collect(FIRST_COMPLETED)

            try:
                future = pool.submit(
                    upload,
                    name,
                    config_id,
                    cfg,
                    steps,
                    series,
                    summary,
                    args.tags,
                    args.project,
                    entity,
                    args.mode,
                    run_id,
                    last_step,
                )
            except BrokenProcessPool as e:
                broken = e
                failed.append((name, e))
                pbar.update()
                continue
            except Exception as e:
                failed.append((name, e))
                pbar.update()
                continue
            seen.add(name)
            pending[future] = name

        collect(ALL_COMPLETED)

    pbar.close()

    resumed_ok = {k: v for k, v in resumed.items() if k in uploaded}
    print(
        f"Uploaded {len(uploaded)} run(s), {len(resumed_ok)} of them resumed. "
        f"Skipped {skipped} finished run(s) already in W&B."
    )
    if resumed_ok:
        print("Resumed unfinished run(s):")
        for name, (state, last_step) in resumed_ok.items():
            print(f"  {name}: was {state}, last step {last_step}")
    if running:
        print(f"{len(running)} row(s) skipped, their run is still running in W&B:")
        for name in running:
            print(f"  {name}")
    if duplicates:
        print(f"{len(duplicates)} row(s) skipped, their run name appears in an earlier row:")
        for name in duplicates:
            print(f"  {name}")
    if failed:
        print(f"{len(failed)} row(s) failed to be read or uploaded:")
        for name, e in failed:
            print(f"  {name}: {e!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
