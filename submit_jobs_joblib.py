"""Run a sweep file locally through Hydra's multirun.

Same sweep style of submit_jobs_slurm.py, i.e., it reads a YAML file in
SWEEP_DIR defining the parameters to sweep over.
It also needs the same arguments, with the addition of the optional `--with`,
that adds overrides. Used to define Hydra local sweep configs. Example:

    python submit_jobs_joblib.py --sweep=example --seeds 0-9 --data_dir data_example --with hydra.launcher.verbose=1000
"""

import argparse
import os
import subprocess
import sys

from src.utils.sweep import format_override, load_sweep

# Applied to every run, but a sweep file can override them.
# These are override literals, not YAML: `false` and `null` are what the grammar
# reads, where Python's False and None would arrive as the strings "False" and "None".
DEFAULT_OVERRIDES = [
    "results.force_run=false",
    "results.progress_report=history_table",
    "wandb.mode=disabled",
]


def parse_seeds(tokens):
    """Seeds from `--seeds`: ints are literal, `a-b` is an inclusive range."""

    seeds = []
    for token in tokens:
        if "-" in token[1:]:
            lo, hi = token.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(token))
    return sorted(dict.fromkeys(seeds))


def sweep_override(key, values):
    """One Hydra override covering every value a key takes.

    A single value goes through `format_override`, so this agrees with the SLURM
    path on nulls, lists and quoting. Several values are comma-joined, which is
    what makes a multirun sweep the key.

    The left-hand side is taken from what `format_override` returned rather than
    from `key`, since it may carry a prefix (`+`, `++`). Values whose overrides
    disagree on it cannot be joined: a deletion (`~key`) and an assignment are
    different overrides, and no comma list expresses both."""

    formatted = [format_override(key, value) for value in values]
    if len(formatted) == 1:
        return formatted[0]
    heads = {text.split("=", 1)[0] for text in formatted}
    if len(heads) > 1 or "=" not in formatted[0]:
        raise ValueError(
            f"{key}: cannot sweep over {values} in one override -- {sorted(heads)} "
            "are different kinds of override, so split the sweep entry."
        )
    return heads.pop() + "=" + ",".join(text.split("=", 1)[1] for text in formatted)


def commands(
    global_sweep,
    sweep,
    seeds,
    scratch,
    data_dir,
    launcher,
    n_jobs,
    max_batch_size,
    passthrough,
):
    """One `python main.py -m ...` argv per sweep entry."""

    launcher = [
        f"hydra/launcher={launcher}",
        f"hydra.launcher.n_jobs={n_jobs}",
        "hydra.sweeper.max_batch_size="
        + ("null" if max_batch_size is None else str(max_batch_size)),
    ]
    for entry in sweep:
        keys = {**global_sweep, **entry}
        yield [
            sys.executable, "main.py", "-m",
            *launcher,
            *DEFAULT_OVERRIDES,
            f"results.data_dir={scratch}/{data_dir}",
            *(sweep_override(key, values) for key, values in keys.items()),
            sweep_override("experiment.rng_seed", seeds),
            *passthrough,
        ]


parser = argparse.ArgumentParser()
parser.add_argument("--sweep", required=True,
    help="Sweep file in SWEEP_DIR (name without .yaml or a path). "
         "It defines environments, algorithms and hyperparameters.")
parser.add_argument("--seeds", nargs="+", required=True,
    help="Seeds to run: ints are literal seeds, a-b an inclusive range.")
parser.add_argument("--data_dir", required=True,
    help="Where the runs write, under $SCRATCH.")
parser.add_argument("--launcher", default="joblib",
    help="Hydra launcher plugin. Default: joblib.")
parser.add_argument("--n_jobs", type=int, default=4,
    help="Runs the launcher executes at once. -1 is one per CPU, but it may "
         "cause OOM. Default: 4.")
parser.add_argument("--max_batch_size", type=int, default=None,
    help="Runs Hydra hands the launcher per batch. Unset sends the whole sweep "
         "as one batch, which is what you want unless a batch has to fit in "
         "something.")
parser.add_argument("--debug", action="store_true",
    help="Print the commands without running them.")
parser.add_argument("--with", dest="passthrough", nargs="+", default=[],
    help="Overrides added to every command, e.g. hydra.launcher.verbose=1000.")
args = parser.parse_args()

try:
    scratch = os.environ["SCRATCH"]
    print(f"Current SCRATCH folder: {scratch}")
except KeyError:
    raise SystemExit("Environment variable $SCRATCH is not set.")

for argv in commands(
    *load_sweep(args.sweep),
    parse_seeds(args.seeds),
    scratch,
    args.data_dir,
    args.launcher,
    args.n_jobs,
    args.max_batch_size,
    args.passthrough,
):
    print(" ".join(argv))
    if not args.debug:
        result = subprocess.run(argv)
        if result.returncode:
            sys.exit(result.returncode)
