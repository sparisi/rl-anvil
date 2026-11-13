"""Write a script that relaunches the runs of a sweep, of a plot config, or of
both, missing from a data directory.

The configurations come from two sources, either or both of which can be given.
With neither, the script ends without writing anything.

  --sweep           every configuration the sweep expands to (see
                    src/utils/sweep.py). A directory is a set of sweeps, and
                    every one in it or in its subdirectories not marked `# skip`
                    is read.
  --plot            every entry of the plot config in every environment it
                    declares, as plot_results.py composes it. An entry that pins
                    another environment is skipped in that one. A directory is a
                    set of plot configs, and every one not marked `# skip` is read.

Every configuration is composed through Hydra with the overrides
submit_jobs_slurm.py adds to every run (the approximator devices, set by --cuda)
and the ones given with --with. A configuration that both sources declare
composes to the same config_id, and is checked and reported once. So is a
configuration that several sweeps of a directory declare.

The runs in the data directory are the directories holding a cfg.yaml, at any
depth. A configuration matches a run when the two agree on every key, with the
keys exempt from matching left out on both sides: those in --ignore, and for a
plot config also those in its `ignored_cfg_keys`. A key also exempts its
sub-keys, and a null value matches an absent key. The seeds of a configuration
are the seed directories holding a data.npz in any run it matches, and every
seed from 0 to N-1 not among them is missing.

The script is written to <folder>/submit_missing_runs.py and relaunches the
missing seeds through submit_jobs_slurm.submit_jobs(...). Its command line takes
the parameters of submit_jobs, the cluster among them, and `--extra` the
overrides to launch every run with, which is where the values of the exempt keys
go. A recap of every configuration missing seeds is printed at the end, grouped
by environment, each group headed by the environment YAML (its environment.id
for a configuration naming none) with its number of configurations and missing
seeds: for every configuration, the run directories it matches, relative to the
data directory, the sources declaring it, the missing seeds and its overrides
other than the environment. A configuration matching no run is named by the
config_id it composes to; when it has keys exempt from matching, that config_id
is marked as composed before `--extra`.

Example

    python prepare_missing_runs.py --sweep encoders --plot example --folder data_encoders --n_seeds 10
"""

import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

from src.utils.id import compose_cfg, flatten_cfg, make_id, read_run_cfg, strip_config
from src.utils.matching import (
    exempt_key,
    ignored_cfg_keys,
    is_null,
    split_config_label,
    value_str,
)
from src.utils.paths import SWEEP_DIR
from src.utils.plot import (
    PLOT_CONFIG_DIR,
    env_groups,
    is_skipped_config,
    launch_overrides,
    load_plot_configs,
    plot_entries,
    record_missing,
    write_missing_runs_script,
)
from src.utils.sweep import expand_sweep, format_override, load_sweep

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sweep",
    help=f"Sweep whose configurations are checked. A name without .yaml is "
         f"looked for in {SWEEP_DIR}, or give a path; a directory is a set of "
         f"sweeps, bar those marked `# skip`.",
)
parser.add_argument(
    "-p", "--plot",
    metavar="NAME",
    help=f"Plot config whose entries are checked. A name without .yaml is "
         f"looked for in {PLOT_CONFIG_DIR}, or give a path; a directory is a set "
         f"of configs, bar those marked `# skip`.",
)
parser.add_argument(
    "-f", "--folder",
    required=True,
    metavar="DIR",
    help="Data directory holding <config_id>/cfg.yaml and "
         "<config_id>/<seed>/data.npz of every run.",
)
parser.add_argument(
    "-n", "--n_seeds",
    type=int,
    required=True,
    help="Number of seeds every configuration must have, from 0 to N-1.",
)
parser.add_argument(
    "--ignore",
    nargs="+",
    default=[],
    metavar="KEY",
    help="Keys left out when matching a configuration against the runs, on top "
         "of a plot config's ignored_cfg_keys. A key also exempts its sub-keys.",
)
parser.add_argument(
    "--cuda",
    action="store_true",
    help="Compose every configuration with the approximator devices set to cuda.",
)
parser.add_argument(
    "--with",
    dest="extra",
    nargs="+",
    default=[],
    help="Overrides every configuration is composed with, after its own.",
)
args = parser.parse_args()

if not args.sweep and not args.plot:
    print("No --sweep or --plot given, nothing to check.")
    sys.exit()

if not os.path.isdir(args.folder):
    raise SystemExit(f"Data directory not found: {args.folder}")

device = "cuda" if args.cuda else "cpu"


def sweep_files(name):
    """Return every sweep `--sweep name` asks for, as [(name, path)].

    `name` is looked for as given, then in SWEEP_DIR, then in SWEEP_DIR with
    .yaml appended. A file is one sweep. A directory is a set of sweeps: every
    .yaml in it or in its subdirectories, bar those whose first non-empty line
    starts with `# skip`. The name each comes back with is its path under
    SWEEP_DIR without the extension, e.g. "gcrl/main"."""

    for candidate in (name, os.path.join(SWEEP_DIR, name), os.path.join(SWEEP_DIR, f"{name}.yaml")):
        path = Path(candidate)
        if path.is_file():
            files = [path]
            break
        if path.is_dir():
            files = [p for p in sorted(path.rglob("*.yaml")) if not is_skipped_config(p)]
            if not files:
                raise SystemExit(f"No .yaml in {path} that is not marked `# skip`, so there is nothing to check.")
            break
    else:
        raise SystemExit(f"No sweep '{name}', and none in {SWEEP_DIR} by that name.")

    out = []
    for path in files:
        try:
            stem = os.path.splitext(os.path.relpath(path, SWEEP_DIR))[0]
        except ValueError:  # another drive: no path relative to SWEEP_DIR
            stem = path.stem
        if stem.startswith(".."):
            stem = path.stem
        out.append((stem.replace(os.sep, "/"), str(path)))
    return out


# Read before the runs, so a sweep that does not exist stops the script at once.
sweeps = sweep_files(args.sweep) if args.sweep else []


def identity(flat, ignore):
    """Return the keys of `flat` that take part in matching, each with its value
    as value_str renders it, as a frozenset. Keys holding a null and keys
    `ignore` exempts are left out."""

    return frozenset(
        (key, value_str(value))
        for key, value in flat.items()
        if not is_null(value) and not exempt_key(key, ignore)
    )


# --- The runs in the data directory --------------------------------------------
# (run directory, flat config of its cfg.yaml, {seeds holding a data.npz}) per
# run directory.
runs = []
cfg_dirs = [dirpath for dirpath, _dirs, files in os.walk(args.folder) if "cfg.yaml" in files]
for dirpath in tqdm(cfg_dirs, desc="Reading cfg.yaml", unit="run"):
    _cfg, cfg_dict, reason = read_run_cfg(os.path.join(dirpath, "cfg.yaml"))
    if reason is not None:
        tqdm.write(f"WARNING: {os.path.join(dirpath, 'cfg.yaml')} {reason}, skipping.")
        continue
    seeds = {
        int(entry) for entry in os.listdir(dirpath)
        if entry.isdigit() and os.path.isfile(os.path.join(dirpath, entry, "data.npz"))
    }
    runs.append((dirpath, flatten_cfg(cfg_dict, drop_interpolations=False), seeds))

# Exempt keys -> {identity: (seeds, run directories)}, built the first time a set
# of exempt keys is asked for.
run_index = {}


def runs_found(flat, ignore):
    """Return the seeds holding a data.npz in the runs `flat` matches with the
    keys in `ignore` exempt, and the directories of those runs relative to the
    data directory."""

    if ignore not in run_index:
        by_identity = {}
        for dirpath, run_flat, seeds in runs:
            found_seeds, found_dirs = by_identity.setdefault(identity(run_flat, ignore), (set(), set()))
            found_seeds.update(seeds)
            found_dirs.add(os.path.relpath(dirpath, args.folder))
        run_index[ignore] = by_identity
    return run_index[ignore].get(identity(flat, ignore), (set(), set()))


# --- The configurations to check -----------------------------------------------
# config_id -> {"hp": overrides, "flat": composed config, "sources": [names],
# "ignores": {exempt keys of each source}}
configs = {}


def add(hp, source, ignore):
    """Compose the configuration `hp` declared by `source` and add it to
    `configs` under its config_id, with the keys `ignore` exempts from
    matching."""

    # The device overrides submit_jobs_slurm.submit_jobs adds to every run.
    config_hp = {
        **hp,
        "agent.critic.approximator.device": device,
        "agent.critic.approximator_visit.device": device,
    }
    cfg = compose_cfg([*(format_override(k, v) for k, v in config_hp.items()), *args.extra])
    config_id = make_id(cfg)
    entry = configs.setdefault(
        config_id,
        {
            "hp": hp,
            "flat": flatten_cfg(strip_config(cfg), drop_interpolations=False),
            "sources": [],
            "ignores": set(),
        },
    )
    if source not in entry["sources"]:
        entry["sources"].append(source)
    entry["ignores"].add(ignore)


# (sweep name, overrides) for every configuration of every sweep.
sweep_hps = [(name, hp) for name, path in sweeps for hp in expand_sweep(*load_sweep(path))]
sweep_ignore = frozenset(args.ignore)

# (plot config name, exempt keys, environment, selector) for every entry in every
# environment its plot config declares.
plot_items = []
if args.plot:
    for name, _path, cfg in load_plot_configs(args.plot):
        ignore = ignored_cfg_keys(cfg) | sweep_ignore
        envs = dict.fromkeys(env for group in env_groups(cfg).values() for env in group)
        for env in envs:
            for _group, _label, selector in plot_entries(cfg):
                plot_items.append((name, ignore, env, selector))

with tqdm(total=len(sweep_hps) + len(plot_items), desc="Configurations", unit="config") as bar:
    for name, hp in sweep_hps:
        add(hp, f"sweep {name}", sweep_ignore)
        bar.update()
    for name, ignore, env, selector in plot_items:
        # An entry naming another environment is skipped in this one.
        named = selector.get("environment")
        if named is None or split_config_label(value_str(named))[0] == env:
            add(launch_overrides({"environment": env}, dict(selector)), f"plot {name}", ignore)
        bar.update()

# --- The seeds each configuration is short of ----------------------------------
missing_runs = {}
recap = []
for config_id, entry in configs.items():
    present, run_dirs = set(), set()
    for ignore in entry["ignores"]:
        seeds, dirs = runs_found(entry["flat"], ignore)
        present |= seeds
        run_dirs |= dirs
    found = [seed for seed in range(args.n_seeds) if seed in present]
    missing = record_missing(missing_runs, entry["hp"], found, args.n_seeds)
    if missing:
        if run_dirs:
            where = ", ".join(sorted(run_dirs))
        elif any(entry["ignores"]):
            where = f"{config_id} (no run on disk; config_id before --extra)"
        else:
            where = f"{config_id} (no run on disk)"
        # The environment YAML the configuration names, or its environment.id
        # when it names none.
        env = entry["hp"].get("environment", entry["flat"].get("environment.id"))
        recap.append((value_str(env), where, entry["sources"], missing, entry["hp"]))

path, n_jobs = write_missing_runs_script(args.folder, missing_runs)

by_env = {}
for env, *rest in recap:
    by_env.setdefault(env, []).append(rest)

print(
    f"\nMissing runs -- {len(recap)} of {len(configs)} configuration(s), "
    f"{sum(len(r[3]) for r in recap)} seed(s), {len(by_env)} environment(s)"
)
for env in sorted(by_env):
    items = by_env[env]
    print(f"\n>>> {env} -- {len(items)} configuration(s), {sum(len(missing) for _, _, missing, _ in items)} seed(s)")
    for where, sources, missing, hp in items:
        print(f"\n  {where}  [{', '.join(sources)}]")
        print(f"    seeds: {', '.join(map(str, missing))}")
        for key, value in hp.items():
            if key != "environment":
                print(f"    {key}={value}")
print(f"\nMissing runs script: {path} ({n_jobs} entries)")
