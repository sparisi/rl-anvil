import gymnasium
import numpy as np
from omegaconf import DictConfig
from copy import deepcopy
import builtins
import traceback
import warnings
import os
from rich.console import Console
from contextlib import nullcontext
import time
from rich import print
import wandb

from src.utils.pbar import MultiProgressBar
from src.actor import Actor
from src.critic import Critic
import src.replay_memory
from src.utils.misc import (
    cantor_pairing,
    format_dict_row,
    random_argmax,
    StatsTracker,
)
from src.utils.slurm import (
    cgroup_memory,
    malloc_trim,
    peak_rss_bytes,
    rss_bytes,
)
from src.utils.video import frames_to_grid
from src.pseudocount import tabular_count


def exploration_stats(counts, mask=None):
    """Coverage and normalized entropy of a visitation count array.

    If we know which states can be visited (even when their current count is 0)
    entropy normalizes exactly to [0, 1] and coverage is exact. Otherwise we
    assume all can be visited, potentially underestimating both.
    """
    if counts.sum() == 0:
        return 0.0, 0.0
    if mask is None:
        mask = np.ones_like(counts, dtype=bool)
    counts = counts[mask]
    size = mask.sum()
    nz = counts[counts > 0]
    p = nz.astype(np.float64) / counts.sum()
    entropy = - (p * np.log(p)).sum() / np.log(size) if size > 1 else 0.0
    coverage = nz.size / size if size > 0 else 0.0

    return coverage, entropy


def memory_stats():
    """Memory diagnostics for one checkpoint, as a {name: value} dict.

    Per-process, in MB:

        rss_mb       Resident set of THIS worker, read after the trim below.
                     Counts anonymous and file-backed pages, so the shared
                     libtorch text is counted again by every parallel worker.
        trim_mb      What malloc_trim() handed back to the OS. free() alone
                     leaves blocks on glibc's free lists, where they still count
                     against the cgroup, so this separates memory the allocator
                     is holding from memory the program is holding. RSS that
                     climbs while this stays flat is the shape of a real leak.
        rss_peak_mb  Largest resident set this process ever reached. Every other
                     number here is an instantaneous sample, so a spike that
                     allocates and frees between two checkpoints leaves no trace
                     in any of them; the kernel maintains this one and it never
                     falls.

    Whole-job, from the cgroup, as a percentage of what --mem became. Absent
    when there is no cgroup limit to read (a laptop, a node without enforcement):

        mem_%        Anonymous memory only. Unreclaimable without swap, so this
                     is what an OOM kill is decided on.
        mem_all_%    anon + file + kernel, i.e. what the limit is really applied
                     to. It runs far above mem_%, so a comfortable mem_% can hide
                     a cgroup pinned at its limit.
        hit_max      Times the kernel had to reclaim to stay under the limit.
                     Zero while the headroom is real. Climbing means the job is
                     evicting library pages and refaulting them from Lustre, and
                     will spend its wallclock doing that instead of training --
                     with nothing in the log to say so, since nothing fails.
    """

    freed = malloc_trim()
    stats = {
        "rss_mb": rss_bytes() / 2**20,
        "trim_mb": freed / 2**20,
        "rss_peak_mb": peak_rss_bytes() / 2**20,
    }

    mem = cgroup_memory()
    if mem is not None:
        anon, limit = mem["anon"], mem["limit"]
        stats["mem_%"] = 100.0 * anon / limit
        stats["mem_all_%"] = 100.0 * (anon + mem["file"] + mem["kernel"]) / limit
        stats["hit_max"] = mem["hit_max"]

    return stats


def time_stats(d_steps, train_sec, test_sec, steps_left, tests_left):
    """Timing diagnostics for one checkpoint, as a {name: value} dict.

    The clock starts when the replay memory becomes ready, not at __init__.
    Warm-up runs a different loop -- no updates, no gradient -- so folding it in
    makes the first speed reading optimistic by whatever fraction of the run was
    warm-up, and every extrapolation built on it inherits that.

        test_sec   Seconds inside the test() that just ran. Kept separate because
                   it does not scale with training steps: it is a fixed cost per
                   checkpoint, so mixing it into the step rate makes the rate a
                   function of log_frequency instead of a property of the run.
        train_sec  Seconds since the previous checkpoint returned, EXCLUDING
                   test() and everything else this checkpoint does (heatmaps,
                   stats, npz bookkeeping). Pure training wall time.
        step/sec   d_steps / train_sec. The rate over the LAST interval only, so it
                   tracks the current regime -- a run that slows down as the
                   replay memory fills shows it here while a whole-run average is
                   still being propped up by the fast early steps.
        eta_sec    training + testing time left. This is the one to compare
                   against the SLURM wallclock (be careful, though: testing time
                   may vary as the agent improves).
    """

    speed = d_steps / train_sec if train_sec and train_sec > 0 else np.nan
    if speed and speed > 0 and np.isfinite(speed):
        left_s = steps_left / speed
    else:
        left_s = np.inf

    return {
        "test_sec": test_sec,
        "train_sec": train_sec,
        "step/sec": speed,
        "eta_sec": left_s + tests_left * test_sec,
    }


class Experiment:
    _PROGRESS_REPORTS = ("history_table", "live_table", "dict", "none")

    def __init__(
        self,
        env_train: gymnasium.Env,
        env_test: gymnasium.Env,
        actor: Actor,
        critic: Critic,
        replay_memory: DictConfig,
        update_frequency: int,
        updates_per_step: int,
        training_steps: int,
        testing_episodes: int,
        testing_points: int,
        data_dir: str = None,
        save_memory: bool = False,
        save_videos: bool = False,
        save_heatmaps: bool = False,
        rng_seed: int = 0,
        progress_report: str = "history_table",
        slurm_debug: bool = False,
        training_return_episode_window: int = 10,
        git_hash: str = None,
        slurm_job_id: str = None,
        **kwargs,
    ):
        """
        Args:
            env_train (gymnasium.Env): to collect training samples,
            env_test (gymnasium.Env): to test the greedy policy,
            actor (Actor): actor to draw actions,
            critic (Critic): critic to evaluate state-action pairs,
            replay_memory (DictConfig): configuration to initialize the replay memory,
            update_frequency (int): how many training steps between updates,
            updates_per_step (int): how many times we sample a mini-batch and update
                the critic per training step,
            training_steps (int): how many environment steps training will last,
            testing_episodes (int): how many episodes to test the greedy policy,
            testing_points (int): determines the interval between testing
                points. Let L = max(training_steps // max(testing_points, 1), 1).
                The first checkpoint always fires at warm_up, the step the
                replay memory becomes ready, before any update has been done:
                that is the untrained baseline every curve starts from.
                Subsequent ones fire at the multiples of L above warm_up, i.e.
                at (warm_up // L + 1) * L, then +L each time.
                A forced final checkpoint always fires at training_steps
                (without launching a new test), so the last logged step is
                always training_steps regardless of L. Pass testing_points = -1
                to log every step.
                Examples (training_steps = 10_000):
                - testing_points = 1_000, warm_up = 1
                  => L = 10, tests at 1, 10, 20, ..., 10_000 (1_001 total).
                - testing_points = 10, warm_up = 1_000
                  => L = 1_000, tests at 1_000, 2_000, ..., 10_000 (10 total).
                - testing_points = 10, warm_up = 1_500
                  => L = 1_000, tests at 1_500, 2_000, 3_000, ..., 10_000 (10 total).
            training_return_episode_window (int): the train return statistic is
                the mean over the last N completed training episodes.
                Other statistics (e.g., training loss) are averaged over all
                updates between checkpoints,
            rng_seed (int): to fix random seeds for reproducibility,
            data_dir (str): where to save npz history files,
            save_memory (bool): if True, the whole replay memory will be saved in
                data_dir,
            save_videos (bool): if True, video recording of testing episodes
                will be saved in data_dir,
            save_heatmaps (bool): if True, heatmaps of the Q-function will
                be saved in data_dir,
            progress_report (str): how a running experiment reports itself.
                - "history_table": rich live progress bar with one row per
                  checkpoint, scrolling above the bars. Keeps the whole history
                  on screen, but a row wider than the terminal wraps.
                - "live_table": rich live progress bar but with the latest
                  checkpoint only, as a key/value list rewritten in place under
                  the bars. Never wraps.
                - "dict": one plain `{key: value}` line per checkpoint, no
                  formatting and no cursor movement, flushed. This is what SLURM
                  log.out is for, and what detect_jobs_err.sh parses to reconstruct
                  a dead run's speed history. Timing statistics are printed in
                  this mode only.
                - None: print nothing. Statistics are still recorded to npz and
                  W&B (if enabled); only stdout stays silent.
            slurm_debug (bool): if True, every checkpoint also records memory
                diagnostics,
            git_hash (str): commit the run was launched from, and
            slurm_job_id (str): the SLURM job that ran it. Both are recorded per
                seed in the .npz rather than in cfg.yaml, which every seed of a
                run_id shares -- see `self._run_meta`,
        """

        self._env_train = env_train
        self._env_test = env_test

        self._actor = actor
        self._critic = critic
        self._rng_seed = rng_seed
        self._critic.reset(self._rng_seed)
        self._actor.reset(self._rng_seed)

        self._update_frequency = update_frequency
        self._updates_per_step = updates_per_step

        self._training_steps = training_steps
        self._testing_episodes = testing_episodes
        if testing_points == -1:
            self._log_frequency = 1
        else:
            self._log_frequency = max(training_steps // max(testing_points, 1), 1)

        # Step at which the next checkpoint is due (first is step 0).
        #A threshold, not an exact `% log_frequency`, which would only fire on
        # steps that are a multiple of both this and update_frequency -- sometimes never.
        self._next_checkpoint = 0

        self._data_dir = data_dir
        self._save_memory = save_memory
        self._save_videos = save_videos
        self._save_heatmaps = save_heatmaps

        # Used for debugging
        self._critic.visit_count = tabular_count(self._env_train)
        if hasattr(self._actor, "goal_selected_count"):
            self._actor.goal_selected_count = tabular_count(self._env_train)
        if hasattr(self._actor, "goal_reached_count"):
            self._actor.goal_reached_count = tabular_count(self._env_train)

        self._replay_memory = getattr(src.replay_memory, replay_memory.id)(**replay_memory)
        self._replay_memory.init(
            obs=self._env_train.observation_space.sample(),
            act=self._env_train.action_space.sample(),
            rwd=np.asarray(0.0),
            term=np.asarray(False),
            trunc=np.asarray(False),
            next_obs=self._env_train.observation_space.sample(),
        )

        # Also for debugging
        if hasattr(self._actor, "goal_selected_count"):
            self._replay_memory.add_keys(
                goal_obs=self._env_train.observation_space.sample(),
                goal_act=self._env_train.action_space.sample(),
                goal_valid=np.asarray(False),  # Whether the actor acted on a goal or not (e.g., random exploration)
            )

        # Counts for scoring goal candidates. The memory serves either the true
        # binned counts read from this counter or pseudocounts over its own
        # samples, depending on whether its rho is None; the counter is None when
        # the environment's observations cannot be binned, which leaves
        # pseudocounts as the only option.
        self._replay_memory.init_counting(
            n_actions=self._critic.n_actions,
            goal_idx=getattr(self._critic, "goal_idx", slice(None)),
            counter=self._critic.visit_count,
        )

        # Pre-allocate critic and arrays for test()
        if self._testing_episodes >= 1 and self._env_test is not None:
            self._critic_test = deepcopy(self._critic)
            self._disc_ret_test_buf = np.zeros(self._testing_episodes)
            self._undisc_ret_test_buf = np.zeros(self._testing_episodes)
            self._done_test_buf = np.zeros(self._testing_episodes, dtype=bool)
            self._seed_test = [
                cantor_pairing(self._rng_seed, ep)
                for ep in range(self._testing_episodes)
            ]

        # The train return reported at each checkpoint is the mean over the last
        # N completed episodes.
        window = max(training_return_episode_window, 1)
        self._disc_ret_train = np.full(window, np.nan)
        self._undisc_ret_train = np.full(window, np.nan)
        self._ep_ret_idx = 0
        self._disc_ret_test = np.nan
        self._undisc_ret_test = np.nan
        self._update_stats = StatsTracker()

        self._tot_steps = 0
        self._tot_episodes = 0
        self._tot_updates = 0

        # Data saved to npz at the end
        self._data = {"steps": []}

        # Scalars that describe the run rather than the configuration
        try:
            critic_device = str(self._critic.q.device)
            print(f":computer: (Critic) Running PyTorch on {critic_device.upper()}")
        except Exception:
            critic_device = None
        self._run_meta = {
            "critic_device": critic_device,
            "git_hash": git_hash,
            "slurm_job_id": slurm_job_id,
        }

        # Printing variables
        if progress_report is None:
            progress_report = "none"
        if progress_report not in self._PROGRESS_REPORTS:
            raise ValueError(
                "progress_report must be one of "
                f"{', '.join(repr(r) for r in self._PROGRESS_REPORTS)}, "
                f"got {progress_report!r}"
            )
        self._progress_report = progress_report
        self._slurm_debug = slurm_debug
        self._print_stats_keys_frequency = 10  # Print keys every N checkpoints
        self._checkpoint_count = 0
        self._new_stat_key = False
        self._console = Console()
        self._max_ep_steps = (
            getattr(env_train, "spec", None) and
            getattr(env_train.spec, "max_episode_steps", None)
        )
        self._pbar = None
        if self._progress_report in ("history_table", "live_table"):
            self._pbar = MultiProgressBar(console=self._console)
            self._pbar.add_task("training", self._training_steps)
            self._pbar.add_task("testing", self._max_ep_steps)

        # Timing. Both start at the first step for which the replay memory is
        # ready (i.e., when updates start).
        self._t_last_ckpt = None
        self._steps_last_ckpt = 0


    def savedata(self):
        """Save statistics history and (optionally) the whole replay memory."""

        if self._data_dir is not None:
            training_time = time.time() - self._start_time
            os.makedirs(self._data_dir, exist_ok=True)
            np.savez(
                os.path.join(self._data_dir, "data"),
                time=training_time,
                **self._data,
                **self._run_meta,
            )
            if self._save_memory:
                self._replay_memory.export_data(
                    os.path.join(self._data_dir, f"memory")
                )
        # TODO save models as well, and call this in checkpoint


    def run(self):
        """Main function with try ... except ... finally to ensure WandB and
        environments are closed."""

        self._start_time = time.time()

        exit_code = 0

        try:
            self.train()
            # TODO support intermediate saves and resume
            if self._data_dir is not None:
                print("... saving data ...")
                # Saving must not mask the error that reached this point: if
                # training raised, a failure to save is reported and swallowed
                # rather than re-raised, so the original traceback survives.
                try:
                    self.savedata()
                    print(":thumbsup: Data saved")
                except Exception:
                    exit_code = 1
                    print(":x: Failed to save data:")
                    traceback.print_exc()
        except KeyboardInterrupt:
            # Deliberately swallowed: Ctrl+C is a clean stop, not a failure.
            print(":x: KeyboardInterrupt (Ctrl+C) detected. Cleaning up...")
            print(":thumbsup: Graceful shutdown complete")
        except BaseException:
            # Anything else is a real failure. Flag it for wandb, then let it
            # propagate: main.py/Hydra print the traceback and SLURM marks the
            # task FAILED.
            exit_code = 1
            raise
        finally:
            self._env_train.close()
            if self._env_test is not None:
                self._env_test.close()
            wandb.finish(exit_code=exit_code)
            print(":white_check_mark: Run over")


    def train(self):
        # Main training loop
        pbar_ctx = self._pbar if self._pbar is not None else nullcontext()
        with pbar_ctx:
            while self._tot_steps < self._training_steps:

                # Episode reset
                ep_seed = cantor_pairing(self._rng_seed, self._tot_episodes)
                actor_rng = self._actor.rng_generator(seed=ep_seed)
                critic_rng = self._critic.rng_generator(seed=ep_seed)

                obs, info = self._env_train.reset(seed=ep_seed)
                self._actor.reset_episode(
                    seed=ep_seed,
                    rng_generator=actor_rng,
                    obs=obs,
                )
                ep_steps = 0
                ep_disc_return = 0.0
                ep_undisc_return = 0.0
                ep_done = False

                # Episode loop
                while True:
                    if self._pbar is not None:
                        self._pbar.update("training", self._tot_steps)

                    act = self._actor(
                        obs=obs,
                        replay_memory=self._replay_memory,
                        rng_generator=actor_rng,
                    )
                    next_obs, rwd, term, trunc, info = self._env_train.step(act)

                    sample = dict(
                        obs=obs,
                        act=act,
                        rwd=rwd,
                        term=term,
                        trunc=trunc,
                        next_obs=next_obs,
                    )
                    if hasattr(self._actor, "goal_selected_count"):
                        # `_goal_used_last_step` is the goal the action was
                        # actually taken for, which is not `goal` once a
                        # per-step actor has already re-selected.
                        goal = getattr(self._actor, "_goal_used_last_step", None)
                        if goal is None:
                            goal = self._actor.goal
                        # Written every step, including the ones with no goal:
                        # add() only stores what it is given, so leaving the keys
                        # out would keep whatever the slot held before -- a goal
                        # from a previous lap of the ring buffer -- and record it
                        # as this step's. goal_valid says which entries mean
                        # anything; the rest are zeroed placeholders.
                        sample["goal_valid"] = goal is not None
                        sample["goal_obs"] = 0 if goal is None else goal["obs"]
                        sample["goal_act"] = 0 if goal is None else goal["act"]

                    # Update statistics and decay epsilon, learning rates, and
                    # any other scheduled parameter. These run on the
                    # environment-step clock, so a schedule's `steps` means
                    # environment steps whatever update_frequency is.
                    self._actor.post_step(**sample)
                    self._critic.post_step(**sample)
                    self._replay_memory.post_step(**sample)
                    if self._critic.visit_count is not None:
                        self._critic.visit_count.update(obs, act)  # Update true count

                    self._replay_memory.add(**sample)
                    self._tot_steps += 1

                    mem_ready = self._replay_memory.is_ready
                    time_to_update = self._tot_steps % self._update_frequency == 0

                    # Start the clock when the first time the memory is ready.
                    if mem_ready and self._t_last_ckpt is None:
                        self._t_last_ckpt = time.perf_counter()
                        self._steps_last_ckpt = self._tot_steps

                    # The first checkpoint happens once the memory is ready BUT
                    # no update has been done yet.
                    if mem_ready and self._tot_steps >= self._next_checkpoint:
                        self.checkpoint()
                        # `while`, not `+=`: several thresholds can be crossed at
                        # once when log_frequency is small.
                        while self._next_checkpoint <= self._tot_steps:
                            self._next_checkpoint += self._log_frequency

                    if mem_ready and time_to_update:
                        for _ in range(self._updates_per_step):
                            update_stats = self._critic.update(
                                replay_memory=self._replay_memory,
                                rng_generator=critic_rng,
                            )
                            self._update_stats.update(**update_stats)
                            self._tot_updates += 1
                            self._critic.post_update()  # target network copy
                            self._actor.post_update()

                        # NO-OP for most memories but OnlineMemory, where we
                        # must mark the segment since the last update as consumed.
                        self._replay_memory.post_update_round()

                    ep_disc_return += (self._critic.gamma**ep_steps) * rwd
                    ep_undisc_return += rwd
                    ep_steps += 1
                    obs = next_obs

                    if term or trunc:
                        # Only COMPLETED episodes are recorded
                        i = self._ep_ret_idx % len(self._disc_ret_train)
                        self._disc_ret_train[i] = ep_disc_return
                        self._undisc_ret_train[i] = ep_undisc_return
                        self._ep_ret_idx += 1

                    if term or trunc or self._tot_steps >= self._training_steps:
                        self._tot_episodes += 1
                        break

            # Force a final checkpoint at training_steps.
            last_logged = self._data["steps"][-1] if self._data["steps"] else -1
            if last_logged < self._training_steps:
                self.checkpoint()


    def checkpoint(self):
        """Run tests, compute statistics, save debug data and log everything."""

        # Read the clock BEFORE test(): everything from here to the end of this
        # method is checkpoint overhead, not training, and must not be charged
        # to the step rate.
        t_enter = time.perf_counter()
        if self._t_last_ckpt is None:
            # Only reachable through the forced final checkpoint when the run
            # ends before the memory was ever ready. No interval to measure.
            train_sec = np.nan
            d_steps = 0
        else:
            train_sec = t_enter - self._t_last_ckpt
            d_steps = self._tot_steps - self._steps_last_ckpt

        t_test = time.perf_counter()
        self.test()
        test_sec = time.perf_counter() - t_test

        if self._save_heatmaps and self._data_dir is not None:
            heatmaps_dir = os.path.join(self._data_dir, "heatmaps")
            os.makedirs(heatmaps_dir, exist_ok=True)
            from src.utils.heatmaps import heatmaps_from_agent
            heatmaps_from_agent(
                actor=self._actor,
                critic=self._critic,
                env=self._env_train,
                savepath=heatmaps_dir,
                tot_steps=self._tot_steps,
            )

        stats = self._compute_stats(d_steps=d_steps, train_sec=train_sec, test_sec=test_sec)
        self._new_stat_key = self._record_stats(stats)
        self._emit_stats(stats)
        self._checkpoint_count += 1

        # Restart the interval --- the next train_sec covers training and nothing else.
        self._t_last_ckpt = time.perf_counter()
        self._steps_last_ckpt = self._tot_steps


    def _compute_stats(self, d_steps: int, train_sec: float, test_sec: float) -> dict:
        """Gather this checkpoint's statistics into one flat dict."""

        expl_stats = {}

        # Here, coverage and entropy are computed from binned counts, i.e.,
        # discretizations of the environment spaces.
        # If the spaces are large, fine-grained binning is too expensive: either
        # the environment has no binned count, or its bins are too few for
        # accurate statistics.
        # In this case, save the whole replay memory and compute statistics
        # post-training (see README.md).
        if self._critic.visit_count is not None:
            visits = self._critic.visit_count()  # shape: (num states, num actions)
            try:  # Gridworlds have a mask to filter out unreachable states
                state_mask = self._env_train.unwrapped.grid_reachable.reshape(visits.shape[:-1])
            except AttributeError:
                state_mask = np.ones(visits.shape[:-1], dtype=bool)
            coverage_sa, entropy_sa = exploration_stats(
                visits,
                np.broadcast_to(state_mask[..., None], visits.shape),
            )
            coverage_s, entropy_s = exploration_stats(visits.sum(-1), state_mask)
            expl_stats |= {
                "sa_%": coverage_sa,
                "sa_h": entropy_sa,
                "s_%": coverage_s,
                "s_h": entropy_s,
            }

        with warnings.catch_warnings():  # ignore "RuntimeWarning: Mean of empty slice"
            warnings.simplefilter("ignore", category=RuntimeWarning)
            stats = {
                "train (γ)": np.nanmean(self._disc_ret_train),
                "train": np.nanmean(self._undisc_ret_train),
                "test (γ)": self._disc_ret_test,
                "test": self._undisc_ret_test,
            } | expl_stats | self._actor.info() | self._update_stats.get()
        self._update_stats.reset()

        if self._slurm_debug:
            stats |= memory_stats()

        # How many more checkpoints will fire, i.e. how many more test() calls
        # the remaining budget has to pay for. Ceiling, and never negative.
        steps_left = max(self._training_steps - self._tot_steps, 0)
        tests_left = np.ceil(steps_left / self._log_frequency)
        stats |= time_stats(d_steps, train_sec, test_sec, steps_left, tests_left)

        return stats


    def _record_stats(self, stats: dict) -> bool:
        """Append stats to the history saved as npz. Returns whether a key is new.

        Keys can show up after the first checkpoint -- for example, losses do not
        appear with the first log, when no update has been done yet.
        Appending straight away would leave that column shorter
        than "steps" forever, and np.savez stores the ragged lengths without
        complaint, so every value would silently plot one checkpoint early.
        Backfill with nan instead.

        All keys should always show up at every checkpoint. If that is not the
        case (e.g., if your algorithm returns some statistics every other step),
        data and steps will be misaligned.
        We suggest NOT to fill this missing data with np.nan, as that should be
        reserved for detecting bugs (e.g., gradient norm should never be np.nan).
        """

        new_key = False
        for k, v in stats.items():
            if k not in self._data:
                self._data[k] = [np.nan] * len(self._data["steps"])
                new_key = True
            self._data[k].append(v)
        self._data["steps"].append(self._tot_steps)
        return new_key


    # Keys produced by time_stats(). Recorded and logged always; printed only in
    # "dict" mode, where the log is the only window into a running job.
    _TIME_KEYS = ("test_sec", "train_sec", "step/sec", "eta_sec")

    def _emit_stats(self, stats: dict) -> None:
        """Log to W&B, the progress bar, and stdout."""

        # np.float64(0.56) -> 0.56: better prints, and prevents W&B errors.
        stats = {
            k: v.item() if isinstance(v, np.generic) else v
            for k, v in stats.items()
        }

        if self._pbar is not None:
            self._pbar.refresh()

        wandb.log(
            stats,
            step=self._tot_steps,
            commit=False,
        )

        if self._progress_report in ("history_table", "live_table"):
            # The bar already shows elapsed and remaining, so the timing columns
            # would be redundant width in an already wide table.
            shown = {"steps": self._tot_steps} | {
                k: v for k, v in stats.items() if k not in self._TIME_KEYS
            }

            if self._progress_report == "live_table":
                self._pbar.set_stats(shown)
            else:
                print_keys = (
                    self._new_stat_key
                    or self._checkpoint_count % self._print_stats_keys_frequency == 0
                )
                (header_str, stats_str) = format_dict_row(shown, print_keys=print_keys)
                if header_str is not None:
                    self._console.print("[bright_magenta]" + header_str)
                self._console.print(stats_str)

        elif self._progress_report == "dict":
            # builtins.print, not the rich print imported at the top of this
            # module: this print is meant to be caught by SLURM log.out to debug jobs.
            builtins.print({"steps": self._tot_steps} | stats, flush=True)


    def test(self):
        """Run the greedy policy for some episodes. Use copies to not affect
        seeding."""

        if self._testing_episodes < 1 or self._env_test is None:
            return

        self._critic_test.copy_from(self._critic)
        self._critic_test.eval()

        # Buffers built in __init__, reset in place to not allocate new memory
        ep_disc_return = self._disc_ret_test_buf
        ep_undisc_return = self._undisc_ret_test_buf
        ep_done = self._done_test_buf
        ep_disc_return.fill(0.0)
        ep_undisc_return.fill(0.0)
        ep_done.fill(False)
        ep_seed = self._seed_test
        test_rng = self._critic_test.rng_generator(seed=self._rng_seed)
        obs, _ = self._env_test.reset(seed=ep_seed)  # ep_sees is fixed per experiment.rng_seed

        if self._save_videos and self._data_dir is not None:
            import cv2
            frames = []

        ep_steps = 0
        while not np.all(ep_done):
            if self._max_ep_steps and ep_steps > self._max_ep_steps + 1:
                raise RuntimeError(f"test() exceeded {self._max_ep_steps} steps without all episodes ending")

            if self._pbar is not None:
                self._pbar.update("testing", ep_steps)

            # TODO define self._actor_test (for example, with eps = 0.05)
            act = random_argmax(self._critic_test(obs), test_rng, axis=-1)

            # Videos from all episodes are automatically arranged into a grid
            if self._save_videos and self._data_dir is not None:
                frame = self._env_test.render()
                if frame is not None:
                    # Downscale to 25% before gridding or videos will be too large
                    frame = [
                        cv2.resize(f, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
                        for f in frame
                    ]
                    frames.append(frames_to_grid(frame))

            next_obs, rwd, term, trunc, info = self._env_test.step(act)
            ep_disc_return += (self._critic.gamma**ep_steps) * rwd * (1.0 - ep_done)
            ep_undisc_return += rwd * (1.0 - ep_done)
            done = np.logical_or(term, trunc)
            np.logical_or(done, ep_done, out=ep_done)  # out to use the pre-allocated array
            obs = next_obs
            ep_steps += 1

        self._disc_ret_test = ep_disc_return.mean()
        self._undisc_ret_test = ep_undisc_return.mean()

        if self._save_videos and self._data_dir is not None and frames:
            video_dir = os.path.join(self._data_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            out = cv2.VideoWriter(
                os.path.join(video_dir, f"{self._tot_steps}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (frames[0].shape[1], frames[0].shape[0]),
            )
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()

        # Testing episodes length is unknown and they can end earlier than expected.
        # To make it clear that testing is done, manually set the bar to 100%.
        if self._pbar is not None:
            self._pbar.update("testing", self._max_ep_steps)
