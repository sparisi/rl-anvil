import traceback
import subprocess
import gymnasium
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import os
from functools import partial
from rich import print
import wandb
import numpy as np

from src.utils.id import prepare_run
from src.wrappers import gym_wrappers
from src.experiment import Experiment
import src.actor
import src.critic
import src.replay_memory


def _run(cfg: DictConfig) -> None:
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = None

    slurm_job_id = os.getenv("SLURM_JOB_ID")

    with open_dict(cfg):
        cfg.meta.git_hash = git_hash
        cfg.meta.slurm_job_id = slurm_job_id

    # Check for existing data with the same config and make needed directories
    config_id = prepare_run(cfg)
    if config_id is None:
        return

    print(f":rocket: Running configuration {config_id} (seed {cfg.experiment.rng_seed})")

    # Make training environment. `make_env` accepts overrides so the test env
    # can differ from training (e.g. no reward noise, rgb_array rendering)
    # without mutating cfg.environment — which would desync the on-disk
    # cfg.yaml and the wandb config from the run's actual training setup.
    def make_env(**overrides):
        return gym_wrappers.make_gym_env(**{**cfg.environment, **overrides})
    env_train = make_env()

    # Make testing environments (vectorized)
    env_test = None
    if cfg.experiment.testing_episodes > 0:
        test_overrides = {}
        if "reward_noise_std" in cfg.environment.keys():
            test_overrides["reward_noise_std"] = 0.0  # Test without reward noise
        if cfg.results.save_videos:
            test_overrides["render_mode"] = "rgb_array"  # Need RGB rendering to save videos
        env_test = gymnasium.vector.SyncVectorEnv(
            [partial(make_env, **test_overrides) for _ in range(cfg.experiment.testing_episodes)]
        )

    # Make agent
    critic = getattr(src.critic, cfg.agent.critic.id)(
        env_train.observation_space,
        env_train.action_space,
        **cfg.agent.critic,
        seed=cfg.experiment.rng_seed,
    )
    actor = getattr(src.actor, cfg.agent.actor.id)(
        critic,
        **cfg.agent.actor,
        seed=cfg.experiment.rng_seed,
    )

    # Init W&B
    wandb.init(
        id=None,  # There is no save & resume feature, so a resubmitted seed must count as a new run
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # W&B does not support DictConfig
        name=f"{config_id}_{cfg.experiment.rng_seed}",
        tags=[f"job_id:{os.getenv('SLURM_JOB_ID', 'manual_run')}"],  # Log job ID for SLURM runs
        # settings=wandb.Settings(x_disable_stats=True),  # Disable logging of system metrics (CPU, GPU, ...)
        **cfg.wandb,
    )

    # Run experiment
    Experiment(
        env_train,
        env_test,
        actor,
        critic,
        **cfg.experiment,
        **cfg.results,
        **cfg.meta,
    ).run()


# Wrap _run() to raise only errors relevant the the run, and skip Hydra stacks.
@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(cfg: DictConfig) -> None:
    try:
        _run(cfg)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None


if __name__ == "__main__":
    run()
