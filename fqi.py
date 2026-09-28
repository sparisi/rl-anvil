"""Fitted Q-Iteration to learn the true optimal Q-function.

For each (s, a) pair, we sample transitions from the environment (via
``env.unwrapped.set_state``) and repeatedly train a tabular critic on
them until the average TD error is negligible. The result is still an
approximation because we don't use exact transition probabilities, but
it is close because we train on ~10k transitions per (s, a) pair.

Everything is written under `<data_dir>/<config_id>/<rng_seed>/fqi/`, so the
maps produced here never overwrite those of the training run they approximate.
"""

import hydra
from omegaconf import DictConfig
import os
import numpy as np
from copy import deepcopy
from functools import partial
from tqdm import tqdm
import gymnasium

from src.utils.id import prepare_run
from src.utils.heatmaps import heatmaps_from_agent
from src.experiment import Experiment
from src.pseudocount import tabular_count
from src.wrappers import gym_wrappers
import src.critic
import src.actor
import src.replay_memory


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(cfg: DictConfig) -> None:
    # Force heatmaps so prepare_run always allocates a directory. `results` is
    # dropped from the config identity, so this does not change the config ID.
    cfg.results.save_heatmaps = True

    # The ID is computed before the critic overrides below, so it matches the
    # training run this Q-function is the reference for.
    config_id = prepare_run(cfg)
    if config_id is None:
        return

    # prepare_run leaves results.data_dir pointing at the seed folder
    fqi_dir = os.path.join(cfg.results.data_dir, "fqi")
    os.makedirs(fqi_dir, exist_ok=True)
    cfg.results.data_dir = fqi_dir

    # Overrides rather than mutating cfg.environment, which would desync the
    # cfg.yaml prepare_run just wrote from the environment actually used.
    def make_env(**overrides):
        return gym_wrappers.make_gym_env(**{**cfg.environment, **overrides})

    no_noise = {}
    if "reward_noise_std" in cfg.environment.keys():
        no_noise["reward_noise_std"] = 0.0

    env = make_env(**no_noise)
    env.reset(seed=cfg.experiment.rng_seed)

    n_obs = int(env.observation_space.n)
    n_act = int(env.action_space.n)
    n_pairs = n_obs * n_act
    transitions_per_pair = 10_000

    cfg.agent.critic.batch_size = n_pairs
    cfg.agent.critic.sequence_length = 1
    cfg.agent.critic.lr.init_value = 1.0
    cfg.agent.critic.lr.end_value = 0.001
    cfg.agent.critic.lr.steps = transitions_per_pair

    if "batch_size_visit" in cfg.agent.critic:
        cfg.agent.critic.batch_size_visit = n_pairs
        cfg.agent.critic.sequence_length_visit = 1
        cfg.agent.critic.lr_visit.init_value = 1.0
        cfg.agent.critic.lr_visit.end_value = 0.001
        cfg.agent.critic.lr_visit.steps = transitions_per_pair

    critic = getattr(src.critic, cfg.agent.critic.id)(
        env.observation_space,
        env.action_space,
        **cfg.agent.critic,
        seed=cfg.experiment.rng_seed,
    )
    actor = getattr(src.actor, cfg.agent.actor.id)(
        critic,
        **cfg.agent.actor,
        seed=cfg.experiment.rng_seed,
    )
    critic_rng = critic.rng_generator(cfg.experiment.rng_seed)

    # heatmaps_from_agent draws the visit maps from this counter
    critic.visit_count = tabular_count(env)

    memory = src.replay_memory.ReplayMemory(min_size=n_pairs, max_size=n_pairs)
    memory.init(
        obs=env.observation_space.sample(),
        act=env.action_space.sample(),
        rwd=np.asarray(0.0),
        term=np.asarray(False),
        trunc=np.asarray(False),
        next_obs=env.observation_space.sample(),
    )

    def save_pics(step):
        heatmaps_from_agent(
            actor=actor,
            critic=critic,
            env=env,
            savepath=fqi_dir,
            tot_steps=step,
        )

    pbar = tqdm(range(transitions_per_pair))
    i = 0
    consecutive_below = 0
    for i in pbar:
        for obs in range(n_obs):
            for act in range(n_act):
                env.unwrapped.set_state(obs)
                next_obs, rwd, term, trunc, _ = env.step(act)
                memory.add(
                    obs=obs,
                    act=act,
                    rwd=rwd,
                    term=term,
                    trunc=trunc,
                    next_obs=next_obs,
                )
        stats = critic.update(replay_memory=memory, rng_generator=critic_rng)
        critic.post_step()
        tot_loss = float(np.abs(stats["td_err"]).sum())
        tot_loss_visit = (
            float(np.abs(stats["td_err_visit"]).sum())
            if "td_err_visit" in stats
            else 0.0
        )
        pbar.set_description(
            f"td_err={tot_loss:.6f} td_err_visit={tot_loss_visit:.6f}"
        )
        if i % 25 == 0:
            save_pics(i)
        if tot_loss < 10e-4 and tot_loss_visit < 10e-4:
            consecutive_below += 1
            if consecutive_below >= 10:
                break
        else:
            consecutive_below = 0
    save_pics(i)

    if cfg.experiment.testing_episodes < 1:
        return

    cfg.results.save_videos = True
    cfg.results.progress_report = None

    test_overrides = {"render_mode": "rgb_array", **no_noise}
    env_test = gymnasium.vector.SyncVectorEnv(
        [
            partial(make_env, **test_overrides)
            for _ in range(cfg.experiment.testing_episodes)
        ]
    )

    # Experiment.__init__ resets the critic it is given, and only deep-copies it
    # into _critic_test afterwards, so everything learned above would be thrown
    # away. Hold the trained tables aside and put them back: test() copies
    # _critic into _critic_test itself.
    trained = deepcopy(critic)
    experiment = Experiment(
        env,
        env_test,
        actor,
        critic,
        **cfg.experiment,
        **cfg.results,
        **cfg.meta,
    )
    experiment._critic.copy_from(trained)
    experiment._tot_steps = i
    experiment.test()
    env_test.close()


if __name__ == "__main__":
    run()
