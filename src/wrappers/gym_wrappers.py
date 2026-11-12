import importlib

import numpy as np
import gymnasium
from gymnasium import logger
from omegaconf import DictConfig, OmegaConf
from gymnasium.wrappers import (
    AddRenderObservation,
    ResizeObservation,
)


def get_wrapper_names(env):
    """
    Utility to retrieve the names of all wrappers used on an environment.
    """

    names = []
    current = env
    while isinstance(current, gymnasium.Wrapper):
        names.append(type(current).__name__)
        current = current.env
    return names


class ChannelsFirstWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape  # (H, W, C)
        new_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (C, H, W)
        self.observation_space = gymnasium.spaces.Box(
            low=env.observation_space.low.transpose((2, 0, 1)),
            high=env.observation_space.high.transpose((2, 0, 1)),
            shape=new_shape,
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


class NormalizeObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = env.observation_space.low
        high = env.observation_space.high
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            logger.warn(
                "Observation space is unbounded, observations will not be normalized."
            )
            self.obs_bound = np.ones_like(low)
            return

        self.obs_bound = np.maximum(np.abs(low), np.abs(high))
        self.observation_space = gymnasium.spaces.Box(
            low / self.obs_bound,
            high / self.obs_bound,
        )

    def observation(self, observation):
        return observation / self.obs_bound


def wrapper_class(name):
    """
    Return the wrapper class whose full dotted path is `name`, imported from
    its module. Raise ValueError if `name` has no module.

    Example

        wrapper_class("gymnasium.wrappers.DiscretizeAction")
        wrapper_class("gym_gridworlds.observation_wrappers.MatrixWrapper")
    """

    module, _, cls = name.rpartition(".")
    if not module:
        raise ValueError(
            f"Wrapper '{name}' has no module: give the full dotted path of the "
            f"class, e.g., gymnasium.wrappers.{name}."
        )
    return getattr(importlib.import_module(module), cls)


def apply_wrappers(env, wrappers):
    """
    Wrap `env` with every wrapper of `wrappers`, in order. Each is a mapping
    whose `id` is the full dotted path of the wrapper class (see
    wrapper_class) and whose other keys are passed to it as keyword arguments.
    """

    for spec in wrappers or []:
        if isinstance(spec, DictConfig):
            spec = OmegaConf.to_container(spec, resolve=True)
        spec = dict(spec)
        env = wrapper_class(spec.pop("id"))(env, **spec)
    return env


def make_gym_env(train_from_pixels=False, wrappers=None, **kwargs):
    """
    Make the environment `kwargs["id"]` with the other keys of `kwargs` and
    wrap it with `wrappers` (see apply_wrappers). With `train_from_pixels`, the
    environment renders as an RGB array, and except for Atari and MiniGrid its
    observations are then replaced by the render, resized to 84x84 and with
    channels first. MinAtar observations otherwise get their channels first.
    """

    if train_from_pixels:
        kwargs["render_mode"] = "rgb_array"
    env_name = kwargs["id"]

    # Packages that register their environments with gymnasium when imported
    if "ALE" in env_name:  # eg, ALE/Breakout-v5
        import ale_py
        gymnasium.register_envs(ale_py)
    if "MiniGrid" in env_name:  # eg, MiniGrid-DoorKey-8x8-v0
        import minigrid
    if "Vizdoom" in env_name:  # eg, VizdoomHealthGatheringSupreme-v0
        from vizdoom import gymnasium_wrapper
    if "MiniWorld" in env_name:  # eg, MiniWorld-Hallway-v0
        import miniworld
    if "PyFlyt" in env_name:  # eg, PyFlyt/QuadX-Hover-v0
        import PyFlyt.gym_envs
    if "Gym-MinAtar" in env_name:  # eg, Gym-MinAtar/Breakout-v1
        import gym_minatar
    if "Gym-Gridworlds" in env_name:  # eg, Gym-Gridworlds/Empty-2x2-v0
        import gym_gridworlds
    if "PointMaze" in env_name:  # eg, PointMaze_Medium-v3
        import gymnasium_robotics

    # OGBench must call its own make
    if "OGBench" in env_name:  # eg, OGBench/antmaze-large-navigate-v0
        import ogbench
        kwargs.pop("id")
        env = ogbench.make_env_and_datasets(
            env_name[len("OGBench/"):],
            env_only=True,
            **kwargs,
        )
    # Every other environment simply calls gymnasium.make
    else:
        env = gymnasium.make(**kwargs)

    env = apply_wrappers(env, wrappers)

    if train_from_pixels and "ALE" not in env_name and "MiniGrid" not in env_name:
        env = AddRenderObservation(env, render_only=True)
        env = ResizeObservation(env, (84, 84))
        env = ChannelsFirstWrapper(env)
    elif "Gym-MinAtar" in env_name:
        env = ChannelsFirstWrapper(env)

    return env
