import numpy as np
import gymnasium
from gymnasium import logger
from gymnasium.wrappers import (
    AtariPreprocessing,
    FlattenObservation,
    AddRenderObservation,
    ResizeObservation,
    TimeLimit,
    TransformObservation,
    DiscretizeAction,
    DiscretizeObservation,
    FilterObservation,
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


def make_gym_env(train_from_pixels=False, observation_bins=None, action_bins=None, **kwargs):
    if train_from_pixels:
        kwargs["render_mode"] = "rgb_array"
    env_name = kwargs["id"]

    # https://github.com/Farama-Foundation/Shimmy/blob/main/shimmy/atari_env.py
    if "ALE" in env_name:  # eg, ALE/Breakout-v5
        import ale_py
        gymnasium.register_envs(ale_py)
        env = gymnasium.make(
            **kwargs,
            frameskip=1,
            repeat_action_probability=0.0,
            full_action_space=False,
            max_num_frames_per_episode=18_000,  # as in Machado et al., 2018
            obs_type="rgb",
        )
        env = AtariPreprocessing(env)  # (optional)
        env = gymnasium.wrappers.FrameStackObservation(env, 4)

    elif "MiniGrid" in env_name:  # eg, MiniGrid-DoorKey-8x8-v0
        from minigrid import wrappers as minigrid_wrappers
        env = gymnasium.make(**kwargs)  # (default) 7x7x3 partial obs
        # env = minigrid_wrappers.FullyObsWrapper(env) # (optional) WxHx3 full obs, size depends on the grid
        # env = minigrid_wrappers.RGBImgObsWrapper(env) # (optional) RGB-like full obs
        # env = minigrid_wrappers.RGBImgPartialObsWrapper(env) # (optional) RGB-like partial obs
        env = minigrid_wrappers.ImgObsWrapper(env)  # (mandatory) removes the 'mission' field

    else:  # eg, Pendulum-v1, HalfCheetah-v4, dm_control/cheetah-run-v0, highway-v0, FetchReach-v3, PandaReach-v3, ...
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
            from gym_gridworlds.observation_wrappers import (
                MatrixWrapper,
                MatrixWithGoalWrapper,
                ContinuousObservationWrapper,
            )
        if "PointMaze" in env_name:  # eg, PointMaze_Medium-v3
            import gymnasium_robotics
        if "OGBench" in env_name:  # eg, OGBench/antmaze-large-navigate-v0
            import ogbench
            env_name = env_name[len("OGBench/"):]
            kwargs.pop("id")
            env = ogbench.make_env_and_datasets(env_name, env_only=True, **kwargs)
        else:
            env = gymnasium.make(**kwargs)

        if isinstance(env.action_space, gymnasium.spaces.Box):
            env = DiscretizeAction(env, bins=action_bins)

        if "PointMaze" in env_name:
            env = FilterObservation(env, ["observation"])

        if train_from_pixels:
            env = AddRenderObservation(env, render_only=True)
            env = ResizeObservation(env, (84, 84))
            env = ChannelsFirstWrapper(env)
        else:
            if "Gym-Gridworlds" in env_name:
                pass
                # if "Taxi" in env_name or "CleanDirt" in env_name:
                #     env = MatrixWithGoalWrapper(env)
                # else:
                #     env = MatrixWrapper(env)
                #     # env = ContinuousObservationWrapper(env)
                # env = FlattenObservation(env)
            elif "Gym-MinAtar" in env_name:
                env = ChannelsFirstWrapper(env)
            # elif isinstance(env.observation_space, gymnasium.spaces.Box):
            #     env = FlattenObservation(env)
            #     env = DiscretizeObservation(env, bins=observation_bins)
            else:
                env = FlattenObservation(env)
                # env = NormalizeObservationWrapper(env)

    return env
