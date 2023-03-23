import gym
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecTransposeImage,
    VecFrameStack,
    VecNormalize,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env


def make_training_envs(env_id, env_args, rl_params, save_dir):

    env = make_vec_env(
        env_id,
        n_envs=rl_params["n_envs"],
        seed=rl_params["seed"],
        vec_env_cls=SubprocVecEnv,
        monitor_dir=save_dir,
        env_kwargs=env_args,
    )

    # normalize obs/rew with running metrics
    env = VecNormalize(
        env,
        training=True,
        norm_obs=rl_params['norm_obs'],
        norm_reward=rl_params['norm_reward'],
    )

    # stack the images for frame history
    env = VecFrameStack(env, n_stack=rl_params["n_stack"])

    # transpose images in observation
    env = VecTransposeImage(env)

    return env


def make_eval_env(
    env_id,
    env_args,
    rl_params
):
    """
    Make a single environment with visualisation specified.
    """
    eval_env = gym.make(env_id, **env_args)

    # wrap in monitor
    eval_env = Monitor(eval_env)

    # dummy vec env generally faster than SubprocVecEnv for small networks
    eval_env = DummyVecEnv([lambda: eval_env])

    # normalize obs/rew with running metrics
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=rl_params['norm_obs'],
        norm_reward=rl_params['norm_reward'],
    )

    # stack observations
    eval_env = VecFrameStack(eval_env, n_stack=rl_params["n_stack"])

    # transpose images in observation
    eval_env = VecTransposeImage(eval_env)

    return eval_env
