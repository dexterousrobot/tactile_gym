import os
import sys

from stable_baselines3.common.callbacks import EvalCallback, EveryNTimesteps
from stable_baselines3 import PPO, SAC

import tactile_gym.rl_envs
from tactile_gym.sb3_helpers.params import import_parameters
from tactile_gym.sb3_helpers.rl_utils import make_training_envs, make_eval_env
from tactile_gym.sb3_helpers.eval_agent_utils import final_evaluation

from tactile_learning.utils.utils_learning import save_json_obj, convert_json, make_dir

from tactile_gym.sb3_helpers.custom.custom_callbacks import (
    FullPlottingCallback,
    ProgressBarManager,
)


def train_agent(
    algo_name="ppo",
    env_id="edge_follow-v0",
    env_args={},
    rl_params={},
    algo_params={}
):

    # create save dir
    save_dir = os.path.join(
        "saved_models/", rl_params["env_id"], algo_name, "s{}_{}".format(
            rl_params["seed"], env_args["env_params"]["observation_mode"])
    )
    make_dir(save_dir)

    # save params
    save_json_obj(convert_json(rl_params), os.path.join(save_dir, "rl_params"))
    save_json_obj(convert_json(algo_params), os.path.join(save_dir, "algo_params"))
    save_json_obj(convert_json(env_args), os.path.join(save_dir, "env_args"))

    # load the envs
    env = make_training_envs(env_id, env_args, rl_params, save_dir)

    eval_env = make_eval_env(
        env_id,
        env_args,
        rl_params
    )

    # define callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "trained_models/"),
        log_path=os.path.join(save_dir, "trained_models/"),
        eval_freq=rl_params["eval_freq"],
        n_eval_episodes=rl_params["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=1,
    )

    plotting_callback = FullPlottingCallback(log_dir=save_dir, total_timesteps=rl_params["total_timesteps"])
    event_plotting_callback = EveryNTimesteps(n_steps=rl_params["eval_freq"] * rl_params["n_envs"], callback=plotting_callback)

    # create the model with hyper params
    if algo_name == "ppo":
        model = PPO(rl_params["policy"], env, **algo_params, verbose=1)
    elif algo_name == "sac":
        model = SAC(rl_params["policy"], env, **algo_params, verbose=1)
    else:
        sys.exit("Incorrect algorithm specified: {}.".format(algo_name))

    # train an agent
    with ProgressBarManager(rl_params["total_timesteps"]) as progress_bar_callback:
        model.learn(
            total_timesteps=rl_params["total_timesteps"],
            callback=[progress_bar_callback, eval_callback, event_plotting_callback],
        )

    # save the final model after training
    model.save(os.path.join(save_dir, "trained_models", "final_model"))
    env.close()
    eval_env.close()

    # run final evaluation over 20 episodes and save a vid
    final_evaluation(
        saved_model_dir=save_dir,
        n_eval_episodes=10,
        seed=None,
        deterministic=True,
        show_gui=False,
        show_tactile=False,
        show_vision=False,
        render=True,
        save_vid=True,
        take_snapshot=False,
    )


if __name__ == "__main__":

    # choose which RL algo to use
    algo_name = 'ppo'
    # algo_name = 'sac'

    # env_id = "edge_follow-v0"
    env_id = 'surface_follow-v0'
    # env_id = 'object_roll-v0'
    # env_id = "object_push-v0"
    # env_id = 'object_balance-v0'

    # import paramters
    env_args, rl_params, algo_params = import_parameters(env_id, algo_name)
    train_agent(
        algo_name,
        env_id,
        env_args,
        rl_params,
        algo_params
    )
