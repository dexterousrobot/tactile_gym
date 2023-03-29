from tactile_gym.sb3_helpers.params.default_params import env_args
from tactile_gym.sb3_helpers.params.default_params import rl_params_ppo
from tactile_gym.sb3_helpers.params.default_params import ppo_params
from tactile_gym.sb3_helpers.params.default_params import rl_params_sac
from tactile_gym.sb3_helpers.params.default_params import sac_params


env_args["env_params"]["max_steps"] = 250
env_args["env_params"]["observation_mode"] = "oracle"
# env_args["env_params"]["observation_mode"] = "tactile_and_feature"
# env_args["env_params"]["observation_mode"] = "visual_and_feature"
# env_args["env_params"]["observation_mode"] = "visuotactile_and_feature"

env_args["robot_arm_params"]["control_mode"] = "tcp_velocity_control"
env_args["robot_arm_params"]["control_dofs"] = ["x", "y"]

env_args["tactile_sensor_params"]["type"] = "flat_tactip"

rl_params_ppo["env_id"] = "object_roll-v0"
rl_params_ppo["total_timesteps"] = int(2e6)
ppo_params["learning_rate"] = 3e-4

rl_params_sac["env_id"] = "object_roll-v0"
rl_params_sac["total_timesteps"] = int(2e6)
sac_params["learning_rate"] = 3e-4
