from tactile_gym.sb3_helpers.params.default_params import env_args
from tactile_gym.sb3_helpers.params.default_params import rl_params_ppo
from tactile_gym.sb3_helpers.params.default_params import ppo_params
from tactile_gym.sb3_helpers.params.default_params import rl_params_sac
from tactile_gym.sb3_helpers.params.default_params import sac_params


env_args["env_params"]["max_steps"] = 10_000
env_args["env_params"]["observation_mode"] = "oracle"
# env_args["env_params"]["observation_mode"] = "tactile"
# env_args["env_params"]["observation_mode"] = "visual"
# env_args["env_params"]["observation_mode"] = "visuotactile"

env_args["robot_arm_params"]["control_mode"] = "tcp_velocity_control"
env_args["robot_arm_params"]["control_dofs"] = ["x", "y", "z", "Rx", "Ry", "Rz"]

env_args["tactile_sensor_params"]["type"] = "standard_tactip"
# env_args["tactile_sensor_params"]["type"] = "standard_digit"
# env_args["tactile_sensor_params"]["type"] = "standard_digitac"

rl_params_ppo["env_id"] = "example_arm-v0"
rl_params_ppo["total_timesteps"] = int(1e6)
ppo_params["learning_rate"] = 3e-4

rl_params_sac["env_id"] = "example_arm-v0"
rl_params_sac["total_timesteps"] = int(1e6)
sac_params["learning_rate"] = 3e-4
