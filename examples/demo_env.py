import argparse
import gym
import pybullet as pb

from tactile_gym.utils.demo_utils import demo_rl_env
from tactile_gym.sb3_helpers.params import import_parameters
import tactile_gym.envs


def main():

    seed = int(0)
    num_iter = 10
    render = False
    print_info = False
    image_size = [128, 128]  # sets both rgb and tactile images

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-env",
        type=str,
        default='example_arm-v0',
        help="""Options: {
                example_arm-v0,
                edge_follow-v0, surface_follow-v0, surface_follow-v1,
                object_roll-v0, object_push-v0, object_balance-v0}"""
    )
    args = parser.parse_args()
    env_id = args.env

    # import default params for env
    env_args = import_parameters(env_id, None)
    env_params = env_args["env_params"]
    robot_arm_params = env_args["robot_arm_params"]
    tactile_sensor_params = env_args["tactile_sensor_params"]
    visual_sensor_params = env_args["visual_sensor_params"]

    # overwrite default params for testing
    env_params["max_steps"] = 10_000
    env_params["show_gui"] = True
    env_params["observation_mode"] = "oracle"
    # env_params["observation_mode"] = "tactile"
    # env_params["observation_mode"] = "visual"
    # env_params["observation_mode"] = "visuotactile"

    robot_arm_params["type"] = "ur5"
    # robot_arm_params["type"] = "franka_panda"
    # robot_arm_params["type"] = "kuka_iiwa"
    # robot_arm_params["type"] = "cr3"
    # robot_arm_params["type"] = "mg400"

    tactile_sensor_params["show_tactile"] = False
    tactile_sensor_params["image_size"] = image_size
    # tactile_sensor_params["type"] = "standard_tactip"
    # tactile_sensor_params["type"] = "standard_digit"
    # tactile_sensor_params["type"] = "standard_digitac"
    # tactile_sensor_params["type"] = "flat_tactip"
    # tactile_sensor_params["type"] = "right_angle_tactip"

    visual_sensor_params["show_vision"] = False
    visual_sensor_params["image_size"] = image_size

    env = gym.make(
        id=env_id,
        env_params=env_params,
        robot_arm_params=robot_arm_params,
        tactile_sensor_params=tactile_sensor_params,
        visual_sensor_params=visual_sensor_params,
    )

    # set seed for deterministic results
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    # create controllable parameters on GUI
    min_action, max_action = env.min_action, env.max_action
    action_ids = [pb.addUserDebugParameter(control_dof, min_action, max_action, 0)
                  for control_dof in robot_arm_params["control_dofs"]]

    # run the control loop
    demo_rl_env(env, num_iter, action_ids, env_params["show_gui"], render, print_info)


if __name__ == "__main__":
    main()
