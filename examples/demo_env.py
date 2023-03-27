import argparse
import gym
import pybullet as pb

from tactile_gym.utils.demo_utils import demo_rl_env
import tactile_gym.rl_envs


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

    env_params = {
        "max_steps": 10_000,
        "show_gui": True,

        "observation_mode": "oracle",
        # 'observation_mode': 'tactile',
        # 'observation_mode': 'visual',
        # 'observation_mode': 'visuotactile',
        # 'observation_mode': 'tactile_and_feature',
        # 'observation_mode': 'visual_and_feature',
        # 'observation_mode': 'visuotactile_and_feature',
    }

    robot_arm_params = {
        "type": "ur5",
        # "type": "franka_panda",
        # "type": "kuka_iiwa",
        # "type": "cr3",
        # "type": "mg400",

        # "control_mode": "tcp_position_control",
        "control_mode": "tcp_velocity_control",
        "control_dofs": ['x', 'y', 'z', 'Rx', 'Ry', 'Rz'],
        # "control_dofs": ['x', 'y'],

        # the type of control used
        # "control_mode": "joint_position_control",
        # "control_mode": "joint_velocity_control",
        # "control_dofs": ['J1', 'J2', 'J3', 'J4', 'J5', 'J6'],
        # "control_dofs": ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7'],
    }

    tactile_sensor_params = {
        "type": "standard_tactip",
        # "type": "standard_digit",
        # "type": "standard_digitac",

        # "type": "flat_tactip",
        # "type": "right_angle_tactip",

        "image_size": image_size,
        "turn_off_border": False,
        "show_tactile": True,
    }

    visual_sensor_params = {
        'image_size': image_size,
        'show_vision': False
    }

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
