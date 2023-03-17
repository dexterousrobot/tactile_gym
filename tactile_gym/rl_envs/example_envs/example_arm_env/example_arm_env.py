import gym
import numpy as np

from tactile_sim.assets.default_rest_poses import rest_poses_dict
from tactile_sim.utils.setup_pb_utils import load_standard_environment
from tactile_sim.utils.setup_pb_utils import set_debug_camera
from tactile_sim.embodiments.embodiments import VisuoTactileArmEmbodiment

from tactile_gym.rl_envs.base_tactile_env import BaseTactileEnv


class ExampleArmEnv(BaseTactileEnv):
    def __init__(
        self,
        env_params={},
        robot_arm_params={},
        tactile_sensor_params={},
        visual_sensor_params={},
    ):

        # used to setup control of robot
        self._sim_time_step = 1.0 / 240.0
        self._control_rate = 1.0 / 10.0
        self._velocity_action_repeat = int(np.floor(self._control_rate / self._sim_time_step))
        self._max_blocking_pos_move_steps = 10

        # add environment specific env parameters
        env_params["workframe"] = np.array([0.35, 0.0, 0.1, -np.pi, 0.0, np.pi/2])
        # env_params["tcp_lims"] = np.column_stack([-np.inf * np.ones(6), np.inf * np.ones(6)])
        env_params["tcp_lims"] = np.array([
            *[-0.1, 0.1]*3,
            *[-np.pi/8, np.pi/8]*3
        ]).reshape((6, 2))

        # add environment specific robot arm parameters
        robot_arm_params["rest_poses"] = rest_poses_dict[robot_arm_params["type"]]
        robot_arm_params["tcp_link_name"] = "tcp_link"

        # add environment specific tactile sensor parameters
        tactile_sensor_params["core"] = "no_core"
        tactile_sensor_params["dynamics"] = {'stiffness': 50, 'damping': 100, 'friction': 10.0}

        # add environment specific visual sensor parameters
        visual_sensor_params["dist"] = 1.0
        visual_sensor_params["yaw"] = 90.0
        visual_sensor_params["pitch"] = -25.0
        visual_sensor_params["pos"] = [0.6, 0.0, 0.0525]
        visual_sensor_params["fov"] = 75.0
        visual_sensor_params["near_val"] = 0.1
        visual_sensor_params["far_val"] = 100.0

        super(ExampleArmEnv, self).__init__(env_params, robot_arm_params, tactile_sensor_params, visual_sensor_params)

        self.embodiment = VisuoTactileArmEmbodiment(
            self._pb,
            robot_arm_params=robot_arm_params,
            tactile_sensor_params=tactile_sensor_params,
            visual_sensor_params=visual_sensor_params
        )

        # init environment
        set_debug_camera(self._pb, visual_sensor_params)
        load_standard_environment(self._pb)
        self.reset()

        # setup variables
        self.setup_action_space()
        self.setup_observation_space()

    def setup_action_space(self):

        # these are used for bounds on the action space in SAC and clipping
        # range for PPO
        self.min_action, self.max_action = -1.0, 1.0

        # define action ranges per act dim to rescale output of policy
        if self._robot_arm_params["control_mode"] == "tcp_position_control":
            self.lin_pos_lim = 0.001  # m
            self.ang_pos_lim = 1 * (np.pi / 180)  # rad

        elif self._robot_arm_params["control_mode"] == "tcp_velocity_control":
            self.lin_vel_lim = 0.01  # m/s
            self.ang_vel_lim = 5.0 * (np.pi / 180)  # rad/s

        elif self._robot_arm_params["control_mode"] == "joint_position_control":
            self.joint_pos_lim = 0.05 * (np.pi / 180)  # rad

        elif self._robot_arm_params["control_mode"] == "joint_velocity_control":
            self.joint_vel_lim = 1.0 * (np.pi / 180)  # rad/s

        # setup action space
        self.act_dim = self.get_act_dim()
        self.action_space = gym.spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(self.act_dim,),
            dtype=np.float32,
        )

    def reset(self):

        # full reset pybullet sim to clear cache, this avoids silent bug where memory fills and visual
        # rendering fails, this is more prevalent when loading/removing larger files
        if self.reset_counter == self.reset_limit:
            self.full_reset()

        self.reset_counter += 1
        self._env_step_counter = 0

        # reset TCP pos and rpy in work frame
        self.embodiment.reset(reset_tcp_pose=self._env_params["workframe"])

        # get the starting observation
        self._observation = self.get_observation()

        return self._observation

    def get_step_data(self):

        # self.embodiment.arm.draw_ee()
        # self.embodiment.arm.draw_tcp()
        # self.embodiment.tactile_sensor.draw_camera_frame()
        # self.draw_workframe()
        # self.draw_tcp_lims()

        # get rl info
        done = self.termination()

        if self._env_params["reward_mode"] == "sparse":
            reward = self.sparse_reward()

        elif self._env_params["reward_mode"] == "dense":
            reward = self.dense_reward()

        return reward, done

    def termination(self):
        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True
        return False

    def sparse_reward(self):
        return 0.0

    def dense_reward(self):
        return 0.0
