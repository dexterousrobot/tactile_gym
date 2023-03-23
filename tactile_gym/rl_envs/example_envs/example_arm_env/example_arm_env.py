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

        # add environment specific env parameters
        env_params["workframe"] = np.array([0.35, 0.0, 0.1, -np.pi, 0.0, np.pi/2])
        env_params["tcp_lims"] = np.array([
            *[-0.2, 0.2]*3,
            *[-np.pi/4, np.pi/4]*3
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

        # setup variables
        self.setup_action_space()
        self.setup_observation_space()
        set_debug_camera(self._pb, visual_sensor_params)

        # init environment
        load_standard_environment(self._pb)
        self.reset()

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
        done = self.get_termination()
        reward = self.get_reward()

        return reward, done

    def get_termination(self):
        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True
        return False

    def get_reward(self):
        return 0.0
