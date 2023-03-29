import numpy as np
from tactile_gym.envs.exploration.surface_follow.base_surface_env import (
    BaseSurfaceEnv,
)


class SurfaceFollowGoalEnv(BaseSurfaceEnv):
    def __init__(
        self,
        env_params={},
        robot_arm_params={},
        tactile_sensor_params={},
        visual_sensor_params={},
    ):

        super(SurfaceFollowGoalEnv, self).__init__(env_params, robot_arm_params, tactile_sensor_params, visual_sensor_params)

    def get_reward(self):
        """
        Calculate the reward when in dense mode.
        """
        W_goal = 1.0
        W_surf = 10.0
        W_norm = 1.0

        # get the distances
        goal_dist = self.xy_dist_to_goal()
        surf_dist = self.z_dist_to_surface()
        cos_dist = self.cos_dist_to_surface_normal()

        # sum rewards with multiplicative factors
        reward = -((W_goal * goal_dist) + (W_surf * surf_dist) + (W_norm * cos_dist))

        return reward

    def get_extended_feature_array(self):
        """
        features needed to help complete task.
        Goal pose and current tcp pose.
        """
        # get sim info on TCP
        tcp_pose, _ = self.embodiment.arm.get_current_tcp_pose_vel()
        tcp_pose_workframe = self.worldframe_to_workframe(tcp_pose)
        tcp_pos_workframe = tcp_pose_workframe[:3]

        # convert the features into array that matches the image observation shape
        feature_array = np.array([*tcp_pos_workframe, *self.goal_pose_workframe[:3]])

        return feature_array
