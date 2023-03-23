from tactile_gym.rl_envs.exploration.surface_follow.base_surface_env import (
    BaseSurfaceEnv,
)


class SurfaceFollowAutoEnv(BaseSurfaceEnv):
    def __init__(
        self,
        env_params={},
        robot_arm_params={},
        tactile_sensor_params={},
        visual_sensor_params={},
    ):

        super(SurfaceFollowAutoEnv, self).__init__(env_params, robot_arm_params, tactile_sensor_params, visual_sensor_params)

    def encode_actions(self, actions):
        """
        return actions as np.array in correct places for sending to robot arm
        """
        encoded_actions = super(SurfaceFollowAutoEnv, self).encode_actions(actions)
        encoded_actions[0] = self.workframe_directions[0] * self.max_action
        encoded_actions[1] = self.workframe_directions[1] * self.max_action
        return encoded_actions

    def get_reward(self):
        """
        Calculate the reward when in dense mode.
        """
        W_surf = 1.0
        W_norm = 1.0

        # get the distances
        surf_dist = self.z_dist_to_surface()
        cos_dist = self.cos_dist_to_surface_normal()

        # sum rewards with multiplicative factors
        reward = -((W_surf * surf_dist) + (W_norm * cos_dist))

        return reward
