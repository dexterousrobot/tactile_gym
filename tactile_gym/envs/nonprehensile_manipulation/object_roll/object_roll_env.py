import numpy as np
import cv2

from tactile_sim.assets.default_rest_poses import rest_poses_dict
from tactile_sim.utils.transforms import inv_transform_eul

from tactile_gym.assets import add_assets_path
from tactile_gym.envs.nonprehensile_manipulation.base_object_env import BaseObjectEnv


class ObjectRollEnv(BaseObjectEnv):
    def __init__(
        self,
        env_params={},
        robot_arm_params={},
        tactile_sensor_params={},
        visual_sensor_params={},
    ):

        # env specific values
        self.termination_pos_dist = 0.001
        self.embed_dist = 0.0015
        self.visualise_goal = True
        self.default_obj_radius = 0.0025

        # add environment specific env parameters
        env_params["workframe"] = np.array([0.65, 0.0, (2*self.default_obj_radius) - self.embed_dist, -np.pi, 0.0, np.pi/2])

        tcp_lims = np.zeros(shape=(6, 2))
        tcp_lims[0, 0], tcp_lims[0, 1] = -0.05, 0.05  # x lims
        tcp_lims[1, 0], tcp_lims[1, 1] = -0.05, 0.05  # y lims
        tcp_lims[2, 0], tcp_lims[2, 1] = -0.01, 0.01  # z lims
        tcp_lims[3, 0], tcp_lims[3, 1] = 0, 0  # roll lims
        tcp_lims[4, 0], tcp_lims[4, 1] = 0, 0  # pitch lims
        tcp_lims[5, 0], tcp_lims[5, 1] = 0, 0  # yaw lims
        env_params["tcp_lims"] = tcp_lims

        # add environment specific robot arm parameters
        robot_arm_params["use_tcp_frame_control"] = False
        robot_arm_params["rest_poses"] = rest_poses_dict[robot_arm_params["type"]]
        robot_arm_params["tcp_link_name"] = "tcp_link"

        # add environment specific tactile sensor parameters
        tactile_sensor_params["core"] = "fixed"
        tactile_sensor_params["dynamics"] = {'stiffness': 10, 'damping': 100, 'friction': 10.0}

        # add environment specific visual sensor parameters
        visual_sensor_params["dist"] = 0.01
        visual_sensor_params["yaw"] = 90.0
        visual_sensor_params["pitch"] = 0.0
        visual_sensor_params["pos"] = [0.75, 0.0, 0.00775]
        visual_sensor_params["fov"] = 75.0
        visual_sensor_params["near_val"] = 0.01
        visual_sensor_params["far_val"] = 100.0

        # init base env
        super(ObjectRollEnv, self).__init__(env_params, robot_arm_params, tactile_sensor_params, visual_sensor_params)

    def setup_object(self):
        """
        Set vars for loading an spherical object and goal.
        """
        # define an initial position for the objects (world coords)
        self.init_obj_pos = [0.65, 0.0, self.default_obj_radius]
        self.init_obj_orn = self._pb.getQuaternionFromEuler([0.0, 0.0, 0.0])

        # textured objects don't render in direct mode
        if self._env_params['show_gui']:
            self.object_path = add_assets_path("nonprehensile_manipulation/object_roll/sphere/sphere_tex.urdf")
        else:
            self.object_path = add_assets_path("nonprehensile_manipulation/object_roll/sphere/sphere.urdf")

        self.goal_path = add_assets_path("nonprehensile_manipulation/object_roll/sphere/sphere.urdf")

    def reset_task(self):
        """
        Change object size.
        Change embed distance.
        """
        self.scaling_factor = self.np_random.uniform(1.0, 2.0)
        self.scaled_obj_radius = self.default_obj_radius * self.scaling_factor
        self.embed_dist = self.np_random.uniform(0.0015, 0.003)

    def update_workframe(self):
        """
        Change workframe on reset to match scaled object.
        """
        # reset workframe origin based on new obj radius
        new_workframe = np.array([0.65, 0.0, (2*self.scaled_obj_radius) - self.embed_dist, -np.pi, 0.0, np.pi/2])
        self._env_params["workframe"] = new_workframe
        self._workframe = new_workframe

    def reset_object(self):
        """
        Reset the base pose of an object on reset,
        can also adjust physics params here.
        """
        # reset the position of the object
        self.init_obj_pos = [
            0.65 + self.np_random.uniform(-0.009, 0.009),
            0.0 + self.np_random.uniform(-0.009, 0.009),
            self.scaled_obj_radius,
        ]
        self.init_obj_orn = self._pb.getQuaternionFromEuler([0.0, 0.0, 0.0])

        # remove and reload the object
        self._pb.removeBody(self.obj_id)
        self.obj_id = self._pb.loadURDF(
            self.object_path, self.init_obj_pos, self.init_obj_orn, globalScaling=self.scaling_factor
        )

        # could perform object dynamics randomisations here
        self._pb.changeDynamics(
            self.obj_id,
            -1,
            lateralFriction=10.0,
            spinningFriction=0.0,
            rollingFriction=0.0,
            restitution=0.0,
            frictionAnchor=0,
            collisionMargin=0.000001,
        )

    def make_goal(self):
        """
        Generate a goal place a set distance from the inititial object pose.
        """

        # place goal randomly
        goal_ang = self.np_random.uniform(-np.pi, np.pi)
        goal_dist = self.np_random.uniform(low=0.0, high=0.015)

        # self.goal_pos_tcpframe = np.array([goal_dist * np.cos(goal_ang), goal_dist * np.sin(goal_ang), 0.0])
        # self.goal_rpy_tcpframe = [0.0, 0.0, 0.0]
        # self.goal_orn_tcpframe = self._pb.getQuaternionFromEuler(self.goal_rpy_tcp)
        self.goal_pose_tcpframe = np.array([
            goal_dist * np.cos(goal_ang), goal_dist * np.sin(goal_ang), 0.0,
            0.0, 0.0, 0.0
        ])

        self.update_goal()

    def update_goal(self):
        """
        Transforms goal in TCP frame to a pose in world frame.
        """

        # get the current tcp pose
        cur_tcp_pose = self.embodiment.arm.get_tcp_pose()

        # transform the goal pose from the tcp frame to the world frame
        self.cur_goal_pose_worldframe = inv_transform_eul(self.goal_pose_tcpframe, cur_tcp_pose)
        goal_pos_worldframe = self.cur_goal_pose_worldframe[3:]
        goal_orn_worldframe = self._pb.getQuaternionFromEuler(self.cur_goal_pose_worldframe[3:])

        # transform the goal pose from the world frame to the work frame
        self.cur_goal_pose_workframe = self.worldframe_to_workframe(self.cur_goal_pose_worldframe)

        # useful for visualisation
        if self.visualise_goal:
            self._pb.resetBasePositionAndOrientation(self.goal_indicator, goal_pos_worldframe, goal_orn_worldframe)

    def get_step_data(self):

        # update the world position of the goal based on current position of TCP
        self.update_goal()

        return super(ObjectRollEnv, self).get_step_data()

    def get_termination(self):
        """
        Criteria for terminating an episode.
        """
        # terminate when distance to goal is < eps
        pos_dist = self.xy_obj_dist_to_goal()

        if pos_dist < self.termination_pos_dist:
            return True

        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True

        return False

    def get_reward(self):
        """
        Calculate the reward when in dense mode.
        """
        W_obj_goal_pos = 1.0
        goal_pos_dist = self.xy_obj_dist_to_goal()
        reward = -(W_obj_goal_pos * goal_pos_dist)
        return reward

    def get_oracle_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """

        # stack into array
        observation = np.hstack(
            [
                *self.cur_tcp_pose_workframe,
                *self.cur_tcp_vel_workframe,
                *self.cur_obj_pose_workframe,
                *self.cur_obj_vel_workframe,
                *self.goal_pose_tcpframe,
                self.scaled_obj_radius,
            ]
        )

        return observation

    def get_extended_feature_array(self):
        """
        features needed to help complete task.
        Goal pose in TCP frame.
        """
        feature_array = np.array([*self.goal_pose_tcpframe[:2]])
        return feature_array

    def overlay_goal_on_image(self, tactile_image):
        """
        Overlay a crosshairs onto the observation in roughly the position
        of the goal
        """
        # get the coords of the goal in image space
        # min/max from 20mm radius tip + extra for border
        min, max = -0.021, 0.021
        norm_tcp_pos_x = (self.goal_pose_tcpframe[0] - min) / (max - min)
        norm_tcp_pos_y = (self.goal_pose_tcpframe[1] - min) / (max - min)

        image_size = self._tactile_sensor_params["image_size"]

        goal_coordinates = (
            int(norm_tcp_pos_x * image_size[0]),
            int(norm_tcp_pos_y * image_size[1]),
        )

        # Draw a circle at the goal
        marker_size = int(image_size[0] / 32)
        overlay_img = cv2.drawMarker(
            tactile_image,
            goal_coordinates,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=marker_size,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

        return overlay_img

    def render(self, mode="rgb_array"):
        """
        Most rendering handeled with show_gui, show_tactile flags.
        This is useful for saving videos.
        """

        if mode != "rgb_array":
            return np.array([])

        # get the rgb camera image
        rgb_array = self.get_visual_obs()

        # get the current tactile images and reformat to match rgb array
        tactile_array = self.get_tactile_obs()
        tactile_array = cv2.cvtColor(tactile_array, cv2.COLOR_GRAY2RGB)

        # rezise tactile to match rgb if rendering in higher res
        if self._tactile_sensor_params["image_size"] != self._visual_sensor_params["image_size"]:
            tactile_array = cv2.resize(tactile_array, tuple(self.rgb_image_size))

        # add goal indicator in approximate position
        tactile_array = self.overlay_goal_on_image(tactile_array)

        # concat the images into a single image
        render_array = np.concatenate([rgb_array, tactile_array], axis=1)

        # setup plot for rendering
        if self._first_render:
            self._first_render = False
            if self._seed is not None:
                self.window_name = "render_window_{}".format(self._seed)
            else:
                self.window_name = "render_window"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # plot rendered image
        if not self._render_closed:
            render_array_rgb = cv2.cvtColor(render_array, cv2.COLOR_BGR2RGB)
            cv2.imshow(self.window_name, render_array_rgb)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(self.window_name)
                self._render_closed = True

        return render_array
