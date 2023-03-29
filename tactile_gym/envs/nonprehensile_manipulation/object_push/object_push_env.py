import os
import numpy as np
from opensimplex import OpenSimplex

from tactile_gym.assets import add_assets_path
from tactile_gym.envs.nonprehensile_manipulation.base_object_env import BaseObjectEnv
from tactile_gym.envs.nonprehensile_manipulation.object_push.rest_poses import rest_poses_dict


class ObjectPushEnv(BaseObjectEnv):
    def __init__(
        self,
        env_params={},
        robot_arm_params={},
        tactile_sensor_params={},
        visual_sensor_params={},
    ):

        # env specific values
        self.termination_pos_dist = 0.025
        self.visualise_goal = False
        self.obj_width = 0.08
        self.obj_height = 0.08
        self.traj_n_points = 10
        self.traj_spacing = 0.025
        self.traj_max_perturb = 0.1

        # add environment specific env parameters
        env_params["workframe"] = np.array([0.55, -0.20, self.obj_height/2, -np.pi, 0.0, np.pi/2])

        tcp_lims = np.zeros(shape=(6, 2))
        tcp_lims[0, 0], tcp_lims[0, 1] = -0.0, 0.3  # x lims
        tcp_lims[1, 0], tcp_lims[1, 1] = -0.1, 0.1  # y lims
        tcp_lims[2, 0], tcp_lims[2, 1] = -0.0, 0.0  # z lims
        tcp_lims[3, 0], tcp_lims[3, 1] = -0.0, 0.0  # roll lims
        tcp_lims[4, 0], tcp_lims[4, 1] = -0.0, 0.0  # pitch lims
        tcp_lims[5, 0], tcp_lims[5, 1] = -45 * np.pi / 180, 45 * np.pi / 180  # yaw lims
        env_params["tcp_lims"] = tcp_lims

        # add environment specific robot arm parameters
        robot_arm_params["use_tcp_frame_control"] = True
        robot_arm_params["rest_poses"] = rest_poses_dict[robot_arm_params["type"]]
        robot_arm_params["tcp_link_name"] = "tcp_link"

        # add environment specific tactile sensor parameters
        tactile_sensor_params["core"] = "fixed"
        tactile_sensor_params["dynamics"] = {"stiffness": 50, "damping": 100, "friction": 10.0}

        # add environment specific visual sensor parameters
        visual_sensor_params["dist"] = 1.0
        visual_sensor_params["yaw"] = 90.0
        visual_sensor_params["pitch"] = -45
        visual_sensor_params["pos"] = [0.1, 0.0, -0.35]
        visual_sensor_params["fov"] = 75.0
        visual_sensor_params["near_val"] = 0.1
        visual_sensor_params["far_val"] = 100.0

        # init base env
        super(ObjectPushEnv, self).__init__(env_params, robot_arm_params, tactile_sensor_params, visual_sensor_params)

    def setup_object(self):
        """
        Set vars for loading an object
        """

        # define an initial position for the objects (world coords)
        self.init_obj_pos = [0.55, -0.20 + self.obj_width / 2, self.obj_height/2]
        self.init_obj_orn = self._pb.getQuaternionFromEuler([-np.pi, 0.0, np.pi / 2])

        # get paths
        self.object_path = add_assets_path("nonprehensile_manipulation/object_push/cube/cube.urdf")
        self.goal_path = add_assets_path("shared_assets/environment_objects/goal_indicators/sphere_indicator.urdf")

    def reset_object(self):
        """
        Reset the base pose of an object on reset,
        can also adjust physics params here.
        """
        # reset the position of the object
        self.init_obj_ang = self.np_random.uniform(-np.pi / 32, np.pi / 32)
        self.init_obj_orn = self._pb.getQuaternionFromEuler([-np.pi, 0.0, np.pi / 2 + self.init_obj_ang])
        self._pb.resetBasePositionAndOrientation(self.obj_id, self.init_obj_pos, self.init_obj_orn)

        # perform object dynamics randomisations
        self._pb.changeDynamics(
            self.obj_id,
            -1,
            lateralFriction=0.065,
            spinningFriction=0.00,
            rollingFriction=0.00,
            restitution=0.0,
            frictionAnchor=1,
            collisionMargin=0.0001,
        )

        # obj_mass = self.np_random.uniform(0.4, 0.8)
        # self._pb.changeDynamics(self.obj_id, -1, mass=obj_mass)

    def load_trajectory(self):

        # place goals at each point along traj
        self.traj_ids = []
        for i in range(int(self.traj_n_points)):
            pos = [0.0, 0.0, 0.0]
            traj_point_id = self._pb.loadURDF(
                os.path.join(os.path.dirname(__file__), self.goal_path),
                pos,
                [0, 0, 0, 1],
                useFixedBase=True,
            )
            self._pb.changeVisualShape(traj_point_id, -1, rgbaColor=[0, 1, 0, 0.5])
            self._pb.setCollisionFilterGroupMask(traj_point_id, -1, 0, 0)
            self.traj_ids.append(traj_point_id)

    def update_trajectory(self):

        # setup traj arrays
        self.targ_traj_list_id = -1
        self.traj_pos_workframe = np.zeros(shape=(self.traj_n_points, 3))
        self.traj_rpy_workframe = np.zeros(shape=(self.traj_n_points, 3))
        self.traj_orn_workframe = np.zeros(shape=(self.traj_n_points, 4))

        # generate a random trajectory
        self.update_trajectory_simplex()

        # calc orientation to place object at
        self.traj_rpy_workframe[:, 2] = np.gradient(self.traj_pos_workframe[:, 1], self.traj_spacing)

        for i in range(int(self.traj_n_points)):
            # get workframe orn
            self.traj_orn_workframe[i] = self._pb.getQuaternionFromEuler(self.traj_rpy_workframe[i])

            # convert worldframe
            traj_pose_worldframe = self.workframe_to_worldframe(
                np.array([*self.traj_pos_workframe[i], *self.traj_rpy_workframe[i]])
            )
            traj_pos_worldframe = traj_pose_worldframe[:3]
            traj_rpy_worldframe = traj_pose_worldframe[3:]
            traj_orn_worldframe = self._pb.getQuaternionFromEuler(traj_rpy_worldframe)

            # place goal
            self._pb.resetBasePositionAndOrientation(self.traj_ids[i], traj_pos_worldframe, traj_orn_worldframe)
            self._pb.changeVisualShape(self.traj_ids[i], -1, rgbaColor=[0, 1, 0, 0.5])

    def update_trajectory_simplex(self):
        """
        Generates smooth trajectory of goals
        """
        # initialise noise
        simplex_noise = OpenSimplex(seed=self.np_random.randint(1e8))
        init_offset = self.obj_width / 2 + self.traj_spacing

        # generate smooth 1d traj using opensimplex
        first_run = True
        for i in range(int(self.traj_n_points)):

            noise = simplex_noise.noise2d(x=i * 0.1, y=1) * self.traj_max_perturb

            if first_run:
                init_noise_pos_offset = -noise
                first_run = False

            x = init_offset + (i * self.traj_spacing)
            y = init_noise_pos_offset + noise
            z = 0.0
            self.traj_pos_workframe[i] = [x, y, z]

    def make_goal(self):
        """
        Generate a goal place a set distance from the inititial object pose.
        """
        # update the curren trajecory
        self.update_trajectory()

        # set goal as first point along trajectory
        self.update_goal()

    def update_goal(self):
        """
        move goal along trajectory
        """
        # increment targ list
        self.targ_traj_list_id += 1

        if self.targ_traj_list_id >= self.traj_n_points:
            return False
        else:
            self.goal_id = self.traj_ids[self.targ_traj_list_id]

            # get goal pose in world frame
            goal_pos_worldframe, goal_orn_worldframe = self._pb.getBasePositionAndOrientation(self.goal_id)
            goal_rpy_worldframe = self._pb.getEulerFromQuaternion(goal_orn_worldframe)
            self.cur_goal_pose_worldframe = np.array([*goal_pos_worldframe, *goal_rpy_worldframe])

            # create variables for goal pose in workframe to use later
            goal_pos_workframe = self.traj_pos_workframe[self.targ_traj_list_id]
            goal_rpy_workframe = self.traj_rpy_workframe[self.targ_traj_list_id]
            self.cur_goal_pose_workframe = np.array([*goal_pos_workframe, *goal_rpy_workframe])

            # change colour of new target goal
            self._pb.changeVisualShape(self.goal_id, -1, rgbaColor=[0, 0, 1, 0.5])

            # change colour of goal just reached
            prev_goal_traj_list_id = self.targ_traj_list_id - 1 if self.targ_traj_list_id > 0 else None
            if prev_goal_traj_list_id is not None:
                self._pb.changeVisualShape(self.traj_ids[prev_goal_traj_list_id], -1, rgbaColor=[1, 0, 0, 0.5])

            return True

    def encode_actions(self, actions):
        """
        return actions as np.array in correct places for sending to robot arm.
        """
        encoded_actions = super(ObjectPushEnv, self).encode_actions(actions)
        encoded_actions[0] = self.max_action
        return encoded_actions

    def cos_tcp_dist_to_obj(self):
        """
        Cos distance from current orientation of the TCP to the current
        orientation of the object
        """

        cur_obj_orn_worldframe = self._pb.getQuaternionFromEuler(self.cur_obj_pose_worldframe[3:])
        cur_tcp_orn_worldframe = self._pb.getQuaternionFromEuler(self.cur_tcp_pose_worldframe[3:])

        # get normal vector of object
        obj_rot_matrix = self._pb.getMatrixFromQuaternion(cur_obj_orn_worldframe)
        obj_rot_matrix = np.array(obj_rot_matrix).reshape(3, 3)
        obj_init_vector = np.array([1, 0, 0])
        obj_vector = obj_rot_matrix.dot(obj_init_vector)

        # get vector of t_s tip, directed through tip body
        tip_rot_matrix = self._pb.getMatrixFromQuaternion(cur_tcp_orn_worldframe)
        tip_rot_matrix = np.array(tip_rot_matrix).reshape(3, 3)
        tip_init_vector = np.array([1, 0, 0])
        tip_vector = tip_rot_matrix.dot(tip_init_vector)

        # get the cosine similarity/distance between the two vectors
        cos_sim = np.dot(obj_vector, tip_vector) / (np.linalg.norm(obj_vector) * np.linalg.norm(tip_vector))
        cos_dist = 1 - cos_sim

        # # draw for debugging
        # line_scale = 0.2
        # start_point = self.cur_obj_pose_worldframe[:3]
        # normal = obj_vector * line_scale
        # self._pb.addUserDebugLine(start_point, start_point + normal,
        #                           [0, 1, 0], parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)
        #
        # start_point = self.cur_tcp_pose_worldframe[:3]
        # normal = tip_vector * line_scale
        # self._pb.addUserDebugLine(start_point, start_point + normal,
        #                           [1, 0, 0], parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)

        return cos_dist

    def get_termination(self):
        """
        Criteria for terminating an episode.
        """

        # check if near goal, change the goal if so
        obj_goal_pos_dist = self.xyz_obj_dist_to_goal()
        if obj_goal_pos_dist < self.termination_pos_dist:

            # update the goal (if not at end of traj)
            goal_updated = self.update_goal()

            # if self.targ_traj_list_id > self.traj_n_points-1:
            if not goal_updated:
                return True

        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True

        return False

    def get_reward(self):
        """
        Calculate the reward when in dense mode.
        """
        obj_goal_pos_dist = self.xyz_obj_dist_to_goal()
        obj_goal_orn_dist = self.orn_obj_dist_to_goal()
        tip_obj_orn_dist = self.cos_tcp_dist_to_obj()

        # weights for rewards
        W_obj_goal_pos = 1.0
        W_obj_goal_orn = 1.0
        W_tip_obj_orn = 1.0

        # sum rewards with multiplicative factors
        reward = -(
            (W_obj_goal_pos * obj_goal_pos_dist) + (W_obj_goal_orn * obj_goal_orn_dist) + (W_tip_obj_orn * tip_obj_orn_dist)
        )

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
                *self.cur_goal_pose_workframe,
            ]
        )

        return observation

    def get_extended_feature_array(self):
        """
        This is added to the image based observations.
        """
        feature_array = np.array(
            [
                *self.cur_tcp_pose_workframe,
                *self.cur_goal_pose_workframe,
            ]
        )
        return feature_array
