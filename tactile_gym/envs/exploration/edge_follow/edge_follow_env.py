import numpy as np

from tactile_sim.assets.default_rest_poses import rest_poses_dict
from tactile_sim.utils.setup_pb_utils import load_standard_environment
from tactile_sim.utils.setup_pb_utils import set_debug_camera
from tactile_sim.embodiments.embodiments import VisuoTactileArmEmbodiment

from tactile_gym.assets import add_assets_path
from tactile_gym.envs.base_tactile_env import BaseTactileEnv


class EdgeFollowEnv(BaseTactileEnv):
    def __init__(
        self,
        env_params={},
        robot_arm_params={},
        tactile_sensor_params={},
        visual_sensor_params={},
    ):

        # add environment specific env parameters
        env_params["workframe"] = np.array([0.65, 0.0, 0.035, -np.pi, 0.0, np.pi/2])

        tcp_lims = np.zeros(shape=(6, 2))
        tcp_lims[0, 0], tcp_lims[0, 1] = -0.175, +0.175  # x lims
        tcp_lims[1, 0], tcp_lims[1, 1] = -0.175, +0.175  # y lims
        tcp_lims[2, 0], tcp_lims[2, 1] = -0.1, +0.1  # z lims
        tcp_lims[3, 0], tcp_lims[3, 1] = 0.0, 0.0  # roll lims
        tcp_lims[4, 0], tcp_lims[4, 1] = 0.0, 0.0  # pitch lims
        tcp_lims[5, 0], tcp_lims[5, 1] = -np.pi, np.pi  # yaw lims
        env_params["tcp_lims"] = tcp_lims

        # add environment specific robot arm parameters
        robot_arm_params["use_tcp_frame_control"] = False
        robot_arm_params["rest_poses"] = rest_poses_dict[robot_arm_params["type"]]
        robot_arm_params["tcp_link_name"] = "tcp_link"

        # add environment specific tactile sensor parameters
        tactile_sensor_params["core"] = "no_core"
        tactile_sensor_params["dynamics"] = {'stiffness': 50, 'damping': 100, 'friction': 10.0}

        # add environment specific visual sensor parameters
        visual_sensor_params["dist"] = 0.4
        visual_sensor_params["yaw"] = 90.0
        visual_sensor_params["pitch"] = -25.0
        visual_sensor_params["pos"] = [0.65, 0.0, 0.035]
        visual_sensor_params["fov"] = 75.0
        visual_sensor_params["near_val"] = 0.1
        visual_sensor_params["far_val"] = 100.0

        super(EdgeFollowEnv, self).__init__(env_params, robot_arm_params, tactile_sensor_params, visual_sensor_params)

        self.embodiment = VisuoTactileArmEmbodiment(
            self._pb,
            robot_arm_params=robot_arm_params,
            tactile_sensor_params=tactile_sensor_params,
            visual_sensor_params=visual_sensor_params
        )

        # distance from goal to cause termination
        self.termination_dist = 0.01

        # how much penetration of the tip to optimize for
        # randomly vary this on each episode
        self.embed_dist = 0.0035

        # load environment objects
        load_standard_environment(self._pb)
        set_debug_camera(self._pb, visual_sensor_params)
        self.setup_edge()
        self.load_edge()
        self.reset()

        # setup variables
        self.setup_action_space()
        self.setup_observation_space()

    def setup_edge(self):
        """
        Defines params for loading/resetting the edge object.
        """
        # define an initial position for the objects (world coords)
        self.edge_pos = [0.65, 0.0, 0.0]
        self.edge_height = 0.035
        self.edge_len = 0.175

    def load_edge(self):
        """
        Loads the edge object
        """
        # load temp edge and goal indicators so they can be more conveniently updated
        edge_path = "exploration/edge_follow/edge_stimuli/long_edge_flat/long_edge.urdf"
        self.edge_stim_id = self._pb.loadURDF(
            add_assets_path(edge_path),
            self.edge_pos,
            [0, 0, 0, 1],
            useFixedBase=True,
        )
        self.goal_indicator = self._pb.loadURDF(
            add_assets_path("shared_assets/environment_objects/goal_indicators/sphere_indicator.urdf"),
            self.edge_pos,
            [0, 0, 0, 1],
            useFixedBase=True,
        )

    def update_edge(self):
        """
        Randomises the orientation of the edge.
        """

        # load in the edge stimulus
        self.edge_ang = self.np_random.uniform(-np.pi, np.pi)
        self.edge_orn = self._pb.getQuaternionFromEuler([0.0, 0.0, self.edge_ang])
        self._pb.resetBasePositionAndOrientation(
            self.edge_stim_id, self.edge_pos, self.edge_orn
        )

        # place a goal at the end of an edge (world coords)
        self.goal_pos_worldframe = [
            self.edge_pos[0] + (self.edge_len * np.cos(self.edge_ang)),
            self.edge_pos[1] + (self.edge_len * np.sin(self.edge_ang)),
            self.edge_pos[2] + self.edge_height,
        ]
        self.goal_rpy_worldframe = [0, 0, 0]
        self.goal_orn_worldframe = self._pb.getQuaternionFromEuler(
            self.goal_rpy_worldframe
        )
        self.goal_pose_worldframe = np.array([*self.goal_pos_worldframe, *self.goal_rpy_worldframe])

        # create variables for goal pose in workframe to use later in easy feature observation
        self.goal_pose_workframe = self.worldframe_to_workframe(self.goal_pose_worldframe)

        self.edge_end_points = np.array(
            [
                [
                    self.edge_pos[0] - (self.edge_len * np.cos(self.edge_ang)),
                    self.edge_pos[1] - (self.edge_len * np.sin(self.edge_ang)),
                    self.edge_pos[2] + self.edge_height,
                ],
                [
                    self.edge_pos[0] + (self.edge_len * np.cos(self.edge_ang)),
                    self.edge_pos[1] + (self.edge_len * np.sin(self.edge_ang)),
                    self.edge_pos[2] + self.edge_height,
                ],
            ]
        )

        # useful for visualisation
        self._pb.resetBasePositionAndOrientation(
            self.goal_indicator, self.goal_pos_worldframe, self.goal_orn_worldframe
        )

    def reset_task(self):
        """
        Randomise amount tip is embedded into edge
        Reorientate edge
        """
        # reset the ur5 arm at the origin of the workframe with variation to the embed distance
        self.embed_dist = self.np_random.uniform(0.0015, 0.0065)

        # load an edge with random orientation and goal
        self.update_edge()

    def update_init_pose(self):
        """
        Update the initial pose to be taken on reset.
        """
        init_tcp_pose = self.workframe_to_worldframe(np.array([0.0, 0.0, self.embed_dist, 0.0, 0.0, 0.0]))
        return init_tcp_pose

    def reset(self):
        """
        Reset the environment after an episode terminates.
        """

        # full reset pybullet sim to clear cache, this avoids silent bug where memory fills and visual
        # rendering fails, this is more prevelant when loading/removing larger files
        if self.reset_counter == self.reset_limit:
            self.full_reset()

        self.reset_counter += 1
        self._env_step_counter = 0

        # update the workframe to a new position if embed dist randomisations are on
        self.reset_task()
        init_TCP_pose = self.update_init_pose()
        self.embodiment.reset(reset_tcp_pose=init_TCP_pose)

        # just to change variables to the reset pose incase needed before taking
        # a step
        self.get_step_data()

        # get the starting observation
        self._observation = self.get_observation()

        return self._observation

    def full_reset(self):
        """
        Sidesteps pybullet issue that causes visual rendering problems when objects are repeatedly removed/reloaded.
        """
        self._pb.resetSimulation()
        self.embodiment.full_reset()
        load_standard_environment(self._pb)
        set_debug_camera(self._pb, self._visual_sensor_params)
        self.load_edge()
        self.reset_counter = 0

    def get_step_data(self):
        """
        Gets the current tcp pose in worldframe for use in reward/termination calculation.
        """
        # get the cur tip pos here for once per step
        self.cur_tcp_pose_worldframe = self.embodiment.arm.get_tcp_pose()
        self.cur_tcp_pos_worldframe = self.cur_tcp_pose_worldframe[:3]

        # get rl info
        done = self.get_termination()
        reward = self.get_reward()

        return reward, done

    def dist_to_goal(self):
        """
        Euclidean distance from the tcp to the goal pose.
        """
        dist = np.linalg.norm(
            np.array(self.cur_tcp_pos_worldframe) - np.array(self.goal_pos_worldframe)
        )
        return dist

    def dist_to_center_edge(self):
        """
        Perpendicular distance from the current tcp to the center edge.
        """
        # use only x/y dont need z
        p1 = self.edge_end_points[0, :2]
        p2 = self.edge_end_points[1, :2]
        p3 = self.cur_tcp_pos_worldframe[:2]

        # calculate perpendicular distance between EE and edge
        dist = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

        return dist

    def get_termination(self):
        """
        Determins whether the current state is terminal or not.
        """

        # terminate when distance to goal is < eps
        if self.dist_to_goal() < self.termination_dist:
            return True

        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True

        return False

    def get_reward(self):
        """
        Weighted distance between current tool center point and goal pose.
        """
        W_goal = 1.0
        W_edge = 10.0

        goal_dist = self.dist_to_goal()
        edge_dist = self.dist_to_center_edge()

        # sum rewards with multiplicative factors
        reward = -(
            (W_goal * goal_dist)
            + (W_edge * edge_dist)
        )

        return reward

    def get_oracle_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """
        # get sim info on TCP
        tcp_pose, tcp_vel = self.embodiment.arm.get_current_tcp_pose_vel()
        tcp_pose_workframe = self.worldframe_to_workframe(tcp_pose)
        tcp_vel_workframe = self.worldvel_to_workvel(tcp_pose)

        observation = np.hstack(
            [
                *tcp_pose_workframe,
                *tcp_vel_workframe,
                *self.goal_pose_workframe,
                self.edge_ang,
            ]
        )
        return observation
