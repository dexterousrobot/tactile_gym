import numpy as np

from tactile_sim.utils.pybullet_draw_utils import draw_vector

from tactile_gym.assets import add_assets_path
from tactile_gym.envs.nonprehensile_manipulation.base_object_env import BaseObjectEnv
from tactile_gym.envs.nonprehensile_manipulation.object_balance.rest_poses import rest_poses_dict


class ObjectBalanceEnv(BaseObjectEnv):
    def __init__(
        self,
        env_params={},
        robot_arm_params={},
        tactile_sensor_params={},
        visual_sensor_params={},
    ):

        # distance from goal to cause termination
        self.termination_dist_deg = 35
        self.termination_dist_pos = 0.1
        # self.termination_dist_deg = np.inf
        # self.termination_dist_pos = np.inf
        self.visualise_goal = False
        self.embed_dist = 0.0035

        # add environment specific env parameters
        env_params["workframe"] = np.array([0.55, 0.0, 0.35, 0.0, 0.0, 0.0])

        tcp_lims = np.zeros(shape=(6, 2))
        tcp_lims[0, 0], tcp_lims[0, 1] = -0.1, 0.1  # x lims
        tcp_lims[1, 0], tcp_lims[1, 1] = -0.1, 0.1  # y lims
        tcp_lims[2, 0], tcp_lims[2, 1] = -0.1, 0.1  # z lims
        tcp_lims[3, 0], tcp_lims[3, 1] = -45 * np.pi / 180, 45 * np.pi / 180  # roll lims
        tcp_lims[4, 0], tcp_lims[4, 1] = -45 * np.pi / 180, 45 * np.pi / 180  # pitch lims
        tcp_lims[5, 0], tcp_lims[5, 1] = -45 * np.pi / 180, 45 * np.pi / 180  # yaw lims
        env_params["tcp_lims"] = tcp_lims

        # add environment specific robot arm parameters
        robot_arm_params["use_tcp_frame_control"] = False
        robot_arm_params["rest_poses"] = rest_poses_dict[robot_arm_params["type"]]
        robot_arm_params["tcp_link_name"] = "tcp_link"

        # add environment specific tactile sensor parameters
        tactile_sensor_params["core"] = "no_core"
        tactile_sensor_params["dynamics"] = {"stiffness": 50, "damping": 100, "friction": 10.0}

        # add environment specific visual sensor parameters
        visual_sensor_params["dist"] = 1.0
        visual_sensor_params["yaw"] = 90.0
        visual_sensor_params["pitch"] = -10
        visual_sensor_params["pos"] = [-0.1, 0.0, 0.25]
        visual_sensor_params["fov"] = 75.0
        visual_sensor_params["near_val"] = 0.1
        visual_sensor_params["far_val"] = 100.0

        # init base env
        super(ObjectBalanceEnv, self).__init__(env_params, robot_arm_params, tactile_sensor_params, visual_sensor_params)

    def setup_object(self):
        """
        Set vars for loading an object
        """

        # if self.object_mode == "pole":
        self.obj_base_width = 0.1
        self.obj_base_height = 0.0025
        self.object_path = add_assets_path("nonprehensile_manipulation/object_balance/pole/pole.urdf")

        # define an initial position for the objects (world coords)
        self.init_obj_pos = np.array([0.55, 0.0, 0.35 + self.obj_base_height / 2 - self.embed_dist])
        self.init_obj_rpy = np.array([0.0, 0.0, 0.0])
        self.init_obj_orn = np.array(self._pb.getQuaternionFromEuler(self.init_obj_rpy))

    def apply_constraints(self):
        """
        Add constraint to connect object and tip.
        """
        obj_to_const_id = self.obj_id
        child_pos = [0, 0, -self.obj_base_height / 2 + self.embed_dist]

        self.obj_tip_constraint_id = self._pb.createConstraint(
            self.embodiment.embodiment_id,
            self.embodiment.tcp_link_id,
            obj_to_const_id,
            -1,
            self._pb.JOINT_POINT2POINT,
            jointAxis=[0, 0, 1],
            parentFramePosition=[0, 0, 0],
            childFramePosition=child_pos,
            parentFrameOrientation=self._pb.getQuaternionFromEuler([0, 0, 0]),
            childFrameOrientation=self._pb.getQuaternionFromEuler([0, 0, 0]),
        )

    def update_constraints(self):
        """
        Add any randomisation to object/tip constraint.
        """
        self._pb.changeConstraint(
            self.obj_tip_constraint_id,
            jointChildPivot=[0, 0, -self.obj_base_height / 2 + self.embed_dist],
            maxForce=1.0
        )

    def reset_task(self):
        """
        Change gravity
        Change embed distance
        """

        self._pb.setGravity(0, 0, -0.01)
        self.embed_dist = self.np_random.uniform(0.003, 0.006)
        self.init_obj_pos = [0.55, 0.0, 0.35 + self.obj_base_height / 2 - self.embed_dist]
        self.update_constraints()

    def reset_object(self):
        """
        Reset the base pose of an object on reset,
        can also adjust physics params here.
        """

        # reset the position of the object
        self._pb.resetBasePositionAndOrientation(self.obj_id, self.init_obj_pos, self.init_obj_orn)

        # make sure linear and angular damping is 0
        num_obj_joints = self._pb.getNumJoints(self.obj_id)
        for link_id in range(-1, num_obj_joints):
            self._pb.changeDynamics(
                self.obj_id,
                link_id,
                linearDamping=0.0,
                angularDamping=0.0,
            )

        # apply random force to objects
        self.apply_random_force_obj(force_mag=0.1)

    def apply_random_force_obj(self, force_mag):
        """
        Apply a random force to the pole object
        """

        # calculate force
        force_pos = self.init_obj_pos + np.array(
            [
                self.np_random.choice([-1, 1]) * self.np_random.rand() * self.obj_base_width / 2,
                self.np_random.choice([-1, 1]) * self.np_random.rand() * self.obj_base_width / 2,
                0,
            ]
        )

        force_dir = np.array([0, 0, -1])
        force = force_dir * force_mag

        # apply force
        self._pb.applyExternalForce(self.obj_id, -1, force, force_pos, flags=self._pb.WORLD_FRAME)

        # plot force
        draw_vector(force_pos, force_dir)

    def apply_random_torque_obj(self, force_mag):
        """
        Apply a random torque to the plate object
        """
        force_dir = np.array([0, 0, -1])
        force = force_dir * force_mag

        # apply force
        self._pb.applyExternalTorque(self.obj_id, -1, force, flags=self._pb.LINK_FRAME)

    def check_obj_fall(self):
        """
        Check if the roll and pitch are greater than allowed threshold.
        Check if distance travelled is greater than threshold.
        """

        cur_obj_pos = self.cur_obj_pose_worldframe[:3]
        cur_obj_rpy_deg = self.cur_obj_pose_worldframe[3:] * 180 / np.pi
        init_obj_rpy_deg = self.init_obj_rpy * 180 / np.pi

        # calc distance in deg accounting for angle representation
        rpy_dist = np.abs(((cur_obj_rpy_deg - init_obj_rpy_deg) + 180) % 360 - 180)

        # terminate if either roll or pitch off by set distance
        if (rpy_dist[0] > self.termination_dist_deg) or (rpy_dist[1] > self.termination_dist_deg):
            return True

        # moved too far
        pos_dist = np.linalg.norm(cur_obj_pos - self.init_obj_pos)
        if pos_dist > self.termination_dist_pos:
            return True

        return False

    def get_termination(self):
        """
        Criteria for terminating an episode.
        """

        if self.check_obj_fall():
            return True

        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True

        return False

    def get_reward(self):
        """
        Positive reward for every step where object is balanced.
        """
        return 1.0

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
            ]
        )

        return observation
