import numpy as np

from tactile_sim.utils.setup_pb_utils import load_standard_environment
from tactile_sim.utils.setup_pb_utils import set_debug_camera
from tactile_sim.embodiments.embodiments import VisuoTactileArmEmbodiment
from tactile_sim.utils.transforms import inv_transform_eul, transform_eul
from tactile_sim.utils.pybullet_draw_utils import draw_link_frame

from tactile_gym.rl_envs.base_tactile_env import BaseTactileEnv


class BaseObjectEnv(BaseTactileEnv):
    def __init__(
        self,
        env_params={},
        robot_arm_params={},
        tactile_sensor_params={},
        visual_sensor_params={},
    ):

        super(BaseObjectEnv, self).__init__(env_params, robot_arm_params, tactile_sensor_params, visual_sensor_params)

        self.embodiment = VisuoTactileArmEmbodiment(
            self._pb,
            robot_arm_params=robot_arm_params,
            tactile_sensor_params=tactile_sensor_params,
            visual_sensor_params=visual_sensor_params
        )

        # load environment objects
        load_standard_environment(self._pb)
        set_debug_camera(self._pb, visual_sensor_params)
        self.setup_object()
        self.load_object(self.visualise_goal)
        self.load_trajectory()
        self.reset()

        # setup variables
        self.setup_action_space()
        self.setup_observation_space()

    def setup_object(self):
        """
        Set vars for loading an object
        """
        pass

    def load_object(self, visualise_goal=True):
        """
        Load an object that is used
        """
        # load temp object and goal indicators so they can be more conveniently updated
        self.obj_id = self._pb.loadURDF(self.object_path, self.init_obj_pos, self.init_obj_orn)

        if visualise_goal:
            self.goal_indicator = self._pb.loadURDF(self.goal_path, self.init_obj_pos, [0, 0, 0, 1], useFixedBase=True)
            self._pb.changeVisualShape(self.goal_indicator, -1, rgbaColor=[1, 0, 0, 0.5])
            self._pb.setCollisionFilterGroupMask(self.goal_indicator, -1, 0, 0)

        # can be used to connect object and tip
        self.apply_constraints()

    def reset_object(self):
        """
        Reset the base pose of an object on reset,
        can also adjust physics params here.
        """
        pass

    def apply_constraints(self):
        """
        Add constraint to connect object and tip.
        """
        pass

    def load_trajectory(self):
        """
        Used in the pushing enviroment to set a trajectory of goals
        """
        pass

    def make_goal(self):
        """
        Generate a goal pose for the object.
        """
        pass

    def reset_task(self):
        """
        Can be used to reset task specific variables
        """
        pass

    def update_workframe(self):
        """
        Change workframe on reset if needed
        """
        pass

    def update_init_pose(self):
        """
        Update the workframe to match object size if varied
        """
        init_tcp_pose = self.workframe_to_worldframe(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        return init_tcp_pose

    def get_obj_pose(self):
        """
        Get the current pose of the object, return as arrays.
        """
        obj_pos, obj_orn = self._pb.getBasePositionAndOrientation(self.obj_id)
        obj_rpy = self._pb.getEulerFromQuaternion(obj_orn)
        return np.array([*obj_pos, *obj_rpy])

    def get_obj_pose_workframe(self):
        obj_pose = self.get_obj_pos_worldframe()
        return self.worldframe_to_workframe(obj_pose)

    def get_obj_vel(self):
        """
        Get the current velocity of the object, return as arrays.
        """
        obj_lin_vel, obj_ang_vel = self._pb.getBaseVelocity(self.obj_id)
        return np.array([*obj_lin_vel, *obj_ang_vel])

    def get_obj_vel_workframe(self):
        """
        Get the current velocity of the object, return as arrays.
        """
        obj_vel = self.get_obj_vel_worldframe()
        return self.worldvel_to_workvel(obj_vel)

    def worldframe_to_objframe(self, pose):
        return transform_eul(pose, self._obj_frame)

    def objframe_to_worldframe(self, pose):
        return inv_transform_eul(pose, self._obj_frame)

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

        # update the workframe to a new position if randomisations are on
        self.reset_task()
        self.update_workframe()

        # reset object and goal
        self.reset_object()
        self.make_goal()

        init_tcp_pose = self.update_init_pose()
        self.embodiment.reset(reset_tcp_pose=init_tcp_pose)

        # just to change variables to the reset pose incase needed before taking
        # a step
        self.get_step_data()

        # get the starting observation
        self._observation = self.get_observation()

        return self._observation

    def get_step_data(self):

        # get state of tcp and obj
        self.cur_tcp_pose_worldframe = self.embodiment.arm.get_tcp_pose()
        self.cur_tcp_pose_workframe = self.worldframe_to_workframe(self.cur_tcp_pose_worldframe)

        self.cur_tcp_vel_worldframe = self.embodiment.arm.get_tcp_vel()
        self.cur_tcp_vel_workframe = self.worldvel_to_workvel(self.cur_tcp_vel_worldframe)

        self.cur_obj_pose_worldframe = self.get_obj_pose()
        self.cur_obj_pose_workframe = self.worldframe_to_workframe(self.cur_obj_pose_worldframe)

        self.cur_obj_vel_worldframe = self.get_obj_vel()
        self.cur_obj_vel_workframe = self.worldvel_to_workvel(self.cur_obj_vel_worldframe)

        # get rl info
        done = self.get_termination()
        reward = self.get_reward()

        return reward, done

    def full_reset(self):
        """
        Pybullet can encounter some silent bugs, particularly when unloading and
        reloading objects. This will do a full reset every once in a while to
        clear caches.
        """
        self._pb.resetSimulation()
        self.embodiment.full_reset()
        load_standard_environment(self._pb)
        set_debug_camera(self._pb, self._visual_sensor_params)
        self.load_object(self.visualise_goal)
        self.reset_counter = 0

    def xyz_tcp_dist_to_goal(self):
        """
        xyz L2 distance from the current tip position to the goal.
        """
        dist = np.linalg.norm(self.cur_tcp_pose_worldframe[:3] - self.cur_goal_pose_worldframe[:3])
        return dist

    def xyz_obj_dist_to_goal(self):
        """
        xyz L2 distance from the current obj position to the goal.
        """
        dist = np.linalg.norm(self.cur_obj_pose_worldframe[:3] - self.cur_goal_pose_worldframe[:3])
        return dist

    def xy_obj_dist_to_goal(self):
        """
        xyz L2 distance from the current obj position to the goal.
        """
        dist = np.linalg.norm(self.cur_obj_pose_worldframe[:2] - self.cur_goal_pose_worldframe[:2])
        return dist

    def xyz_tcp_dist_to_obj(self):
        """
        xyz L2 distance from the current tip position to the obj center.
        """
        dist = np.linalg.norm(self.cur_tcp_pose_worldframe[:3] - self.cur_obj_pose_worldframe[:3])
        return dist

    def orn_obj_dist_to_goal(self):
        """
        Distance between the current obj orientation and goal orientation.
        """
        cur_goal_orn_worldframe = self._pb.getQuaternionFromEuler(self.cur_goal_pose_worldframe[3:])
        cur_obj_orn_worldframe = self._pb.getQuaternionFromEuler(self.cur_obj_pose_worldframe[3:])
        dist = np.arccos(np.clip((2 * (np.inner(cur_goal_orn_worldframe, cur_obj_orn_worldframe) ** 2)) - 1, -1, 1))
        return dist

    """
    ==================== Debug Tools ====================
    """

    def draw_objframe(self, lifetime=0.1):
        draw_link_frame(self.obj_id, -1, lifetime=lifetime)

    def draw_goalframe(self, lifetime=0.1):
        draw_link_frame(self.goal_id, -1, lifetime=lifetime)
