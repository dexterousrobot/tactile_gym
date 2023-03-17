import sys
import gym
import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as bc
import pkgutil
import cv2

from tactile_sim.utils.setup_pb_utils import load_standard_environment
from tactile_sim.utils.pybullet_draw_utils import draw_frame, draw_box
from tactile_sim.utils.transforms import inv_transform_eul, transform_eul, inv_transform_vec_eul, transform_vec_eul

tcp_action_mapping = {
    'x': 0, 'y': 1, 'z': 2,
    'Rx': 3, 'Ry': 4, 'Rz': 5,
}
joint_action_mapping = {
    'J1': 0, 'J2': 1, 'J3': 2,
    'J4': 3, 'J5': 4, 'J6': 5,
    'J7': 6,
}


class BaseTactileEnv(gym.Env):
    def __init__(self, env_params, robot_arm_params, tactile_sensor_params, visual_sensor_params):

        self._env_params = env_params
        self._robot_arm_params = robot_arm_params
        self._tactile_sensor_params = tactile_sensor_params
        self._visual_sensor_params = visual_sensor_params

        # env params
        self._max_steps = env_params["max_steps"]
        self._show_gui = env_params["show_gui"]
        self._observation = []
        self._env_step_counter = 0
        self._first_render = True
        self._render_closed = False

        self._workframe = self._env_params["workframe"]
        self._tcp_lims = self._env_params["tcp_lims"]

        self.connect_pybullet()
        self.set_pybullet_params()

        # set vars for full pybullet reset to clear cache
        self.reset_counter = 0
        self.reset_limit = 1000

    def connect_pybullet(self):
        """Connect to pybullet with/without gui enabled."""
        if self._show_gui:
            self._pb = bc.BulletClient(connection_mode=pb.GUI)
            self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        else:
            self._pb = bc.BulletClient(connection_mode=pb.DIRECT)
            egl = pkgutil.get_loader("eglRenderer")
            if egl:
                self._pb.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            else:
                self._pb.loadPlugin("eglRendererPlugin")

        # bc automatically sets client but keep here incase needed
        self._physics_client_id = self._pb._client

    def set_pybullet_params(self):
        self._pb.setGravity(0, 0, -9.81)
        self._pb.setPhysicsEngineParameter(
            fixedTimeStep=self._sim_time_step,
            numSolverIterations=150,
            enableConeFriction=1,
            contactBreakingThreshold=0.0001
        )

    def full_reset(self):
        self._pb.resetSimulation()
        load_standard_environment(self._pb)
        self.embodiment.full_reset()
        self.reset_counter = 0

    def seed(self, seed=None):
        self._seed = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def __del__(self):
        self.close()

    def close(self):
        if self._pb.isConnected():
            self._pb.disconnect()
        if not self._render_closed:
            cv2.destroyAllWindows()

    def setup_observation_space(self):

        obs_dim_dict = self.get_obs_dim()

        spaces = {
            "oracle": gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_dim_dict["oracle"], dtype=np.float32),
            "tactile": gym.spaces.Box(low=0, high=255, shape=obs_dim_dict["tactile"], dtype=np.uint8),
            "visual": gym.spaces.Box(low=0, high=255, shape=obs_dim_dict["visual"], dtype=np.uint8),
            "extended_feature": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_dim_dict["extended_feature"], dtype=np.float32
            ),
        }

        observation_mode = self._env_params["observation_mode"]

        if observation_mode == "oracle":
            self.observation_space = gym.spaces.Dict({"oracle": spaces["oracle"]})

        elif observation_mode == "tactile":
            self.observation_space = gym.spaces.Dict({"tactile": spaces["tactile"]})

        elif observation_mode == "visual":
            self.observation_space = gym.spaces.Dict({"visual": spaces["visual"]})

        elif observation_mode == "visuotactile":
            self.observation_space = gym.spaces.Dict({"tactile": spaces["tactile"], "visual": spaces["visual"]})

        elif observation_mode == "tactile_and_feature":
            self.observation_space = gym.spaces.Dict(
                {"tactile": spaces["tactile"], "extended_feature": spaces["extended_feature"]}
            )

        elif observation_mode == "visual_and_feature":
            self.observation_space = gym.spaces.Dict(
                {"visual": spaces["visual"], "extended_feature": spaces["extended_feature"]}
            )

        elif observation_mode == "visuotactile_and_feature":
            self.observation_space = gym.spaces.Dict(
                {"tactile": spaces["tactile"], "visual": spaces["visual"], "extended_feature": spaces["extended_feature"]}
            )

    def get_obs_dim(self):
        obs_dim_dict = {
            "oracle": self.get_oracle_obs().shape,
            "tactile": self.get_tactile_obs().shape,
            "visual": self.get_visual_obs().shape,
            "extended_feature": self.get_extended_feature_array().shape,
        }
        return obs_dim_dict

    def get_act_dim(self):
        return len(self._robot_arm_params["control_dofs"])

    def encode_actions(self, actions):
        """
        Return actions as np.array in correct places for sending to robot arm
        i.e. NN could be predicting [y, Rz] actions. Make sure they are in
        correct place [1, 5].
        """

        if 'tcp' in self._robot_arm_params['control_mode']:
            encoded_actions = np.zeros(6)
            for i, control_dof in enumerate(self._robot_arm_params["control_dofs"]):
                encoded_actions[tcp_action_mapping[control_dof]] = actions[i]

        elif 'joint' in self._robot_arm_params['control_mode']:
            encoded_actions = np.zeros(self.embodiment.arm.num_control_dofs)
            for i, control_dof in enumerate(self._robot_arm_params["control_dofs"]):
                encoded_actions[joint_action_mapping[control_dof]] = actions[i]

        else:
            sys.exit("Incorrect control_mode specified")

        return encoded_actions

    def scale_actions(self, actions):
        """Scale actions from input range to range specific to actions space."""

        actions = np.clip(actions, self.min_action, self.max_action)
        input_range = self.max_action - self.min_action

        # define action ranges per act dim to rescale output of policy
        if self._robot_arm_params["control_mode"] == "tcp_position_control":
            scaled_actions = np.zeros(6)
            scaled_actions[:3] = (((actions[:3] - self.min_action) * (2*self.lin_pos_lim)) / input_range) - self.lin_pos_lim
            scaled_actions[3:] = (((actions[3:] - self.min_action) * (2*self.ang_pos_lim)) / input_range) - self.ang_pos_lim

        elif self._robot_arm_params["control_mode"] == "tcp_velocity_control":
            scaled_actions = np.zeros(6)
            scaled_actions[:3] = (((actions[:3] - self.min_action) * (2*self.lin_vel_lim)) / input_range) - self.lin_vel_lim
            scaled_actions[3:] = (((actions[3:] - self.min_action) * (2*self.ang_vel_lim)) / input_range) - self.ang_vel_lim

        elif self._robot_arm_params["control_mode"] == "joint_position_control":
            scaled_actions = (((actions - self.min_action) * (2*self.joint_pos_lim)) / input_range) - self.joint_pos_lim

        elif self._robot_arm_params["control_mode"] == "joint_velocity_control":
            scaled_actions = (((actions - self.min_action) * (2*self.joint_vel_lim)) / input_range) - self.joint_vel_lim

        return np.array(scaled_actions)

    def transform_actions(self, actions):
        """
        Converts an action defined in the workframe to an action defined in the worldframe
        """

        if self._robot_arm_params["control_mode"] == "tcp_position_control":

            # get the current tcp pose and increment it using the action
            cur_tcp_pose = self.embodiment.arm.get_tcp_pose()
            cur_tcp_pose_workframe = self.worldframe_to_workframe(cur_tcp_pose)
            target_pose = cur_tcp_pose_workframe + actions

            # limit actions to safe ranges
            clipped_target_pose = self.check_TCP_pos_lims(target_pose)

            # convert to worldframe coords for IK
            target_pose_worldframe = self.workframe_to_worldframe(clipped_target_pose)
            transformed_actions = target_pose_worldframe

        elif self._robot_arm_params["control_mode"] == "tcp_velocity_control":

            # check that this won't push the TCP out of limits
            # zero any velocities that will
            clipped_target_vels = self.check_TCP_vel_lims(actions)

            # convert desired vels from workframe to worldframe
            target_linvel, target_angvel = self.workvel_to_worldvel(
                clipped_target_vels[:3], clipped_target_vels[3:]
            )
            transformed_actions = np.concatenate([target_linvel, target_angvel])

        elif self._robot_arm_params["control_mode"] == "joint_position_control":
            transformed_actions = actions
            cur_joint_angles = self.embodiment.arm.get_joint_angles()
            transformed_actions = cur_joint_angles + actions

        elif self._robot_arm_params["control_mode"] == "joint_velocity_control":
            transformed_actions = actions

        return transformed_actions

    def step(self, action):
        """
        Encode actions, send to embodiment to be applied to the environment.
        Return observation, reward, terminal, info
        """

        # scale and embed actions appropriately
        encoded_actions = self.encode_actions(action)
        scaled_actions = self.scale_actions(encoded_actions)
        transformed_actions = self.transform_actions(scaled_actions)

        # send action
        self._env_step_counter += 1
        self.apply_action(
            transformed_actions,
            control_mode=self._robot_arm_params["control_mode"],
            velocity_action_repeat=self._velocity_action_repeat,
            max_steps=self._max_blocking_pos_move_steps,
        )
        # get data
        reward, done = self.get_step_data()
        self._observation = self.get_observation()
        return self._observation, reward, done, {}

    def apply_action(
        self,
        motor_commands,
        control_mode="tcp_velocity_control",
        velocity_action_repeat=1,
        max_steps=100,
    ):

        # set the simulation with desired control points
        if control_mode == "tcp_position_control":
            self.embodiment.arm.set_target_tcp_pose(motor_commands)
        elif control_mode == "tcp_velocity_control":
            self.embodiment.arm.set_target_tcp_velocities(motor_commands)
        elif control_mode == "joint_position_control":
            self.embodiment.arm.set_target_joint_positions(motor_commands)
        elif control_mode == "joint_velocity_control":
            self.embodiment.arm.set_target_joint_velocities(motor_commands)
        else:
            sys.exit("Incorrect control mode specified: {}".format(control_mode))

        # run the simulation for a number of steps
        if "position" in control_mode:
            self.embodiment.arm.blocking_position_move(
                max_steps=max_steps,
                constant_vel=None,
                j_pos_tol=1e-6,
                j_vel_tol=1e-3,
            )
        elif "velocity" in control_mode:
            self.embodiment.arm.blocking_velocity_move(blocking_steps=velocity_action_repeat)
        else:
            self.step_sim()

    def worldframe_to_workframe(self, pose):
        return transform_eul(pose, self._workframe)

    def workframe_to_worldframe(self, pose):
        return inv_transform_eul(pose, self._workframe)

    def worldvel_to_workvel(self, linvel, angvel):
        work_linvel = transform_vec_eul(linvel, self._workframe)
        work_angvel = transform_vec_eul(angvel, self._workframe)
        return work_linvel, work_angvel

    def workvel_to_worldvel(self, linvel, angvel):
        world_linvel = inv_transform_vec_eul(linvel, self._workframe)
        world_angvel = inv_transform_vec_eul(angvel, self._workframe)
        return world_linvel, world_angvel

    def check_TCP_pos_lims(self, pose):
        """
        Clip the pose at the TCP limits specified.
        """
        return np.clip(pose, self._tcp_lims[:, 0], self._tcp_lims[:, 1])

    def check_TCP_vel_lims(self, vels):
        """
        check whether action will take TCP outside of limits,
        zero any velocities that will.
        """
        cur_tcp_pose = self.embodiment.arm.get_tcp_pose()
        cur_tcp_pose_workframe = self.worldframe_to_workframe(cur_tcp_pose)

        # get bool arrays for if limits are exceeded and if velocity is in
        # the direction that's exceeded
        exceed_llims = np.logical_and(cur_tcp_pose_workframe < self._tcp_lims[:, 0], vels < 0)
        exceed_ulims = np.logical_and(cur_tcp_pose_workframe > self._tcp_lims[:, 1], vels > 0)
        exceeded = np.logical_or(exceed_llims, exceed_ulims)

        # cap the velocities at 0 if limits are exceeded
        capped_vels = np.array(vels)
        capped_vels[np.array(exceeded)] = 0

        return capped_vels

    def get_extended_feature_array(self):
        """
        Get feature to extend current observations.
        """
        return np.array([])

    def get_oracle_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """
        return np.array([])

    def get_tactile_obs(self):
        """
        Returns the tactile observation with an image channel.
        """
        tactile_obs = self.embodiment.get_tactile_observation()
        observation = tactile_obs[..., np.newaxis]
        return observation

    def get_visual_obs(self):
        """
        Returns the rgb image from a static environment camera.
        """
        visual_rgb, _, _ = self.embodiment.get_visual_observation()
        return visual_rgb

    def get_observation(self):
        """
        Returns the observation dependent on which mode is set.
        """

        observation_mode = self._env_params["observation_mode"]

        # check correct obs type set
        if observation_mode not in [
            "oracle",
            "tactile",
            "visual",
            "visuotactile",
            "tactile_and_feature",
            "visual_and_feature",
            "visuotactile_and_feature",
        ]:
            sys.exit("Incorrect observation mode specified: {}".format(observation_mode))

        observation = {}
        # use direct pose info to check if things are working
        if "oracle" in observation_mode:
            observation["oracle"] = self.get_oracle_obs()

        # observation is just the tactile sensor image
        if "tactile" in observation_mode:
            observation["tactile"] = self.get_tactile_obs()

        # observation is rgb environment camera image
        if any(obs in observation_mode for obs in ["visual", "visuo"]):
            observation["visual"] = self.get_visual_obs()

        # observation is mix image + features (pretending to be image shape)
        if "feature" in observation_mode:
            observation["extended_feature"] = self.get_extended_feature_array()

        return observation

    def render(self, mode="rgb_array"):
        """
        Most rendering handled with show_gui, show_tactile flags.
        This is useful for saving videos.
        """

        if mode != "rgb_array":
            return np.array([])

        # get the rgb camera image
        rgb_array = self.get_visual_obs()

        # get the current tactile images and reformat to match rgb array
        tactile_array = self.get_tactile_obs()
        tactile_array = cv2.cvtColor(tactile_array, cv2.COLOR_GRAY2RGB)

        # resize tactile to match rgb if rendering in higher res
        if self._tactile_sensor_params["image_size"] != self._visual_sensor_params["image_size"]:
            tactile_array = cv2.resize(tactile_array, tuple(self._visual_sensor_params["image_size"]))

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

    def draw_workframe(self, lifetime=0.1):
        draw_frame(self._workframe, lifetime=lifetime)

    def draw_tcp_lims(self, lifetime=0.1):
        draw_box(self._workframe, self._tcp_lims)
