import numpy as np
from opensimplex import OpenSimplex

from tactile_sim.assets.default_rest_poses import rest_poses_dict
from tactile_sim.utils.setup_pb_utils import load_standard_environment
from tactile_sim.utils.setup_pb_utils import set_debug_camera
from tactile_sim.embodiments.embodiments import VisuoTactileArmEmbodiment

from tactile_gym.assets import add_assets_path
from tactile_gym.rl_envs.base_tactile_env import BaseTactileEnv


class BaseSurfaceEnv(BaseTactileEnv):
    def __init__(
        self,
        env_params={},
        robot_arm_params={},
        tactile_sensor_params={},
        visual_sensor_params={},
    ):

        # distance from goal to cause termination
        self.termination_dist = 0.01
        self.embed_dist = 0.0015
        self.height_perturbation_range = 0.025  # max/min height of surface
        self.x_y_extent = 0.15  # limits for x,y TCP coords and goal pos

        # add environment specific env parameters
        env_params["workframe"] = np.array([0.65, 0.0, self.height_perturbation_range, -np.pi, 0.0, np.pi/2])

        # limits for tool center point relative to workframe
        tcp_lims = np.zeros(shape=(6, 2))
        tcp_lims[0, 0], tcp_lims[0, 1] = -self.x_y_extent, +self.x_y_extent  # x lims
        tcp_lims[1, 0], tcp_lims[1, 1] = -self.x_y_extent, +self.x_y_extent  # y lims
        tcp_lims[2, 0], tcp_lims[2, 1] = (-self.height_perturbation_range, +self.height_perturbation_range)  # z lims
        tcp_lims[3, 0], tcp_lims[3, 1] = -np.pi / 4, +np.pi / 4  # roll lims
        tcp_lims[4, 0], tcp_lims[4, 1] = -np.pi / 4, +np.pi / 4  # pitch lims
        tcp_lims[5, 0], tcp_lims[5, 1] = 0.0, 0.0  # yaw lims
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
        visual_sensor_params["pos"] = [0.65, 0.0, 0.05]
        visual_sensor_params["fov"] = 75.0
        visual_sensor_params["near_val"] = 0.1
        visual_sensor_params["far_val"] = 100.0

        super(BaseSurfaceEnv, self).__init__(env_params, robot_arm_params, tactile_sensor_params, visual_sensor_params)

        self.embodiment = VisuoTactileArmEmbodiment(
            self._pb,
            robot_arm_params=robot_arm_params,
            tactile_sensor_params=tactile_sensor_params,
            visual_sensor_params=visual_sensor_params
        )

        # load environment objects
        load_standard_environment(self._pb)
        set_debug_camera(self._pb, visual_sensor_params)
        self.setup_surface()
        self.init_surface_and_goal()
        self.reset()

        # setup variables
        self.setup_action_space()
        self.setup_observation_space()

    def setup_surface(self):
        """
        Sets variables for generating a surface from heightfield data.
        """

        # define params for generating coherent random surface using opensimplex
        self.heightfield_grid_scale = 0.006  # mesh grid size
        self.num_heightfield_rows, self.num_heightfield_cols = (64, 64)  # n grid points
        self.interpolate_noise = 0.05  # "zoom" into the map (opensimplex param)

        self.surface_pos = [0.65, 0.0, self.height_perturbation_range]
        self.surface_orn = self._pb.getQuaternionFromEuler([0.0, 0.0, 0.0])

        # get the limits of the surface
        min_x = self.surface_pos[0] - ((self.num_heightfield_rows / 2) * self.heightfield_grid_scale)
        max_x = self.surface_pos[0] + ((self.num_heightfield_rows / 2) * self.heightfield_grid_scale)
        min_y = self.surface_pos[1] - ((self.num_heightfield_cols / 2) * self.heightfield_grid_scale)
        max_y = self.surface_pos[1] + ((self.num_heightfield_cols / 2) * self.heightfield_grid_scale)

        # make a grid of x/y positions for pulling height info from world pos
        self.x_bins = np.linspace(min_x, max_x, self.num_heightfield_rows)
        self.y_bins = np.linspace(min_y, max_y, self.num_heightfield_cols)

    def xy_to_surface_idx(self, x, y):
        """
        input: x,y in world coords
        output: i,j for corresponding nearest heightfield data point.
        """

        # find the idxs corresponding to position
        i = np.digitize(y, self.y_bins)
        j = np.digitize(x, self.x_bins)

        # digitize can return max idx which will throw error if pos is outside of range
        if i == self.num_heightfield_cols:
            i -= 1
        if j == self.num_heightfield_rows:
            j -= 1

        return i, j

    def gen_simplex_heigtfield(self):
        """
        Generates a heightmap using OpenSimplex algorithm which results in
        coherent noise across neighbouring vertices.
        Noise in both X and Y directions.
        """

        heightfield_data = np.zeros(shape=(self.num_heightfield_rows, self.num_heightfield_cols))

        for x in range(int(self.num_heightfield_rows)):
            for y in range(int(self.num_heightfield_cols)):

                height = (
                    self.simplex_noise.noise2d(x=x * self.interpolate_noise, y=y * self.interpolate_noise)
                    * self.height_perturbation_range
                )
                heightfield_data[x, y] = height

        return heightfield_data

    def init_surface_and_goal(self):
        """
        Loads a surface based on previously set data.
        Also laod a goal indicator that can be moved to new positions on updates.
        """
        # generate heightfield data as zeros, this gets updated to noisey terrain
        self.heightfield_data = np.zeros(shape=(self.num_heightfield_rows, self.num_heightfield_cols))

        self.create_surface()

        # load a goal so that it can have its position updated
        self.goal_indicator = self._pb.loadURDF(
            add_assets_path("shared_assets/environment_objects/goal_indicators/sphere_indicator.urdf"),
            self.surface_pos,
            [0, 0, 0, 1],
            useFixedBase=True,
        )

    def create_surface(self):

        # load surface
        self.surface_shape = self._pb.createCollisionShape(
            shapeType=self._pb.GEOM_HEIGHTFIELD,
            meshScale=[self.heightfield_grid_scale, self.heightfield_grid_scale, 1],
            heightfieldTextureScaling=(self.num_heightfield_rows - 1) / 2,
            heightfieldData=self.heightfield_data.flatten(),
            numHeightfieldRows=self.num_heightfield_rows,
            numHeightfieldColumns=self.num_heightfield_cols,
        )

        self.surface_id = self._pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=self.surface_shape)
        self._pb.resetBasePositionAndOrientation(self.surface_id, self.surface_pos, self.surface_orn)

        # change color of surface (keep opacity at 1.0)
        self._pb.changeVisualShape(self.surface_id, -1, rgbaColor=[0, 0.0, 1, 1.0])

        # turn off collisions with surface
        self._pb.setCollisionFilterGroupMask(self.surface_id, -1, 0, 0)

    def update_surface(self):
        """
        Update an already loaded surface with random noise.
        """
        # set seed for simplex noise
        self.simplex_noise = OpenSimplex(seed=self.np_random.randint(1e8))
        self.heightfield_data = self.gen_simplex_heigtfield()

        # update heightfield
        self._pb.removeBody(self.surface_id)
        self.create_surface()

        # self.surface_shape = self._pb.createCollisionShape(
        #     shapeType=self._pb.GEOM_HEIGHTFIELD,
        #     meshScale=[self.heightfield_grid_scale, self.heightfield_grid_scale, 1],  # unit size
        #     heightfieldTextureScaling=(self.num_heightfield_rows - 1) / 2,  # no need
        #     heightfieldData=self.heightfield_data.flatten(),  # from gen_heigtfield_simplex_1d(), get cordinate and its height
        #     numHeightfieldRows=self.num_heightfield_rows,  # 64
        #     numHeightfieldColumns=self.num_heightfield_cols,  # 64
        #     replaceHeightfieldIndex=self.surface_shape,  # no need
        #     physicsClientId=self._physics_client_id  # no need
        # )

        # create an array for the surface in world coords
        X, Y = np.meshgrid(self.x_bins, self.y_bins)
        self.surface_array = np.dstack((X, Y, self.heightfield_data + self.surface_pos[2]))

        # Create a grid of surface normal vectors for calculating reward
        surface_grad_y, surface_grad_x = np.gradient(self.heightfield_data, self.heightfield_grid_scale)
        self.surface_normals = np.dstack((-surface_grad_x, -surface_grad_y, np.ones_like(self.heightfield_data)))

        # normalise
        n = np.linalg.norm(self.surface_normals, axis=2)
        self.surface_normals[:, :, 0] /= n
        self.surface_normals[:, :, 1] /= n
        self.surface_normals[:, :, 2] /= n

    def make_goal(self):
        """
        Generate a random position on the current surface.
        Set the directions for automatically moving towards goal.
        """

        self.workframe_directions = np.array([0.0, 1.0, 0.0])
        ang = self.np_random.uniform(-np.pi, np.pi)
        self.workframe_directions[0] = np.cos(ang)
        self.workframe_directions[1] = np.sin(ang)

        # translate from world direction to workframe frame direction
        # in order to auto move towards goal
        self.worldframe_directions = self.workvec_to_worldvec(self.workframe_directions)

        # create the goal in world coords
        self.goal_pos_worldframe = [
            self.surface_pos[0] + self.x_y_extent * self.worldframe_directions[0],
            self.surface_pos[1] + self.x_y_extent * self.worldframe_directions[1],
        ]
        goal_i, goal_j = self.xy_to_surface_idx(self.goal_pos_worldframe[0], self.goal_pos_worldframe[1])
        self.goal_pos_worldframe.append(self.surface_array[goal_i, goal_j, 2])  # get z pos from surface

        self.goal_rpy_worldframe = [0, 0, 0]
        self.goal_orn_worldframe = self._pb.getQuaternionFromEuler(self.goal_rpy_worldframe)
        self.goal_pose_worldframe = np.array([*self.goal_pos_worldframe, *self.goal_rpy_worldframe])

        # create variables for goal pose in coord frame to use later in easy feature observation
        self.goal_pose_workframe = self.worldframe_to_workframe(self.goal_pose_worldframe)

        # useful for visualisation, transparent to not interfere with tactile images
        self._pb.resetBasePositionAndOrientation(self.goal_indicator, self.goal_pos_worldframe, self.goal_orn_worldframe)

    def reset_task(self):
        """
        Create new random surface.
        Place goal on new surface.
        """
        # regenerate a new surface
        self.update_surface()

        # define a goal pos on new surface
        self.make_goal()

    def update_init_pose(self):
        """
        update the initial pose to be taken on reset, relative to the workframe
        """
        # reset the tcp in the center of the surface at the height of the surface at this point
        center_surf_height = self.heightfield_data[int(self.num_heightfield_rows / 2), int(self.num_heightfield_cols / 2)]

        init_tcp_pose = [
            self.surface_pos[0],
            self.surface_pos[1],
            self.surface_pos[2] + center_surf_height - self.embed_dist,
            -np.pi, 0.0, np.pi/2
        ]

        return init_tcp_pose

    def reset(self):
        """
        Reset the environment after an episode terminates.
        """

        # full reset pybullet sim to clear cache, this avoids silent bug where memory fills and visual
        # rendering fails, this is more prevelant when loading/removing larger files
        if self.reset_counter > self.reset_limit:
            self.full_reset()

        # reset vars
        self.reset_counter += 1
        self._env_step_counter = 0

        # update the workframe to a new position relative to surface
        self.reset_task()
        init_tcp_pose = self.update_init_pose()
        self.embodiment.reset(reset_tcp_pose=init_tcp_pose)

        # just to change variables to the reset pose incase needed before taking a step
        self.get_step_data()

        # get the starting observation
        self._observation = self.get_observation()

        return self._observation

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
        self.init_surface_and_goal()
        self.reset_counter = 0

    def get_step_data(self):
        """
        Gets the current tcp pose in worldframe for use in reward/termination calculation.
        """
        # get the cur tip pos here for once per step
        self.cur_tcp_pose_worldframe = self.embodiment.arm.get_tcp_pose()
        self.cur_tcp_pos_worldframe = self.cur_tcp_pose_worldframe[:3]
        self.cur_tcp_rpy_worldframe = self.cur_tcp_pose_worldframe[3:]
        self.cur_tcp_orn_worldframe = self._pb.getQuaternionFromEuler(self.cur_tcp_rpy_worldframe)
        self.tip_i, self.tip_j = self.xy_to_surface_idx(self.cur_tcp_pos_worldframe[0], self.cur_tcp_pos_worldframe[1])

        # self.draw_target_normal()
        # self.draw_tip_normal()

        # get rl info
        done = self.get_termination()
        reward = self.get_reward()

        return reward, done

    def xyz_dist_to_goal(self):
        """
        xyz L2 distance from the current tip position to the goal.
        """
        dist = np.linalg.norm(np.array(self.cur_tcp_pos_worldframe) - np.array(self.goal_pos_worldframe))
        return dist

    def xy_dist_to_goal(self):
        """
        xy L2 distance from the current tip position to the goal.
        Don't care about height in this case.
        """
        dist = np.linalg.norm(np.array(self.cur_tcp_pos_worldframe[:2]) - np.array(self.goal_pos_worldframe[:2]))
        return dist

    def cos_dist_to_surface_normal(self):
        """
        Distance from current orientation of the TCP to the normal of the nearest
        surface point.
        """

        # get normal vector of nearest surface vertex
        targ_surface_normal = self.surface_normals[self.tip_i, self.tip_j, :]

        # get vector of t_s tip, directed through tip body
        tip_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe)
        tip_rot_matrix = np.array(tip_rot_matrix).reshape(3, 3)

        init_vector = np.array([0, 0, -1])
        rot_tip_vector = tip_rot_matrix.dot(init_vector)

        # get the cosine similarity/distance between the two vectors
        cos_sim = np.dot(targ_surface_normal, rot_tip_vector) / (
            np.linalg.norm(targ_surface_normal) * np.linalg.norm(rot_tip_vector)
        )
        cos_dist = 1 - cos_sim

        return cos_dist

    def z_dist_to_surface(self):
        """
        L1 dist from current tip height to surface height.
        This could be improved by using raycasting and measuring the distance
        to the surface in the current orientation of the sensor but in practice
        this works just as well with less complexity.
        """

        surf_z_pos = self.surface_array[self.tip_i, self.tip_j, 2]
        init_vector = np.array([0, 0, -self.embed_dist])

        # find the position embedded in the tip based on current orientation
        tip_rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe)
        tip_rot_matrix = np.array(tip_rot_matrix).reshape(3, 3)

        rot_tip_vector = tip_rot_matrix.dot(init_vector)
        embedded_tip_pos = self.cur_tcp_pos_worldframe + rot_tip_vector

        # draw to double check
        # self._pb.addUserDebugLine(self.cur_tcp_pos_worldframe, embedded_tip_pos, [1, 0, 0], parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)

        # get the current z position of the tip
        # and calculate the distance between the two
        tcp_z_pos = embedded_tip_pos[2]
        dist = np.abs(tcp_z_pos - surf_z_pos)

        return dist

    def get_termination(self):
        """
        Criteria for terminating an episode.
        """
        # terminate when near to goal
        dist = self.xyz_dist_to_goal()
        if dist < self.termination_dist:
            return True

        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True

        return False

    def get_oracle_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """
        # get the surface height and normal at current tip pos
        targ_surf_height = self.surface_array[self.tip_i, self.tip_j, 2]
        targ_surface_normal = self.surface_normals[self.tip_i, self.tip_j, :]
        targ_surface_normal = self.worldvec_to_workvec(targ_surface_normal)

        # get sim info on TCP
        tcp_pose, tcp_vel = self.embodiment.arm.get_current_tcp_pose_vel()
        tcp_pose_workframe = self.worldframe_to_workframe(tcp_pose)
        tcp_vel_workframe = self.worldvel_to_workvel(tcp_pose)

        observation = np.hstack(
            [
                *tcp_pose_workframe,
                *tcp_vel_workframe,
                *self.goal_pose_workframe,
                targ_surf_height,
                *targ_surface_normal,
            ]
        )
        return observation

    """
    Debugging tools
    """

    def draw_tip_normal(self):
        """
        Draws a line in GUI calculated as the normal at the current tip
        orientation
        """
        line_scale = 1.0

        # world pos method
        rot_matrix = self._pb.getMatrixFromQuaternion(self.cur_tcp_orn_worldframe)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        init_vector = np.array([0, 0, -1]) * line_scale

        rot_vector = rot_matrix.dot(init_vector)
        self._pb.addUserDebugLine(
            self.cur_tcp_pos_worldframe,
            self.cur_tcp_pos_worldframe + rot_vector,
            [0, 1, 0],
            parentObjectUniqueId=-1,
            parentLinkIndex=-1,
            lifeTime=0.1,
        )

    def draw_target_normal(self):
        """
        Draws a line in GUI calculated as the normal at the nearest surface
        position.
        """
        line_scale = 1.0

        targ_surface_start_point = self.surface_array[self.tip_i, self.tip_j, :]
        targ_surface_normal = self.surface_normals[self.tip_i, self.tip_j, :] * line_scale

        self._pb.addUserDebugLine(
            targ_surface_start_point,
            targ_surface_start_point + targ_surface_normal,
            [1, 0, 0],
            parentObjectUniqueId=-1,
            parentLinkIndex=-1,
            lifeTime=0.1,
        )

    def plot_surface_normals(self):
        """
        Use to visualise all surface and normal vectors
        """
        X, Y = np.meshgrid(self.x_bins, self.y_bins)

        # find an array of surface normals of height map using numeric method
        surface_grad_y, surface_grad_x = np.gradient(self.heightfield_data, self.heightfield_grid_scale)
        surface_normals = np.dstack((-surface_grad_x, -surface_grad_y, np.ones_like(self.heightfield_data)))

        # normalise
        n = np.linalg.norm(surface_normals, axis=2)
        surface_normals[:, :, 0] /= n
        surface_normals[:, :, 1] /= n
        surface_normals[:, :, 2] /= n

        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d

        fig = plt.figure()
        ax = plt.axes(projection="3d")

        ax.plot_surface(X, Y, self.heightfield_data, cmap="viridis", edgecolor="none")
        ax.quiver(
            X,
            Y,
            self.heightfield_data,
            surface_normals[:, :, 0],
            surface_normals[:, :, 1],
            surface_normals[:, :, 2],
            length=0.0025,
            normalize=False,
        )
        # ax.set_xlim3d(-1,1)
        # ax.set_ylim3d(-1,1)
        # ax.set_zlim3d(-1,1)
        ax.set_title("Surface plot")
        plt.show()
        exit()
