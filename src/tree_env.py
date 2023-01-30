from typing import List
import numpy as np
import matplotlib.pyplot as plt
from rlvortex.envs.base_env import BaseEnvTrait, EnvWrapper

import utils


class PolyLineTreeEnv(BaseEnvTrait):
    def __init__(
        self,
        *,
        max_vertex_num: int,
        delta_dis_range: np.ndarray[int, np.dtype[np.float64]] = np.array([0, 1]),
        delta_rotate_range: np.ndarray[int, np.dtype[np.float64]] = np.array([-5, 5]),
        new_branch_rot_range: np.ndarray[int, np.dtype[np.float64]] = np.array([-5, 5]),
        headless: bool = True,
    ) -> None:
        super().__init__()
        self.max_vertex_num = max_vertex_num
        assert max_vertex_num >= 2
        self.delta_dis_range = delta_dis_range
        assert len(delta_dis_range) == 2
        self.delta_rotate_range = delta_rotate_range  # rotate around direction with angle in degreee
        assert len(delta_rotate_range) == 2
        assert delta_rotate_range[0] >= -180 and delta_rotate_range[1] <= 180
        # the angle between the new branch and the parent branch,
        # the init rotation of the new node is compute by rotating alpha degree around parent node x axis ([1,0,0] in local frame)
        self.new_branch_rot_range = new_branch_rot_range
        # vertices on the tree
        self.eposide_length = 1000  # the maximum number of steps for each episode
        self.steps = 0
        self.headless = headless
        self.init_plot = False
        if not self.headless:
            self.init_plot = True
            self.f = plt.figure()
            self.ax1: plt.Axes = plt.axes(projection="3d")
        # task variables
        self.__set_init_variables()
        # tentative variables, these varibles are updated in step for other methods to use
        self.cur_rot_mats_h = np.zeros((self.max_vertex_num, 3, 3))

    def __set_init_variables(self):
        """
        this method is called in reset and __init__ to set the initial task variables
        env variables are set in __init__() and reset()
        """
        self.num_vertex = 1
        self.num_feats = 8  # (exists[1], pos[1:4], rot[4:7] [-1, 1] for each channel with (-180,180), up (0,1,0), right (1,0,0), forward(0,0,1), sleep[8]))
        self.vertices_states_h: np.ndarray = np.zeros((self.max_vertex_num, self.num_feats))
        self.vertices_states_h[0, 0] = 1  # exists
        self.vertices_states_hist: np.ndarray = np.zeros((self.eposide_length, self.max_vertex_num, self.num_feats))
        self.vertices_born_step_hist: np.ndarray = np.zeros((self.max_vertex_num, 1)).astype(np.int32)

    # this method is called after the action is performed.
    def compute_step_info(self):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def get_done(self, valid: bool):
        raise NotImplementedError

    @property
    def renderable(self):
        return not self.headless

    @property
    def observation_dim(self):
        raise NotImplementedError

    @property
    def action_dim(self):
        return (self.max_vertex_num, 6)  # each vertex can move to one direction (2d normalzied vector)

    @property
    def action_n(self):
        return 0

    def awake(self) -> None:
        pass

    def reset(self):
        self.steps = 0
        self.__set_init_variables()

        return self.vertices_states_h

    def step(self, action):
        """
        action is a 2d array of shape: (num_vertices, 5)
            (num_vertices,
             move_dis: (1) [0,1],
             delta_rot: (3)[-1,1] from [delta_lower, delta_upper],
             branch_prob: (1),
             sleep_prob: (1) )
        """
        self.steps += 1
        # awake vertex indices (filter out sleeping nodes and not exists nodes)
        awake_vertex_indices = np.array(list(filter(lambda row_idx: row_idx < self.num_vertex, np.where(self.vertices_states_h[:, -1] == 0)[0]))).astype(np.int32)
        num_awake_vertices = awake_vertex_indices.shape[0]
        if num_awake_vertices == 0:
            return
        # check and extract different actions
        assert action.shape == self.action_dim, f"action dimension is wrong, expect {self.action_dim}, got {action.shape}"
        assert action.shape[0] == self.vertices_states_h.shape[0], f"action dimension is wrong, expect {self.vertices_states_h.shape[0]}, got {action.shape[0]}"
        # check the direction is normalized to [0,1]
        move_dis_g = action[awake_vertex_indices, 0].reshape(-1, 1)
        assert np.all(move_dis_g >= 0.0) and np.all(move_dis_g <= 1.0), "move_dis should be in the range of [0, 1]"
        # check the rotation is normalized to [-1,1]
        delta_euler_normalized_g = action[awake_vertex_indices, 1:4]
        assert np.all(delta_euler_normalized_g >= -1.0) and np.all(delta_euler_normalized_g <= 1.0), "delta_euler should be in the range of [-1, 1]"
        # check the branch prob is normalized to [0,1]
        branch_probs_g = action[awake_vertex_indices, 4]
        assert np.all(branch_probs_g >= 0.0) and np.all(branch_probs_g <= 1.0), "branch_prob should be in the range of [0, 1]"
        # check the sleep prob is normalized to [0,1]
        sleep_probs_g = action[awake_vertex_indices, 5]
        assert np.all(sleep_probs_g >= 0.0) and np.all(sleep_probs_g <= 1.0), "sleep_prob should be in the range of [0, 1]"

        # 1. rotate the direction of the vertices
        # 1.1 get delta rotation matrix, unscale to [-180, 180] in degrees
        delta_euler_degree = utils.unscale_by_range(delta_euler_normalized_g, self.delta_rotate_range[0], self.delta_rotate_range[1])
        delta_rot_mat = utils.rot_mats_from_eulers(delta_euler_degree)
        # 1.2 get prev rotation matrix, unscale to [-180, 180] in degrees
        prev_euler_degree = utils.unscale_by_range(self.vertices_states_h[awake_vertex_indices, 4:7], -180, 180)
        prev_rot_mat = utils.rot_mats_from_eulers(prev_euler_degree)
        # 1.3 get current rotation matrix
        assert delta_rot_mat.shape == prev_rot_mat.shape, f"delta_rot_mat.shape={delta_rot_mat.shape} does not match prev_rot_mat.shape={prev_rot_mat.shape}"
        self.cur_rot_mats_h[awake_vertex_indices] = np.matmul(delta_rot_mat, prev_rot_mat)  # multiply num_vertex rotation matrixs to get new num_vertex rotation matrixs
        # store the current rotation matrixs to self.curr_rot_mat for rendering
        curr_euler_degree = utils.euler_from_rot_mats(self.cur_rot_mats_h[awake_vertex_indices])
        curr_euler_degree = (curr_euler_degree % 360 + 180) % 360 - 180
        # 1.4 perform the rotation via the rotation matrixs
        curr_up_dirs = self.cur_rot_mats_h[awake_vertex_indices] @ np.array([0, 1, 0])
        assert curr_up_dirs.shape == (
            num_awake_vertices,
            3,
        ), f"curr_up_dirs.shape={curr_up_dirs.shape} does not match (num_vertex, 3)"

        # 2. update the vertices states
        # 2.1 update the position
        self.vertices_states_h[awake_vertex_indices, 1:4] += utils.unscale_by_range(move_dis_g, self.delta_dis_range[0], self.delta_dis_range[1]) * curr_up_dirs
        # 2.2 update the rotation
        self.vertices_states_h[awake_vertex_indices, 4:7] = utils.scale_by_range(curr_euler_degree, -180, 180)

        # 2.3 grow new branch by sampling the prob
        # the indices of the vertices that will grow new branches  from awake vertices
        grow_indices = awake_vertex_indices[np.where(np.random.uniform(0, 1, num_awake_vertices) < branch_probs_g)[0]]
        num_child_vertices = len(grow_indices)
        # if num_vertex is max, then no new branch will be grown
        if num_child_vertices > 0 and self.num_vertex < self.max_vertex_num:
            child_vertices_indices = np.arange(self.num_vertex, self.num_vertex + num_child_vertices)
            self.num_vertex += num_child_vertices
            self.vertices_states_h[child_vertices_indices, 0] = 1  # set exists to 1 to label there is a node
            # set the position to the previous node
            self.vertices_states_h[child_vertices_indices, 1:4] = self.vertices_states_h[grow_indices, 1:4]

            parent_left_dir = self.cur_rot_mats_h[grow_indices] @ np.array([1, 0, 0])
            assert np.allclose(np.linalg.norm(parent_left_dir, axis=1), 1), f"the length of parent_left_dir={parent_left_dir} is not 1. get {np.linalg.norm(parent_left_dir)}"
            assert parent_left_dir.shape == (
                num_child_vertices,
                3,
            ), f"parent_left_dir.shape={parent_left_dir.shape} does not match ({num_child_vertices}, 3)"
            # initialize the rotation matrixs of the new vertices, from the parent rotation by rotating from parent up dir n dgrees
            parent_up_dirs = self.cur_rot_mats_h[grow_indices] @ np.array([0, 1, 0])
            rot_axes = np.cross(parent_up_dirs, parent_left_dir)
            assert rot_axes.shape == (num_child_vertices, 3), f"rot_axes.shape={rot_axes.shape} does not match ({num_child_vertices}, 3)"
            new_branch_rot_angle = np.random.uniform(self.new_branch_rot_range[0], self.new_branch_rot_range[1], size=num_child_vertices).reshape(num_child_vertices, -1)
            new_branch_rot_mat = utils.rot_mat_from_axis_angles(rot_axes, new_branch_rot_angle)
            assert new_branch_rot_angle.shape == (num_child_vertices, 1), f"new_branch_rot_angle.shape={new_branch_rot_angle.shape} does not match ({num_child_vertices}, 1)"
            new_branch_rot_mat = self.cur_rot_mats_h[grow_indices] @ utils.rot_mat_from_axis_angles(rot_axes, new_branch_rot_angle)
            assert new_branch_rot_mat.shape == (num_child_vertices, 3, 3), f"new_branch_rot_mat.shape={new_branch_rot_mat.shape} does not match ({num_child_vertices}, 3, 3)"
            new_branch_euler_degree = utils.euler_from_rot_mats(new_branch_rot_mat)
            assert new_branch_euler_degree.shape == (num_child_vertices, 3), f"new_branch_euler_degree.shape={new_branch_euler_degree.shape} does not match ({num_child_vertices}, 3)"
            new_branch_euler_degree = (new_branch_euler_degree % 360 + 180) % 360 - 180
            self.vertices_states_h[child_vertices_indices, 4:7] = utils.scale_by_range(new_branch_euler_degree, -180, 180)
            self.vertices_born_step_hist[child_vertices_indices] = self.steps

        # 3. set vertices to sleep
        # the num_awake_vertices is the number of vertices that are awake before growing new branches
        sleep_indices = awake_vertex_indices[np.where(np.random.uniform(0, 1, num_awake_vertices) < sleep_probs_g)[0]]
        self.vertices_states_h[sleep_indices, 7] = 1  # set selected vertices to sleep

        # 4. record information into history
        self.vertices_states_hist[self.steps, : self.num_vertex] = self.vertices_states_h[: self.num_vertex]
        if not self.headless:
            self.render()

    def sample_action(self):
        acts = np.random.uniform(-1, 1, self.action_dim)
        acts[:, (0, -2, -1)] += 1.0
        acts[:, (0, -2, -1)] /= 2.0
        acts[:, -2] = 0.9
        acts[:, -1] = 0.0001
        return acts

    def destory(self):
        if self.renderable:
            plt.close(self.f)

    def __plot(self):
        if self.init_plot:
            self.ax1.clear()
            print("self.steps", self.steps)
            print("self.num_vertex", self.num_vertex)
            for v_idx in range(self.num_vertex):
                born_step = int(self.vertices_born_step_hist[v_idx])
                points_x = self.vertices_states_hist[born_step : self.steps, v_idx, 1].squeeze()
                points_y = self.vertices_states_hist[born_step : self.steps, v_idx, 2].squeeze()
                points_z = self.vertices_states_hist[born_step : self.steps, v_idx, 3].squeeze()
                self.ax1.plot(points_x, points_z, points_y)
                self.ax1.set_aspect("equal", adjustable="box")

    def render(self):
        if not self.init_plot:
            self.init_plot = True
            self.f = plt.figure()
            self.ax1: plt.Axes = plt.axes(projection="3d")
        if not self.headless:
            self.__plot()
            plt.pause(0.01)

    def final_plot(
        self,
    ):
        if not self.init_plot:
            self.init_plot = True
            self.f = plt.figure()
            self.ax1: plt.Axes = plt.axes(projection="3d")
        self.__plot()
        plt.show()
