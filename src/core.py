import random
from typing import List
import warnings
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils


class ArborEngine:
    def __init__(
        self,
        *,
        max_grow_steps: int,
        max_bud_num: int,
        num_growth_per_bud: int,
        init_dis: float,
        delta_dis_range: np.ndarray[int, np.dtype[np.float64]],
        delta_rotate_range: np.ndarray[int, np.dtype[np.float64]],
        init_branch_rot: float,
        branch_rot_range: np.ndarray[int, np.dtype[np.float64]],
        branch_prob_range: np.ndarray[int, np.dtype[np.float64]],
        sleep_prob_range: np.ndarray[int, np.dtype[np.float64]],
        collision_space_interval: float,
        collision_space_half_size: int,
        shadow_space_interval: float,
        shadow_space_half_size: int,
        shadow_pyramid_half_size: int,
        delta_shadow_value: float,
    ) -> None:

        super().__init__()
        assert max_grow_steps >= 1, "max grow steps must be at least 1"  # the maximum number of steps for each episode
        # the max grow steps of the tree
        self.max_grow_steps = max_grow_steps
        assert max_bud_num >= 2
        # the max number of buds a tree can have
        self.max_bud_num = max_bud_num
        assert num_growth_per_bud > 0
        # the max number of times a bud can grow. when brach, the new buds will inherit the remaining growth num
        self.num_growth_per_bud = num_growth_per_bud
        assert init_dis > 0, "init_dis must be positive"
        # the default distance a buds should grow
        self.init_dis = init_dis
        assert len(delta_dis_range) == 2
        assert np.all(self.init_dis + delta_dis_range > 0)
        # the range of the delta distance of init_dis
        self.delta_dis_range = delta_dis_range
        assert delta_rotate_range.shape == (3, 2)
        assert (delta_dis_range >= -180).all()
        assert (delta_dis_range <= 180).all()
        # the rotation angle (in degree) range (in x,y,z dimension separately) when a bud grow
        self.delta_rotate_range = delta_rotate_range
        assert init_branch_rot > 0 and init_branch_rot < 180
        # the default angle (in degree) between the new branch and its parent branch when branch
        # the init rotation of the new node is compute by rotating alpha degree around parent node x axis ([1,0,0] in local frame)
        self.init_branch_rot = init_branch_rot
        assert len(branch_prob_range) == 2
        # the delta rotation ange of init_branch_rot
        self.branch_rot_range = branch_rot_range
        # the probability of a bud branch at each step
        self.branch_prob_range = branch_prob_range
        assert len(sleep_prob_range) == 2
        # the probabiliy of a bud sleep at each step
        self.sleep_prob_range = sleep_prob_range
        assert collision_space_interval > 0.0
        self.collision_space_interval = collision_space_interval
        self.collision_space_half_size = collision_space_half_size
        assert shadow_space_interval > 0.0
        self.shadow_space_interval = shadow_space_interval
        self.shadow_space_half_size = shadow_space_half_size
        # the half size of the shadow pyramid
        self.shadow_pyramid_half_size = shadow_pyramid_half_size
        # the delta value of shadow voxel
        self.delta_shadow_value = delta_shadow_value
        # the shadow pyramid template with values to be added when applied on the shadow space
        self.shadow_pyramid_template: np.ndarray[int, np.dtype[np.float32]] = self.__gen_shadow_pyramid_template()

        # buds on the tree
        self.steps = 0
        self.done = False
        # # tentative variables, these varibles are updated in step for other methods to use
        # self.cur_rot_mats_h = np.zeros((self.max_bud_num, 3, 3))
        self.__set_init_variables()

    def __gen_shadow_pyramid_template(self) -> np.ndarray[int, np.dtype[np.float32]]:
        x, y, z = (
            np.indices(
                (2 * self.shadow_pyramid_half_size - 1, self.shadow_pyramid_half_size, 2 * self.shadow_pyramid_half_size - 1)
            )
            - self.shadow_pyramid_half_size
            + 1
        )
        pyramid_mask = np.abs(x) + self.shadow_pyramid_half_size - np.abs(y) + np.abs(z) <= self.shadow_pyramid_half_size
        template = np.zeros(
            (2 * self.shadow_pyramid_half_size - 1, self.shadow_pyramid_half_size, 2 * self.shadow_pyramid_half_size - 1),
            dtype=np.float32,
        )
        template[pyramid_mask] = self.delta_shadow_value
        return template

    def __set_init_variables(self) -> None:
        """
        this method is called in reset and __init__ to set the initial task variables
        env variables are set in __init__() and reset()
        """
        self.num_bud = 1
        self.num_feats = 9
        """
        1. exists[0],
        2. pos[1:4],
        3. rot[4:7] [-1, 1] for each channel with (-180,180), up (0,1,0), right (1,0,0), forward(0,0,1), 
        4. sleep[7]
        5. num_growth[8]
        """
        # buds_state_h stores the information of all the nodes (indicates the buds can act)
        # budsa_state_h -> (max_bud_num, num_feats)
        self.nodes_states_h: np.ndarray = np.zeros((self.max_bud_num, self.num_feats))
        self.nodes_states_h[0, 0] = 1  # set first bud exists
        self.nodes_states_h[0, 8] = self.num_growth_per_bud
        # buds_state_hist stores all the history information of the tree (buds)
        # the 1st dimension is the ith growth step
        # the 2nd dimension is the jth bud
        # the 3rd dimension stores the information of ith step, jth bud's features
        self.nodes_states_hist: np.ndarray = np.zeros((self.max_grow_steps, self.max_bud_num, self.num_feats))
        self.buds_born_step_hist: np.ndarray = np.zeros((self.max_bud_num, 1)).astype(np.int32)
        self.collision_space: np.ndarray = np.zeros(
            (2 * self.collision_space_half_size, 2 * self.collision_space_half_size, 2 * self.collision_space_half_size)
        )
        # a voxel space that record the shadow values in per voxel (0 - 1 (fully shadowed)) with zero-padding
        self.shadow_space: np.ndarray = np.zeros(
            (
                2 * self.shadow_space_half_size + 2 * self.shadow_pyramid_half_size,
                2 * self.shadow_space_half_size + 2 * self.shadow_pyramid_half_size,
                2 * self.shadow_space_half_size + 2 * self.shadow_pyramid_half_size,
            )
        )
        self.nodes_occupy_space()

    @property
    def action_dim(self):
        return (self.max_bud_num * 6,)

    @property
    def observation_dim(self):
        return (self.max_bud_num * self.num_feats,)

    def reset(self):
        self.done = False
        self.steps = 0
        self.__set_init_variables()

    def step(self, action: np.ndarray) -> bool:
        """perform a step of growth of the tree

        this method perform the growth/ branch/ sleep action for each buds. after actions is performed, the collision will be computed and the data will be stored

        Args:
            action is a 1d array of shape: (max_buds_num * 6)
            all actions must be normalized to [-1,1]
            need to be reshaped to (max_buds_num , 6)

            (
                num_buds,
                move_dis: (1) [-1,1],
                delta_rot: (3)[-1,1] from [delta_lower, delta_upper],
                branch_prob: (1),
                sleep_prob: (1)
            )
        Returns: Bool if the modeling process is completed (all data will be stored in the member variables)
        """
        ############################################## sanity check: input ##############################################
        assert not self.done, f"the modeling process is done, no further action can be performed"
        assert action.shape == (
            self.max_bud_num * 6,
        ), f"action dimension is wrong, expect {self.max_bud_num * 6,}, got {action.shape}"
        action_2d = action.reshape(self.max_bud_num, -1).clip(-1, 1)
        assert (
            action_2d.shape[0] == self.nodes_states_h.shape[0]
        ), f"action dimension is wrong, expect {self.nodes_states_h.shape[0]}, got {action_2d.shape[0]}"

        self.steps += 1
        if self.steps == self.max_grow_steps:
            self.done = True
            return True
        # active bud indices by filtering out  (1. non-existing nodes 2. sleeping nodes 3. no remaining growth chance)
        active_nodes_index = np.array(
            list(
                filter(
                    lambda row_idx: row_idx < self.num_bud,
                    np.where((self.nodes_states_h[:, 7] == 0) & (self.nodes_states_h[:, 8] > 0))[0],
                )
            )
        ).astype(np.int32)
        num_active_buds = active_nodes_index.shape[0]

        ############################################## early stop: no active buds ##############################################
        if num_active_buds <= 0:
            self.done = True
            return True

        ############################################## sanity check: action attributes ##############################################
        # 0. check incoming action parameters
        # 0.1 check delta move distance between [-1, 1]
        act_delta_move_dis_g = action_2d[active_nodes_index, 0].reshape(-1, 1)
        assert np.all(act_delta_move_dis_g >= -1.0) and np.all(
            act_delta_move_dis_g <= 1.0
        ), "act of delta move distance should be in the range of [-1, 1]"
        delta_move_dis_h = utils.unscale_by_range(act_delta_move_dis_g, self.delta_dis_range[0], self.delta_dis_range[1])

        # 0.2 check rotation between [-1, 1] in xyz
        delta_euler_normalized_g = action_2d[active_nodes_index, 1:4]
        assert np.all(delta_euler_normalized_g >= -1.0) and np.all(
            delta_euler_normalized_g <= 1.0
        ), "delta_euler should be in the range of [-1, 1]"

        # 0.3 unscale degree to [-180, 180] in degrees
        delta_euler_degrees = utils.unscale_by_ranges(
            delta_euler_normalized_g, self.delta_rotate_range[:, 0], self.delta_rotate_range[:, 1]
        )

        # 0.4 check the branch prob is normalized to [0,1]
        branch_probs_g = utils.unscale_by_range(
            action_2d[active_nodes_index, 4], self.branch_prob_range[0], self.branch_prob_range[1]
        )
        assert np.all(branch_probs_g >= 0.0) and np.all(branch_probs_g <= 1.0), "branch_prob should be in the range of [0, 1]"

        ############################################## buds grow ##############################################

        # 1. prepare move distance & rotation parameters
        self.nodes_grow(active_nodes_index, delta_move_dis_h, delta_euler_degrees)

        ############################################## buds grow ##############################################
        # cast shadow for each new node
        self.nodes_cast_shadow(nodes_index_g=active_nodes_index)

        ############################################## buds sleep ##############################################
        # 2.4. set buds to sleep
        # the num_awake_buds is the number of buds that are awake before growing new branches
        # check the sleep prob is normalized to [0,1]
        sleep_probs_g = utils.unscale_by_range(
            action_2d[active_nodes_index, 5], self.sleep_prob_range[0], self.sleep_prob_range[1]
        )
        assert np.all(sleep_probs_g >= 0.0) and np.all(sleep_probs_g <= 1.0), "sleep_prob should be in the range of [0, 1]"
        sleep_indices = active_nodes_index[np.where(np.random.uniform(0, 1, num_active_buds) < sleep_probs_g)[0]]
        self.nodes_sleep(sleep_indices)

        ############################################## buds branch ##############################################
        # 2.5. grow new branch by sampling the prob
        # the indices of the buds that will grow new branches from awake buds and the remaining number of growth > 0
        grow_indices = active_nodes_index[np.where(np.random.uniform(0, 1, num_active_buds) < branch_probs_g)[0]]
        grow_indices = list(filter(lambda row_idx: self.nodes_states_h[row_idx, 8] > 0, grow_indices))
        num_child_buds = min(len(grow_indices), self.max_bud_num - self.num_bud)
        grow_indices = grow_indices[:num_child_buds]
        assert (
            len(grow_indices) == num_child_buds
        ), f"len(grow_indices)={len(grow_indices)} does not match num_child_buds={num_child_buds}"

        # if num_bud is max, then no new branch will be grown
        if num_child_buds > 0 and self.num_bud <= self.max_bud_num:
            # grow new buds by setting it as exist, after all the existing buds
            child_buds_indices = np.arange(self.num_bud, self.num_bud + num_child_buds)
            self.nodes_branch(grow_indices, child_buds_indices)

        self.nodes_states_hist[self.steps, : self.num_bud] = self.nodes_states_h[: self.num_bud]
        #
        self.nodes_occupy_space()
        # if any node is under ground, then stop
        if np.any(self.nodes_states_h[: self.num_bud, 2] < 0):
            return True
        return False

    def nodes_grow(self, bud_indices_g: np.ndarray, delta_move_dis_g: np.ndarray, delta_euler_degrees_g: np.ndarray):
        """perform the grow action of the buds, includeing direction change and grow forward and data recording

        args:
            delta_move_distance: unnormalized moving distance
            delta_euler_degrees: unnormalized branch rotaion angle (-180,180) in degrees
        """

        # 1. prepare move distance & rotation parameters
        # 1.1 move buds
        move_dis_h = self.init_dis + delta_move_dis_g
        # 1.2. rotate the direction of the buds via previous rot mat and delta rot mat
        # 1.2.2 get delta rotation matrix
        delta_rot_mat = utils.rot_mats_from_eulers(delta_euler_degrees_g)
        # 1.2.3 get prev rotation matrix, unscale to [-180, 180] in degrees
        prev_euler_degree = utils.unscale_by_range(self.nodes_states_h[bud_indices_g, 4:7], -180, 180)
        prev_rot_mat = utils.rot_mats_from_eulers(prev_euler_degree)
        # 1.3 get current rotation matrix
        assert (
            delta_rot_mat.shape == prev_rot_mat.shape
        ), f"delta_rot_mat.shape={delta_rot_mat.shape} does not match prev_rot_mat.shape={prev_rot_mat.shape}"
        buds_rot_mat = np.matmul(
            delta_rot_mat, prev_rot_mat
        )  # multiply num_bud rotation matrixs to get new num_bud rotation matrixs
        # 1.4 store the current rotation matrixs to self.curr_rot_mat for rendering
        curr_euler_degree = utils.euler_from_rot_mats(buds_rot_mat)
        # 1.5 perform the rotation of up vector via the rotation matrixs
        curr_up_dirs = buds_rot_mat @ np.array([0, 1, 0])
        assert curr_up_dirs.shape == (
            len(bud_indices_g),
            3,
        ), f"curr_up_dirs.shape={curr_up_dirs.shape} does not match (num_bud, 3)"

        # 2. update the buds states
        # 2.1 update the position
        self.nodes_states_h[bud_indices_g, 1:4] += move_dis_h * curr_up_dirs
        # 2.2 update the rotation
        self.nodes_states_h[bud_indices_g, 4:7] = utils.scale_by_range(curr_euler_degree, -180, 180)
        # 2.3 update the remaining num of growth
        self.nodes_states_h[bud_indices_g, 8] -= 1

    def nodes_sleep(self, nodes_indice_g: np.ndarray):
        self.nodes_states_h[nodes_indice_g, 7] = 1  # set selected buds to sleep

    def nodes_branch(self, parent_nodes_indice: List[int], child_nodes_indice: np.ndarray) -> None:
        num_child_buds = len(child_nodes_indice)
        self.num_bud += num_child_buds
        self.nodes_states_h[child_nodes_indice, 0] = 1  # set exists to 1 to label there is a node
        # set the position to the previous node
        self.nodes_states_h[child_nodes_indice, 1:4] = self.nodes_states_h[parent_nodes_indice, 1:4]
        parent_euler = utils.unscale_by_range(self.nodes_states_h[parent_nodes_indice, 4:7], -180, 180)
        parent_rot_mats_g = utils.rot_mats_from_eulers(parent_euler)
        rot_axes = parent_rot_mats_g @ np.array([1, 0, 0])  # rotate axis depends on the parent branch
        assert np.allclose(
            np.linalg.norm(rot_axes, axis=1), 1
        ), f"the length of rotate axis = {rot_axes} is not 1. get {np.linalg.norm(rot_axes)}"
        assert rot_axes.shape == (
            num_child_buds,
            3,
        ), f"rotate axis.shape={rot_axes.shape} does not match ({num_child_buds}, 3)"
        # initialize the rotation matrixs of the new buds, from the parent rotation by rotating from parent up dir n dgrees
        assert rot_axes.shape == (
            num_child_buds,
            3,
        ), f"rot_axes.shape={rot_axes.shape} does not match ({num_child_buds}, 3)"
        # TODO: restore
        new_branch_rot_degree = random.choice([1, -1]) * (
            self.init_branch_rot
            + np.random.uniform(self.branch_rot_range[0], self.branch_rot_range[1], size=num_child_buds).reshape(
                num_child_buds, -1
            )
        )
        new_branch_rot_mat = utils.rot_mat_from_axis_angles(rot_axes, new_branch_rot_degree)
        assert new_branch_rot_degree.shape == (
            num_child_buds,
            1,
        ), f"new_branch_rot_angle.shape={new_branch_rot_degree.shape} does not match ({num_child_buds}, 1)"
        new_branch_rot_mat = parent_rot_mats_g @ utils.rot_mat_from_axis_angles(rot_axes, new_branch_rot_degree)
        assert new_branch_rot_mat.shape == (
            num_child_buds,
            3,
            3,
        ), f"new_branch_rot_mat.shape={new_branch_rot_mat.shape} does not match ({num_child_buds}, 3, 3)"
        new_branch_euler_degree = utils.euler_from_rot_mats(new_branch_rot_mat)
        assert new_branch_euler_degree.shape == (
            num_child_buds,
            3,
        ), f"new_branch_euler_degree.shape={new_branch_euler_degree.shape} does not match ({num_child_buds}, 3)"
        # set the rotation of new bud via rotating the parent branch angle
        self.nodes_states_h[child_nodes_indice, 4:7] = utils.scale_by_range(new_branch_euler_degree, -180, 180)
        # set the growth num as the parent branch's num_growth - 1
        self.nodes_states_h[child_nodes_indice, 8] = self.nodes_states_h[parent_nodes_indice, 8] - 1
        assert np.all(self.nodes_states_h[:, 8] >= 0), "the number of remaining growth chance must be positive"
        # record the birthday of the new buds
        self.buds_born_step_hist[child_nodes_indice] = self.steps

    def nodes_occupy_space(self) -> None:
        # 6. compute collision occupy
        all_exists_nodes_index = np.where(self.nodes_states_h[:, 0] == 1)  # get all exists buds' indices
        occupied_voxels_index = np.around(
            (self.nodes_states_h[all_exists_nodes_index][:, 1:4]) / self.collision_space_interval
        ).astype(np.int32)
        collision_indices = (
            occupied_voxels_index + np.array([self.collision_space_half_size, 0, self.collision_space_half_size])
        ).transpose()
        if np.any(collision_indices < 0) or np.any(collision_indices > self.collision_space_half_size * 2 - 1):
            warnings.warn(f"[step/collision_detection] {self.steps}: tree node position outside collision voxel space")
        collision_indices = np.clip(collision_indices, 0, self.collision_space_half_size * 2 - 1)
        self.collision_space[tuple(collision_indices)] += 1

    def nodes_cast_shadow(self, nodes_index_g: np.ndarray) -> None:
        active_nodes_pos = self.nodes_states_h[nodes_index_g][:, 1:4]
        # get selected nodes voxel space index
        occupied_voxel_indices = (active_nodes_pos / self.shadow_space_interval).astype(np.int32) + np.array(
            [
                self.shadow_space_half_size + self.shadow_pyramid_half_size,
                self.shadow_pyramid_half_size,
                self.shadow_space_half_size + self.shadow_pyramid_half_size,
            ]
        )
        for occupied_voxel_index in occupied_voxel_indices:
            self.shadow_space[
                occupied_voxel_index[0]
                - self.shadow_pyramid_half_size
                + 1 : occupied_voxel_index[0]
                + self.shadow_pyramid_half_size,
                occupied_voxel_index[1] - self.shadow_pyramid_half_size : occupied_voxel_index[1],
                occupied_voxel_index[2]
                - self.shadow_pyramid_half_size
                + 1 : occupied_voxel_index[2]
                + self.shadow_pyramid_half_size,
            ] += self.shadow_pyramid_template
        # i, j, k = np.meshgrid(
        #     np.arange(-self.shadow_pyramid_half_size + 1, self.shadow_pyramid_half_size),
        #     np.arange(-self.shadow_pyramid_half_size, 0),
        #     np.arange(-self.shadow_pyramid_half_size + 1, self.shadow_pyramid_half_size),
        #     indexing="ij",
        # )
        # indices = np.array(occupied_voxel_indices)[:, np.newaxis, np.newaxis, np.newaxis] + np.stack([i, j, k], axis=-1)
        # self.shadow_space[indices[..., 0], indices[..., 1], indices[..., 2]] += self.shadow_pyramid_template

    def sample_action(self) -> np.ndarray:
        acts = np.random.uniform(-1, 1, self.max_bud_num * 6).reshape(self.max_bud_num, -1)
        return acts.flatten()

    def matplot_tree(self, plt_axes: axes.Axes) -> None:
        plt_axes.clear()
        plt_axes.dist = 8  # type: ignore
        tree_zlimit = 0
        for v_idx in range(self.num_bud):
            born_step = int(self.buds_born_step_hist[v_idx])
            points_x = self.nodes_states_hist[born_step : self.steps + 1, v_idx, 1].squeeze()
            points_y = self.nodes_states_hist[born_step : self.steps + 1, v_idx, 2].squeeze()
            points_z = self.nodes_states_hist[born_step : self.steps + 1, v_idx, 3].squeeze()
            plt_axes.plot(points_x, points_z, points_y, "-o", markersize=2)
            tree_zlimit = max(tree_zlimit, np.max(np.abs(points_y)) * 1.1)
        plt_axes.set_aspect("equal", adjustable="box")
        plt_axes.set_zlim(-0.5, tree_zlimit)  # type: ignore

    def matplot_collision(self, plt_axes: axes.Axes) -> None:
        plt_axes.clear()
        plt_axes.dist = 8  # type: ignore
        xv, yv, zv = np.where(self.collision_space > 0)
        col_vertices = (
            np.array([xv, yv, zv], dtype=np.float32)
            - np.array([self.collision_space_half_size, 0, self.collision_space_half_size]).reshape(3, 1)
        ) * self.collision_space_interval
        plt_axes.plot(col_vertices[0, :], col_vertices[2, :], col_vertices[1, :], "o", markersize=2)
        plt_axes.set_aspect("equal", adjustable="box")
        plt_axes.set_zlim(-0.5, np.max(col_vertices[1, :]))  # type: ignore

    def matplot_shadow(self, plt_axes: axes.Axes) -> None:
        plt_axes.clear()
        plt_axes.dist = 8  # type: ignore
        xv, yv, zv = np.where(self.shadow_space > 0)
        shadow_vertices = (
            np.array([xv, yv, zv], dtype=np.float32)
            - np.array([self.shadow_space_half_size, 0, self.shadow_space_half_size]).reshape(3, 1)
        ) * self.shadow_space_interval
        plt_axes.scatter(
            shadow_vertices[0, :],
            shadow_vertices[2, :],
            shadow_vertices[1, :],
            "o",
            markersize=2,
        )
        plt_axes.set_aspect("equal", adjustable="box")
        plt_axes.set_zlim(-0.5, np.max(shadow_vertices[1, :]))  # type: ignore

    def plotly_tree(self, fig: plotly.graph_objs._figure.Figure) -> None:
        for v_idx in range(self.num_bud):
            born_step = int(self.buds_born_step_hist[v_idx])
            points_x = self.nodes_states_hist[born_step : self.steps + 1, v_idx, 1].squeeze()
            points_y = self.nodes_states_hist[born_step : self.steps + 1, v_idx, 2].squeeze()
            points_z = self.nodes_states_hist[born_step : self.steps + 1, v_idx, 3].squeeze()
            fig.add_trace(go.Scatter3d(x=points_x, y=points_z, z=points_y, showlegend=False), row=1, col=1)

    def plotly_collision(self, fig: plotly.graph_objs._figure.Figure) -> None:
        xv, yv, zv = np.where(self.collision_space > 0)
        col_vertices = (
            np.array([xv, yv, zv], dtype=np.float32)
            - np.array([self.collision_space_half_size, 0, self.collision_space_half_size]).reshape(3, 1)
        ) * self.collision_space_interval
        fig.add_trace(
            go.Scatter3d(
                x=col_vertices[0, :],
                y=col_vertices[2, :],
                z=col_vertices[1, :],
                mode="markers",
                marker=dict(size=20, opacity=0.8),
            ),
            row=1,
            col=1,
        )

    def plotly_shadow(self, fig: plotly.graph_objs._figure.Figure):
        # todo: march 20 2023, fix shadow viz bug
        xv, yv, zv = np.where(self.shadow_space > 0)
        shadow_vertices = (
            np.array([xv, yv, zv], dtype=np.float32)
            - np.array(
                [
                    self.shadow_space_half_size + self.shadow_pyramid_half_size,
                    0 + self.shadow_pyramid_half_size,
                    self.shadow_space_half_size + self.shadow_pyramid_half_size,
                ]
            ).reshape(3, 1)
        ) * self.shadow_space_interval
        fig.add_trace(
            go.Scatter3d(
                x=shadow_vertices[0, :],
                y=shadow_vertices[2, :],
                z=shadow_vertices[1, :],
                mode="markers",
                marker=dict(
                    size=12,
                    color=self.shadow_space[(xv, yv, zv)],  # set color to an array/list of desired values
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.15,
                ),
            ),
            row=1,
            col=1,
        )
