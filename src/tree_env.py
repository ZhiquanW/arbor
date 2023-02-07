from typing import List, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import pylab
from rlvortex.envs.base_env import BaseEnvTrait, EnvWrapper

import utils
import os


class PolyLineTreeEnv(BaseEnvTrait):
    def __init__(
        self,
        *,
        max_grow_steps: int = 20,
        max_bud_num: int,
        init_dis: float = 0.5,
        delta_dis_range: np.ndarray[int, np.dtype[np.float64]],
        delta_rotate_range: np.ndarray[int, np.dtype[np.float64]],
        new_branch_rot_range: np.ndarray[int, np.dtype[np.float64]],
        branch_prob_range: np.ndarray[int, np.dtype[np.float64]],
        sleep_prob_range: np.ndarray[int, np.dtype[np.float64]],
        matplot: bool = True,
        headless: bool = True,
        render_path: str = os.path.join(
            os.getcwd(), "tree.jpg"
        ),  # the path to save the render result, used only when render and headless is True
    ) -> None:
        super().__init__()
        assert max_grow_steps >= 1, "max grow steps must be at least 1"  # the maximum number of steps for each episode
        self.max_grow_steps = max_grow_steps
        assert max_bud_num >= 2
        self.max_bud_num = max_bud_num
        assert init_dis > 0, "init_dis must be positive"
        self.init_dis = init_dis
        assert len(delta_dis_range) == 2
        assert np.all(self.init_dis + delta_dis_range > 0)
        self.delta_dis_range = delta_dis_range
        assert len(delta_rotate_range) == 2
        assert delta_rotate_range[0] >= -180 and delta_rotate_range[1] <= 180
        self.delta_rotate_range = delta_rotate_range  # rotate around direction with angle in degreee
        assert len(branch_prob_range) == 2
        self.branch_prob_range = branch_prob_range
        assert len(sleep_prob_range) == 2
        self.sleep_prob_range = sleep_prob_range
        # the angle between the new branch and the parent branch,
        # the init rotation of the new node is compute by rotating alpha degree around parent node x axis ([1,0,0] in local frame)
        self.new_branch_rot_range = new_branch_rot_range
        # buds on the tree
        self.steps = 0
        # render variables
        self.matplot: bool = matplot
        self.headless = headless
        if self.headless:
            matplotlib.use("Agg")
        self.render_path = render_path
        if self.matplot:
            self.f = plt.figure()
            self.ax1 = plt.axes(projection="3d")
        # observation variables
        self.__set_init_variables()
        # reward variables
        self.w_height = 1
        self.w_branch_dir = 1
        self.w_collision = -1
        self.w_light = 1
        # tentative variables, these varibles are updated in step for other methods to use
        self.cur_rot_mats_h = np.zeros((self.max_bud_num, 3, 3))

    def __set_init_variables(self):
        """
        this method is called in reset and __init__ to set the initial task variables
        env variables are set in __init__() and reset()
        """
        self.num_bud = 1
        self.num_feats = 8  # (exists[1], pos[1:4], rot[4:7] [-1, 1] for each channel with (-180,180), up (0,1,0), right (1,0,0), forward(0,0,1), sleep[8]))
        self.buds_states_h: np.ndarray = np.zeros((self.max_bud_num, self.num_feats))
        self.buds_states_h[0, 0] = 1  # exists
        self.buds_states_hist: np.ndarray = np.zeros((self.max_grow_steps, self.max_bud_num, self.num_feats))
        self.buds_born_step_hist: np.ndarray = np.zeros((self.max_bud_num, 1)).astype(np.int32)

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
        return self.matplot

    @property
    def observation_dim(self):
        return (self.max_bud_num * self.num_feats,)

    @property
    def action_dim(self):
        return (self.max_bud_num * 6,)  # each bud can move to one direction (2d normalzied vector)

    @property
    def action_n(self):
        return 0

    def awake(self) -> None:
        pass

    def reset(self):
        self.steps = 0
        self.__set_init_variables()

        return self.buds_states_h.flatten(), {}

    def step(self, action: np.ndarray):
        """
        action is a 1d array of shape: (num_buds * 6)
        need to be reshaped to (num_buds , 6)
        (
            num_buds,
            move_dis: (1) [-1,1],
            delta_rot: (3)[-1,1] from [delta_lower, delta_upper],
            branch_prob: (1),
            sleep_prob: (1)
        )
        """
        assert action.shape == self.action_dim, f"action dimension is wrong, expect {self.action_dim}, got {action.shape}"
        action_2d = action.reshape(self.max_bud_num, -1).clip(-1, 1)
        assert (
            action_2d.shape[0] == self.buds_states_h.shape[0]
        ), f"action dimension is wrong, expect {self.buds_states_h.shape[0]}, got {action_2d.shape[0]}"
        self.steps += 1
        if self.steps == self.max_grow_steps:
            return (
                self.buds_states_h.flatten(),
                0,
                True,
                {"num_buds": self.num_bud, "mean_buds_height": self.buds_states_h[: self.num_bud, 2]},
            )
        # awake bud indices (filter out sleeping nodes and not exists nodes)
        awake_bud_indices = np.array(
            list(filter(lambda row_idx: row_idx < self.num_bud, np.where(self.buds_states_h[:, -1] == 0)[0]))
        ).astype(np.int32)
        num_awake_buds = awake_bud_indices.shape[0]
        if num_awake_buds > 0:
            # check and extract different actions
            act_delta_move_dis_g = action_2d[awake_bud_indices, 0].reshape(-1, 1)
            assert np.all(act_delta_move_dis_g >= -1.0) and np.all(
                act_delta_move_dis_g <= 1.0
            ), "act of delta move distance should be in the range of [-1, 1]"
            delta_move_dis_g = utils.unscale_by_range(act_delta_move_dis_g, self.delta_dis_range[0], self.delta_dis_range[1])
            move_dis_h = self.init_dis + delta_move_dis_g
            # check the rotation is normalized to [-1,1]
            delta_euler_normalized_g = action_2d[awake_bud_indices, 1:4]
            assert np.all(delta_euler_normalized_g >= -1.0) and np.all(
                delta_euler_normalized_g <= 1.0
            ), "delta_euler should be in the range of [-1, 1]"
            # check the branch prob is normalized to [0,1]
            branch_probs_g = utils.unscale_by_range(
                action_2d[awake_bud_indices, 4], self.branch_prob_range[0], self.branch_prob_range[1]
            )
            assert np.all(branch_probs_g >= 0.0) and np.all(
                branch_probs_g <= 1.0
            ), "branch_prob should be in the range of [0, 1]"
            # check the sleep prob is normalized to [0,1]
            sleep_probs_g = utils.unscale_by_range(
                action_2d[awake_bud_indices, 5], self.sleep_prob_range[0], self.sleep_prob_range[1]
            )
            assert np.all(sleep_probs_g >= 0.0) and np.all(sleep_probs_g <= 1.0), "sleep_prob should be in the range of [0, 1]"

            # 1. rotate the direction of the buds
            # 1.1 get delta rotation matrix, unscale to [-180, 180] in degrees
            delta_euler_degree = utils.unscale_by_range(
                delta_euler_normalized_g, self.delta_rotate_range[0], self.delta_rotate_range[1]
            )
            delta_rot_mat = utils.rot_mats_from_eulers(delta_euler_degree)
            # 1.2 get prev rotation matrix, unscale to [-180, 180] in degrees
            prev_euler_degree = utils.unscale_by_range(self.buds_states_h[awake_bud_indices, 4:7], -180, 180)
            prev_rot_mat = utils.rot_mats_from_eulers(prev_euler_degree)
            # 1.3 get current rotation matrix
            assert (
                delta_rot_mat.shape == prev_rot_mat.shape
            ), f"delta_rot_mat.shape={delta_rot_mat.shape} does not match prev_rot_mat.shape={prev_rot_mat.shape}"
            self.cur_rot_mats_h[awake_bud_indices] = np.matmul(
                delta_rot_mat, prev_rot_mat
            )  # multiply num_bud rotation matrixs to get new num_bud rotation matrixs
            # store the current rotation matrixs to self.curr_rot_mat for rendering
            curr_euler_degree = utils.euler_from_rot_mats(self.cur_rot_mats_h[awake_bud_indices])
            curr_euler_degree = (curr_euler_degree % 360 + 180) % 360 - 180
            # 1.4 perform the rotation via the rotation matrixs
            curr_up_dirs = self.cur_rot_mats_h[awake_bud_indices] @ np.array([0, 1, 0])
            assert curr_up_dirs.shape == (
                num_awake_buds,
                3,
            ), f"curr_up_dirs.shape={curr_up_dirs.shape} does not match (num_bud, 3)"

            # 2. update the buds states
            # 2.1 update the position
            self.buds_states_h[awake_bud_indices, 1:4] += move_dis_h * curr_up_dirs
            # 2.2 update the rotation
            self.buds_states_h[awake_bud_indices, 4:7] = utils.scale_by_range(curr_euler_degree, -180, 180)

            # 2.3 grow new branch by sampling the prob
            # the indices of the buds that will grow new branches  from awake buds
            grow_indices = awake_bud_indices[np.where(np.random.uniform(0, 1, num_awake_buds) < branch_probs_g)[0]]
            # print(f"{}: {}"., self.steps, len(grow_indices))
            num_child_buds = min(len(grow_indices), self.max_bud_num - self.num_bud)
            grow_indices = grow_indices[:num_child_buds]
            assert (
                len(grow_indices) == num_child_buds
            ), f"len(grow_indices)={len(grow_indices)} does not match num_child_buds={num_child_buds}"
            # if num_bud is max, then no new branch will be grown
            if num_child_buds > 0 and self.num_bud < self.max_bud_num:
                child_buds_indices = np.arange(self.num_bud, self.num_bud + num_child_buds)
                self.num_bud += num_child_buds
                self.buds_states_h[child_buds_indices, 0] = 1  # set exists to 1 to label there is a node
                # set the position to the previous node
                self.buds_states_h[child_buds_indices, 1:4] = self.buds_states_h[grow_indices, 1:4]

                rot_axes = self.cur_rot_mats_h[grow_indices] @ np.array([1, 0, 0])  # rotate axis depends on the parent branch
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
                new_branch_rot_angle = np.random.uniform(
                    self.new_branch_rot_range[0], self.new_branch_rot_range[1], size=num_child_buds
                ).reshape(num_child_buds, -1)
                new_branch_rot_mat = utils.rot_mat_from_axis_angles(rot_axes, new_branch_rot_angle)
                assert new_branch_rot_angle.shape == (
                    num_child_buds,
                    1,
                ), f"new_branch_rot_angle.shape={new_branch_rot_angle.shape} does not match ({num_child_buds}, 1)"
                new_branch_rot_mat = self.cur_rot_mats_h[grow_indices] @ utils.rot_mat_from_axis_angles(
                    rot_axes, new_branch_rot_angle
                )
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
                new_branch_euler_degree = (new_branch_euler_degree % 360 + 180) % 360 - 180
                self.buds_states_h[child_buds_indices, 4:7] = utils.scale_by_range(new_branch_euler_degree, -180, 180)
                self.buds_born_step_hist[child_buds_indices] = self.steps
            # 3. set buds to sleep
            # the num_awake_buds is the number of buds that are awake before growing new branches
            sleep_indices = awake_bud_indices[np.where(np.random.uniform(0, 1, num_awake_buds) < sleep_probs_g)[0]]
            self.buds_states_h[sleep_indices, 7] = 1  # set selected buds to sleep
        # 4. record information into history
        self.buds_states_hist[self.steps, : self.num_bud] = self.buds_states_h[: self.num_bud]

        # Compute Rewards Functions
        # 1. compute the height reward by summizeing
        buds_height = self.buds_states_h[: self.num_bud, 2]
        r_height = self.w_height * np.mean(buds_height)
        reward = r_height
        done = False
        if num_awake_buds == 0:
            done = True
        if any(buds_height < 0):
            done = True
            reward -= 100
        if not self.renderable:
            self.render()
        return self.buds_states_h.flatten(), reward, done, {"num_buds": self.num_bud, "mean_buds_height": np.mean(buds_height)}

    def sample_action(self):
        acts = np.random.uniform(-1, 1, self.action_dim).reshape(self.max_bud_num, -1)
        return acts.flatten()

    def destory(self):
        if self.renderable:
            plt.close(self.f)

    def __plot(self):
        assert self.f is not None, "self.f is None, the environment is not renderable"
        assert self.ax1 is not None, "self.ax1 is None, the environment is not renderable"
        self.ax1.clear()

        for v_idx in range(self.num_bud):
            born_step = int(self.buds_born_step_hist[v_idx])
            points_x = self.buds_states_hist[born_step : self.steps, v_idx, 1].squeeze()
            points_y = self.buds_states_hist[born_step : self.steps, v_idx, 2].squeeze()
            points_z = self.buds_states_hist[born_step : self.steps, v_idx, 3].squeeze()
            self.ax1.plot(points_x, points_z, points_y)
            self.ax1.set_aspect("equal", adjustable="box")
            z_limit = np.max(np.abs(points_y)) * 1.1
            self.ax1.set_zlim(-0.5, z_limit)  # type: ignore

    def render(self):
        if self.f is None:
            self.f = plt.figure()
            self.ax1: plt.Axes = plt.axes(projection="3d")
        if not self.renderable:
            self.__plot()
            if not self.headless:
                plt.pause(0.1)

    def final_plot(
        self,
    ):
        if self.f is None:
            self.f = plt.figure()
            self.ax1: plt.Axes = plt.axes(projection="3d")
        self.__plot()
        if self.headless:
            plt.savefig(self.render_path)
        else:
            plt.show()
