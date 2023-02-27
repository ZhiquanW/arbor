# buit-in import
import io
import os
from typing import List, Optional
from time import gmtime, strftime
import random
from matplotlib import cm

# third-party import
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab
from PIL import Image
from rlvortex.envs.base_env import BaseEnvTrait, EnvWrapper
import tqdm

# project package import
import utils


class PolyLineTreeEnv(BaseEnvTrait):
    def __init__(
        self,
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
        collision_space_half_size: float,
        matplot: bool = True,
        headless: bool = True,
        render_path: str = os.getcwd(),  # the path to save the render result, used only when render and headless is True
    ) -> None:
        super().__init__()
        assert max_grow_steps >= 1, "max grow steps must be at least 1"  # the maximum number of steps for each episode
        self.max_grow_steps = max_grow_steps
        assert max_bud_num >= 2
        self.max_bud_num = max_bud_num
        assert num_growth_per_bud > 0
        self.num_growth_per_bud = num_growth_per_bud
        assert init_dis > 0, "init_dis must be positive"
        self.init_dis = init_dis
        assert len(delta_dis_range) == 2
        assert np.all(self.init_dis + delta_dis_range > 0)
        self.delta_dis_range = delta_dis_range
        assert init_branch_rot > 0 and init_branch_rot < 180
        self.init_branch_rot = init_branch_rot
        assert len(delta_rotate_range) == 2
        assert delta_rotate_range[0] >= -180 and delta_rotate_range[1] <= 180
        self.delta_rotate_range = delta_rotate_range  # rotate around direction with angle in degreee
        assert len(branch_prob_range) == 2
        self.branch_prob_range = branch_prob_range
        assert len(sleep_prob_range) == 2
        self.sleep_prob_range = sleep_prob_range
        assert collision_space_interval > 0.0
        self.collision_space_interval = collision_space_interval
        self.collision_space_half_size = collision_space_half_size
        # the angle between the new branch and the parent branch,
        # the init rotation of the new node is compute by rotating alpha degree around parent node x axis ([1,0,0] in local frame)
        self.branch_rot_range = branch_rot_range
        # buds on the tree
        self.steps = 0
        # render variables
        self.matplot: bool = matplot
        self.headless = headless
        if self.matplot and self.headless:
            matplotlib.use("Agg")
        self.render_path = render_path
        self.f = None
        if self.matplot:
            self.f = plt.figure()
            self.tree_ax = self.f.add_subplot(121, projection="3d")  # subplot has 1 row 2 columns, it takes the first position
            self.collision_ax = self.f.add_subplot(
                122, projection="3d"
            )  # subplot has 1 row 2 columns, it takes the second position
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
        """
        1. exists[0],
        2. pos[1:4],
        3. rot[4:7] [-1, 1] for each channel with (-180,180), up (0,1,0), right (1,0,0), forward(0,0,1), 
        4. sleep[7]
        5. num_growth[8]
        """
        self.num_feats = 9
        self.buds_states_h: np.ndarray = np.zeros((self.max_bud_num, self.num_feats))
        self.buds_states_h[0, 0] = 1  # set first bud exists
        self.buds_states_h[0, 8] = self.num_growth_per_bud
        self.buds_states_hist: np.ndarray = np.zeros((self.max_grow_steps, self.max_bud_num, self.num_feats))
        self.buds_born_step_hist: np.ndarray = np.zeros((self.max_bud_num, 1)).astype(np.int32)
        self.collision_space: np.ndarray = np.zeros(
            (2 * self.collision_space_half_size, 2 * self.collision_space_half_size, 2 * self.collision_space_half_size)
        )

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
        # active bud indices by filtering out  (1. not exists nodes 2. sleeping nodes 3. no remaining growth chance)
        active_bud_indices = np.array(
            list(
                filter(
                    lambda row_idx: row_idx < self.num_bud,
                    np.where((self.buds_states_h[:, 7] == 0) & (self.buds_states_h[:, 8] > 0))[0],
                )
            )
        ).astype(np.int32)
        num_active_buds = active_bud_indices.shape[0]
        delta_euler_normalized_g = 0  # pre-define for reward computing
        # 3. active buds perform actions
        if num_active_buds > 0:
            # 0. check incoming action parameters
            # 0.1 check delta move distance between [-1, 1]
            act_delta_move_dis_g = action_2d[active_bud_indices, 0].reshape(-1, 1)
            assert np.all(act_delta_move_dis_g >= -1.0) and np.all(
                act_delta_move_dis_g <= 1.0
            ), "act of delta move distance should be in the range of [-1, 1]"
            delta_move_dis_h = utils.unscale_by_range(act_delta_move_dis_g, self.delta_dis_range[0], self.delta_dis_range[1])
            # 0.2 check rotation between [-1, 1] in xyz
            delta_euler_normalized_g = action_2d[active_bud_indices, 1:4]
            assert np.all(delta_euler_normalized_g >= -1.0) and np.all(
                delta_euler_normalized_g <= 1.0
            ), "delta_euler should be in the range of [-1, 1]"
            # 0.3 check the branch prob is normalized to [0,1]
            branch_probs_g = utils.unscale_by_range(
                action_2d[active_bud_indices, 4], self.branch_prob_range[0], self.branch_prob_range[1]
            )
            assert np.all(branch_probs_g >= 0.0) and np.all(
                branch_probs_g <= 1.0
            ), "branch_prob should be in the range of [0, 1]"
            # 1. prepare move distance & rotation parameters
            # 1.1 move buds
            move_dis_h = self.init_dis + delta_move_dis_h
            # 1.2. rotate the direction of the buds via previous rot mat and delta rot mat
            # 1.2.1 unscale degree to [-180, 180] in degrees
            delta_euler_degree = utils.unscale_by_range(
                delta_euler_normalized_g, self.delta_rotate_range[0], self.delta_rotate_range[1]
            )
            # 1.2.2 get delta rotation matrix
            delta_rot_mat = utils.rot_mats_from_eulers(delta_euler_degree)
            # 1.2.3 get prev rotation matrix, unscale to [-180, 180] in degrees
            prev_euler_degree = utils.unscale_by_range(self.buds_states_h[active_bud_indices, 4:7], -180, 180)
            prev_rot_mat = utils.rot_mats_from_eulers(prev_euler_degree)
            # 1.3 get current rotation matrix
            assert (
                delta_rot_mat.shape == prev_rot_mat.shape
            ), f"delta_rot_mat.shape={delta_rot_mat.shape} does not match prev_rot_mat.shape={prev_rot_mat.shape}"
            self.cur_rot_mats_h[active_bud_indices] = np.matmul(
                delta_rot_mat, prev_rot_mat
            )  # multiply num_bud rotation matrixs to get new num_bud rotation matrixs
            # 1.4 store the current rotation matrixs to self.curr_rot_mat for rendering
            curr_euler_degree = utils.euler_from_rot_mats(self.cur_rot_mats_h[active_bud_indices])
            # 1.5 perform the rotation of up vector via the rotation matrixs
            curr_up_dirs = self.cur_rot_mats_h[active_bud_indices] @ np.array([0, 1, 0])
            assert curr_up_dirs.shape == (
                num_active_buds,
                3,
            ), f"curr_up_dirs.shape={curr_up_dirs.shape} does not match (num_bud, 3)"

            # 2. update the buds states
            # 2.1 update the position
            self.buds_states_h[active_bud_indices, 1:4] += move_dis_h * curr_up_dirs
            # 2.2 update the rotation
            self.buds_states_h[active_bud_indices, 4:7] = utils.scale_by_range(curr_euler_degree, -180, 180)
            # 2.3 update the remaining num of growth
            self.buds_states_h[active_bud_indices, 8] -= 1
            # 2.4. set buds to sleep
            # the num_awake_buds is the number of buds that are awake before growing new branches
            # check the sleep prob is normalized to [0,1]
            sleep_probs_g = utils.unscale_by_range(
                action_2d[active_bud_indices, 5], self.sleep_prob_range[0], self.sleep_prob_range[1]
            )
            assert np.all(sleep_probs_g >= 0.0) and np.all(sleep_probs_g <= 1.0), "sleep_prob should be in the range of [0, 1]"
            sleep_indices = active_bud_indices[np.where(np.random.uniform(0, 1, num_active_buds) < sleep_probs_g)[0]]
            self.buds_states_h[sleep_indices, 7] = 1  # set selected buds to sleep

            # 2.5. grow new branch by sampling the prob
            # the indices of the buds that will grow new branches  from awake buds and the remaining number of growth > 0
            grow_indices = active_bud_indices[np.where(np.random.uniform(0, 1, num_active_buds) < branch_probs_g)[0]]
            grow_indices = list(filter(lambda row_idx: self.buds_states_h[row_idx, 8] > 0, grow_indices))
            num_child_buds = min(len(grow_indices), self.max_bud_num - self.num_bud)
            grow_indices = grow_indices[:num_child_buds]
            assert (
                len(grow_indices) == num_child_buds
            ), f"len(grow_indices)={len(grow_indices)} does not match num_child_buds={num_child_buds}"
            # if num_bud is max, then no new branch will be grown
            if num_child_buds > 0 and self.num_bud < self.max_bud_num:
                # grow new buds by setting it as exist, after all the existing buds
                child_buds_indices = np.arange(self.num_bud, self.num_bud + num_child_buds)
                self.num_bud += num_child_buds
                self.buds_states_h[child_buds_indices, 0] = 1  # set exists to 1 to label there is a node
                # set the position to the previous node
                self.buds_states_h[child_buds_indices, 1:4] = self.buds_states_h[grow_indices, 1:4]
                # parent_euler = utils.unscale_by_range(self.buds_states_h[grow_indices, 4:7], -180, 180)
                # parent_rot_mats_g = utils.rot_mats_from_eulers(parent_euler)
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
                db_rot_mat = utils.rot_mat_from_axis_angles(rot_axes, new_branch_rot_degree)
                new_branch_rot_mat = self.cur_rot_mats_h[grow_indices] @ utils.rot_mat_from_axis_angles(
                    rot_axes, new_branch_rot_degree
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
                # set the rotation of new bud via rotating the parent branch angle
                self.buds_states_h[child_buds_indices, 4:7] = utils.scale_by_range(new_branch_euler_degree, -180, 180)
                # set the growth num as the parent branch's num_growth - 1
                self.buds_states_h[child_buds_indices, 8] = self.buds_states_h[grow_indices, 8] - 1
                assert np.all(self.buds_states_h[:, 8] >= 0), "the number of remaining growth chance must be positive"
                # record the birthday of the new buds
                self.buds_born_step_hist[child_buds_indices] = self.steps
        # 5. record information into history
        self.buds_states_hist[self.steps, : self.num_bud] = self.buds_states_h[: self.num_bud]
        # 6. compute collision occupy
        all_exists_buds_indices = np.where(self.buds_states_hist[:, :, 0] == 1)  # get all exists buds' indices
        occupied_voxel_indices = (
            self.buds_states_hist[:, :, 1:4][all_exists_buds_indices] / self.collision_space_interval
        ).astype(np.int32)
        collision_indices = (
            occupied_voxel_indices + np.array([self.collision_space_half_size, 0, self.collision_space_half_size])
        ).transpose()
        self.collision_space[tuple(collision_indices)] = 1

        # 6.Compute Rewards Functions
        # 6.1 compute the height reward by summizeing
        buds_height = self.buds_states_h[: self.num_bud, 2]
        r_height = self.w_height * np.mean(buds_height)
        # 6.2 compute the reward of mantain growing direciton
        r_branch_dir = self.w_branch_dir / np.exp(np.mean(delta_euler_normalized_g**2))
        reward = r_height + r_branch_dir
        done = num_active_buds == 0 or any(buds_height < 0)
        if not self.renderable:
            self.render()
        return (
            self.buds_states_h.flatten(),
            reward,
            done,
            {
                "num_buds": self.num_bud,
                "mean_buds_height": np.mean(buds_height),
                "reward/height": r_height,
                "reward/branch_dir": r_branch_dir,
            },
        )

    def sample_action(self):
        acts = np.random.uniform(-1, 1, self.action_dim).reshape(self.max_bud_num, -1)
        return acts.flatten()

    def destory(self):
        if self.f is not None:
            plt.close(self.f)

    def __plot(self, step: Optional[int] = None, max_steps: Optional[int] = None):
        assert self.f is not None, "self.f is None, the environment is not renderable"
        assert self.tree_ax is not None, "self.ax1 is None, the environment is not renderable"
        # matplotlib.rcParams["axes.linewidth"] = 0.1  # set the value globally
        self.tree_ax.clear()
        self.collision_ax.clear()
        current_step = self.steps if step is None else step
        max_steps = self.max_grow_steps if max_steps else max_steps
        delta_angle = 360 / self.max_grow_steps
        self.tree_ax.view_init(15, current_step * delta_angle, 0)
        self.tree_ax.dist = 10
        self.collision_ax.view_init(15, current_step * delta_angle, 0)
        self.collision_ax.dist = 10
        # Option 1. plot collision space with 3d heatmap
        # elements = (
        #     np.arange(2 * self.collision_space_half_size) - self.collision_space_half_size
        # ) * self.collision_space_interval
        # xv, yv, zv = np.meshgrid(elements, elements + self.collision_space_half_size * self.collision_space_interval, elements)
        # heats = self.collision_space.flatten()
        # colors = cm.hsv(heats / max(heats))
        # self.collision_ax.scatter(xv.flatten(), zv.flatten(), yv.flatten(), c=colors, s=4)
        # Option 2. plot collision spacw with scatter
        xv, yv, zv = np.where(self.collision_space > 0)
        xv -= self.collision_space_half_size
        zv -= self.collision_space_half_size
        self.collision_ax.plot(xv, zv, yv, "o", markersize=2)
        self.collision_ax.set_aspect("equal", adjustable="box")
        z_limit = np.max(np.abs(yv)) * 1.1
        self.collision_ax.set_zlim(-0.5, z_limit)  # type: ignore
        #  plot tree buds and trajectory
        tree_zlimit = 0
        for v_idx in range(self.num_bud):
            born_step = int(self.buds_born_step_hist[v_idx])
            points_x = self.buds_states_hist[born_step : self.steps + 1, v_idx, 1].squeeze()
            points_y = self.buds_states_hist[born_step : self.steps + 1, v_idx, 2].squeeze()
            points_z = self.buds_states_hist[born_step : self.steps + 1, v_idx, 3].squeeze()
            self.tree_ax.spines["top"].set_linewidth(0.1)
            self.tree_ax.spines["bottom"].set_linewidth(0.1)
            self.tree_ax.spines["left"].set_linewidth(0.1)
            self.tree_ax.spines["right"].set_linewidth(0.1)
            # self.tree_ax.axis("off")
            self.tree_ax.plot(points_x, points_z, points_y, "-o", markersize=2)
            self.tree_ax.set_aspect("equal", adjustable="box")
            tree_zlimit = max(tree_zlimit, np.max(np.abs(points_y)) * 1.1)
            self.tree_ax.set_zlim(-0.5, tree_zlimit)  # type: ignore

    def render(self):
        if self.renderable:
            if self.f is None:
                self.f = plt.figure()
                self.tree_ax: plt.Axes = plt.axes(projection="3d")
            self.__plot()
            if not self.headless:
                plt.pause(0.1)

    def final_plot(self, interactive: bool = True, name: Optional[str] = None, num_frame: int = 1):
        if self.f is None:
            self.f = plt.figure()
            self.tree_ax: plt.Axes = plt.axes(projection="3d")
        if interactive:
            self.__plot()
            plt.show()
            return
        fig_name = "tree_" + strftime("%Y-%m-%d|%H:%M:%S", gmtime()) if name is None else name
        if num_frame == 1:
            self.__plot()
            if self.headless:
                plt.savefig(os.path.join(self.render_path, fig_name + ".jpg"), dpi=512)
            else:
                plt.show()
        else:
            matplotlib.use("macosx")
            Image.MAX_IMAGE_PIXELS = 1000000000
            frames = []
            io_buf = io.BytesIO()

            for i in tqdm.tqdm(range(num_frame)):
                self.__plot(step=i, max_steps=num_frame)
                self.f.savefig(io_buf, format="raw", dpi=100, bbox_inches=0)
                io_buf.seek(0)
                img_arr = np.reshape(
                    np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                    newshape=(self.f.canvas.get_width_height()[::-1] + (4,)),
                )
                frames.append(img_arr)
            io_buf.close()
            imgs = [Image.fromarray(frame) for frame in frames]
            imgs[0].save(fig_name + ".gif", format="gif", save_all=True, append_images=imgs[1:], duration=16, loop=0)
