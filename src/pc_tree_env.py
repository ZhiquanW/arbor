from typing import Optional
import matplotlib
from plyfile import PlyData, PlyElement
import os

import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib import axes
import numpy as np
from rlvortex.envs.base_env import BaseEnvTrait

import core


def load_pointcloud(file_path: str):
    data = PlyData.read(file_path)
    x = data["vertex"]["x"]
    y = data["vertex"]["y"]
    z = data["vertex"]["z"]

    y -= np.min(y)
    # move all point below the ground up above the groud
    return np.stack([x, z, y]).T


class PointCloudSingleTreeEnv(BaseEnvTrait):
    def __init__(
        self,
        *,
        point_cloud_mat: np.ndarray,
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
    ) -> None:
        super().__init__()
        self.arbor_engine: core.ArborEngine = core.ArborEngine(
            max_grow_steps=max_grow_steps,
            max_bud_num=max_bud_num,
            num_growth_per_bud=num_growth_per_bud,
            init_dis=init_dis,
            delta_dis_range=delta_dis_range,
            delta_rotate_range=delta_rotate_range,
            init_branch_rot=init_branch_rot,
            branch_rot_range=branch_rot_range,
            branch_prob_range=branch_prob_range,
            sleep_prob_range=sleep_prob_range,
            collision_space_interval=collision_space_interval,
            collision_space_half_size=collision_space_half_size,
        )
        self.collision_space_interval = collision_space_interval
        self.collision_space_half_size = collision_space_half_size
        assert len(point_cloud_mat.shape) == 2, "point cloud mat must be a 2d array"
        assert point_cloud_mat.shape[1] == 3, "point must be in a 3d space"
        self.point_cloud_mat: np.ndarray = point_cloud_mat
        self.point_cloud_collision_space: np.ndarray = np.zeros(
            (2 * collision_space_half_size, 2 * collision_space_half_size, 2 * collision_space_half_size)
        )
        self.__init_pc_collision_space()
        # plot variables
        self.f: Optional[matplotlib.figure.Figure] = None

    def __init_pc_collision_space(self) -> None:
        # compute occupied voxel indices in orginal coordinate
        occupied_voxel_indices = self.point_cloud_mat / self.collision_space_interval
        # compute collision indices in array by occupied voxel indices
        collision_indices = (
            occupied_voxel_indices + np.array([self.collision_space_half_size, self.collision_space_half_size, 0])
        ).astype(np.int32)
        collision_indices = np.clip(collision_indices, 0, self.collision_space_half_size * 2 - 1)
        self.point_cloud_collision_space: np.ndarray = np.zeros(
            (2 * self.collision_space_half_size, 2 * self.collision_space_half_size, 2 * self.collision_space_half_size)
        )
        collision_indices = (collision_indices[:, 0], collision_indices[:, 1], collision_indices[:, 2])
        self.point_cloud_collision_space[tuple(collision_indices)] = 1

    @property
    def action_dim(self) -> tuple:
        return self.arbor_engine.action_dim

    @property
    def observation_dim(self) -> tuple:
        return self.arbor_engine.observation_dim

    @property
    def action_n(self) -> int:
        return 0

    @property
    def renderable(self) -> bool:
        return self.f is not None

    def awake(self):
        pass

    def reset(self):
        self.arbor_engine.reset()
        self.__init_pc_collision_space()

    def destory(self):
        if self.renderable:
            plt.close(self.f)

    def sample_action(self) -> np.ndarray:
        return self.arbor_engine.sample_action()

    def step(self, action: np.ndarray):
        done = self.arbor_engine.step(action=action)

        return 0, done, 0, {}

    def render(self):
        if self.f is None:
            self.f: Optional[matplotlib.figure.Figure] = plt.figure()
            self.pc_axes: plt.Axes = self.f.add_subplot(131, projection="3d")
            self.arbor_axes: plt.Axes = self.f.add_subplot(132, projection="3d")
            self.pc_collision_axes: plt.Axes = self.f.add_subplot(133, projection="3d")
        self.arbor_engine.matplot(self.arbor_axes)
        self.plot_point_collision()
        self.pc_axes.plot(self.point_cloud_mat[:, 0], self.point_cloud_mat[:, 1], self.point_cloud_mat[:, 2], "o", markersize=1)
        plt.pause(0.1)

    def plot_point_collision(self) -> None:
        self.pc_collision_axes.dist = 15  # type: ignore
        xv, yv, zv = np.where(self.point_cloud_collision_space > 0)
        xv -= self.collision_space_half_size
        zv -= self.collision_space_half_size
        self.pc_collision_axes.plot(xv, yv, zv, "o", markersize=2)
        self.pc_collision_axes.set_aspect("equal", adjustable="box")
