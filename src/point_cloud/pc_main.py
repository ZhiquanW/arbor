import copy
import pickle

import os
from typing import Dict, List, Set, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import tqdm


class Cylinder:
    def __init__(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        radius: torch.Tensor,
    ) -> None:
        assert start.shape == (3,), f"{start}: {start.shape}"
        assert end.shape == (3,), f"{end}: {end.shape}"
        assert radius > 0 and radius.shape == (), f"{radius}: {radius.shape}"
        self.__start: torch.Tensor = start
        self.__end: torch.Tensor = end
        self.__radius: torch.Tensor = radius

    @property
    def start(self) -> torch.Tensor:
        return self.__start

    @property
    def end(self) -> torch.Tensor:
        return self.__end

    @end.setter
    def end(self, end: torch.Tensor) -> None:
        assert end.shape == (3,)
        self.__end = end

    @property
    def radius(self) -> torch.Tensor:
        return self.__radius

    @radius.setter
    def radius(self, radius: torch.Tensor) -> None:
        self.__radius = torch.min(
            torch.max(torch.tensor(0.0001), radius), torch.tensor(0.05)
        )

    @property
    def length(self):
        return torch.norm(self.end - self.start)

    @property
    def dir(self) -> torch.Tensor:
        return (self.__end - self.__start) / torch.norm(self.__end - self.__start)

    def points_inside_cylinder(
        self, points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert points.shape[1] == 3
        points = points - self.__start
        length = torch.norm(self.start - self.end)
        # use cylinder dir as y axis (up) to indicate above or below
        point2up_projs = torch.matmul(points, self.dir)
        points_above_start = point2up_projs >= 0.0
        points_below_end = point2up_projs <= length
        points_in_vertical_range = points_above_start & points_below_end
        point2horion_proj_dirs = points - point2up_projs.reshape(-1, 1) * self.dir
        points_in_outer_range = torch.norm(point2horion_proj_dirs, dim=-1) <= (
            self.__radius
        )
        points_in_inner_range = (
            torch.norm(point2horion_proj_dirs, dim=-1) >= 0.8 * self.__radius
        )
        points_in = torch.where(
            points_in_vertical_range & points_in_outer_range & points_in_inner_range
        )
        return (
            point2up_projs[points_in],
            point2horion_proj_dirs[points_in],
            points_in[0],
        )


class CylinderOperator:
    def __init__(
        self,
        cylinder: Cylinder,
        end_offset: torch.Tensor,
        radius_offset: torch.Tensor,
        num_cylinder: int = 1,
    ) -> None:
        self.num_cylinders = num_cylinder
        assert end_offset.shape == (3, 2), f"{end_offset}: {end_offset.shape}"
        assert radius_offset.shape == (2,), f"{radius_offset}: {radius_offset.shape}"
        # used to store the previous state of cylinder in case of bad score restore
        self.__cylinders = [copy.deepcopy(cylinder) for _ in range(num_cylinder)]
        self.__backup_cylinders = copy.deepcopy(self.__cylinders)
        self.__end_offset = end_offset
        self.__radius_offset = radius_offset

    def sample_actions(self) -> tuple[torch.Tensor, torch.Tensor]:
        random_end_offsets = (
            torch.rand(self.num_cylinders, 3)
            * (self.__end_offset[:, 1] - self.__end_offset[:, 0])
            + self.__end_offset[:, 0]
        )

        random_radius_offsets = (
            torch.rand(self.num_cylinders, 1)
            * (self.__radius_offset[1] - self.__radius_offset[0])
            + self.__radius_offset[0]
        )
        return (random_end_offsets, random_radius_offsets)

    def predict(
        self,
        end_offsets: torch.Tensor,
        radius_offsets: torch.Tensor,
    ):
        for i in range(self.num_cylinders):
            self.__cylinders[i].end = self.__cylinders[i].end + end_offsets[i]
            self.__cylinders[i].radius = self.__cylinders[i].radius + radius_offsets[i]

    def accept_prediction(self):
        self.__backup_cylinders = copy.deepcopy(self.__cylinders)

    def reject_prediction(self):
        self.__cylinders = copy.deepcopy(self.__backup_cylinders)

    @property
    def cylinders(self) -> List[Cylinder]:
        return self.__cylinders


def compute_score(
    points: torch.Tensor, cylinders: List[Cylinder]
) -> Tuple[float, Dict]:
    num_points = points.shape[0]
    score = 0
    score_dict = {
        "points_in_cylinder_score": [],
        "end2closest_point_score": [],
    }
    global_points_in: Set = set()
    for i in range(len(cylinders)):
        c = cylinders[i]
        (
            point2up_projs,
            point2horizon_proj,
            points_in,
        ) = c.points_inside_cylinder(points)
        unique_points = []
        unique_points_idx = []
        for i in range(len(points_in)):
            p = points_in[i].item()
            if p not in global_points_in:
                unique_points.append(p)
                unique_points_idx.append(i)
                global_points_in.add(p)

        num_points_in = len(unique_points)
        point2up_projs = point2up_projs[unique_points_idx]
        end2closest_point = (
            torch.min(torch.abs(c.length - point2up_projs)).item()
            if len(point2up_projs) > 0
            else 99999
        )

        points_in_cylinder_score = num_points_in / num_points
        end2closest_point_score = np.exp(-end2closest_point) / len(cylinders)
        score += points_in_cylinder_score + end2closest_point_score
        score_dict["points_in_cylinder_score"].append(points_in_cylinder_score)
        score_dict["end2closest_point_score"].append(end2closest_point_score)

    return score, score_dict


def draw_3d_circle(ax, center, radius, dir):
    # draw dir as line

    theta = np.linspace(0, 2 * np.pi, 100)
    right_vecotor = np.cross(dir, np.array([0, 0, 1]))
    up_dir = np.cross(right_vecotor, dir)

    x = center[0] + radius * (
        right_vecotor[0] * np.cos(theta) + up_dir[0] * np.sin(theta)
    )
    y = center[1] + radius * (
        right_vecotor[1] * np.cos(theta) + up_dir[1] * np.sin(theta)
    )
    z = center[2] + radius * (
        right_vecotor[2] * np.cos(theta) + up_dir[2] * np.sin(theta)
    )

    # Plot the circle in 3D space
    ax.plot(
        x,
        y,
        z,
    )


def draw_cylinder(ax, c: Cylinder):
    start = c.start.numpy()
    end = c.end.numpy()
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])
    draw_3d_circle(
        ax,
        start,
        c.radius.item(),
        c.dir.numpy(),
    )

    draw_3d_circle(
        ax,
        end,
        c.radius.item(),
        c.dir.numpy(),
    )


def main():
    with open(
        os.path.join(
            os.path.join(os.path.dirname(os.path.abspath(__file__))),
            "../../data/points.pkl",
        ),
        "rb",
    ) as f:
        point_cloud_info: dict = pickle.load(f)
    pc_data = list(point_cloud_info.values())[-3]
    points = torch.tensor(pc_data["pointcloud"])
    center = torch.tensor(pc_data["position"])
    # center = torch.tensor([-0.019, -0.0426, 0.004], dtype=torch.double)
    cylinder = Cylinder(center, center + torch.tensor([0, 0.05, 0]), torch.tensor(0.01))
    cylinder_operator = CylinderOperator(
        cylinder=cylinder,
        num_cylinder=2,
        end_offset=torch.tensor([[-0.005, 0.005], [-0.005, 0.005], [-0.005, 0.005]]),
        radius_offset=torch.tensor([-0.001, 0.001]),
    )
    # mcmc parameters
    epsilon = 0.000  # stop condition 0
    max_steps = 5000  # stop condition 1
    accept_bad_prob = 0.05  # accept bad score prob
    # start mcmc optimization util the delta score < epsilon or max_steps reached
    score = 0
    for optim_step in tqdm.tqdm(range(max_steps), desc=f"{score}"):
        cylinder_operator.predict(*cylinder_operator.sample_actions())
        new_score, score_dict = compute_score(points, cylinder_operator.cylinders)
        if 0 < new_score - score < epsilon:
            print(
                f"early stop at step {optim_step}, delta:{new_score-score}, final socre: {new_score}"
            )
            break
        if new_score > score:
            cylinder_operator.accept_prediction()
            score = new_score
        elif torch.rand(1) < accept_bad_prob:
            cylinder_operator.accept_prediction()
            score = new_score
        else:
            cylinder_operator.reject_prediction()

        print(f"final score: {score}, {score_dict}")

    # visualization
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("equal")
    max_lim = torch.max(torch.abs(points))

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker="o")
    for c in cylinder_operator.cylinders:
        draw_cylinder(ax, c)
    ax.axes.set_xlim3d(-max_lim, max_lim)
    ax.axes.set_ylim3d(-max_lim, max_lim)
    ax.axes.set_zlim3d(-max_lim, max_lim)
    plt.show()


if __name__ == "__main__":
    main()
