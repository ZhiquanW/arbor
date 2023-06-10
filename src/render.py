from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt
from matplotlib import axes
import sim.torch_arbor as arbor
import torch
import numpy as np
import rlvortex.envs.base_env as base_env
import rlvortex
import tree_envs
import sim.torch_arbor as arbor
import sim.aux_space as auxs_space
import sim.energy_module as energy
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import Optional
import streamlit


def plotly_tree_skeleton(
    fig: Optional[plotly.graph_objs._figure.Figure],
    arbor_engine: arbor.TorchArborEngine,
):
    if fig is None:
        fig = make_subplots(
            rows=2,
            row_heights=[0.2, 0.8],
            specs=[[{"is_3d": False}], [{"is_3d": True}]],
            vertical_spacing=0.00,
        )

    for v_idx in range(arbor_engine.max_branches_num):
        born_step = int(arbor_engine.branch_birth_hist[v_idx])
        points_x = arbor_engine.nodes_state[
            born_step : arbor_engine.steps + 1, v_idx, 5
        ]
        points_y = arbor_engine.nodes_state[
            born_step : arbor_engine.steps + 1, v_idx, 6
        ]
        points_z = arbor_engine.nodes_state[
            born_step : arbor_engine.steps + 1, v_idx, 7
        ]
        fig.add_trace(
            go.Scatter3d(
                x=points_x,
                y=points_y,
                z=points_z,
                marker_size=3,
                line=dict(
                    width=7,
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    if arbor_engine.occupancy_space is not None:
        voxel_positions = arbor_engine.occupancy_space.get_occupied_voxel_positions()
        if voxel_positions is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=voxel_positions[:, 0],
                    y=voxel_positions[:, 1],
                    z=voxel_positions[:, 2],
                    mode="markers",
                    name="occupancy",
                ),
                row=2,
                col=1,
            )
    if arbor_engine.shadow_space is not None:
        voxel_positions = arbor_engine.shadow_space.get_occupied_voxel_positions()
        xv, yv, zv = torch.where(arbor_engine.shadow_space.space > 0)
        if voxel_positions is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=voxel_positions[:, 0],
                    y=voxel_positions[:, 1],
                    z=voxel_positions[:, 2],
                    mode="markers",
                    name="shadow",
                    marker=dict(
                        size=12,
                        color=arbor_engine.shadow_space.space[
                            (xv, yv, zv)
                        ],  # set color to an array/list of desired values
                        colorscale="Bluyl",  # choose a colorscale
                        opacity=0.20,
                    ),
                ),
                row=2,
                col=1,
            )
    max_range = torch.max(torch.abs(arbor_engine.nodes_state[:, :, 5:8])).item()
    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
        ),
        autosize=False,
        height=800,
    )
    fig.update_layout(
        scene=dict(
            aspectmode="cube",
            xaxis=dict(
                range=[-max_range, max_range],
            ),
            yaxis=dict(
                range=[-max_range, max_range],
            ),
            zaxis=dict(
                range=[-2.0, 2.0 * max_range],
            ),
        ),
    )
    # fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    if arbor_engine.energy_module is not None:
        plot_x = list(range(len(arbor_engine.energy_module.energy_hist.total_energys)))
        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=arbor_engine.energy_module.energy_hist.total_energys,
                mode="lines+markers",
                name="total energys",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=arbor_engine.energy_module.energy_hist.maintainence_consumptions,
                mode="lines+markers",
                name="maintainence consumptions",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=arbor_engine.energy_module.energy_hist.move_consumptions,
                mode="lines+markers",
                name="move consumptions",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=arbor_engine.energy_module.energy_hist.branch_consumptions,
                mode="lines+markers",
                name="branch consumptions",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=arbor_engine.energy_module.energy_hist.collected_energys,
                mode="lines+markers",
                name="collected energys",
            ),
            row=1,
            col=1,
        )
    return fig


def plotly_energy_module(
    fig: Optional[plotly.graph_objs._figure.Figure], energy_hist: energy.EnergyHist
):
    if fig is None:
        fig = go.Figure()

    plot_x = list(range(len(energy_hist.total_energys)))
    fig.add_trace(
        go.Scatter(
            x=plot_x,
            y=energy_hist.total_energys,
            mode="lines+markers",
            name="total energys",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_x,
            y=energy_hist.maintainence_consumptions,
            mode="lines+markers",
            name="maintainence consumptions",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_x,
            y=energy_hist.move_consumptions,
            mode="lines+markers",
            name="move consumptions",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_x,
            y=energy_hist.branch_consumptions,
            mode="lines+markers",
            name="branch consumptions",
        )
    )

    return fig


def get_matplot_fig_size(dpi=256):
    f = plt.figure(figsize=(8, 8), dpi=256)
    return f.canvas.get_width_height()


# def matplot_arbor(
#     skeleton_axes: axes.Axes,
#     nodes_axes: axes.Axes,
#     arbor_engine: core.TorchArborEngine,
# ):
#     skeleton_axes.clear()
#     nodes_axes.clear()
#     for i in range(arbor_engine.num_nodes - 1, 0, -1):
#         pos = arbor_engine.nodes_state[i][6:9]
#         parent_idx = arbor_engine.nodes_state[i][5].to(torch.int64)
#         print(i, parent_idx)
#         if parent_idx == -1:
#             continue
#         parent_pos = arbor_engine.nodes_state[parent_idx][6:9]
#         skeleton_axes.plot(
#             [parent_pos[0], pos[0]],
#             [parent_pos[2], pos[2]],
#             [parent_pos[1], pos[1]],
#             "-o",
#             markersize=2,
#         )
#     nodes_pos = arbor_engine.nodes_state[: arbor_engine.num_nodes, 6:9].cpu().numpy()
#     nodes_axes.plot(
#         nodes_pos[:, 0], nodes_pos[:, 2], nodes_pos[:, 1], "o", markersize=2
#     )
#     skeleton_axes.set_aspect("equal", adjustable="box")
#     nodes_axes.set_aspect("equal", adjustable="box")
#     skeleton_axes.set_zlim(-0.5, 4)  # type: ignore
#     nodes_axes.set_zlim(-0.5, 4)  # type: ignore


def matplot_tree(plt_axes: axes.Axes, arbor_engine: arbor.TorchArborEngine) -> None:
    plt_axes.clear()
    tree_zlimit = 0
    for branch_idx in range(arbor_engine.num_branches):
        birth_step = int(arbor_engine.branch_birth_hist[branch_idx].cpu().numpy())
        nodes_pos = (
            arbor_engine.nodes_state[birth_step : arbor_engine.steps, branch_idx, 5:8]
            .cpu()
            .numpy()
        )
        if len(nodes_pos) == 0:
            continue
        points_x = nodes_pos[:, 0]
        points_y = nodes_pos[:, 1]
        points_z = nodes_pos[:, 2]
        plt_axes.plot(points_x, points_z, points_y, "-o", markersize=2)
        tree_zlimit = max(tree_zlimit, np.max(np.abs(points_y)) * 1.1)
    plt_axes.set_aspect("equal", adjustable="box")
    plt_axes.set_zlim(-0.5, tree_zlimit)  # type: ignore


def matplot_tree_energy(
    plt_axes: axes.Axes,
    arbor_engine: arbor.TorchArborEngine,
    nodes_energy: torch.Tensor,
) -> PathCollection:
    plt_axes.clear()
    tree_zlimit = 0
    points_x = []
    points_y = []
    points_z = []
    energys = []
    for branch_idx in range(arbor_engine.num_branches):
        birth_step = int(arbor_engine.branch_birth_hist[branch_idx].cpu().numpy())
        nodes_pos = (
            arbor_engine.nodes_state[
                birth_step : arbor_engine.steps + 1, branch_idx, 5:8
            ]
            .cpu()
            .numpy()
        )
        if len(nodes_pos) == 0:
            continue
        points_x.append(nodes_pos[:, 0])
        points_y.append(nodes_pos[:, 1])
        points_z.append(nodes_pos[:, 2])

        energys.append(
            nodes_energy[birth_step : arbor_engine.steps + 1, branch_idx].cpu().numpy()
        )
        tree_zlimit = max(tree_zlimit, np.max(np.abs(nodes_pos[:, 1])) * 1.1)
    points_x = np.concatenate(points_x)
    points_y = np.concatenate(points_y)
    points_z = np.concatenate(points_z)
    energys = np.concatenate(energys).astype(np.float16)
    negtive_energy_nodes_idx = np.where(energys <= 0)[0]
    p = plt_axes.scatter(
        points_x,
        points_z,
        points_y,
        c=energys,
        vmin=0,
        vmax=1,
        cmap="viridis",
        s=160,  # type: ignore
        alpha=1.0,
    )
    plt_axes.plot(
        points_x[negtive_energy_nodes_idx],
        points_z[negtive_energy_nodes_idx],
        points_y[negtive_energy_nodes_idx],
        "o",
        markersize=16,
        c="r",
    )
    plt_axes.set_aspect("equal", adjustable="box")
    plt_axes.set_zlim(-0.5, tree_zlimit)  # type: ignore
    return p


def matplot_tree_buffer_render(
    plt_axes: axes.Axes, arbor_engine: arbor.TorchArborEngine
):
    plt_axes.clear()
    import io

    f = plt.figure(dpi=256)
    tree_ax = f.add_subplot(
        111,
        projection="3d",
    )  # type: ignore
    # nodes_ax = f.add_subplot(122, projection="3d")  # type: ignore
    # matplot_tree(tree_ax, env_wrapper.env.arbor_engine)  # type: ignore
    # f.canvas.draw()
    io_buf = io.BytesIO()
    f.savefig(io_buf, format="raw", dpi=128)
    io_buf.seek(0)
    img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
    io_buf.close()
    w, h = f.canvas.get_width_height()
    im = img_arr.reshape((int(h), int(w), -1)).astype(np.float32) / 255.0
    return im
