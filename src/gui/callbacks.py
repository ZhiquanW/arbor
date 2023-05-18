import os
import sys
import io

import numpy as np
import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
import rlvortex.envs.base_env as base_env
import rlvortex

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import render
import tags
import sim.torch_arbor as arbor
import tree_envs
import energy_management

import torch


def render_skeleton(sender, app_data, user_data):
    env_wrapper, f, tree_ax, dpi, regenerate = user_data
    tree_ax.set_box_aspect(None, zoom=dpg.get_value(tags.Drag.camera_zoom))
    tree_ax.view_init(
        dpg.get_value(tags.Drag.camera_elevation),
        dpg.get_value(tags.Drag.camera_azimuthal),
        0,
    )
    if regenerate:
        env_wrapper.reset()
        env_wrapper.step(env_wrapper.sample_action())
        for i in range(1000):
            _, _, done, _ = env_wrapper.step(env_wrapper.sample_action())
            if done:
                break
    render.matplot_tree(tree_ax, env_wrapper.env.arbor_engine)  # type: ignore
    io_buf = io.BytesIO()
    f.savefig(io_buf, transparent=True, format="raw", dpi=dpi)
    io_buf.seek(0)
    img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
    io_buf.close()
    w, h = f.canvas.get_width_height()
    im = img_arr.reshape((int(h), int(w), -1)).astype(np.float32) / 255.0
    dpg.set_value(tags.Texture.render, im)


def render_energy(sender, app_data, user_data):
    _, f, energy_axes, dearbor, dpi, regenerate = user_data
    energy_axes.set_box_aspect(None, zoom=dpg.get_value(tags.Drag.camera_zoom))
    energy_axes.view_init(
        dpg.get_value(tags.Drag.camera_elevation),
        dpg.get_value(tags.Drag.camera_azimuthal),
        0,
    )
    if dearbor.colorbar_p is not None:
        dearbor.colorbar_p.remove()
    if regenerate:
        dearbor.nodes_energy = energy_management.compute_energy_v0(
            dearbor.env_wrapper.env.arbor_engine,
            dpg.get_value(tags.Slider.node_energy_collection_ratio),
            dpg.get_value(tags.Slider.energy_backpropagate_decay),
            dpg.get_value(tags.Drag.node_alive_energy_consumption),
        )
    p = render.matplot_tree_energy(
        energy_axes,
        dearbor.env_wrapper.env.arbor_engine,
        dearbor.nodes_energy,
    )
    dearbor.colorbar_p = f.colorbar(
        p,
        ax=energy_axes,
        orientation="vertical",
        label="labelname",
    )
    io_buf = io.BytesIO()
    f.savefig(
        io_buf,
        transparent=True,
        format="raw",
        dpi=dpi,
    )
    io_buf.seek(0)
    img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
    io_buf.close()
    w, h = f.canvas.get_width_height()
    im = img_arr.reshape((int(h), int(w), -1)).astype(np.float32) / 255.0
    dpg.set_value(tags.Texture.render, im)


def render_all(send, app_data, user_data):
    _, f, node_axes, energy_axes, dearbor, dpi, regenerate = user_data
    if regenerate:
        arbor_engine = arbor.TorchArborEngine(
            max_steps=dpg.get_value(tags.Drag.max_steps),
            max_branches_num=dpg.get_value(tags.Drag.max_branches_num),
            move_dis_range=dpg.get_value(tags.Drag.move_distance_range)[:2],
            move_rot_range=[
                dpg.get_value(tags.Drag.move_rot_range_x)[:2],
                dpg.get_value(tags.Drag.move_rot_range_y)[:2],
                dpg.get_value(tags.Drag.move_rot_range_z)[:2],
            ],
            new_branch_rot_range=dpg.get_value(tags.Drag.new_branch_rot_range)[:2],
            node_branch_prob_range=dpg.get_value(tags.Drag.node_branch_prob_range)[:2],
            node_sleep_prob_range=dpg.get_value(tags.Drag.node_sleep_prob_range)[:2],
            device=torch.device("cpu"),
        )
        dearbor.env_wrapper: base_env.BaseEnvTrait = rlvortex.envs.base_env.EnvWrapper(
            env=tree_envs.CoreTorchEnv(arbor_engine=arbor_engine)
        )
        dearbor.env_wrapper.awake()
        dearbor.env_wrapper.reset()
    render_skeleton(None, None, (dearbor.env_wrapper, f, node_axes, dpi, regenerate))
    render_energy(
        None, None, (dearbor.env_wrapper, f, energy_axes, dearbor, dpi, regenerate)
    )
