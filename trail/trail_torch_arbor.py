import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src/")

from matplotlib import pyplot as plt

import random


import plotly.graph_objects as go

import rlvortex.envs.base_env as base_env
import rlvortex

import utils.utils as utils
import envs.tree_envs as tree_envs
import sim.torch_arbor as arbor
import torch
import utils.render as render

random.seed(11)


if __name__ == "__main__":
    arbor = arbor.TorchArborEngine(
        max_steps=20,
        max_branches_num=50,
        move_dis_range=[0.3, 0.5],
        move_rot_range=[[-1, 1], [-30, 30], [-1, 1]],
        new_branch_rot_range=[20, 40],
        node_branch_prob_range=[0.1, 0.5],
        node_sleep_prob_range=[0.001, 0.01],
        device=torch.device("cpu"),
    )
    env_wrapper: base_env.BaseEnvTrait = rlvortex.envs.base_env.EnvWrapper(
        env=tree_envs.BranchProbArborEnv(arbor_engine=arbor)
    )
    env_wrapper.awake()
    env_wrapper.reset()
    env_wrapper.step(env_wrapper.sample_action())
    for i in range(1000):
        _, _, done, _ = env_wrapper.step(env_wrapper.sample_action())
        if done:
            break
    f = plt.figure()
    tree_ax = f.add_subplot(121, projection="3d")
    nodes_ax = f.add_subplot(122, projection="3d")
    render.matplot_tree(tree_ax, env_wrapper.env.arbor_engine)
    plt.show()
    # energy_hist = []
    # env_wrapper.awake()
    # o = env_wrapper.reset()
    # f = plt.figure()
    # tree_ax = f.add_subplot(121, projection="3d")
    # nodes_ax = f.add_subplot(122, projection="3d")
    # for i in range(100):
    #     a = env_wrapper.sample_action()

    #     o, r, d, _ = env_wrapper.step(a)
    #     # plt.show()
    #     if d:
    #         plt.show()
    #         break
    # env_wrapper.render()
    # fig = go.Figure()

    # fig.add_trace(
    #     go.Scatter(
    #         x=list(range(len(energy_hist))),
    #         y=energy_hist,
    #         name="energy history",  # Style name/legend entry with html tags
    #     )
    # )

    # fig.show()
    # env.destory()
