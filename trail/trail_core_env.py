import os, sys

from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src/")
import random


import numpy as np
import plotly.graph_objects as go

import rlvortex.envs.base_env as base_env
import rlvortex

import utils
import tree_envs

random.seed(11)
np.random.seed(11)


if __name__ == "__main__":

    env_wrapper: base_env.BaseEnvTrait = rlvortex.envs.base_env.EnvWrapper(
        env=tree_envs.CoreTreeEnv(
            max_grow_steps=20,
            max_outer_nodes_num=200,
            num_growth_per_node=20,
            init_dis=0.5,
            delta_dis_range=np.array([-0.1, 1.0]),
            delta_rotate_range=np.array([[-10, 10], [-30, 30], [-10, 10]]),
            init_branch_rot=30,
            branch_rot_range=np.array([-10, 10]),
            branch_prob_range=np.array([0.1, 0.5]),
            sleep_prob_range=np.array([0.001, 0.01]),
            collision_space_interval=0.1,
            collision_space_half_size=200,
            shadow_space_interval=0.2,
            shadow_space_half_size=100,
            shadow_pyramid_half_size=20,
            delta_shadow_value=0.1,
            init_energy=1000,
            energy_absorption_ratio=0.001,
            branch_extension_consumption_factor=50,
            new_branch_consumption=20,
            node_maintainence_consumption=100,
        )
    )
    energy_hist = []
    env_wrapper.awake()
    o = env_wrapper.reset()
    for i in range(100):
        a = env_wrapper.sample_action()
        print(f"total energy at {i} : {env_wrapper.env.arbor_engine.total_energy:.2f}")  # type: ignore
        energy_hist.append(env_wrapper.env.arbor_engine.total_energy)
        o, r, d, _ = env_wrapper.step(a)
        if d:
            break
    env_wrapper.render()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(energy_hist))),
            y=energy_hist,
            name="energy history",  # Style name/legend entry with html tags
        )
    )

    fig.show()
    # env.destory()
