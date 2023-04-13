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
import core

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
            branch_prob_range=np.array([0.0001, 0.001]),
            sleep_prob_range=np.array([0.001, 0.01]),
            collision_space_interval=0.1,
            collision_space_half_size=200,
            shadow_space_interval=0.2,
            shadow_space_half_size=100,
            shadow_pyramid_half_size=20,
            delta_shadow_value=0.1,
            energy_mode=core.EnergyMode.node,
            init_energy=1,
            max_energy=100,
            init_energy_collection_rate=1,
            energy_collection_rate_decay=core.EnergyCollectionDecay.linear,
            node_moving_consumption_factor=0.01,
            node_generation_consumption=0.1,
            node_maintainence_consumption=0.01,
        )
    )
    energy_hist = []
    env_wrapper.awake()
    o = env_wrapper.reset()
    for i in range(100):
        a = env_wrapper.sample_action()
        print(f"total energy at {i} : {env_wrapper.env.arbor_engine.total_energy:.2f}")  # type: ignore
        energy_hist.append(env_wrapper.env.arbor_engine.total_energy)  # type: ignore
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
