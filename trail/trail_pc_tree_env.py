import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src/")

import random

import numpy as np
import rlvortex.envs.base_env as base_env
import rlvortex

import core

import pc_tree_env

random.seed(11)
np.random.seed(11)
import matplotlib.pyplot as plt

if __name__ == "__main__":

    env: base_env.BaseEnvTrait = rlvortex.envs.base_env.EnvWrapper(
        env=pc_tree_env.PointCloudSingleTreeEnv(
            point_cloud_mat=pc_tree_env.load_pointcloud(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/raw_points.ply")
            ),
            max_grow_steps=25,
            max_bud_num=200,
            num_growth_per_bud=25,
            init_dis=0.5,
            delta_dis_range=np.array([-0.1, 1.0]),
            delta_rotate_range=np.array([-10, 10]),
            init_branch_rot=30,
            branch_rot_range=np.array([-10, 10]),
            branch_prob_range=np.array([0.1, 0.5]),
            sleep_prob_range=np.array([0.001, 0.01]),
            collision_space_interval=0.01,
            collision_space_half_size=100,
        )
    )
    env.awake()
    o = env.reset()
    for _ in range(10):
        a = env.sample_action()
        o, r, d, _ = env.step(a)
        env.render()
        if d:
            break
    plt.show()
