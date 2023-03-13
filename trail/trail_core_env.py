import os, sys

from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src/")
import random


import numpy as np

import rlvortex.envs.base_env as base_env
import rlvortex

import utils
import tree_envs

random.seed(11)
np.random.seed(11)


if __name__ == "__main__":

    env: base_env.BaseEnvTrait = rlvortex.envs.base_env.EnvWrapper(
        env=tree_envs.CoreTreeEnv(
            max_grow_steps=20,
            max_bud_num=200,
            num_growth_per_bud=20,
            init_dis=0.5,
            delta_dis_range=np.array([-0.1, 1.0]),
            delta_rotate_range=np.array([[-10, 10], [-30, 30], [-10, 10]]),
            init_branch_rot=30,
            branch_rot_range=np.array([-10, 10]),
            branch_prob_range=np.array([0.1, 0.5]),
            sleep_prob_range=np.array([0.001, 0.01]),
            voxel_space_interval=0.1,
            voxel_space_half_size=200,
        )
    )
    env.awake()
    o = env.reset()
    for _ in range(100):
        a = env.sample_action()
        o, r, d, _ = env.step(a)
        if d:
            print("Done")
            env.render()
            # plt.show()
            env.reset()
            input()
    plt.show()
    # env.destory()
