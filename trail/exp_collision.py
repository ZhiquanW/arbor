import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src/")


import numpy as np

import rlvortex.envs.base_env as base_env
import rlvortex

import tree_envs
import utils
import random

random.seed(11)
np.random.seed(11)


def collision_occupy():
    occ_interval = 0.1
    occ_space = np.zeros((40, 40, 40))
    pos = np.array([[0, 1, 1], [1, 1, 1], [-1, 0.2, 0.4]])
    idx = (pos / occ_interval).astype(np.int32)
    occ_space[tuple(idx.transpose())] += 1

    print(idx)
    print(occ_space[0, 10, 10])
    print(occ_space[0, 10, 11])
    print(occ_space[0, 10, 9])
    print(occ_space[10, 10, 10])
    print(occ_space[-10, 2, 4])
    print(occ_space[11, 10, 10])
    print(occ_space[10, 10, 11])
    elements = list((np.arange(2 * 40) - 40 + 1) / occ_interval)
    elements = [0, 1, 2, 3]
    xv, yv, zv = np.meshgrid(elements, elements, elements)
    print("*" * 20)
    print(xv.flatten())
    print("*" * 20)
    print(yv.flatten())
    print("*" * 20)
    print(zv.flatten())


if __name__ == "__main__":
    # collision_occupy()

    env: base_env.BaseEnvTrait = rlvortex.envs.base_env.EnvWrapper(
        env=tree_envs.PolyLineTreeEnv(
            max_grow_steps=20,
            max_bud_num=200,
            num_growth_per_bud=20,
            init_dis=0.5,
            delta_dis_range=np.array([-0.1, 1.0]),
            delta_rotate_range=np.array([-10, 10]),
            init_branch_rot=30,
            branch_rot_range=np.array([-10, 10]),
            branch_prob_range=np.array([0.1, 0.5]),
            sleep_prob_range=np.array([0.001, 0.01]),
            collision_space_interval=0.01,
            collision_space_half_size=500,
            matplot=True,
            headless=False,
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
    # env.destory()
    raw_env_g: tree_envs.PolyLineTreeEnv = env.env  # type: ignore
    raw_env_g.final_plot(interactive=True)
    # while True:
