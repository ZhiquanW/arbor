import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src/")


import numpy as np

import rlvortex.envs.base_env as base_env
import rlvortex

import tree_env
import utils
import random

# random.seed(11)
# np.random.seed(11)


def exp_env():
    env: base_env.BaseEnvTrait = rlvortex.envs.base_env.EnvWrapper(
        env=tree_env.PolyLineTreeEnv(
            max_vertex_num=2000,
            delta_dis_range=np.array([0.0, 0.4]),
            delta_rotate_range=np.array([-30, 30]),
            new_branch_rot_range=np.array([-40, 40]),
            headless=True,
        )
    )
    env.awake()
    o = env.reset()
    for _ in range(10):
        a = env.sample_action()
        env.step(a)
    # env.destory()
    raw_env_g: tree_env.PolyLineTreeEnv = env.env  # type: ignore
    raw_env_g.final_plot()
    # while True:
    #     pass


if __name__ == "__main__":
    exp_env()
