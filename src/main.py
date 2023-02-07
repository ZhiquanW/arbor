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
            max_bud_num=500,
            delta_dis_range=np.array([0.0, 0.4]),
            delta_rotate_range=np.array([-20, 20]),
            new_branch_rot_range=np.array([-40, 40]),
            branch_prob_range=np.array([0.7, 0.9]),
            sleep_prob_range=np.array([0.0, 0.001]),
            headless=True,
        )
    )
    env.awake()
    o = env.reset()
    for _ in range(10):
        a = env.sample_action()
        o, r, d, _ = env.step(a)
        print(o.shape)
    # env.destory()
    raw_env_g: tree_env.PolyLineTreeEnv = env.env  # type: ignore
    raw_env_g.final_plot()
    # while True:
    #     pass


if __name__ == "__main__":
    exp_env()
