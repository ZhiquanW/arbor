import os, sys
from typing import NamedTuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src/")

from collections import namedtuple

import numpy as np

import rlvortex.envs.base_env as base_env
import rlvortex

import tree_envs
import utils
import random

random.seed(11)
np.random.seed(11)


def plot_tree(params: NamedTuple, param_type: str, idx: int = 0):
    env: base_env.BaseEnvTrait = rlvortex.envs.base_env.EnvWrapper(env=tree_envs.PolyLineTreeEnv(*params))
    env.awake()
    o = env.reset()
    for _ in range(1000):
        a = env.sample_action()
        o, r, d, _ = env.step(a)
        if d:
            break
    raw_env_g: tree_envs.PolyLineTreeEnv = env.env  # type: ignore
    raw_env_g.final_plot(name=param_type + str(idx), num_frame=600)
    env.destory()


if __name__ == "__main__":
    num_tree_per_congif = 8
    TreeEnvParams = namedtuple(
        "TreeEnvParams",
        [
            "max_grow_steps",
            "max_bud_num",
            "num_growth_per_bud",
            "init_dis",
            "delta_dis_range",
            "delta_rotate_range",
            "init_branch_rot",
            "branch_rot_range",
            "branch_prob_range",
            "sleep_prob_range",
            "matplot",
            "headless",
        ],
    )
    verification_params = TreeEnvParams(
        max_grow_steps=20,
        max_bud_num=200,
        num_growth_per_bud=20,
        init_dis=0.5,
        delta_dis_range=np.array([-0.2, 0.2]),
        delta_rotate_range=np.array([-1, 1]),
        init_branch_rot=90,
        branch_rot_range=np.array([-1, 1]),
        branch_prob_range=np.array([0.1, 0.5]),
        sleep_prob_range=np.array([0.001, 0.01]),
        matplot=True,
        headless=False,
    )
    default_params = TreeEnvParams(
        max_grow_steps=20,
        max_bud_num=200,
        num_growth_per_bud=20,
        init_dis=0.5,
        delta_dis_range=np.array([-0.2, 0.2]),
        delta_rotate_range=np.array([-10, 10]),
        init_branch_rot=30,
        branch_rot_range=np.array([-10, 10]),
        branch_prob_range=np.array([0.1, 0.5]),
        sleep_prob_range=np.array([0.001, 0.01]),
        matplot=True,
        headless=False,
    )
    morebuds_params = TreeEnvParams(
        max_grow_steps=200,
        max_bud_num=2000,
        num_growth_per_bud=60,
        init_dis=0.5,
        delta_dis_range=np.array([-0.2, 0.2]),
        delta_rotate_range=np.array([-10, 10]),
        init_branch_rot=30,
        branch_rot_range=np.array([-10, 10]),
        branch_prob_range=np.array([0.1, 0.5]),
        sleep_prob_range=np.array([0.001, 0.01]),
        matplot=True,
        headless=False,
    )
    up_params = TreeEnvParams(
        max_grow_steps=20,
        max_bud_num=200,
        num_growth_per_bud=20,
        init_dis=0.5,
        delta_dis_range=np.array([-0.2, 0.2]),
        delta_rotate_range=np.array([-10, 10]),
        init_branch_rot=1,
        branch_rot_range=np.array([-1, 1]),
        branch_prob_range=np.array([0.1, 0.5]),
        sleep_prob_range=np.array([0.001, 0.01]),
        matplot=True,
        headless=False,
    )

    run_params = {"default": default_params, "verif": verification_params, "morebuds": morebuds_params, "up": up_params}
    num_tree_per_params = 9
    for param_name, params in run_params.items():
        print(param_name)
        for i in range(num_tree_per_params):
            plot_tree(params, param_name, i)
