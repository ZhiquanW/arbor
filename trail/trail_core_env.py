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

    env_wrapper: base_env.BaseEnvTrait = rlvortex.envs.base_env.EnvWrapper(
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
            collision_space_interval=0.1,
            collision_space_half_size=200,
            shadow_space_interval=0.2,
            shadow_space_half_size=100,
            shadow_pyramid_half_size=20,
            delta_shadow_value=0.1,
            init_energy=10,
            branch_extension_consumption_factor=0.1,
            new_branch_consumption=0.5,
        )
    )
    env_wrapper.awake()
    o = env_wrapper.reset()
    for _ in range(100):
        a = env_wrapper.sample_action()
        print(env_wrapper.env.arbor_engine.total_energy)  # type: ignore
        o, r, d, _ = env_wrapper.step(a)
        if d:
            break
    env_wrapper.render()
    # env.destory()
