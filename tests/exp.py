import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src/")


import numpy as np

import rlvortex.envs.base_env as base_env
import rlvortex

import tree_env
import utils
import random

random.seed(11)
np.random.seed(11)


def exp_env():
    env: base_env.BaseEnvTrait = rlvortex.envs.base_env.EnvWrapper(
        env=tree_env.PolyLineTreeEnv(
            max_vertex_num=2000,
            delta_dis_range=np.array([0.0, 0.4]),
            delta_rotate_range=np.array([-10, 10]),
            new_branch_rot_range=np.array([-20, 20]),
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


def exp_others():
    a = np.arange(3 * 4 * 5).reshape(3, -1)
    print(a)
    b = np.where(a[:, 0] > 10)[0]
    print(b)
    c = a[b]
    print(c)
    # b = a[:2, ...]
    # c = np.zeros_like(b)
    # b[:] = c
    # print(a)


def exp_np_cross():
    print()
    a = np.repeat(np.array([[0, 0, 1]]), 7, axis=0)
    a[1, :] = [0, 1, 0]
    b = np.repeat(np.array([[1, 0, 0]]), 7, axis=0)
    print(a)
    print(b)
    print(np.cross(a, b))


def exp_rot_around_axis():
    vec = np.array([1, 0, 0])
    axis = np.array([0, 1, 0])
    degrees = np.ndarray([0, 30, 60, 60])
    print(vec)
    for _ in range(3):
        rot_mat = utils.rot_mat_from_axis_angles(axis, degrees)
        vec = rot_mat @ vec
        print(vec)


def exp_indices_in_indices():
    a = np.zeros((17, 4))
    a[5, 0] = 1
    a[15, 0] = 1
    a[2, 0] = 1
    a[7, 0] = 1
    a[9, 0] = 1
    a[10, 0] = 1
    print(a[5:10, 0])
    act_indices = np.where(a[5:10, 0] == 1)[0]
    print(act_indices)
    act_mat = a[act_indices, :]
    print(act_mat)


def exp_will_grow():
    a = np.zeros((17, 4))
    a[5, 0] = 1
    a[15, 0] = 1
    a[2, 0] = 1
    a[7, 0] = 1
    a[9, 0] = 1
    a[10, 0] = 1
    act_indices = np.where(a[:, 0] == 1)[0]
    print(a[act_indices, :])
    grow_indices = act_indices[np.where(np.random.uniform(0, 1, len(act_indices)) < 0.5)[0]]
    print(grow_indices)
    print(a[grow_indices, :])
    a[grow_indices, 0] = -1
    print(a)


def exp_arange():
    a = np.arange(4, 10)
    b = np.arange(17 * 2).reshape(17, -1)
    print(b)
    print(b[a])


def exp_float_mut_mat_by_row():
    a = np.arange(3 * 4).reshape(3, -1)
    b = np.arange(3).reshape(3, 1)
    print(a)
    print(b)
    print(a * b)


def exp_indices():
    a = np.arange(3 * 4).reshape(3, -1)
    b = np.where(a[:, 0] > 10)[0]
    print(a[b])


if __name__ == "__main__":
    exp_env()
    # exp_indices()

    # exp_others()
    # exp_np_cross()
    # exp_rot_around_axis()
    # exp_indices_in_indices()
    # exp_will_grow()
    # exp_arange()
    # exp_float_mut_mat_by_row()
