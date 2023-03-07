import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src/")

import numpy as np
import matplotlib.pyplot as plt
import core

if __name__ == "__main__":
    print("run trail core")
    arbor_engine = core.ArborEngine(
        max_grow_steps=20,
        max_bud_num=200,
        num_growth_per_bud=20,
        init_dis=0.5,
        delta_dis_range=np.array([-0.1, 1.0]),
        delta_rotate_range=np.array([[-10, 10], [-20, 20], [-10, 10]]),
        init_branch_rot=30,
        branch_rot_range=np.array([-10, 10]),
        branch_prob_range=np.array([0.1, 0.5]),
        sleep_prob_range=np.array([0.001, 0.01]),
        collision_space_interval=0.1,
        collision_space_half_size=200,
    )
    for _ in range(100):
        a = arbor_engine.sample_action()
        done = arbor_engine.step(a)
        if done:
            break
    f = plt.figure()
    tree_ax = f.add_subplot(121, projection="3d")
    col_ax = f.add_subplot(122, projection="3d")
    arbor_engine.matplot_tree(tree_ax)
    arbor_engine.matplot_collision(col_ax)
    plt.show()
