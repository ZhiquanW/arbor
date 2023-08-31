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
import sim.torch_arbor as torch_arbor
import core
import render

random.seed(11)
np.random.seed(11)


if __name__ == "__main__":
    env_wrapper: base_env.BaseEnvTrait = rlvortex.envs.base_env.EnvWrapper(
        env=tree_envs.CoreTorchEnv(
            arbor_engine=torch_arbor.TorchArborEngine(),
        )
    )
    energy_hist = []
    env_wrapper.awake()
    o = env_wrapper.reset()
    for i in range(100):
        a = env_wrapper.sample_action()
        o, r, d, _ = env_wrapper.step(a)
        if d:
            break

    fig = render.plotly_tree_skeleton(None, env_wrapper.env.arbor_engine)
    fig.show()
    print("done")
