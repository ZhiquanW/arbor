import os, sys

from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src/")
import random
import torch
import numpy as np
import plotly.graph_objects as go


import utils.utils as utils
import envs.tree_envs as tree_envs
import sim.torch_arbor as arbor
import utils.render as render
import rlvortex.envs.base_env as base_env

random.seed(11)
np.random.seed(11)


if __name__ == "__main__":
    device = torch.device("cpu")
    arbor_engine = arbor.TorchArborEngine(device = device)
    env_wrapper: base_env.EnvWrapper = base_env.EnvWrapper(env=tree_envs.BranchProbArborEnv(arbor_engine=arbor_engine)).awake()
    env_wrapper.check(True)
