"""
this file is a experimental file for training the policy 
to learn the branch probability for energy collection target.
"""
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))
import copy
import torch
import rlvortex.envs.base_env as base_env
import rlvortex.utils.vlogger as vlogger
from rlvortex.policy.ppo_policy import (
    BasePPOPolicy,
    GaussianActor,
    BaseCritic,
)
from rlvortex.policy.quick_build import mlp
from rlvortex.trainer.ppo_trainer import NativePPOTrainer

import sim.torch_arbor as arbor
import envs.tree_envs as tree_envs
import train.trainer_params as trainer_params

import time
import tqdm


def main():
    start_time = time.time()
    test_steps = 1000
    env = trainer_params.BranchProbEnvSpeedTestParams.env
    env.awake()
    o, _ = env.reset()
    for i in tqdm.tqdm(range(test_steps)):
        a = env.sample_action()
        o, r, d, _ = env.step(a)
        if d:
            o, _ = env.reset()
    running_time = time.time() - start_time
    print(running_time)


if __name__ == "__main__":
    main()
