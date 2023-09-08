"""
this file is a experimental file for training the policy 
to learn the branch probability for energy collection target.
"""
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
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
import train.trainer_params


def main():
    train_batch = 50
    trainer_params = train.trainer_params.BranchProbEnvParams
    trainer: NativePPOTrainer = trainer_params.trainer
    print("env observation dim:", trainer_params.env.observation_dim)
    print("env action dim:", trainer_params.env.action_dim)
    print("policy", trainer_params.policy.actor)
    eva_env = copy.deepcopy(trainer_params.env)
    sub_steps = int(trainer_params.epochs // train_batch)
    for i in range(train_batch):
        print("batch idx:", i)
        trainer.train(sub_steps)
        ep_rtn, ep_mean = trainer.evaluate(
            10,
        )
        print("evalution: ep_rtn:", ep_rtn, "ep_mean:", ep_mean)


if __name__ == "__main__":
    main()
