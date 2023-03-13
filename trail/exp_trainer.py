# sys imports
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src/")
# fundamental imports
import numpy as np
import torch

# rlvortex imports
from rlvortex.envs.base_env import BaseEnvTrait, EnvWrapper
from rlvortex.policy.ppo_policy import BasePPOPolicy
from rlvortex.policy.ppo_policy import (
    BasePPOPolicy,
    GaussianActor,
    GaussianSepActor,
    CategoricalActor,
    BaseCritic,
)
from rlvortex.policy.quick_build import mlp
from rlvortex.trainer.ppo_trainer import NativePPOTrainer
from rlvortex.utils import trainer_utils, vlogger
import rlvortex
import rlvortex.envs.base_env as base_env

# core imports
import tree_envs
import random
import torch.nn as nn

random.seed(315)
np.random.seed(315)
torch.manual_seed(315)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(1600, 512)
        self.act_1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 64)
        self.act_2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 1200)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.act_1(out)
        # out = self.layer2(out)
        # out = self.act_2(out)
        out = self.layer3(out)
        # print(torch.max(out), torch.min(out))
        return out


def make_env(headless=True):
    grow_steps = 15
    max_bud_num = 200
    num_growth_per_bud = 20
    init_dis = 0.5
    delta_dis_range = np.array([0.0, 0.4])
    delta_rotate_range = np.array([-20, 20])
    init_branch_rot = 20
    branch_rot_range = np.array([-5, 5])
    branch_prob_range = np.array([0.0, 1.0])
    sleep_prob_range = np.array([0.001, 1.0])
    matplot = True
    env: base_env.BaseEnvTrait = rlvortex.envs.base_env.EnvWrapper(
        env=tree_envs.PolyLineTreeEnv(
            max_grow_steps=grow_steps,
            max_bud_num=max_bud_num,
            num_growth_per_bud=num_growth_per_bud,
            init_dis=init_dis,
            delta_dis_range=delta_dis_range,
            delta_rotate_range=delta_rotate_range,
            init_branch_rot=init_branch_rot,
            branch_rot_range=branch_rot_range,
            branch_prob_range=branch_prob_range,
            sleep_prob_range=sleep_prob_range,
            matplot=matplot,
            headless=headless,
        )
    )
    return env


if __name__ == "__main__":
    env: base_env.BaseEnvTrait = make_env(True)
    policy = BasePPOPolicy(
        actor=GaussianActor(
            net=mlp([*env.observation_dim, 256, 256, *env.action_dim], torch.nn.ReLU),
            init_log_stds=-0.5 * torch.ones(env.action_dim),
        ),
        critic=BaseCritic(net=mlp([*env.observation_dim, 256, 256, 1], torch.nn.ReLU)),
    )
    steps_per_env = 1024
    num_batches_per_env = 4
    learning_iterations = 32
    val_loss_coef = 1.0
    init_lr = 3e-4
    random_sampler = True
    normalize_adv = False
    optimizer = torch.optim.Adam
    desired_kl = None
    epochs = 200
    clip_ratio = 0.05
    trainer = NativePPOTrainer(
        env=env,
        policy=policy,
        optimizer=optimizer,
        steps_per_env=steps_per_env,
        num_batches_per_env=num_batches_per_env,
        learning_iterations=learning_iterations,
        val_loss_coef=val_loss_coef,
        init_lr=init_lr,
        desired_kl=desired_kl,
        clip_ratio=clip_ratio,
        random_sampler=random_sampler,
        normalize_adv=normalize_adv,
        enable_tensorboard=True,
        save_freq=10,
        log_type=vlogger.LogType.Screen,
        trainer_dir=os.path.join(os.getcwd(), "trainer_cache"),
        comment="rect_env",
        device_id=-1,
    )

    train_batch = 20
    sub_steps = int(epochs // train_batch)
    trainer.evaluate(1, make_env(False))
    print("start training")
    for _ in range(train_batch):
        trainer.train(sub_steps)
        ep_rtn, ep_mean = trainer.evaluate(1, env=make_env(True))
        print("ep_rtn:", ep_rtn, "ep_mean:", ep_mean)
    trainer.evaluate(-1, env=make_env(False))
