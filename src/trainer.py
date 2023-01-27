# sys imports
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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

# core imports

# if __name__ == "__main__":
#     vertex_num = 8
#     grid_size = 0.1
#     env: EnvWrapper = EnvWrapper(
#         env=RectangleEnv(
#             radius=2,
#             vertex_num=vertex_num,
#             target_center=np.array([0, 0]),
#             target_size=np.array([7, 5]),
#             grid_size=grid_size,
#             polygon_cross_stop=True,
#             headless=True,
#         )
#     )
#     policy = BasePPOPolicy(
#         actor=GaussianActor(
#             net=mlp([*env.observation_dim, 128, 128, *env.action_dim], torch.nn.Tanh),
#             init_log_stds=-0.5 * torch.ones(env.action_dim),
#         ),
#         critic=BaseCritic(net=mlp([*env.observation_dim, 128, 128, 1], torch.nn.Tanh)),
#     )
#     steps_per_env = 512
#     num_batches_per_env = 2
#     learning_iterations = 32
#     val_loss_coef = 1.0
#     init_lr = 3e-4
#     random_sampler = True
#     normalize_adv = True
#     optimizer = torch.optim.Adam
#     desired_kl = 0.01
#     epochs = 200
#     trainer = NativePPOTrainer(
#         env=env,
#         policy=policy,
#         optimizer=optimizer,
#         steps_per_env=steps_per_env,
#         num_batches_per_env=num_batches_per_env,
#         learning_iterations=learning_iterations,
#         val_loss_coef=val_loss_coef,
#         init_lr=init_lr,
#         desired_kl=desired_kl,
#         random_sampler=random_sampler,
#         normalize_adv=normalize_adv,
#         enable_tensorboard=True,
#         save_freq=10,
#         log_type=vlogger.LogType.Screen,
#         trainer_dir=os.path.join(os.getcwd(), "trainer_cache"),
#         comment="rect_env",
#     )

#     train_batch = 20
#     sub_steps = int(epochs // train_batch)
#     trainer.evaluate(
#         1,
#         env=RectangleEnv(
#             radius=2,
#             vertex_num=vertex_num,
#             target_center=np.array([0, 0]),
#             target_size=np.array([7, 5]),
#             grid_size=grid_size,
#             polygon_cross_stop=True,
#             headless=False,
#         ),
#     )
#     for _ in range(train_batch):
#         trainer.train(sub_steps)
#         ep_rtn, ep_mean = trainer.evaluate(
#             1,
#             env=RectangleEnv(
#                 radius=2,
#                 vertex_num=vertex_num,
#                 target_center=np.array([0, 0]),
#                 target_size=np.array([7, 5]),
#                 grid_size=grid_size,
#                 polygon_cross_stop=True,
#                 headless=False,
#             ),
#         )
#         print("ep_rtn:", ep_rtn, "ep_mean:", ep_mean)
#     trainer.evaluate(
#         -1,
#         env=RectangleEnv(
#             radius=2,
#             vertex_num=vertex_num,
#             target_center=np.array([0, 0]),
#             target_size=np.array([7, 5]),
#             grid_size=grid_size,
#             polygon_cross_stop=True,
#             headless=False,
#         ),
#     )
