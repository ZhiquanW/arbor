import os
import torch
import envs.tree_envs as tree_envs
import sim.torch_arbor as arbor
import rlvortex.envs.base_env as base_env
from rlvortex.policy.quick_build import mlp
from rlvortex.utils import trainer_utils, vlogger
from rlvortex.policy.ppo_policy import (
    BasePPOPolicy,
    GaussianActor,
    GaussianSepActor,
    CategoricalActor,
    BaseCritic,
)
from rlvortex.trainer.ppo_trainer import NativePPOTrainer
import sim.aux_space as aux_space
import sim.energy_module as energy_module

tree_env_seed = 19970314


class BranchProbEnvSpeedTestParams:
    env_fn = tree_envs.BranchProbArborEnv
    device = torch.device("mps")
    env = base_env.EnvWrapper(
        env=env_fn(
            arbor_engine=arbor.TorchArborEngine(
                move_dis_range=[0.05, 0.1],
                max_steps=50,
                occupancy_space=aux_space.TorchOccupancySpace(
                    space_half_size=50, device=device
                ),
                shadow_space=aux_space.TorchShadowSpace(
                    space_half_size=50, device=device
                ),
                energy_module=energy_module.EnergyModule(),
                device=device,
            )
        )
    )


class BranchProbEnvParams:
    env_fn = tree_envs.BranchProbArborEnv
    device = torch.device("cpu")
    env = base_env.EnvWrapper(
        env=env_fn(
            arbor_engine=arbor.TorchArborEngine(
                max_steps=1000,
                occupancy_space=aux_space.TorchOccupancySpace(
                    space_half_size=50, device=device
                ),
                shadow_space=aux_space.TorchShadowSpace(
                    space_half_size=50, device=device
                ),
                energy_module=energy_module.EnergyModule(),
                device=device,
            )
        )
    )
    policy = BasePPOPolicy(
        actor=GaussianSepActor(
            preprocess_net=mlp(
                [*env.observation_dim, 32],
                torch.nn.Tanh,
                output_activation=torch.nn.ReLU,
            ),
            logits_net=mlp(
                [32, *env.action_dim],
                torch.nn.Tanh,
                output_activation=torch.nn.Tanh,
            ),
            log_std_net=mlp(
                [32, *env.action_dim],
                torch.nn.Tanh,
                output_activation=torch.nn.Tanh,
            ),
        ),
        critic=BaseCritic(net=mlp([*env.observation_dim, 32, 1], torch.nn.ReLU)),
    )
    optimizer = torch.optim.Adam
    init_lr = 1e-3
    gamma = 0.9
    epochs = 150
    max_grad_norm = None
    trainer = NativePPOTrainer(
        env=env,
        policy=policy,
        optimizer=optimizer,
        init_lr=init_lr,
        gamma=gamma,
        desired_kl=1e-4,
        enable_tensorboard=True,
        save_freq=50,
        log_type=vlogger.LogType.Screen,
        trainer_dir=os.path.join(os.getcwd(), "trainer_cache"),
        comment="branch_prob_env",
        seed=tree_env_seed,
        device=device,
    )
