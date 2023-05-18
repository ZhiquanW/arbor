from enum import Enum
import core
import sim.torch_arbor as torch_arbor
import numpy as np
import torch


def compute_energy_v0(
    arbor: torch_arbor.TorchArborEngine,
    node_energy_collection_ratio: float,
    energy_backpropagation_decay: float,
    node_alive_energy_consumption: float,
):
    nodes_energy = torch.zeros(
        (arbor.max_steps + 1, arbor.max_branches_num), device=arbor.device
    )
    nodes_exist_mat = arbor.nodes_state[..., 0]

    end_nodes_idx = arbor.get_end_nodes_idx()
    nodes_energy[end_nodes_idx] = (
        node_energy_collection_ratio - node_alive_energy_consumption
    )
    for step in reversed(range(1, arbor.max_steps + 1)):
        parents_idx = arbor.nodes_parent_idx(step)
        existing_nodes_idx = arbor.existing_nodes_idx(step)

        backpropagated_energy = (1 - energy_backpropagation_decay) * nodes_energy[
            step, existing_nodes_idx
        ]
        backpropagated_energy = torch.where(
            backpropagated_energy < 0, 0, backpropagated_energy
        )
        nodes_energy[step - 1, parents_idx] = (
            backpropagated_energy - node_alive_energy_consumption
        )
    nodes_energy *= nodes_exist_mat
    return nodes_energy
