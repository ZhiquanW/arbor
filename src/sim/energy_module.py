import torch
import sim.aux_space as aux_space


class EnergyModule:
    def __init__(
        self,
        init_energy: float,
        max_energy: float,
        collection_ratio: float,
        collection_decay: float,
        maintainence_consumption_factor: float,
        move_consumption_factor: float,
        branch_consumption_factor: float,
    ) -> None:
        self.init_energy: float = init_energy
        self.max_energy: float = max_energy
        self.collection_ratio: float = collection_ratio
        self.collection_decay: float = collection_decay
        self.maintainence_consumption_factor: float = maintainence_consumption_factor
        self.move_consumption_factor: float = move_consumption_factor
        self.branch_consumption_factor: float = branch_consumption_factor
        self.total_energy: float = 0

    def compute_energy_collection(
        self, node_positions: torch.Tensor, shadow_space: aux_space.TorchShadowSpace
    ) -> None:
        pass

    # def absorb_energy(self,nodes_state:torch.Tensor,)
