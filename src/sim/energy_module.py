from typing import List, Optional
import torch
import sim.aux_space as aux_space


class EnergyHist:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_energys: List[float] = []
        self.energy_balance: List[float] = [0]
        self.collected_energys: List[float] = []
        self.maintainence_consumptions: List[float] = []
        self.move_consumptions: List[float] = []
        self.branch_consumptions: List[float] = []


class EnergyModule:
    def __init__(
        self,
        init_energy: float = 10.0,
        max_energy: float = 100.0,
        collection_voxel_half_size: int = 4,
        init_collection_ratio: float = 0.8,
        maintainence_consumption_factor: float = 0.1,
        move_consumption_factor: float = 0.5,
        branch_consumption_factor: float = 1.0,
        record_history: bool = False,
    ) -> None:
        self.init_energy = init_energy
        self.max_energy: float = max_energy
        self.collection_voxel_half_size = collection_voxel_half_size
        self.init_collection_ratio: float = init_collection_ratio
        self.maintainence_consumption_factor: float = maintainence_consumption_factor
        self.move_consumption_factor: float = move_consumption_factor
        self.branch_consumption_factor: float = branch_consumption_factor
        self.total_energy: float = init_energy
        self.energy_hist: Optional[EnergyHist] = (
            EnergyHist() if record_history else None
        )

    def reset(self) -> None:
        self.total_energy = self.init_energy
        if self.energy_hist is not None:
            self.energy_hist.reset()

    def colelct_energy(
        self, nodes_position: torch.Tensor, shadow_space: aux_space.TorchShadowSpace
    ) -> None:
        node_voxels_idx = shadow_space.positions_to_voxels_idx(nodes_position)
        energy_collection = 0
        for voxel_index in node_voxels_idx:
            energy_collection += torch.mean(
                1
                - shadow_space.space[
                    voxel_index[0]
                    - self.collection_voxel_half_size
                    + 1 : voxel_index[0]
                    + self.collection_voxel_half_size,
                    voxel_index[1]
                    - self.collection_voxel_half_size
                    + 1 : voxel_index[1]
                    + self.collection_voxel_half_size,
                    voxel_index[2]
                    - self.collection_voxel_half_size
                    + 1 : voxel_index[2]
                    + 1,
                ]
            ).item()
        energy_fullfill_ratio = self.total_energy / self.max_energy
        collection_ratio = self.init_collection_ratio * (1 - energy_fullfill_ratio)
        collected_energy = collection_ratio * energy_collection
        if self.energy_hist is not None:
            self.energy_hist.collected_energys.append(collected_energy)
            self.energy_hist.energy_balance[-1] += collected_energy
            self.energy_hist.energy_balance.append(0)
        self.total_energy += collected_energy
        self.total_energy = min(self.max_energy, self.total_energy)
        if self.energy_hist is not None:
            self.energy_hist.total_energys.append(self.total_energy)

    def maintainence_consumption(self, num_active_nodes: int) -> bool:
        maintainence_consumption = (
            self.maintainence_consumption_factor * num_active_nodes
        )
        if self.energy_hist is not None:
            self.energy_hist.maintainence_consumptions.append(maintainence_consumption)
            self.energy_hist.energy_balance[-1] -= maintainence_consumption
        self.total_energy -= maintainence_consumption
        if self.total_energy < 0 and self.energy_hist is not None:
            self.energy_hist.total_energys.append(self.total_energy)
            return False
        return True

    def move_consumption_consumption(self, distances: torch.Tensor) -> bool:
        move_consumption = self.move_consumption_factor * torch.sum(distances).item()
        self.total_energy -= move_consumption
        if self.energy_hist is not None:
            self.energy_hist.move_consumptions.append(move_consumption)
            self.energy_hist.energy_balance[-1] -= move_consumption
        if self.total_energy < 0 and self.energy_hist is not None:
            self.energy_hist.total_energys.append(self.total_energy)
            return False
        return True

    def branch_consumption(self, num_new_branches: int) -> bool:
        branch_consumption = self.branch_consumption_factor * num_new_branches
        if self.energy_hist is not None:
            self.energy_hist.branch_consumptions.append(branch_consumption)
            self.energy_hist.energy_balance[-1] -= branch_consumption
        self.total_energy -= branch_consumption
        if self.total_energy < 0 and self.energy_hist is not None:
            self.energy_hist.total_energys.append(self.total_energy)
            return False
        return True
