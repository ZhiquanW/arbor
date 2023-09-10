from typing import List, Optional
import torch
import sim.aux_space as aux_space


class EnergyHist:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.reset_energy_info()
        self.reset_tree_info()

    def reset_energy_info(self) -> None:
        self.accumulated_energies: List[float] = []
        self.energy_balances: List[float] = []
        self.collected_energies: List[float] = []
        self.node_consumptions: List[float] = []

    def reset_tree_info(self) -> None:
        self.num_apical_nodes: List[int] = []
        self.num_nodes: List[int] = []


class EnergyModule:
    def __init__(
        self,
        init_energy: float = 10.0,
        max_energy: float = 100.0,
        collection_factor: float = 0.8,
        node_consumption_factor: float = 0.5,
        record_history: bool = False,
    ) -> None:
        self.init_energy = init_energy
        self.max_energy: float = max_energy
        self.collection_factor: float = collection_factor
        self.node_consumption_factor: float = node_consumption_factor
        self.accumulated_energy: float = init_energy
        self.current_energy_balance: float = 0
        self.energy_hist: Optional[EnergyHist] = (
            EnergyHist() if record_history else None
        )

    def reset(self) -> None:
        self.accumulated_energy = self.init_energy
        self.current_energy_balance = 0
        if self.energy_hist is not None:
            self.energy_hist.reset_energy_info()

    def measure_energy(
        self, num_apical_nodes: int, num_nodes: int, record_act: bool = True
    ) -> None:
        collected_energy = num_apical_nodes * self.collection_factor
        node_consumption = num_nodes * self.node_consumption_factor
        self.current_energy_balance = collected_energy - node_consumption
        self.accumulated_energy += self.current_energy_balance
        self.accumulated_energy = min(self.accumulated_energy, self.max_energy)

        if self.energy_hist is not None:
            self.energy_hist.accumulated_energies.append(self.accumulated_energy)
            self.energy_hist.energy_balances.append(self.current_energy_balance)
            self.energy_hist.collected_energies.append(collected_energy)
            self.energy_hist.node_consumptions.append(node_consumption)
            if record_act:
                self.energy_hist.num_apical_nodes.append(num_apical_nodes)
                self.energy_hist.num_nodes.append(num_nodes)

    def recompute_energy_hist(self) -> None:
        if self.energy_hist is not None:
            self.reset()
            for num_apical_nodes, num_nodes in zip(
                self.energy_hist.num_apical_nodes, self.energy_hist.num_nodes
            ):
                self.measure_energy(num_apical_nodes, num_nodes, record_act=False)
