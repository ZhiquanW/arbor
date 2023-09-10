from typing import List, Optional, Tuple, Set

import torch

import utils.utils as utils
import sim.aux_space as aux_space
import sim.energy_module as e_module
import random

# change grow step to tree level


class TorchArborEngine:
    def __init__(
        self,
        *,
        max_steps: int = 20,
        max_branches_num: int = 50,
        move_dis_range: List[float] = [0.3, 0.5],
        move_rot_range: List[List[float]] = [[-10, 10], [-10, 10], [-10, 10]],
        new_branch_rot_range: List[float] = [0, 30],
        node_branch_prob_range: List[float] = [0.1, 0.5],
        node_sleep_prob_range: List[float] = [0.001, 0.01],
        energy_module: Optional[e_module.EnergyModule] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        pass
        """
        An pytorch based implementation of arbor engine for GPU accelerated tee growth simulation. 
        check this document for details: https://www.craft.do/s/no3hkSVGhQMJVD
        """  # noqa: E501
        #################### store input parameters ####################
        self.device: torch.device = device
        # the maximum number of steps for each episode
        # also
        # the max number of times a moving node(the begnning of a branch) can grow.
        # when brach, the child node will inherit the remaining growth num
        self.max_steps: int = max_steps
        assert self.max_steps >= 1, "max grow steps must be at least 1"
        # the max number of branches (apical nodes) a tree can have
        self.max_branches_num: int = max_branches_num
        assert self.max_branches_num >= 2
        # the range of moving distance of a node
        self.move_dis_range: torch.Tensor = torch.tensor(
            move_dis_range, dtype=torch.float32, device=device
        )
        assert self.move_dis_range.shape == (
            2,
        ), "the range of grow distance has 2 values(min, max)"
        assert (
            self.move_dis_range[0] < self.move_dis_range[1]
        ), "the min grow distance must be less than the max grow distance"
        assert self.move_dis_range[0] >= 0, "the grow distance must be postive"
        # the rotation angle (in degree) range (in x,y,z dimension separately)
        # when a node move
        self.move_rot_range: torch.Tensor = torch.tensor(
            move_rot_range, dtype=torch.float32, device=device
        )
        assert self.move_rot_range.shape == (3, 2)
        assert (self.move_rot_range >= -180).all()
        assert (self.move_rot_range <= 180).all()
        # 1) the default angle (in degree) between the new branch
        # and its parent branch when branch
        # 2) the init rotation of the new node is compute by rotating alpha degree
        # around parent node x axis ([1,0,0] in local frame) after sampling
        # from the range
        self.new_branch_rot_range: torch.Tensor = torch.tensor(
            new_branch_rot_range, dtype=torch.float32, device=device
        )
        assert self.new_branch_rot_range.shape == (2,)
        assert self.new_branch_rot_range[0] >= 0
        assert self.new_branch_rot_range[0] < self.new_branch_rot_range[1]
        # the probability of a node branchs at each step
        self.node_branch_prob_range: torch.Tensor = torch.tensor(
            node_branch_prob_range, dtype=torch.float32, device=device
        )
        assert len(self.node_branch_prob_range) == 2
        assert self.node_branch_prob_range[0] >= 0
        assert self.node_branch_prob_range[0] <= self.node_branch_prob_range[1]
        self.node_sleep_prob_range: List[float] = node_sleep_prob_range
        assert len(self.node_sleep_prob_range) == 2
        assert self.node_sleep_prob_range[0] >= 0
        assert (
            self.node_sleep_prob_range[0] < self.node_sleep_prob_range[1]
        ), f"sleep prob range: {self.node_sleep_prob_range}"
        self.energy_module: Optional[e_module.EnergyModule] = energy_module
        # init the simulation variables
        self.reset()

    @property
    def action_dim(self) -> Tuple[int]:
        return (self.max_branches_num * 6,)

    @property
    def num_remaining_branches(self):
        return self.max_branches_num - self.num_branches

    @property
    def num_apical_nodes(self):
        # node must exist, awake, alive and has remaining growth step
        return len(torch.where((self.apical_nodes_state[:, 0] == 1))[0])

    @property
    def alive_nodes_idx(self) -> List[torch.Tensor]:
        # node must exist, awake and alive
        return torch.where(
            (self.nodes_state[:, :, 0] == 1)
            & (self.nodes_state[:, :, 2] == 1)
            & (self.nodes_state[:, :, 3] == 1)
        )

    @property
    def alive_nodes_position(self) -> torch.Tensor:
        return self.nodes_state[self.alive_nodes_idx][:, 5:8]

    def nodes_parent_idx(self, steps: int):
        steps = max(0, min(steps, self.max_steps))
        return torch.where(
            (self.nodes_state[steps, :, 1] >= 0) & (self.nodes_state[steps, :, 0] == 1)
        )[0]

    def __reset_sim_vriable(self) -> None:
        self.steps: int = 0
        self.done: bool = False
        if self.energy_module is not None:
            self.energy_module.reset()

    def __reset_tree_variables(self) -> None:
        self.num_nodes: int = 1
        self.num_branches: int = 1
        # node feats = [exist(1),parent_idx(1) awake(1), alive(1), num_groth(1), pos(3),rot(3),len(1)]  # noqa: E501
        # feats idx = exit[0],parent_idx[1], awake[2], alive[3], num_growth[4], pos[5:8], rot[8:11], len[11]  # noqa: E501
        self.NUM_NODE_FEATS: int = 12
        self.nodes_state: torch.Tensor = torch.zeros(
            (self.max_steps + 1, self.max_branches_num, self.NUM_NODE_FEATS),
            device=self.device,
        )
        self.branch_birth_hist: torch.Tensor = torch.zeros(
            (self.max_branches_num, 1), dtype=torch.int32, device=self.device
        )
        self.apical_nodes_state: torch.Tensor = torch.zeros(
            (self.max_branches_num, self.NUM_NODE_FEATS),
            device=self.device,
        )
        self.next_apical_nodes_state: torch.Tensor = torch.zeros(
            (self.max_branches_num, self.NUM_NODE_FEATS),
            device=self.device,
        )
        self.apical_nodes_state[0] = torch.tensor(
            [
                1,
                -1,
                1,
                1,
                self.max_steps,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.nodes_state[self.steps] = self.apical_nodes_state

    def reset(self):
        self.__reset_sim_vriable()
        self.__reset_tree_variables()

    def __sample_action(self):
        return torch.rand(self.max_branches_num * 6, device=self.device) * 2 - 1

    def step(self, action: torch.Tensor) -> bool:
        """
        perform a step of growth of the tree

        this method perform the growth/ branch/ sleep action for each buds.
        after actions is performed, the collision will be computed and the
        data will be stored

        Args:
            action is a 1d array of shape: (max_buds_num * 6)
            all actions must be normalized to [-1,1]
            need to be reshaped to (max_buds_num , 6)

            (
                max_branches_num,
                move_dis: (1) [-1,1],
                delta_rot: (3)[-1,1] from [delta_lower, delta_upper],
                branch_prob: (1),[-1,1]
                sleep_prob: (1),[-1,1]
            )
        Returns: Bool if the modeling process is completed
        (all data will be stored in the member variables)
        """
        #################### sanity check: input ####################
        normalized_action = self.__sample_action()
        action = normalized_action.view(self.max_branches_num, 6).clip(-1, 1)
        active_nodes_idx = self.get_active_nodes_idx()
        num_active_nodes = len(active_nodes_idx)
        if num_active_nodes == 0:
            return True

        active_nodes_prev_rot_mat = self.__step_nodes_move(
            active_nodes_idx=active_nodes_idx,
            action_move_dis=action[:, 0],
            action_delta_rot=action[:, 1:4],
        )

        active_nodes_idx = self.__step_nodes_sleep(
            active_nodes_idx=active_nodes_idx, nodes_sleep_prob_action=action[:, 5]
        )
        self.__step_nodes_branch(
            active_nodes_idx=active_nodes_idx,
            nodes_branch_prob_action=action[:, 4],
            nodes_prev_rot_mat=active_nodes_prev_rot_mat,
        )

        self.nodes_state[self.steps + 1, ...] = self.next_apical_nodes_state
        self.apical_nodes_state = self.next_apical_nodes_state

        self.steps += 1
        if self.steps == self.max_steps:
            self.done = True

        if self.energy_module is not None:
            self.energy_module.measure_energy(self.num_apical_nodes, self.num_nodes)
        return self.done

    def step_energy(self) -> None:
        if self.energy_module is not None:
            self.energy_module.measure_energy(self.num_apical_nodes, self.num_nodes)

    def get_active_nodes_idx(self) -> torch.Tensor:
        # node must exist, awake, alive and has remaining growth step
        return torch.where(
            (self.apical_nodes_state[:, 0] == 1)
            & (self.apical_nodes_state[:, 2] == 1)
            & (self.apical_nodes_state[:, 3] == 1)
            & (self.apical_nodes_state[:, 4] > 0)
        )[0]

    def existing_nodes_idx(self, step) -> torch.Tensor:
        step = max(0, min(step, self.max_steps))
        return torch.where((self.nodes_state[step, :, 0] == 1))[0]

    def get_end_nodes_idx(self):
        nodes_exist_mat = self.nodes_state[..., 0]
        nodes_exist_mat_left_shift = torch.concatenate(
            [
                nodes_exist_mat[:-1, :],
                torch.zeros((1, self.max_branches_num), device=self.device),
            ],
        )
        nodes_stop_mat = nodes_exist_mat - nodes_exist_mat_left_shift
        return torch.where(nodes_stop_mat == 1)

    def __compute_nodes_move_dir(
        self, active_nodes_idx: torch.Tensor, move_delta_euler_degrees: torch.Tensor
    ):
        num_active_nodes = len(active_nodes_idx)
        delta_rot_mat: torch.Tensor = utils.torch_euler_angles_to_matrix(
            torch.deg2rad(move_delta_euler_degrees)
        )
        assert delta_rot_mat.shape == (
            num_active_nodes,
            3,
            3,
        ), f"expect shape f{(self.max_branches_num,3,3)}, given f{delta_rot_mat.shape}"
        nodes_prev_rot_mat: torch.Tensor = utils.torch_euler_angles_to_matrix(
            torch.deg2rad(
                utils.torch_unscale_by_range(
                    self.apical_nodes_state[active_nodes_idx, 8:11],
                    lower=-180,
                    upper=180,
                )
            )
        )

        assert delta_rot_mat.shape == (
            num_active_nodes,
            3,
            3,
        ), f"expect shape f{(self.max_branches_num,3,3)}, given f{delta_rot_mat.shape}"  # noqa: E501
        new_rot_mat: torch.Tensor = torch.bmm(delta_rot_mat, nodes_prev_rot_mat)
        move_dir = torch.matmul(
            new_rot_mat, torch.tensor([0.0, 0.0, 1.0], device=self.device)
        )
        assert move_dir.shape == (
            num_active_nodes,
            3,
        ), f"expect shape f{(self.max_branches_num,3)}, given f{move_dir.shape}"  # noqa: E501
        new_rot_euler_degrees: torch.Tensor = torch.rad2deg(
            utils.torch_matrix_to_euler_angles(new_rot_mat)
        )
        assert new_rot_euler_degrees.shape == (
            num_active_nodes,
            3,
        ), f"expect shape f{(self.max_branches_num,3)}, given f{new_rot_euler_degrees.shape}"  # noqa: E501
        return move_dir, new_rot_euler_degrees, nodes_prev_rot_mat

    def __step_nodes_move(
        self,
        active_nodes_idx: torch.Tensor,
        action_move_dis: torch.Tensor,
        action_delta_rot: torch.Tensor,
    ) -> torch.Tensor:
        assert action_move_dis.shape == (
            self.max_branches_num,
        ), f"expect shape f{(self.max_branches_num,1)}, given f{action_move_dis.shape}"
        assert action_delta_rot.shape == (
            self.max_branches_num,
            3,
        ), f"expect shape f{(self.max_branches_num,3)}, given f{action_delta_rot.shape}"

        num_active_nodes = len(active_nodes_idx)
        move_distances = utils.torch_unscale_by_range(
            values=action_move_dis[active_nodes_idx],
            lower=self.move_dis_range[0],
            upper=self.move_dis_range[1],
        )
        assert move_distances.shape == (
            num_active_nodes,
        ), f"expect shape f{(num_active_nodes,)}, given f{move_distances.shape}"
        move_delta_euler_degrees = utils.torch_unscale_by_range(
            values=action_delta_rot[active_nodes_idx],
            lower=self.move_rot_range[:, 0],
            upper=self.move_rot_range[:, 1],
        )
        (
            move_dir,
            new_rot_euler_degrees,
            nodes_prev_rot_mat,
        ) = self.__compute_nodes_move_dir(
            active_nodes_idx=active_nodes_idx,
            move_delta_euler_degrees=move_delta_euler_degrees,
        )
        new_nodes_pos = (
            self.apical_nodes_state[active_nodes_idx, 5:8]
            + move_distances.view(num_active_nodes, 1) * move_dir
        )
        # update the state
        self.next_apical_nodes_state[active_nodes_idx, 0:4] = 1
        self.next_apical_nodes_state[active_nodes_idx, 1] = active_nodes_idx.to(
            torch.float32
        )
        self.next_apical_nodes_state[active_nodes_idx,]
        self.next_apical_nodes_state[active_nodes_idx, 4] = (
            self.apical_nodes_state[active_nodes_idx, 4] - 1
        )  # noqa: E501
        self.next_apical_nodes_state[active_nodes_idx, 5:8] = new_nodes_pos
        self.next_apical_nodes_state[
            active_nodes_idx, 8:11
        ] = utils.torch_scale_by_range(new_rot_euler_degrees, lower=-180, upper=180)
        self.next_apical_nodes_state[active_nodes_idx, 11] = move_distances
        # self.branch_birth_hist
        self.num_nodes += num_active_nodes
        return nodes_prev_rot_mat

    def __compute_branch_rot(
        self, branches_idx: torch.Tensor, nodes_prev_rot_mat: torch.Tensor
    ):
        num_new_branches = len(branches_idx)
        branch_dir_rot_axes: torch.Tensor = torch.matmul(
            nodes_prev_rot_mat, torch.tensor([1.0, 0.0, 0.0], device=self.device).t()
        )
        branch_rot_euler = (
            torch.rand(num_new_branches, device=self.device)
            * (self.new_branch_rot_range[1] - self.new_branch_rot_range[0])
            + self.new_branch_rot_range[0]
        ) * (torch.randint(0, 2, (num_new_branches,), device=self.device) * 2 - 1)

        axis_angle_rot_mat = utils.torch_rot_mat_from_axis_angles(
            branch_dir_rot_axes, torch.deg2rad(branch_rot_euler)
        )
        return utils.torch_scale_by_range(
            torch.rad2deg(
                utils.torch_matrix_to_euler_angles(
                    torch.bmm(axis_angle_rot_mat, nodes_prev_rot_mat)
                )
            ),
            -180,
            180,
        )

    def __step_nodes_branch(
        self,
        active_nodes_idx: torch.Tensor,
        nodes_branch_prob_action: torch.Tensor,
        nodes_prev_rot_mat: torch.Tensor,
    ) -> None:
        num_active_nodes = len(active_nodes_idx)
        if num_active_nodes == 0:
            return
        nodes_branch_prob: torch.Tensor = utils.torch_unscale_by_range(
            nodes_branch_prob_action[active_nodes_idx],
            lower=self.node_branch_prob_range[0],
            upper=self.node_branch_prob_range[1],
        )
        local_branch_idx = torch.where(
            torch.rand(num_active_nodes, device=self.device) < nodes_branch_prob
        )[0]
        num_new_branches = min(self.num_remaining_branches, len(local_branch_idx))

        local_branch_idx = local_branch_idx[:num_new_branches]
        if num_new_branches == 0:
            return
        new_nodes_rot_euler_normalized = self.__compute_branch_rot(
            local_branch_idx, nodes_prev_rot_mat[local_branch_idx]
        )
        start_idx = self.num_branches
        end_idx = start_idx + num_new_branches
        self.next_apical_nodes_state[start_idx:end_idx, 0:4] = 1
        self.next_apical_nodes_state[start_idx:end_idx, 1] = active_nodes_idx[
            local_branch_idx
        ].to(torch.float32)
        self.next_apical_nodes_state[start_idx:end_idx, 4] = (
            self.apical_nodes_state[local_branch_idx, 4] - 1
        )
        self.next_apical_nodes_state[start_idx:end_idx, 5:8] = self.apical_nodes_state[
            active_nodes_idx[local_branch_idx], 5:8
        ]

        self.next_apical_nodes_state[
            start_idx:end_idx, 8:11
        ] = new_nodes_rot_euler_normalized
        self.next_apical_nodes_state[start_idx:end_idx, 11] = 0
        self.branch_birth_hist[start_idx:end_idx] = self.steps + 1
        self.num_branches = end_idx
        return

    def __step_nodes_sleep(
        self, active_nodes_idx: torch.Tensor, nodes_sleep_prob_action: torch.Tensor
    ) -> torch.Tensor:
        if self.node_sleep_prob_range[1] == 0:
            return active_nodes_idx
        num_active_nodes = len(active_nodes_idx)
        nodes_sleep_prob: torch.Tensor = utils.torch_unscale_by_range(
            nodes_sleep_prob_action[active_nodes_idx],
            lower=self.node_sleep_prob_range[0],
            upper=self.node_sleep_prob_range[1],
        )
        sleep_mat = torch.rand(num_active_nodes, device=self.device) < nodes_sleep_prob
        local_sleep_idx = torch.where(sleep_mat)[0]
        local_unsleep_idx = torch.where(sleep_mat == False)[0]  # noqa: E712
        self.next_apical_nodes_state[active_nodes_idx[local_sleep_idx], 2] = 0
        # return active nodex indx that are not slept by local sleep idx
        return active_nodes_idx[local_unsleep_idx]

    def random_cut(self) -> None:
        # select a random step before current step
        # select a random branch idx at the current step
        # set all branch to be not exist
        # find child branch indices and set to not exist
        # recurrsively
        target_branch: int = random.randint(0, self.num_branches - 1)
        start_step = random.randint(
            int(self.branch_birth_hist[target_branch, 0]), self.steps
        )
        self.__dfs_cut(start_step, set([target_branch]))

    def __dfs_cut(self, start_step, cut_branch_indices_set: Set) -> None:
        cut_branch_indices_list: List = list(cut_branch_indices_set)
        if len(cut_branch_indices_list) == 0:
            return
        self.nodes_state[start_step:, cut_branch_indices_list, 0] = 0
        self.apical_nodes_state[cut_branch_indices_list, 0] = 0
        child_indices = set()
        for i in range(self.num_branches):
            for j in range(start_step, self.steps + 1):
                if (
                    self.nodes_state[j, i, 0] == 1
                    and self.nodes_state[j, i, 1] in cut_branch_indices_list
                ):
                    child_indices.add(i)
        self.__dfs_cut(start_step, child_indices)
