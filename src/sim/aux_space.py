from typing import List, Optional
import torch


class TorchVoxelSpace:
    def __init__(
        self,
        voxel_size: float,
        space_half_size: int,
        padding_size: int,
        device: torch.device,
    ) -> None:
        """
        the occupancy space is a 3D voxel space use to represent the occupancy of the world entity.
        the actual size of the occupancy is formed by (voxel_size * space_half_size * 2) ^ 3
        the occupancy space is placed in the world whose center is at (0,0,self.space_half_size * self.voxel_size)
        Args:
            voxel_size: size of a voxel
            space_half_size: the half number of the voxels in the space
        """  # noqa: E501
        self.voxel_size: float = voxel_size
        self.space_half_size: int = space_half_size
        self.padding_size: int = padding_size
        self.space_padded_half_size: int = space_half_size + padding_size
        self.device: torch.device = device
        self.space_range = torch.tensor(
            [
                [
                    -self.space_padded_half_size * voxel_size,
                    self.space_padded_half_size * voxel_size,
                ],  # noqa: E501
                [
                    -self.space_padded_half_size * voxel_size,
                    self.space_padded_half_size * voxel_size,
                ],  # noqa: E501
                [
                    -self.padding_size,
                    (2 * self.space_padded_half_size + self.padding_size) * voxel_size,
                ],  # noqa: E501
            ],
            dtype=torch.float,
            device=self.device,
        )
        self.voxel_indices_offset = torch.tensor(
            [
                self.space_padded_half_size,
                self.space_padded_half_size,
                self.padding_size,
            ],
            dtype=torch.int,
            device=self.device,
        )
        self.space: torch.Tensor = torch.zeros(
            (
                self.space_padded_half_size * 2,
                self.space_padded_half_size * 2,
                self.space_padded_half_size * 2,
            ),  # noqa: E501
            dtype=torch.float,
            device=self.device,
        )

    def clear(self) -> None:
        self.space: torch.Tensor = torch.zeros(
            (
                self.space_half_size * 2,
                self.space_half_size * 2,
                self.space_half_size * 2,
            ),
            dtype=torch.float,
            device=self.device,
        )

    def positions_to_voxel_indices(self, positions: torch.Tensor) -> torch.Tensor:
        """
        convert a list of positions to voxel indices
        Args:
            positions: the positions of the entity
        """
        assert positions.shape[1] == 3, "positions should be a tensor of shape (N,3)"
        voxel_indices = (
            torch.round(positions / self.voxel_size).int() + self.voxel_indices_offset
        )
        assert torch.all(voxel_indices >= 0) and torch.all(
            voxel_indices < self.space_half_size * 2
        ), (
            f"[TorchOccupancySpace] position outside occupancy voxel space: {self.space_half_size}\n"  # noqa: E501
            f"voxel indices: {voxel_indices}\n"
            f"voxel positions: {positions}"
        )
        return voxel_indices

    def voxel_indices_to_positions(self, voxel_indices: torch.Tensor) -> torch.Tensor:
        assert (
            voxel_indices.shape[1] == 3
        ), "voxel_indices should be a tensor of shape (N, 3)"  # noqa: E501
        return (voxel_indices - self.voxel_indices_offset) * self.voxel_size

    def get_occupied_voxel_indices(self) -> Optional[torch.Tensor]:
        """
        return the indices of the voxel whose value is greater than 0.0
        """
        vx, vy, vz = torch.where(self.space > 0.0)
        if not len(vx):
            return None
        return torch.stack((vx, vy, vz), dim=1)

    def get_occupied_voxel_positions(self) -> Optional[torch.Tensor]:
        voxel_indices = self.get_occupied_voxel_indices()
        if voxel_indices is None:
            return None
        return self.voxel_indices_to_positions(voxel_indices)

    def get_free_voxels(self) -> List[torch.Tensor]:
        return torch.where(self.space == 0)


class TorchOccupancySpace(TorchVoxelSpace):
    def __init__(
        self, voxel_size: float, space_half_size: int, device: torch.device
    ) -> None:
        """
        the occupancy space is a 3D voxel space use to represent the occupancy of the tree nodes.
        the actual size of the occupancy is formed by (voxel_size * space_half_size * 2) ^ 3
        the occupancy space is placed in the world whose center is at (0,0,self.space_half_size * self.voxel_size)
        Args:
            voxel_size: size of a voxel
            space_half_size: the half number of the voxels in the space
        """  # noqa: E501
        super().__init__(
            voxel_size=voxel_size,
            space_half_size=space_half_size,
            padding_size=0,
            device=device,
        )

    def add_occupancy_entity(self, positions: torch.Tensor) -> None:
        """
        add a occupancy entity to the occupancy space
        Args:
            positions: the positions of the entity
        """
        assert positions.shape[1] == 3, "positions should be a tensor of shape (N,3)"
        voxel_indices = self.positions_to_voxel_indices(positions).transpose(0, 1)
        self.space[tuple(voxel_indices)] = 1


class TorchShadowSpace(TorchVoxelSpace):
    def __init__(
        self,
        voxel_size: float,
        space_half_size: int,
        pyramid_half_size: int,
        shadow_delta: float,
        device: torch.device,
    ) -> None:
        """
        the shadow space is a 3D voxel space use to represent the shadow of the tree nodes.
        https://docs.craft.do/editor/d/95bf139a-3b5a-0847-798a-75f92cdb0cf1/254648B8-D35A-4468-A455-060C6BCD00DD/b/B98D24A3-4C20-48E0-ACCE-45C8CF338599#E62A89A0-B07E-4A33-800E-CC6C8274AA4E
        """
        self.pyramid_half_size: int = pyramid_half_size
        self.shadow_delta: float = shadow_delta
        # space half size must has pyramid half size padding
        super().__init__(
            voxel_size=voxel_size,
            space_half_size=space_half_size,
            padding_size=pyramid_half_size,
            device=device,
        )
        self.shadow_pyramid_template = self.__gen_shadow_pyramid_template()

    def __gen_shadow_pyramid_template(
        self,
    ) -> torch.Tensor:
        indices_xy = (
            torch.arange(2 * self.pyramid_half_size - 1, device=self.device)
            - self.pyramid_half_size
            + 1
        )
        indices_z = (
            torch.arange(self.pyramid_half_size, device=self.device)
            - self.pyramid_half_size
            + 1
        )
        x, y, z = torch.meshgrid(indices_xy, indices_xy, indices_z, indexing="ij")
        pyramid_mask = (
            torch.abs(x) + torch.abs(y) + self.pyramid_half_size - torch.abs(z)
            <= self.pyramid_half_size
        )
        template = torch.zeros(
            (
                2 * self.pyramid_half_size - 1,
                2 * self.pyramid_half_size - 1,
                self.pyramid_half_size,
            ),
            device=self.device,
        )
        template[pyramid_mask] = self.shadow_delta
        return template

    def get_shadowed_voxel_positions(self) -> Optional[torch.Tensor]:
        voxel_indices = self.get_occupied_voxel_indices()
        if voxel_indices is None:
            return None
        return self.voxel_indices_to_positions(voxel_indices)
        pass

    def cast_shadows(self, positions: torch.Tensor) -> None:
        voxel_indices = self.positions_to_voxel_indices(positions)
        for voxel_index in voxel_indices:
            self.space[
                voxel_index[0]
                - self.pyramid_half_size
                + 1 : voxel_index[0]
                + self.pyramid_half_size,
                voxel_index[1]
                - self.pyramid_half_size
                + 1 : voxel_index[1]
                + self.pyramid_half_size,
                voxel_index[2] - self.pyramid_half_size + 1 : voxel_index[2] + 1,
            ] += self.shadow_pyramid_template

        self.space = torch.clamp(self.space, min=0.0, max=1.0)
