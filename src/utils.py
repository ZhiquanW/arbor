import numpy as np
import torch
from typing import Union
import scipy


def torch_unscale_by_range(
    values: torch.Tensor,
    lower: Union[torch.Tensor, float],
    upper: Union[torch.Tensor, float],
) -> torch.Tensor:
    """
    unscale each value in the values to the range of lower and upper, each value must between -1, 1
    input:
        values shape: (num_sample, feat_0, feat_1, ...)
    return:
        scaled_values shape: (num_sample, feat_0, feat_1, ...)
    """
    return (values + 1) * (upper - lower) / 2 + lower


def torch_scale_by_range(
    values: torch.Tensor,
    lower: Union[torch.Tensor, float],
    upper: Union[torch.Tensor, float],
) -> torch.Tensor:
    """
    scale each value in the values to -1, 1
    input:
        values shape: (num_sample, feat_0, feat_1, ...)
    return:
        scaled_values shape: (num_sample, feat_0, feat_1, ...)
    """
    return (values - lower) / (upper - lower) * 2 - 1


def torch_axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def torch_euler_angles_to_matrix(
    euler_angles: torch.Tensor, convention: str = "XYZ"
) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")

    matrices = [
        torch_axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[2], matrices[1]), matrices[0])


def torch_matrix_to_euler_angles(matrix: torch.Tensor) -> torch.Tensor:
    # https://en.wikipedia.org/wiki/Euler_angles#Tait–Bryan_angles
    # http://zhaoxuhui.top/blog/2018/03/13/RelationBetweenQ4&R&Euler.html#4旋转矩阵转欧拉角
    r21 = matrix[..., 1, 0]
    r33 = matrix[..., 2, 2]
    r32 = matrix[..., 2, 1]
    r31 = matrix[..., 2, 0]
    r11 = matrix[..., 0, 0]
    alpha = torch.atan2(r21, r11)
    # beta = torch.rad2deg(torch.arcsin(-r31))
    beta = torch.atan2(-r31, torch.sqrt(r32**2 + r33**2))
    gamma = torch.atan2(r32, r33)
    return torch.stack([gamma, beta, alpha], dim=-1)


def unscale_by_range(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    unscale each value in the values to the range of lower and upper, each value must between -1, 1
    input:
        values shape: (num_sample, feat_0, feat_1, ...)
    return:
        scaled_values shape: (num_sample, feat_0, feat_1, ...)
    """
    assert not np.allclose(lower, upper)
    return (values + 1) * (upper - lower) / 2 + lower


def unscale_by_ranges(
    values: np.ndarray, lowers: np.ndarray, uppers: np.ndarray
) -> np.ndarray:
    """
    unscale each value in the values to the range of lowers, uppers by column, each value must between -1,1
    args:
        values: a 2D np.ndarray in shape (N, M) (each values between -1, 1), defines the values will be unscaled
        lowers: a 1D np.ndarray in shape (M,), defines the unscale lower bounds for each column in values
        uppers: a 1D np.ndarray in shape (M,), defines the unscale upper bounds for each column in values
    return:
        unscaled values
    """
    assert len(values.shape) == 2, "values must be an 2D np.ndarray"
    assert len(lowers.shape) == 1, "lowers must be an 1D np.ndarray"
    assert len(uppers.shape) == 1, "uppers must be an 1D np.ndarray"
    assert (
        values.shape[1] == lowers.shape[0]
    ), f"the number of columns {values.shape[1]} must fit the size of lower {lowers.shape[0]}"
    assert (
        values.shape[1] == uppers.shape[0]
    ), f"the number of columns {values.shape[1]} must fit the size of uppers {lowers.shape[0]}"
    assert not np.allclose(
        lowers, uppers
    ), "the upper bound and the lower bound can not be close"
    assert np.all(
        uppers - lowers > 0
    ), "the upper bound must be greater than lower bound"
    assert np.all(values <= 1), "each value must be less than 1"
    assert np.all(values >= -1), "each value must be greater than -1"
    return (values + 1) * (uppers - lowers) / 2 + lowers


def scale_by_range(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    scale each value in the values to -1, 1
    input:
        values shape: (num_sample, feat_0, feat_1, ...)
    return:
        scaled_values shape: (num_sample, feat_0, feat_1, ...)
    """
    assert not np.allclose(lower, upper)
    return (values - lower) / (upper - lower) * 2 - 1


def scale_by_ranges(
    values: np.ndarray, lowers: np.ndarray, uppers: np.ndarray
) -> np.ndarray:
    """
    scale each value in the values that is in the range of lowers, uppers by column, to the range of [-1 ,1]
    args:
        values: a 2D np.ndarray in shape (N, M) (each values v_i between lowers[i] and uppers[i]), defines the values will be scaled
        lowers: a 1D np.ndarray in shape (M,), defines the unscale lower bounds for each column in values
        uppers: a 1D np.ndarray in shape (M,), defines the unscale upper bounds for each column in values
    return:
        scaled values
    """
    assert len(values.shape) == 2, "values must be an 2D np.ndarray"
    assert len(lowers.shape) == 1, "lowers must be an 1D np.ndarray"
    assert len(uppers.shape) == 1, "uppers must be an 1D np.ndarray"
    assert (
        values.shape[1] == lowers.shape[0]
    ), f"the number of columns {values.shape[1]} must fit the size of lower {lowers.shape[0]}"
    assert (
        values.shape[1] == uppers.shape[0]
    ), f"the number of columns {values.shape[1]} must fit the size of uppers {lowers.shape[0]}"
    assert not np.allclose(
        lowers, uppers
    ), "the upper bound and the lower bound can not be close"
    assert np.all(
        uppers - lowers > 0
    ), "the upper bound must be greater than lower bound"
    assert np.all(values <= uppers), "each value must be less than uppers"
    assert np.all(values >= lowers), "each value must be greater than lowers"
    return (values - lowers) / (uppers - lowers) * 2 - 1


def rot_vector_by_eulers(vector: np.ndarray, xyzs: np.ndarray) -> np.ndarray:
    """
    vector: (3)
    xyzs: (3, N)
    """
    assert vector.shape[0] == 3
    r = scipy.spatial.transform.Rotation.from_euler("xyz", xyzs, degrees=True)
    return np.matmul(r.as_matrix(), vector)


def rot_mats_from_eulers(xyzs: np.ndarray) -> np.ndarray:
    r = scipy.spatial.transform.Rotation.from_euler("xyz", xyzs, degrees=True)
    return r.as_matrix()


def euler_from_rot_mats(rot_mats: np.ndarray) -> np.ndarray:
    assert rot_mats.shape[-2:] == (3, 3), f"rot_mats shape: {rot_mats.shape}"
    r = scipy.spatial.transform.Rotation.from_matrix(rot_mats)
    return (r.as_euler("xyz", degrees=True) % 360 + 180) % 360 - 180


def rescale_by_range(
    values: np.ndarray, lower: float, upper: float, re_lower: float, re_upper: float
) -> np.ndarray:
    """
    convert delta_rot from [-1,1] which is in the range of [lower, upper]
    to the global norm of the rotation [-1,1 ] which is in the range of [-re_lower,re_upper]
    """
    assert values.shape[1] == 3
    assert not np.allclose(re_lower, re_upper)
    assert not np.allclose(lower, upper)
    delta_rot_degree = unscale_by_range(values, lower, upper)
    return scale_by_range(delta_rot_degree, re_lower, re_upper)


def rot_around_axis_by_angle(
    vec: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    """
    axis: (3)
    angle: (1)
    """
    assert axis.shape[0] == 3
    axis = axis / np.linalg.norm(
        axis, axis=-1, keepdims=True
    )  # normalized may lead to error
    r = scipy.spatial.transform.Rotation.from_rotvec(axis * angle, degrees=True)
    return r.apply(vec)


def rot_mat_from_axis_angles(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    axis: (3)
    angle: (N,1)
    """
    axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)
    r = scipy.spatial.transform.Rotation.from_rotvec(axes * angles, degrees=True)
    return r.as_matrix()


def torch_rot_mat_from_axis_angles(
    axes: torch.Tensor, angles: torch.Tensor
) -> torch.Tensor:
    return axis_angle_to_matrix(angles.reshape(-1, 1) * axes)


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))
