import numpy as np
import scipy


def unscale_by_range(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    unscale each value in the values, each value must between -1, 1
    input:
        values shape: (num_sample, feat_0, feat_1, ...)
    return:
        scaled_values shape: (num_sample, feat_0, feat_1, ...)
    """
    assert not np.allclose(lower, upper)
    return (values + 1) * (upper - lower) / 2 + lower


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
    return r.as_euler("xyz", degrees=True)


def rescale_by_range(values: np.ndarray, lower: float, upper: float, re_lower: float, re_upper: float) -> np.ndarray:
    """
    convert delta_rot from [-1,1] which is in the range of [lower, upper]
    to the global norm of the rotation [-1,1 ] which is in the range of [-re_lower,re_upper]
    """
    assert values.shape[1] == 3
    assert not np.allclose(re_lower, re_upper)
    assert not np.allclose(lower, upper)
    delta_rot_degree = unscale_by_range(values, lower, upper)
    return scale_by_range(delta_rot_degree, re_lower, re_upper)


def rot_around_axis_by_angle(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    axis: (3)
    angle: (1)
    """
    assert axis.shape[0] == 3
    axis = axis / np.linalg.norm(axis)
    r = scipy.spatial.transform.Rotation.from_rotvec(axis * angle, degrees=True)
    return r.apply(vec)


def rot_mat_from_axis_angles(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    axis: (3)
    angle: (N,1)
    """
    axes = axes / np.linalg.norm(axes)
    r = scipy.spatial.transform.Rotation.from_rotvec(axes * angles, degrees=True)
    return r.as_matrix()
