import numpy as np
import scipy


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


def unscale_by_ranges(values: np.ndarray, lowers: np.ndarray, uppers: np.ndarray) -> np.ndarray:
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
    assert not np.allclose(lowers, uppers), "the upper bound and the lower bound can not be close"
    assert np.all(uppers - lowers > 0), "the upper bound must be greater than lower bound"
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


def scale_by_ranges(values: np.ndarray, lowers: np.ndarray, uppers: np.ndarray) -> np.ndarray:
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
    assert not np.allclose(lowers, uppers), "the upper bound and the lower bound can not be close"
    assert np.all(uppers - lowers > 0), "the upper bound must be greater than lower bound"
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
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)  # normalized may lead to error
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
