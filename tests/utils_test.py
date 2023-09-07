import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src/")

import numpy as np
import rlvortex
import rlvortex.envs.base_env as base_env
import utils.utils as utils
import envs.tree_envs as tree_envs


def test_unscale_scale_by_range():
    # case 1
    values = np.array([[-1, 1, 0.5], [0.2, -0.2, 0.9]])
    lower = 5
    upper = 20
    unscaled_values = utils.unscale_by_range(values, lower, upper)
    right_values = np.array([[5.0, 20.0, 16.25], [14.0, 11.0, 19.25]])
    assert np.allclose(unscaled_values, right_values)
    scaled_values = utils.scale_by_range(unscaled_values, lower, upper)
    assert np.allclose(scaled_values, values)
    # case 2
    values = np.array([[-1, 1, 0.5], [0.2, -0.2, 0.9]])
    lower = -1
    upper = 1
    unscaled_values = utils.unscale_by_range(values, lower, upper)
    assert np.allclose(unscaled_values, values)
    scaled_values = utils.scale_by_range(unscaled_values, lower, upper)
    assert np.allclose(scaled_values, values)


def test_rescale_by_range():
    # case 1
    delta_rot = np.array([[-1, 1, 0.5], [0.2, -0.2, 0.9]])
    lower = 5
    upper = 20
    delta_rot_degree = utils.rescale_by_range(delta_rot, lower, upper, -180, 180)
    right_delta_rot_degree = np.array([[0.02777778, 0.11111111, 0.09027778], [0.07777778, 0.06111111, 0.10694444]])
    assert np.allclose(delta_rot_degree, right_delta_rot_degree)
    # case 2
    delta_rot = np.array([[-1, 1, 0.5], [0.2, -0.2, 0.9]])
    lower = -1
    upper = 1
    delta_rot_degree = utils.rescale_by_range(delta_rot, lower, upper, -1, 1)
    assert np.allclose(delta_rot_degree, delta_rot)


def test_rot_vector_by_xyz():
    # case 1
    vector = np.array([[1, 0, 0]]).transpose()
    rot = np.array([0, 0, 0])
    rot_vector = utils.rot_vector_by_eulers(vector, rot)
    right_rot_vector = np.array([[1, 0, 0]]).transpose()
    assert np.allclose(rot_vector, right_rot_vector), f"{rot_vector} != \n {right_rot_vector}"
    # case 2
    vector = np.array([[1, 0, 0]]).transpose()
    rot = np.array([90, 0, 0])
    rot_vector = utils.rot_vector_by_eulers(vector, rot)
    right_rot_vector = np.array([[1, 0, 0]]).transpose()
    assert np.allclose(rot_vector, right_rot_vector), f"{rot_vector} != \n {right_rot_vector}"
    vector = np.array([[1, 0, 0]]).transpose()
    rot = np.array([0, 90, 0])
    rot_vector = utils.rot_vector_by_eulers(vector, rot)
    right_rot_vector = np.array([[0, 0, -1]]).transpose()
    assert np.allclose(rot_vector, right_rot_vector), f"{rot_vector} != \n {right_rot_vector}"
    # case 3
    vector = np.array([[0, 1, 0]]).transpose()
    rot = np.array([90, 0, 0])
    rot_vector = utils.rot_vector_by_eulers(vector, rot)
    right_rot_vector = np.array([[0, 0, 1]]).transpose()
    assert np.allclose(rot_vector, right_rot_vector), f"{rot_vector} != \n {right_rot_vector}"
    vector = np.array([[0, 1, 0]]).transpose()
    rot = np.array([0, 0, 90])
    rot_vector = utils.rot_vector_by_eulers(vector, rot)
    right_rot_vector = np.array([[-1, 0, 0]]).transpose()
    assert np.allclose(rot_vector, right_rot_vector), f"{rot_vector} != \n {right_rot_vector}"
    # case 4
    vector = np.array([[0, 1, 0]]).transpose()
    rot = np.array([180, 0, 0])
    rot_vector = utils.rot_vector_by_eulers(vector, rot)
    right_rot_vector = np.array([[0, -1, 0]]).transpose()
    assert np.allclose(rot_vector, right_rot_vector), f"{rot_vector} != \n {right_rot_vector}"
    # case 5
    vector = np.array([[0, 1, 0]]).transpose()
    rot = np.array([0, 90, 0])
    rot_vector = utils.rot_vector_by_eulers(vector, rot)
    right_rot_vector = np.array([[0, 1, 0]]).transpose()
    assert np.allclose(rot_vector, right_rot_vector), f"{rot_vector} != \n {right_rot_vector}"
    # case 5
    vector = np.array([[0, 1, 0]]).transpose()
    rot = np.array([0, 0, 90])
    rot_vector = utils.rot_vector_by_eulers(vector, rot)
    right_rot_vector = np.array([[-1, 0, 0]]).transpose()
    assert np.allclose(rot_vector, right_rot_vector), f"{rot_vector} != \n {right_rot_vector}"


if __name__ == "__main__":
    test_unscale_scale_by_range()
    print("passed:", test_unscale_scale_by_range.__name__)
    test_rescale_by_range()
    print("passed:", test_rescale_by_range.__name__)
    test_rot_vector_by_xyz()
    print("passed:", test_rot_vector_by_xyz.__name__)
    test_env()

    print("passed:", test_env.__name__)
