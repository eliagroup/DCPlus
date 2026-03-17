# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
from numpy.testing import assert_array_equal

from dc_plus.numpy.low_rank_helper import (
    _branch_state_indices,
    _extract_columns_from_matrix,
    _extract_rows_from_matrix,
)


def test_branch_state_indices_all_valid():
    branch_from = np.array([0], dtype=int)
    branch_to = np.array([1], dtype=int)
    angle_component_indices = np.array([2, 3], dtype=int)  # theta indices
    magnitude_component_indices = np.array([5, 7], dtype=int)  # voltage indices

    idx_arr, valid_positions = _branch_state_indices(
        0, branch_from, branch_to, angle_component_indices, magnitude_component_indices
    )

    # expected: [theta_from, theta_to, u_from, u_to]
    np.testing.assert_array_equal(idx_arr, np.array([2, 3, 5, 7], dtype=int))
    np.testing.assert_array_equal(valid_positions, np.array([True, True, True, True], dtype=bool))
    assert np.issubdtype(idx_arr.dtype, np.integer)
    assert np.issubdtype(valid_positions.dtype, np.bool_)


def test_branch_state_indices_some_invalid_positions():
    branch_from = np.array([0, 1], dtype=int)
    branch_to = np.array([1, 2], dtype=int)
    angle_component_indices = np.array([0, -1, 4], dtype=int)  # second node has no theta eq
    magnitude_component_indices = np.array([-1, 6, 8], dtype=int)  # first node has no voltage eq

    # branch 0: from=0, to=1 -> theta_from=0, theta_to=-1, u_from=-1, u_to=6
    idx_arr0, valid_positions0 = _branch_state_indices(
        0, branch_from, branch_to, angle_component_indices, magnitude_component_indices
    )
    np.testing.assert_array_equal(idx_arr0, np.array([0, 0, 0, 6], dtype=int))
    np.testing.assert_array_equal(valid_positions0, np.array([True, False, False, True], dtype=bool))

    # branch 1: from=1, to=2 -> theta_from=-1, theta_to=4, u_from=6, u_to=8
    idx_arr1, valid_positions1 = _branch_state_indices(
        1, branch_from, branch_to, angle_component_indices, magnitude_component_indices
    )
    np.testing.assert_array_equal(idx_arr1, np.array([0, 4, 6, 8], dtype=int))
    np.testing.assert_array_equal(valid_positions1, np.array([False, True, True, True], dtype=bool))


def test_extract_rows_from_matrix():
    matrix = np.arange(16, dtype=float).reshape(4, 4)
    row_indices = np.array([0, 2], dtype=int)
    subset = _extract_rows_from_matrix(matrix, row_indices)
    expected = np.array([[0.0, 1.0, 2.0, 3.0], [8.0, 9.0, 10.0, 11.0]])
    assert_array_equal(subset, expected)
    assert subset.shape == (2, 4)

    row_indices = np.array([], dtype=int)
    subset = _extract_rows_from_matrix(matrix, row_indices)
    expected = np.array([]).reshape(0, 4)
    assert_array_equal(subset, expected)
    assert subset.shape == (0, 4)

    row_indices = np.array([-1, 1], dtype=int)  # last row and second row
    subset = _extract_rows_from_matrix(matrix, row_indices)
    expected = np.array([[12.0, 13.0, 14.0, 15.0], [4.0, 5.0, 6.0, 7.0]])
    assert_array_equal(subset, expected)
    assert subset.shape == (2, 4)


def test_extract_columns_from_matrix_basic():
    matrix = np.arange(9, dtype=float).reshape(3, 3)
    col_indices = np.array([0, 2], dtype=int)
    subset = _extract_columns_from_matrix(matrix, col_indices)
    expected = np.array([[0.0, 2.0], [3.0, 5.0], [6.0, 8.0]])
    assert_array_equal(subset, expected)
    assert subset.shape == (3, col_indices.size)

    matrix = np.arange(16, dtype=float).reshape(4, 4)
    col_indices = np.array([], dtype=int)
    subset = _extract_columns_from_matrix(matrix, col_indices)
    expected = np.array([]).reshape(4, 0)
    assert_array_equal(subset, expected)
    assert subset.shape == (4, col_indices.size)

    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    col_indices = np.array([-1, 1], dtype=int)  # last column and second column
    subset = _extract_columns_from_matrix(matrix, col_indices)
    expected = np.array([[3.0, 2.0], [6.0, 5.0]])
    assert_array_equal(subset, expected)
    assert subset.shape == (2, col_indices.size)
