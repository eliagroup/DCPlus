# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import scipy.sparse as sp

from dc_plus.example_grids.pypowsbl.example_grids import (
    basic_node_breaker_network_powsybl,
)
from dc_plus.importing.powsybl.powsybl_network_helpers import (
    _load_test_grid,
)
from dc_plus.interfaces.network_information import BusType
from dc_plus.preprocess.preprocess_jacobian_bsdf import preprocess_jacobian_bsdf


def test_preprocess_jacobian_bsdf():
    get_net = basic_node_breaker_network_powsybl

    net, static_info, dynamic_info, string_info, jacobian_data = _load_test_grid(get_net)
    splits = 2
    original_jacobian = jacobian_data.jacobian.toarray()
    original_inverse = jacobian_data.inverse_jacobian.copy()
    n_angle = int(np.count_nonzero(jacobian_data.is_angle_component))
    n_voltage = int(np.count_nonzero(jacobian_data.is_magnitude_component))
    n_eq_original = n_angle + n_voltage
    n_buses_original = dynamic_info.n_buses

    new_jacobian_data, extended_dynamic_info = preprocess_jacobian_bsdf(
        jacobian_data=jacobian_data,
        max_bus_splits=splits,
        dynamic_network_data=dynamic_info,
    )
    is_valid_index = new_jacobian_data.jacobian_index_in_use

    new_jacobian_sparse = new_jacobian_data.jacobian
    assert sp.isspmatrix_csr(new_jacobian_sparse) or isinstance(new_jacobian_sparse, sp.csr_array)

    new_jacobian_dense = new_jacobian_sparse.toarray()
    expected_size = n_eq_original + 2 * splits
    assert new_jacobian_dense.shape == (expected_size, expected_size)

    angle_indices = np.arange(n_angle)
    magnitude_indices = np.arange(n_angle + splits, n_angle + splits + n_voltage)
    angle_padding = np.arange(n_angle, n_angle + splits)
    magnitude_padding = np.arange(n_angle + splits + n_voltage, expected_size)

    np.testing.assert_allclose(
        new_jacobian_dense[np.ix_(angle_indices, angle_indices)], original_jacobian[:n_angle, :n_angle]
    )
    np.testing.assert_allclose(
        new_jacobian_dense[np.ix_(angle_indices, magnitude_indices)], original_jacobian[:n_angle, n_angle:]
    )
    np.testing.assert_allclose(
        new_jacobian_dense[np.ix_(magnitude_indices, angle_indices)], original_jacobian[n_angle:, :n_angle]
    )
    np.testing.assert_allclose(
        new_jacobian_dense[np.ix_(magnitude_indices, magnitude_indices)],
        original_jacobian[n_angle:, n_angle:],
    )

    np.testing.assert_allclose(new_jacobian_dense[angle_padding, :][:, angle_padding], np.eye(splits))
    np.testing.assert_allclose(new_jacobian_dense[magnitude_padding, :][:, magnitude_padding], np.eye(splits))

    expected_valid = np.zeros(expected_size, dtype=bool)
    expected_valid[:n_angle] = True
    expected_valid[n_angle + splits : n_angle + splits + n_voltage] = True
    np.testing.assert_array_equal(is_valid_index, expected_valid)

    new_inverse = new_jacobian_data.inverse_jacobian
    assert new_inverse.shape == (expected_size, expected_size)
    np.testing.assert_allclose(new_inverse[np.ix_(angle_indices, angle_indices)], original_inverse[:n_angle, :n_angle])
    np.testing.assert_allclose(new_inverse[np.ix_(angle_indices, magnitude_indices)], original_inverse[:n_angle, n_angle:])
    np.testing.assert_allclose(new_inverse[np.ix_(magnitude_indices, angle_indices)], original_inverse[n_angle:, :n_angle])
    np.testing.assert_allclose(
        new_inverse[np.ix_(magnitude_indices, magnitude_indices)], original_inverse[n_angle:, n_angle:]
    )
    np.testing.assert_allclose(new_inverse[angle_padding, :][:, angle_padding], np.eye(splits))
    np.testing.assert_allclose(new_inverse[magnitude_padding, :][:, magnitude_padding], np.eye(splits))

    assert new_jacobian_data.bus_is_used.size == n_buses_original + splits
    np.testing.assert_array_equal(new_jacobian_data.bus_is_used[:n_buses_original], jacobian_data.bus_is_used)
    assert not new_jacobian_data.bus_is_used[n_buses_original:].any()

    np.testing.assert_array_equal(
        new_jacobian_data.pointer_to_original_bus[:n_buses_original],
        jacobian_data.pointer_to_original_bus,
    )
    np.testing.assert_array_equal(
        new_jacobian_data.pointer_to_original_bus[n_buses_original:],
        np.full(splits, -1, dtype=new_jacobian_data.pointer_to_original_bus.dtype),
    )

    assert extended_dynamic_info.n_buses == n_buses_original + splits
    np.testing.assert_array_equal(extended_dynamic_info.bus_type[:-splits], dynamic_info.bus_type)
    assert np.all(extended_dynamic_info.bus_type[-splits:] == BusType.PQ)
    np.testing.assert_allclose(extended_dynamic_info.bus_voltage_magnitudes[-splits:], 1.0)
    np.testing.assert_allclose(extended_dynamic_info.bus_voltage_angles_rad[-splits:], 0.0)
    np.testing.assert_allclose(extended_dynamic_info.bus_active_power[-splits:], 0.0)
    np.testing.assert_allclose(extended_dynamic_info.bus_reactive_power[-splits:], 0.0)

    # Ensure original structures remain unchanged
    assert dynamic_info.n_buses == n_buses_original
    np.testing.assert_allclose(jacobian_data.jacobian.toarray(), original_jacobian)
    np.testing.assert_allclose(jacobian_data.inverse_jacobian, original_inverse)
