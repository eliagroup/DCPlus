# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pytest

from dc_plus.numpy.bsdf_full_rank import compute_bsdf_update
from tests.test_helper.bsdf_helper import (
    get_bsdf_cases,
    prepare_bsdf_test_context,
    run_reference_one_step,
)


@pytest.mark.parametrize("bsdf_test_case", get_bsdf_cases())
def test_bsdf_full_rank(bsdf_test_case):
    setup = prepare_bsdf_test_context(bsdf_test_case=bsdf_test_case)

    jacobian_data_split_direct = setup.jacobian_data_split_manual

    jacobian_inv_bsdf = compute_bsdf_update(
        jacobian_inv=setup.jacobian_data_with_extra_buses.inverse_jacobian,
        bus_to_split=setup.bus_to_split,
        new_bus_b_index=setup.new_bus_index,
        new_bus_type=2,  # force select PQ node
        branches_connected_to_bus_b=setup.branches_to_move,
        shunt_connected_to_bus_b=np.array([], dtype=np.int32),
        branch_from=setup.branch_from_original,
        branch_to=setup.branch_to_original,
        shunt_to_bus=setup.dynamic_info.shunt_bus_indices,
        v_mag_hat=setup.v_mag_hat,
        theta_hat=setup.theta_hat,
        y_ff=setup.y_ff,
        y_ft=setup.y_ft,
        y_tf=setup.y_tf,
        y_tt=setup.y_tt,
        y_shunt=setup.dynamic_info.shunt_effective_bus_admittance,
        angle_component_indices=setup.jacobian_data_with_extra_buses.angle_component_indices,
        magnitude_component_indices=setup.jacobian_data_with_extra_buses.magnitude_component_indices,
    )

    assert setup.jacobian_data_with_extra_buses.jacobian.shape == jacobian_data_split_direct.jacobian.shape

    in_use_indices = np.flatnonzero(setup.jacobian_data_with_extra_buses.jacobian_index_in_use)
    np.testing.assert_allclose(
        jacobian_inv_bsdf[np.ix_(in_use_indices, in_use_indices)],
        jacobian_data_split_direct.inverse_jacobian[np.ix_(in_use_indices, in_use_indices)],
        rtol=1e-10,
        atol=1e-10,
    )

    # test against powsybl
    dynamic_info_one_step = run_reference_one_step(setup.net, bsdf_test_case=bsdf_test_case)
    dx = -jacobian_inv_bsdf @ setup.mismatch_n1

    # Map Jacobian increments back to bus ordering using the Jacobian mapping

    dynamic_info_split_manual = setup.dynamic_info_split_manual
    theta_actual = dynamic_info_split_manual.bus_voltage_angles_rad
    vm_actual = dynamic_info_split_manual.bus_voltage_magnitudes
    theta_updated_J = theta_actual.copy()
    vm_updated_J = vm_actual.copy()
    theta_updated_J[setup.pvpq_indices] = (
        theta_actual[setup.pvpq_indices] + dx[setup.jacobian_data_with_extra_buses.is_angle_component]
    )
    vm_updated_J[setup.pq_indices] = (
        vm_actual[setup.pq_indices] + dx[setup.jacobian_data_with_extra_buses.is_magnitude_component]
    )

    # reorder bus voltages to match new ordering of manual split
    np.testing.assert_allclose(
        dynamic_info_one_step.bus_voltage_magnitudes[bsdf_test_case.bus_order],
        vm_updated_J,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        dynamic_info_one_step.bus_voltage_angles_rad[bsdf_test_case.bus_order],
        theta_updated_J,
        rtol=1e-10,
        atol=1e-10,
    )
