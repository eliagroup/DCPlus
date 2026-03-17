# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dc_plus.jax.bsdf import compute_bsdf_update as compute_bsdf_update_jax
from tests.test_helper.bsdf_helper import (
    get_bsdf_cases,
    prepare_bsdf_test_context,
    run_reference_one_step,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("bsdf_test_case", get_bsdf_cases())
def test_bsdf_full_rank_jax(bsdf_test_case):
    setup = prepare_bsdf_test_context(bsdf_test_case=bsdf_test_case)

    jacobian_with_extra_bus_inverse = setup.jacobian_data_with_extra_buses.inverse_jacobian
    jacobian_inv_device_transposed = jax.device_put(jacobian_with_extra_bus_inverse.T)

    jacobian_inv_jax_transposed = compute_bsdf_update_jax(
        jacobian_inv_transposed=jacobian_inv_device_transposed,
        bus_to_split=setup.bus_to_split,
        new_bus_b_index=setup.new_bus_index,
        new_bus_type=2,
        branches_connected_to_bus_b=jnp.asarray(setup.branches_to_move, dtype=jnp.int32),
        shunt_connected_to_bus_b=jnp.asarray([], dtype=jnp.int32),
        branch_from=setup.dynamic_info.branch_from_bus,
        branch_to=setup.dynamic_info.branch_to_bus,
        shunt_to_bus=setup.dynamic_info.shunt_bus_indices,
        v_mag_hat=setup.dynamic_info_with_placeholders.bus_voltage_magnitudes.flatten(),
        theta_hat=setup.dynamic_info_with_placeholders.bus_voltage_angles_rad.flatten(),
        y_ff=setup.y_ff,
        y_ft=setup.y_ft,
        y_tf=setup.y_tf,
        y_tt=setup.y_tt,
        y_shunt=setup.dynamic_info.shunt_effective_bus_admittance,
        angle_component_indices=setup.jacobian_data_with_extra_buses.angle_component_indices,
        magnitude_component_indices=setup.jacobian_data_with_extra_buses.magnitude_component_indices,
    )

    jacobian_inv_jax_transposed.block_until_ready()
    jacobian_inv_jax = jnp.transpose(jacobian_inv_jax_transposed)

    in_use_indices = np.flatnonzero(setup.jacobian_data_with_extra_buses.jacobian_index_in_use)
    np.testing.assert_allclose(
        np.asarray(jacobian_inv_jax)[np.ix_(in_use_indices, in_use_indices)],
        setup.jacobian_data_split_manual.inverse_jacobian[np.ix_(in_use_indices, in_use_indices)],
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(jacobian_inv_jax_transposed)[np.ix_(in_use_indices, in_use_indices)],
        setup.jacobian_data_split_manual.inverse_jacobian[np.ix_(in_use_indices, in_use_indices)].T,
        rtol=1e-6,
        atol=1e-8,
    )

    dynamic_info_split_manual = setup.dynamic_info_split_manual
    dynamic_info_one_step = run_reference_one_step(setup.net, bsdf_test_case=bsdf_test_case)
    bus_order = bsdf_test_case.bus_order
    dx = -jacobian_inv_jax @ setup.mismatch_n1

    theta_actual = dynamic_info_split_manual.bus_voltage_angles_rad
    vm_actual = dynamic_info_split_manual.bus_voltage_magnitudes
    theta_updated_J = theta_actual.copy()
    vm_updated_J = vm_actual.copy()
    pvpq = dynamic_info_split_manual.pvpq_buses_indices_pvpq_order
    pq = dynamic_info_split_manual.pq_buses_indices
    theta_updated_J[pvpq] = theta_actual[pvpq] + dx[setup.jacobian_data_with_extra_buses.is_angle_component]
    vm_updated_J[pq] = vm_actual[pq] + dx[setup.jacobian_data_with_extra_buses.is_magnitude_component]

    # reorder bus voltages to match new ordering of manual split
    np.testing.assert_allclose(
        dynamic_info_one_step.bus_voltage_magnitudes[bus_order],
        vm_updated_J,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        dynamic_info_one_step.bus_voltage_angles_rad[bus_order],
        theta_updated_J,
        rtol=1e-10,
        atol=1e-10,
    )
