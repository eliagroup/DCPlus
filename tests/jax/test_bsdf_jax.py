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
from dc_plus.interfaces.network_inputs import (
    JacobianComponentInputs,
    NetworkAdmittanceInputs,
    NetworkTopologyInputs,
    VoltageStateInputs,
)
from dc_plus.numpy.bsdf_full_rank import compute_bsdf_update as compute_bsdf_update_numpy
from tests.test_helper.bsdf_helper import (
    get_bsdf_cases,
    prepare_bsdf_test_context,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("bsdf_test_case", get_bsdf_cases())
def test_bsdf_full_rank_jax_matches_numpy(bsdf_test_case):
    setup = prepare_bsdf_test_context(
        bsdf_test_case=bsdf_test_case,
    )

    jacobian_inv_numpy = compute_bsdf_update_numpy(
        jacobian_inv=setup.jacobian_data_with_extra_buses.inverse_jacobian,
        bus_to_split=setup.bus_to_split,
        new_bus_b_index=setup.new_bus_index,
        new_bus_type=2,
        branches_connected_to_bus_b=setup.branches_connected_to_bus_b,
        shunt_connected_to_bus_b=setup.shunt_connected_to_bus_b,
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

    jacobian_with_extra_bus_inverse = setup.jacobian_data_with_extra_buses.inverse_jacobian
    jacobian_inv_device_transposed = jax.device_put(jacobian_with_extra_bus_inverse.T)
    network_topology = NetworkTopologyInputs(
        branch_from=jnp.asarray(setup.dynamic_info.branch_from_bus, dtype=jnp.int32),
        branch_to=jnp.asarray(setup.dynamic_info.branch_to_bus, dtype=jnp.int32),
        branch_connected=jnp.asarray(setup.dynamic_info.branch_connected),
        shunt_to_bus=jnp.asarray(setup.dynamic_info.shunt_bus_indices, dtype=jnp.int32),
        shunt_connected=jnp.asarray(setup.dynamic_info.shunt_connected),
    )
    voltage_state = VoltageStateInputs(
        bus_voltage_magnitudes=jnp.asarray(setup.dynamic_info_with_placeholders.bus_voltage_magnitudes.flatten()),
        bus_voltage_angles_rad=jnp.asarray(setup.dynamic_info_with_placeholders.bus_voltage_angles_rad.flatten()),
    )
    network_admittance = NetworkAdmittanceInputs(
        y_ff=jnp.asarray(setup.y_ff),
        y_ft=jnp.asarray(setup.y_ft),
        y_tf=jnp.asarray(setup.y_tf),
        y_tt=jnp.asarray(setup.y_tt),
        y_shunt=jnp.asarray(setup.dynamic_info.shunt_effective_bus_admittance),
    )
    jacobian_components = JacobianComponentInputs(
        angle_component_indices=jnp.asarray(setup.jacobian_data_with_extra_buses.angle_component_indices, dtype=jnp.int32),
        magnitude_component_indices=jnp.asarray(
            setup.jacobian_data_with_extra_buses.magnitude_component_indices,
            dtype=jnp.int32,
        ),
    )

    jacobian_inv_jax_transposed = compute_bsdf_update_jax(
        jacobian_inv_transposed=jacobian_inv_device_transposed,
        bus_to_split=setup.bus_to_split,
        new_bus_b_index=setup.new_bus_index,
        new_bus_type=2,
        branches_connected_to_bus_b=jnp.asarray(setup.branches_connected_to_bus_b, dtype=jnp.int32),
        shunt_connected_to_bus_b=jnp.asarray(setup.shunt_connected_to_bus_b, dtype=jnp.int32),
        network_topology=network_topology,
        voltage_state=voltage_state,
        network_admittance=network_admittance,
        jacobian_components=jacobian_components,
    )

    jacobian_inv_jax_transposed.block_until_ready()
    jacobian_inv_jax = jnp.transpose(jacobian_inv_jax_transposed)

    in_use_indices = np.flatnonzero(setup.jacobian_data_with_extra_buses.jacobian_index_in_use)
    np.testing.assert_allclose(
        np.asarray(jacobian_inv_jax)[np.ix_(in_use_indices, in_use_indices)],
        jacobian_inv_numpy[np.ix_(in_use_indices, in_use_indices)],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(jacobian_inv_jax_transposed)[np.ix_(in_use_indices, in_use_indices)],
        jacobian_inv_numpy[np.ix_(in_use_indices, in_use_indices)].T,
        rtol=1e-10,
        atol=1e-10,
    )
