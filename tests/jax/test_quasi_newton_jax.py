# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pypowsybl

from dc_plus.interfaces.jacobian_network_data import (
    _get_admittance_matrix_from_network_data,
    _get_jacobian_data_from_network_data,
    calculate_nodal_mismatch_network_data,
)
from dc_plus.jax.quasi_newton import run_quasi_newton_updates as run_quasi_newton_updates_jax
from dc_plus.numpy.bsdf_full_rank import compute_bsdf_update
from dc_plus.numpy.quasi_newton import run_quasi_newton_updates as run_quasi_newton_updates_numpy
from dc_plus.preprocess.create_network_data import create_network_data_pypowsybl

jax.config.update("jax_enable_x64", True)


def _build_jax_quasi_newton_kwargs(dynamic_info):
    return {
        "pvpq_indices": jnp.asarray(dynamic_info.pvpq_buses_indices_pvpq_order, dtype=jnp.int32),
        "pq_indices": jnp.asarray(dynamic_info.pq_buses_indices, dtype=jnp.int32),
        "branch_from": jnp.asarray(dynamic_info.branch_from_bus, dtype=jnp.int32).reshape(-1),
        "branch_to": jnp.asarray(dynamic_info.branch_to_bus, dtype=jnp.int32).reshape(-1),
        "branch_connected": jnp.asarray(dynamic_info.branch_connected, dtype=bool).reshape(-1),
        "shunt_to_bus": jnp.asarray(dynamic_info.shunt_bus_indices, dtype=jnp.int32).reshape(-1),
        "shunt_connected": jnp.asarray(dynamic_info.shunt_connected, dtype=bool).reshape(-1),
        "bus_active_power": jnp.asarray(dynamic_info.bus_active_power, dtype=jnp.float64).reshape(-1),
        "bus_reactive_power": jnp.asarray(dynamic_info.bus_reactive_power, dtype=jnp.float64).reshape(-1),
        "bus_voltage_magnitudes": jnp.asarray(dynamic_info.bus_voltage_magnitudes, dtype=jnp.float64).reshape(-1),
        "bus_voltage_angles_rad": jnp.asarray(dynamic_info.bus_voltage_angles_rad, dtype=jnp.float64).reshape(-1),
        "y_ff": jnp.asarray(dynamic_info.branch_effective_admittance_from_from, dtype=jnp.complex128).reshape(-1),
        "y_ft": jnp.asarray(dynamic_info.branch_effective_admittance_from_to, dtype=jnp.complex128).reshape(-1),
        "y_tf": jnp.asarray(dynamic_info.branch_effective_admittance_to_from, dtype=jnp.complex128).reshape(-1),
        "y_tt": jnp.asarray(dynamic_info.branch_effective_admittance_to_to, dtype=jnp.complex128).reshape(-1),
        "y_shunt": jnp.asarray(dynamic_info.shunt_effective_bus_admittance, dtype=jnp.complex128).reshape(-1),
    }


def test_quasi_newton_reduces_phase_tap_mismatch(
    micro_grid_be_network_with_replaced_transformers: pypowsybl.network.Network,
):
    net = micro_grid_be_network_with_replaced_transformers
    _static_info_base, dynamic_info_base, _string_info_base = create_network_data_pypowsybl(net)
    jacobian_data_base = _get_jacobian_data_from_network_data(dynamic_info_base)

    phase_tap_changer = net.get_phase_tap_changers().reset_index().iloc[0]
    updated_tap = min(int(phase_tap_changer["tap"]) + 10, int(phase_tap_changer["high_tap"]))
    net.update_phase_tap_changers(df=pd.DataFrame({"id": [phase_tap_changer["id"]], "tap": [updated_tap]}).set_index("id"))

    _static_info_tap, dynamic_info_tap, _string_info_tap = create_network_data_pypowsybl(net)
    y_matrix_tap = _get_admittance_matrix_from_network_data(dynamic_info_tap)
    jacobian_inv_bsdf = compute_bsdf_update(
        jacobian_inv=jacobian_data_base.inverse_jacobian,
        bus_to_split=0,
        new_bus_b_index=0,
        new_bus_type=2,
        branches_connected_to_bus_b=np.asarray([], dtype=np.int32),
        shunt_connected_to_bus_b=np.asarray([], dtype=np.int32),
        branch_from=np.asarray(dynamic_info_base.branch_from_bus, dtype=np.int32),
        branch_to=np.asarray(dynamic_info_base.branch_to_bus, dtype=np.int32),
        shunt_to_bus=np.asarray(dynamic_info_base.shunt_bus_indices, dtype=np.int32),
        v_mag_hat=np.asarray(dynamic_info_tap.bus_voltage_magnitudes, dtype=float),
        theta_hat=np.asarray(dynamic_info_tap.bus_voltage_angles_rad, dtype=float),
        y_ff=np.asarray(dynamic_info_tap.branch_effective_admittance_from_from, dtype=np.complex128),
        y_ft=np.asarray(dynamic_info_tap.branch_effective_admittance_from_to, dtype=np.complex128),
        y_tf=np.asarray(dynamic_info_tap.branch_effective_admittance_to_from, dtype=np.complex128),
        y_tt=np.asarray(dynamic_info_tap.branch_effective_admittance_to_to, dtype=np.complex128),
        y_shunt=np.asarray(dynamic_info_tap.shunt_effective_bus_admittance, dtype=np.complex128),
        angle_component_indices=jacobian_data_base.angle_component_indices,
        magnitude_component_indices=jacobian_data_base.magnitude_component_indices,
        y_ff_base=np.asarray(dynamic_info_base.branch_effective_admittance_from_from, dtype=np.complex128),
        y_ft_base=np.asarray(dynamic_info_base.branch_effective_admittance_from_to, dtype=np.complex128),
        y_tf_base=np.asarray(dynamic_info_base.branch_effective_admittance_to_from, dtype=np.complex128),
        y_tt_base=np.asarray(dynamic_info_base.branch_effective_admittance_to_to, dtype=np.complex128),
        apply_split_bus_adjustment=False,
    )
    dynamic_info_quasi_newton_np, mismatch_history_np = run_quasi_newton_updates_numpy(
        jacobian_inv=jacobian_inv_bsdf,
        dynamic_network_data=dynamic_info_tap,
        n_iterations=10,
        y_matrix=y_matrix_tap,
    )

    results_jax = run_quasi_newton_updates_jax(
        jacobian_inv_transposed=jnp.asarray(jacobian_inv_bsdf.T),
        n_iterations=10,
        **_build_jax_quasi_newton_kwargs(dynamic_info_tap),
    )

    np.testing.assert_allclose(
        np.asarray(results_jax.bus_voltage_magnitudes),
        dynamic_info_quasi_newton_np.bus_voltage_magnitudes,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(results_jax.bus_voltage_angles_rad),
        dynamic_info_quasi_newton_np.bus_voltage_angles_rad,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(results_jax.mismatch_history),
        np.asarray(mismatch_history_np),
        rtol=1e-8,
        atol=1e-10,
    )

    mismatch_before = calculate_nodal_mismatch_network_data(dynamic_info_tap, y_matrix_tap)
    dynamic_info_quasi_newton_jax = replace(
        dynamic_info_tap,
        bus_voltage_magnitudes=np.asarray(results_jax.bus_voltage_magnitudes),
        bus_voltage_angles_rad=np.asarray(results_jax.bus_voltage_angles_rad),
    )
    mismatch_after = calculate_nodal_mismatch_network_data(dynamic_info_quasi_newton_jax, y_matrix_tap)

    assert np.max(np.abs(mismatch_after)) < np.max(np.abs(mismatch_before))
    assert np.all(np.isfinite(np.asarray(results_jax.mismatch_history)))
    assert results_jax.mismatch_history[-1] < 1e-10
