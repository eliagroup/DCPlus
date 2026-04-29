# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from dc_plus.numpy.fixed_jacobian import run_fixed_jacobian_iterations
import numpy as np
import pandas as pd
import pypowsybl

from dc_plus.interfaces.jacobian_network_data import (
    _get_admittance_matrix_from_network_data,
    _get_jacobian_data_from_network_data,
    calculate_nodal_mismatch_network_data,
)
from dc_plus.numpy.bsdf_full_rank import compute_bsdf_update
from dc_plus.preprocess.create_network_data import create_network_data_pypowsybl


def test_fixed_jacobian_single_iteration_matches_manual_dx_for_phase_tap_change_without_split(
    micro_grid_be_network_with_replaced_transformers: pypowsybl.network.Network,
):
    net = micro_grid_be_network_with_replaced_transformers
    _static_info_base, dynamic_info_base, string_info_base = create_network_data_pypowsybl(net)
    jacobian_data_base = _get_jacobian_data_from_network_data(dynamic_info_base)

    phase_tap_changer = net.get_phase_tap_changers().reset_index().iloc[0]
    branch_index_by_id = {str(branch_id): idx for idx, branch_id in enumerate(string_info_base.branch_ids)}
    phase_branch_idx = branch_index_by_id[str(phase_tap_changer["id"])]

    updated_tap = int(phase_tap_changer["high_tap"])
    if updated_tap == int(phase_tap_changer["tap"]):
        updated_tap = int(phase_tap_changer["low_tap"])

    net.update_phase_tap_changers(df=pd.DataFrame({"id": [phase_tap_changer["id"]], "tap": [updated_tap]}).set_index("id"))

    _static_info_tap, dynamic_info_tap, _string_info_tap = create_network_data_pypowsybl(net)
    jacobian_data_tap = _get_jacobian_data_from_network_data(dynamic_info_tap)

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

    np.testing.assert_allclose(
        jacobian_inv_bsdf,
        jacobian_data_tap.inverse_jacobian,
        rtol=1e-10,
        atol=1e-10,
    )

    mismatch_tap = calculate_nodal_mismatch_network_data(
        dynamic_network_data=dynamic_info_tap,
        y_matrix=_get_admittance_matrix_from_network_data(dynamic_info_tap),
    )
    dx_bsdf = -jacobian_inv_bsdf @ mismatch_tap
    dx_direct = -jacobian_data_tap.inverse_jacobian @ mismatch_tap

    np.testing.assert_allclose(dx_bsdf, dx_direct, rtol=1e-10, atol=1e-10)

    dynamic_info_bsdf_single_step = run_fixed_jacobian_iterations(
        jacobian_inv=jacobian_inv_bsdf,
        dynamic_network_data=dynamic_info_tap,
        n_iterations=1,
    )
    theta_expected = np.asarray(dynamic_info_tap.bus_voltage_angles_rad, dtype=float).copy()
    vm_expected = np.asarray(dynamic_info_tap.bus_voltage_magnitudes, dtype=float).copy()
    theta_expected[dynamic_info_tap.pvpq_buses_indices_pvpq_order] += dx_bsdf[
        : dynamic_info_tap.n_pv_buses + dynamic_info_tap.n_pq_buses
    ]
    vm_expected[dynamic_info_tap.pq_buses_indices] += dx_bsdf[dynamic_info_tap.n_pv_buses + dynamic_info_tap.n_pq_buses :]

    np.testing.assert_allclose(dynamic_info_bsdf_single_step.bus_voltage_angles_rad, theta_expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        dynamic_info_bsdf_single_step.bus_voltage_magnitudes,
        vm_expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_fixed_jacobian_iterations_improve_far_phase_tap_change(
    micro_grid_be_network_with_replaced_transformers: pypowsybl.network.Network,
):
    net = micro_grid_be_network_with_replaced_transformers
    phase_tap_changer = net.get_phase_tap_changers().reset_index().iloc[0]
    base_tap = int(phase_tap_changer["low_tap"])
    updated_tap = min(base_tap + 10, int(phase_tap_changer["high_tap"]))

    net.update_phase_tap_changers(df=pd.DataFrame({"id": [phase_tap_changer["id"]], "tap": [base_tap]}).set_index("id"))
    pypowsybl.loadflow.run_ac(net)[0]
    _static_info_base, dynamic_info_base, _string_info_base = create_network_data_pypowsybl(net)
    jacobian_data_base = _get_jacobian_data_from_network_data(dynamic_info_base)

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

    mismatch_before = calculate_nodal_mismatch_network_data(dynamic_info_tap, y_matrix_tap)
    dynamic_info_single_iteration = run_fixed_jacobian_iterations(
        jacobian_inv=jacobian_inv_bsdf,
        dynamic_network_data=dynamic_info_tap,
        n_iterations=1,
        y_matrix=y_matrix_tap,
    )
    dynamic_info_two_iterations = run_fixed_jacobian_iterations(
        jacobian_inv=jacobian_inv_bsdf,
        dynamic_network_data=dynamic_info_tap,
        n_iterations=2,
        y_matrix=y_matrix_tap,
    )

    mismatch_after_one = calculate_nodal_mismatch_network_data(dynamic_info_single_iteration, y_matrix_tap)
    mismatch_after_two = calculate_nodal_mismatch_network_data(dynamic_info_two_iterations, y_matrix_tap)

    assert np.max(np.abs(mismatch_after_one)) < np.max(np.abs(mismatch_before))
    assert np.max(np.abs(mismatch_after_one)) < 0.1 * np.max(np.abs(mismatch_before))
    assert np.max(np.abs(mismatch_after_two)) < np.max(np.abs(mismatch_after_one))
    assert np.max(np.abs(mismatch_after_two)) < 0.2 * np.max(np.abs(mismatch_after_one))
