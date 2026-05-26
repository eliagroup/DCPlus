# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pandas as pd
import pypowsybl
from dc_plus.numpy.fixed_jacobian import run_fixed_jacobian_iterations

from dc_plus.interfaces.jacobian_network_data import (
    _get_admittance_matrix_from_network_data,
    calculate_nodal_mismatch_network_data,
    get_jacobian_data_from_network_data,
)
from dc_plus.importing.powsybl.powsybl_loadflow_parameter import get_powsybl_loadflow_parameter
from dc_plus.numpy.bsdf_full_rank import compute_bsdf_update
from dc_plus.preprocess.create_network_data import create_network_data_pypowsybl
import pytest


def _build_reactive_limit_one_step_parameter() -> pypowsybl.loadflow.Parameters:
    """Build the notebook-style one-step loadflow parameter with reactive limits enabled."""
    provider_param_one_step = {
        "newtonRaphsonConvEpsPerEq": "1e300",
        "maxNewtonRaphsonIterations": "20",
        "svcVoltageMonitoring": "false",
        "newtonRaphsonStoppingCriteriaType": "UNIFORM_CRITERIA",
        "generatorReactivePowerRemoteControl": "false",
        "phaseShifterRegulationOn": "false",
        "alwaysUpdateNetwork": "true",
        "useActiveLimits": "false",
    }
    loadflow_parameter = get_powsybl_loadflow_parameter("default")
    loadflow_parameter.provider_parameters = provider_param_one_step
    loadflow_parameter.distributed_slack = False
    loadflow_parameter.use_reactive_limits = True
    return loadflow_parameter


def test_fixed_jacobian_single_iteration_matches_manual_dx_for_phase_tap_change_without_split(
    micro_grid_be_network_with_replaced_transformers: pypowsybl.network.Network,
):
    net = micro_grid_be_network_with_replaced_transformers
    network_info_base = create_network_data_pypowsybl(net)
    dynamic_info_base = network_info_base.dynamic_network_data
    string_info_base = network_info_base.string_network_data
    jacobian_data_base = get_jacobian_data_from_network_data(
        dynamic_info_base,
    )

    phase_tap_changer = net.get_phase_tap_changers().reset_index().iloc[0]
    branch_index_by_id = {str(branch_id): idx for idx, branch_id in enumerate(string_info_base.branch_ids)}
    phase_branch_idx = branch_index_by_id[str(phase_tap_changer["id"])]

    updated_tap = int(phase_tap_changer["high_tap"])
    if updated_tap == int(phase_tap_changer["tap"]):
        updated_tap = int(phase_tap_changer["low_tap"])

    net.update_phase_tap_changers(df=pd.DataFrame({"id": [phase_tap_changer["id"]], "tap": [updated_tap]}).set_index("id"))

    dynamic_info_tap = create_network_data_pypowsybl(net).dynamic_network_data
    jacobian_data_tap = get_jacobian_data_from_network_data(
        dynamic_info_tap,
    )

    jacobian_inv_bsdf = compute_bsdf_update(
        jacobian_inv=jacobian_data_base.inverse_jacobian,
        bus_to_split=0,
        new_bus_b_index=0,
        new_bus_type=2,
        branches_connected_to_bus_b=np.asarray([], dtype=np.int32),
        shunt_connected_to_bus_b=np.asarray([], dtype=np.int32),
        branch_from=dynamic_info_base.branch_from_bus,
        branch_to=dynamic_info_base.branch_to_bus,
        shunt_to_bus=dynamic_info_base.shunt_bus_indices,
        v_mag_hat=dynamic_info_tap.bus_voltage_magnitudes,
        theta_hat=dynamic_info_tap.bus_voltage_angles_rad,
        y_ff=dynamic_info_tap.branch_effective_admittance_from_from,
        y_ft=dynamic_info_tap.branch_effective_admittance_from_to,
        y_tf=dynamic_info_tap.branch_effective_admittance_to_from,
        y_tt=dynamic_info_tap.branch_effective_admittance_to_to,
        y_shunt=dynamic_info_tap.shunt_effective_bus_admittance,
        angle_component_indices=jacobian_data_base.angle_component_indices,
        magnitude_component_indices=jacobian_data_base.magnitude_component_indices,
        reactive_power_bus_mask=dynamic_info_base.bus_type == 2,
        y_ff_base=dynamic_info_base.branch_effective_admittance_from_from,
        y_ft_base=dynamic_info_base.branch_effective_admittance_from_to,
        y_tf_base=dynamic_info_base.branch_effective_admittance_to_from,
        y_tt_base=dynamic_info_base.branch_effective_admittance_to_to,
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
        jacobian_data=jacobian_data_tap,
    )
    dx_bsdf = -jacobian_inv_bsdf @ mismatch_tap
    dx_direct = -jacobian_data_tap.inverse_jacobian @ mismatch_tap

    np.testing.assert_allclose(dx_bsdf, dx_direct, rtol=1e-10, atol=1e-10)

    y_matrix_tap = _get_admittance_matrix_from_network_data(dynamic_info_tap)

    dynamic_info_bsdf_single_step = run_fixed_jacobian_iterations(
        jacobian_data=jacobian_data_tap.copy_with_inverse_jacobian(jacobian_inv_bsdf),
        dynamic_network_data=dynamic_info_tap,
        n_iterations=1,
        y_matrix=y_matrix_tap,
    )
    non_slack_buses = dynamic_info_tap.pvpq_buses_indices_pvpq_order
    pq_buses = dynamic_info_tap.pq_buses_indices
    theta_expected = dynamic_info_tap.bus_voltage_angles_rad.copy()
    vm_expected = dynamic_info_tap.bus_voltage_magnitudes.copy()
    theta_expected[non_slack_buses] += dx_bsdf[: non_slack_buses.size]
    vm_expected[pq_buses] += dx_bsdf[non_slack_buses.size :]

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
    dynamic_info_base = create_network_data_pypowsybl(net).dynamic_network_data
    jacobian_data_base = get_jacobian_data_from_network_data(
        dynamic_info_base,
    )

    net.update_phase_tap_changers(df=pd.DataFrame({"id": [phase_tap_changer["id"]], "tap": [updated_tap]}).set_index("id"))
    dynamic_info_tap = create_network_data_pypowsybl(net).dynamic_network_data
    y_matrix_tap = _get_admittance_matrix_from_network_data(dynamic_info_tap)

    jacobian_inv_bsdf = compute_bsdf_update(
        jacobian_inv=jacobian_data_base.inverse_jacobian,
        bus_to_split=0,
        new_bus_b_index=0,
        new_bus_type=2,
        branches_connected_to_bus_b=np.asarray([], dtype=np.int32),
        shunt_connected_to_bus_b=np.asarray([], dtype=np.int32),
        branch_from=dynamic_info_base.branch_from_bus,
        branch_to=dynamic_info_base.branch_to_bus,
        shunt_to_bus=dynamic_info_base.shunt_bus_indices,
        v_mag_hat=dynamic_info_tap.bus_voltage_magnitudes,
        theta_hat=dynamic_info_tap.bus_voltage_angles_rad,
        y_ff=dynamic_info_tap.branch_effective_admittance_from_from,
        y_ft=dynamic_info_tap.branch_effective_admittance_from_to,
        y_tf=dynamic_info_tap.branch_effective_admittance_to_from,
        y_tt=dynamic_info_tap.branch_effective_admittance_to_to,
        y_shunt=dynamic_info_tap.shunt_effective_bus_admittance,
        angle_component_indices=jacobian_data_base.angle_component_indices,
        magnitude_component_indices=jacobian_data_base.magnitude_component_indices,
        reactive_power_bus_mask=dynamic_info_base.bus_type == 2,
        y_ff_base=dynamic_info_base.branch_effective_admittance_from_from,
        y_ft_base=dynamic_info_base.branch_effective_admittance_from_to,
        y_tf_base=dynamic_info_base.branch_effective_admittance_to_from,
        y_tt_base=dynamic_info_base.branch_effective_admittance_to_to,
        apply_split_bus_adjustment=False,
    )

    mismatch_before = calculate_nodal_mismatch_network_data(
        dynamic_info_tap,
        y_matrix_tap,
        jacobian_data=jacobian_data_base,
    )
    dynamic_info_single_iteration = run_fixed_jacobian_iterations(
        jacobian_data=jacobian_data_base.copy_with_inverse_jacobian(jacobian_inv_bsdf),
        dynamic_network_data=dynamic_info_tap,
        n_iterations=1,
        y_matrix=y_matrix_tap,
    )
    dynamic_info_two_iterations = run_fixed_jacobian_iterations(
        jacobian_data=jacobian_data_base.copy_with_inverse_jacobian(jacobian_inv_bsdf),
        dynamic_network_data=dynamic_info_tap,
        n_iterations=2,
        y_matrix=y_matrix_tap,
    )

    mismatch_after_one = calculate_nodal_mismatch_network_data(
        dynamic_info_single_iteration,
        y_matrix_tap,
        jacobian_data=jacobian_data_base,
    )
    mismatch_after_two = calculate_nodal_mismatch_network_data(
        dynamic_info_two_iterations,
        y_matrix_tap,
        jacobian_data=jacobian_data_base,
    )

    assert np.max(np.abs(mismatch_after_one)) < np.max(np.abs(mismatch_before))
    assert np.max(np.abs(mismatch_after_one)) < 0.1 * np.max(np.abs(mismatch_before))
    assert np.max(np.abs(mismatch_after_two)) < np.max(np.abs(mismatch_after_one))
    assert np.max(np.abs(mismatch_after_two)) < 0.2 * np.max(np.abs(mismatch_after_one))
