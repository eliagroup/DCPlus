# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from copy import deepcopy

import numpy as np
import pypowsybl
import pytest

from dc_plus.example_grids.pypowsbl.example_grids import (
    PANDAPOWER_NETWORKS_FOR_POWSYBL,
    POWSYBL_NETWORKS,
    basic_node_breaker_network_powsybl,
)
from dc_plus.importing.powsybl.powsybl_import import DANGLING_BUS_STRING_SUFFIX
from dc_plus.importing.powsybl.powsybl_loadflow_parameter import get_powsybl_loadflow_parameter
from dc_plus.importing.powsybl.powsybl_network_helpers import (
    _load_test_grid,
    get_bus_branch_ids_for_n1_results,
    powsybl_n1_analysis,
)
from dc_plus.interfaces.jacobian_network_data import (
    _get_admittance_matrix_from_network_data,
    _get_jacobian_data_from_network_data,
    calculate_nodal_mismatch_network_data,
)
from dc_plus.numpy.lodf import branch_outage_monitored_bus_dx, branch_outage_update_inverse
from dc_plus.preprocess.helper_functions import _find_bridges

powsybl_networks = POWSYBL_NETWORKS
pandapower_networks = PANDAPOWER_NETWORKS_FOR_POWSYBL


def test_lodf_full_rank_update_multi_outage():
    get_net = basic_node_breaker_network_powsybl
    net, static_info, dynamic_info, string_info, jacobian_data = _load_test_grid(get_net)
    theta_actual = dynamic_info.bus_voltage_angles_rad
    vm_actual = dynamic_info.bus_voltage_magnitudes

    is_bridge = _find_bridges(dynamic_info)
    for outage_idx in np.flatnonzero(~is_bridge):
        # reset dynamic info
        dynamic_info_n2 = deepcopy(dynamic_info)
        # N-2 test

        non_bridge_indices = np.flatnonzero(~is_bridge)
        if outage_idx == non_bridge_indices[0]:
            multi_outage_idx = non_bridge_indices[1]
        else:
            multi_outage_idx = non_bridge_indices[0]

        dynamic_info_n2.branch_connected[outage_idx] = False
        dynamic_info_n2.branch_connected[multi_outage_idx] = False

        jacobian_data_n2 = _get_jacobian_data_from_network_data(dynamic_info_n2)
        J_inverse_direct_n2 = jacobian_data_n2.inverse_jacobian

        Yff = dynamic_info.branch_effective_admittance_from_from
        Yft = dynamic_info.branch_effective_admittance_from_to
        Ytf = dynamic_info.branch_effective_admittance_to_from
        Ytt = dynamic_info.branch_effective_admittance_to_to

        outage_idx_array_n2 = np.array([outage_idx, multi_outage_idx], dtype=np.int64)
        jacobian_inv_n2 = branch_outage_update_inverse(
            jacobian_inv=jacobian_data.inverse_jacobian,
            outage_branches_indices=outage_idx_array_n2,
            branch_from=dynamic_info.branch_from_bus,
            branch_to=dynamic_info.branch_to_bus,
            v_mag_hat=dynamic_info.bus_voltage_magnitudes,
            theta_hat=dynamic_info.bus_voltage_angles_rad,
            y_ft=Yft,
            y_tf=Ytf,
            y_ff=Yff,
            y_tt=Ytt,
            angle_component_indices=jacobian_data.angle_component_indices,
            magnitude_component_indices=jacobian_data.magnitude_component_indices,
        )

        assert abs(jacobian_inv_n2 - J_inverse_direct_n2).max() < 1e-10, (
            f"max diff: {abs(jacobian_inv_n2 - J_inverse_direct_n2)}"
        )


@pytest.mark.parametrize("get_net", powsybl_networks + pandapower_networks)
# @pytest.mark.parametrize("get_net", [pypowsybl.network.create_ieee14])
# @pytest.mark.parametrize("get_net", [create_complex_grid_battery_hvdc_svc_3w_trafo])
# @pytest.mark.parametrize("get_net", [pypowsybl.network.create_micro_grid_be_network])
def test_lodf_numpy_full_rank_update_compare_powsybl(get_net):
    net, static_info, dynamic_info, string_info, jacobian_data = _load_test_grid(get_net)
    theta_actual = dynamic_info.bus_voltage_angles_rad
    vm_actual = dynamic_info.bus_voltage_magnitudes
    v_pu_actual = vm_actual * np.exp(theta_actual * 1j)
    j_inverse = jacobian_data.inverse_jacobian

    ### N-1 test
    is_bridge = _find_bridges(dynamic_info)

    loadflow_parameter_one_step = get_powsybl_loadflow_parameter("one_step")
    outage_ids = string_info.branch_ids[~is_bridge]
    sa_res = powsybl_n1_analysis(net=net, outage_grid_ids=outage_ids, loadflow_parameter=loadflow_parameter_one_step)
    sa_bus_results = get_bus_branch_ids_for_n1_results(net, sa_res)

    y_matrix_n0 = _get_admittance_matrix_from_network_data(dynamic_info)
    mismatch_n0 = calculate_nodal_mismatch_network_data(dynamic_network_data=dynamic_info, y_matrix=y_matrix_n0)

    # get a list of bus ids without dangling buses
    # string_info.bus_ids is a numpy array
    bus_ids = string_info.bus_ids
    dangling_bus_mask = np.char.endswith(bus_ids.astype(str), DANGLING_BUS_STRING_SUFFIX)

    # hotstart precition
    if isinstance(get_net(), pypowsybl.network.Network):
        precition = 1e-11
    else:
        precition = 1e-9
    assert abs(mismatch_n0).max() < precition
    dx = -j_inverse @ mismatch_n0

    assert abs(dx).max() < precition

    cases_comapred = 0

    for outage_idx in np.flatnonzero(~is_bridge):
        outage_id = string_info.branch_ids[outage_idx]
        if outage_id not in sa_bus_results.index:
            continue
        is_n1_converged = (
            sa_res.post_contingency_results[outage_id].status
            == pypowsybl._pypowsybl.PostContingencyComputationStatus.CONVERGED
        )
        if not is_n1_converged:
            continue
        jacobian_inv_n1 = branch_outage_update_inverse(
            jacobian_inv=j_inverse,
            outage_branches_indices=np.array([outage_idx], dtype=np.int64),
            branch_from=dynamic_info.branch_from_bus,
            branch_to=dynamic_info.branch_to_bus,
            v_mag_hat=dynamic_info.bus_voltage_magnitudes,
            theta_hat=dynamic_info.bus_voltage_angles_rad,
            y_ft=dynamic_info.branch_effective_admittance_from_to,
            y_tf=dynamic_info.branch_effective_admittance_to_from,
            y_ff=dynamic_info.branch_effective_admittance_from_from,
            y_tt=dynamic_info.branch_effective_admittance_to_to,
            angle_component_indices=jacobian_data.angle_component_indices,
            magnitude_component_indices=jacobian_data.magnitude_component_indices,
        )

        dynamic_info_n1 = deepcopy(dynamic_info)
        dynamic_info_n1.branch_connected[outage_idx] = False
        jacobian_data_n1_direct = _get_jacobian_data_from_network_data(dynamic_info_n1)

        J_inverse_direct_n1 = jacobian_data_n1_direct.inverse_jacobian
        assert abs(jacobian_inv_n1 - J_inverse_direct_n1).max() < 1e-10, (
            f"max diff: {abs(jacobian_inv_n1 - J_inverse_direct_n1)}"
        )
        y_matrix_n1 = _get_admittance_matrix_from_network_data(dynamic_info_n1)
        mismatch_n1 = calculate_nodal_mismatch_network_data(dynamic_network_data=dynamic_info_n1, y_matrix=y_matrix_n1)
        dx = -jacobian_inv_n1 @ mismatch_n1

        # Map Jacobian increments back to bus ordering using the Jacobian mapping
        theta_updated_J = theta_actual.copy()
        vm_updated_J = vm_actual.copy()
        pvpq = dynamic_info_n1.pvpq_buses_indices_pvpq_order
        pq = dynamic_info_n1.pq_buses_indices
        theta_updated_J[pvpq] = theta_actual[pvpq] + dx[jacobian_data_n1_direct.is_angle_component]
        vm_updated_J[pq] = vm_actual[pq] + dx[jacobian_data_n1_direct.is_magnitude_component]

        sa_bus_results_n1 = sa_bus_results.loc[outage_id]
        v_pu_actual_n1_sa = sa_bus_results_n1["v_mag_pu"].values * np.exp(1j * sa_bus_results_n1["v_angle_rad"].values)

        assert abs(theta_updated_J[~dangling_bus_mask] - sa_bus_results_n1["v_angle_rad"].values).max() < 1e-10, (
            f"max diff: {abs(theta_updated_J[~dangling_bus_mask] - sa_bus_results_n1['v_angle_rad'].values)}"
        )
        assert abs(vm_updated_J[~dangling_bus_mask] - sa_bus_results_n1["v_mag_pu"].values).max() < 1e-10, (
            f"max diff: {abs(vm_updated_J[~dangling_bus_mask] - sa_bus_results_n1['v_mag_pu'].values)}"
        )
        cases_comapred += 1

        j_at_mismatch_n1 = j_inverse @ mismatch_n1

        # compare low rank update
        for monitored_bus_idx in range(dynamic_info.n_buses):
            # skip slack bus
            if dynamic_info.slack_indices[0] == monitored_bus_idx:
                continue
            # get the voltage at the monitored bus
            vm_monitored_bus_n1_lodf = vm_updated_J[monitored_bus_idx]
            angle_monitored_bus_n1_lodf = theta_updated_J[monitored_bus_idx]
            v_pu = vm_monitored_bus_n1_lodf * np.exp(1.0j * angle_monitored_bus_n1_lodf)

            delta_theta, delta_u = branch_outage_monitored_bus_dx(
                jacobian_inv=j_inverse,
                j_at_mismatch=j_at_mismatch_n1,
                branch_idx=outage_idx,
                branch_from=dynamic_info.branch_from_bus,
                branch_to=dynamic_info.branch_to_bus,
                v_mag_hat=dynamic_info.bus_voltage_magnitudes,
                theta_hat=dynamic_info.bus_voltage_angles_rad,
                angle_component_indices=jacobian_data.angle_component_indices,
                magnitude_component_indices=jacobian_data.magnitude_component_indices,
                monitor_bus=monitored_bus_idx,
                y_ff=dynamic_info.branch_effective_admittance_from_from,
                y_ft=dynamic_info.branch_effective_admittance_from_to,
                y_tf=dynamic_info.branch_effective_admittance_to_from,
                y_tt=dynamic_info.branch_effective_admittance_to_to,
            )

            new_theta = dynamic_info.bus_voltage_angles_rad[monitored_bus_idx] + delta_theta
            new_mag = dynamic_info.bus_voltage_magnitudes[monitored_bus_idx] + delta_u
            monitored_bus_voltage = new_mag * np.exp(1.0j * new_theta)
            # Compare complex voltage (magnitude+angle) against the SA complex voltage
            assert abs(monitored_bus_voltage - v_pu).max() < 1e-10, f"max diff: {abs(monitored_bus_voltage - v_pu)}"
