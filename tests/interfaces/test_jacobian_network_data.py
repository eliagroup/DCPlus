# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from copy import copy, deepcopy

import numpy as np
import pypowsybl

from dc_plus.importing.powsybl.powsybl_loadflow_parameter import get_powsybl_loadflow_parameter
from dc_plus.importing.powsybl.powsybl_network_helpers import _load_test_grid
from dc_plus.interfaces.jacobian_network_data import (
    _get_admittance_matrix_from_network_data,
    _get_jacobian_data_from_network_data,
    calculate_nodal_mismatch_network_data,
)
from dc_plus.preprocess.create_network_data import create_network_data_pypowsbl
from dc_plus.preprocess.helper_functions import _find_bridges


def test_jacobian_update():
    get_net = pypowsybl.network.create_ieee14
    net, static_info, dynamic_info, string_info, jacobian_data = _load_test_grid(get_net)
    theta_actual = dynamic_info.bus_voltage_angles_rad
    vm_actual = dynamic_info.bus_voltage_magnitudes

    ### N-1 test
    is_bridge = _find_bridges(dynamic_info)

    pvpq_bus = dynamic_info.pvpq_buses_indices_pvpq_order
    pq_bus = dynamic_info.pq_buses_indices

    for outage_idx in np.flatnonzero(~is_bridge):
        dynamic_info_n1 = deepcopy(dynamic_info)
        dynamic_info_n1.branch_connected[outage_idx] = False
        jacobian_data_n1 = _get_jacobian_data_from_network_data(dynamic_info_n1)

        net_n1 = deepcopy(net)
        net_n1.remove_elements(string_info.branch_ids[outage_idx])
        static_info_n1_direct, dynamic_info_n1_direct, string_info_n1_direct = create_network_data_pypowsbl(net_n1)
        jacobian_data_n1_direct = _get_jacobian_data_from_network_data(dynamic_info_n1_direct)
        assert jacobian_data_n1.__eq__(jacobian_data_n1_direct), (
            "Jacobian data from n-1 network data and direct n-1 network data do not match."
        )

        loadflow_parameter_one_step = get_powsybl_loadflow_parameter("one_step")
        loadflow_res = pypowsybl.loadflow.run_ac(net_n1, parameters=loadflow_parameter_one_step)[0]
        if loadflow_res.status != pypowsybl._pypowsybl.LoadFlowComponentStatus.CONVERGED:
            raise ValueError(
                f"Load flow did not converge. Status: {loadflow_res.status}, "
                f"Status text: {loadflow_res.status_text}, "
                f"Reference bus ID: {loadflow_res.reference_bus_id}"
            )

        static_info_n1_direct_lf, dynamic_info_n1_direct_lf, string_info_n1_direct_lf = create_network_data_pypowsbl(net_n1)
        theta_actual_n1 = dynamic_info_n1_direct_lf.bus_voltage_angles_rad
        vm_actual_n1 = dynamic_info_n1_direct_lf.bus_voltage_magnitudes

        J_inverse_direct_n1 = jacobian_data_n1.inverse_jacobian
        y_matrix_n1 = _get_admittance_matrix_from_network_data(dynamic_info_n1)
        mismatch_n1 = calculate_nodal_mismatch_network_data(dynamic_network_data=dynamic_info_n1, y_matrix=y_matrix_n1)
        dx = -J_inverse_direct_n1 @ mismatch_n1

        theta_updated_J = copy(theta_actual)
        vm_updated_J = copy(vm_actual)
        theta_updated_J[pvpq_bus] = theta_actual[pvpq_bus] + dx[jacobian_data_n1_direct.is_angle_component]
        vm_updated_J[pq_bus] = vm_actual[pq_bus] + dx[jacobian_data_n1_direct.is_magnitude_component]
        assert abs(theta_updated_J - theta_actual_n1).max() < 1e-10
        assert abs(vm_updated_J - vm_actual_n1).max() < 1e-10
