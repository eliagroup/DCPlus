# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from copy import deepcopy

import pypowsybl

from dc_plus.importing.powsybl.powsybl_network_helpers import _load_test_grid
from dc_plus.interfaces.jacobian_network_data import _get_jacobian_data_from_network_data
from dc_plus.preprocess.create_network_data import create_network_data_pypowsbl


def test_disconnected_branches():
    get_net = pypowsybl.network.create_ieee14
    net, static_info, dynamic_info, string_info, jacobian_data = _load_test_grid(get_net)

    dynamic_info_n1 = deepcopy(dynamic_info)
    dynamic_info_n1.branch_connected[0] = False  # disconnect first branch
    jacobian_data_n1 = _get_jacobian_data_from_network_data(dynamic_info_n1)

    net_n1 = deepcopy(net)
    net_n1.remove_elements(string_info.branch_ids[0])
    static_info_n1_direct, dynamic_info_n1_direct, string_info_n1_direct = create_network_data_pypowsbl(net_n1)
    jacobian_data_n1_direct = _get_jacobian_data_from_network_data(dynamic_info_n1_direct)

    assert abs(jacobian_data_n1.jacobian - jacobian_data_n1_direct.jacobian).max() < 1e-10
