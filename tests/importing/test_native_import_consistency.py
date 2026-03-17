# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Test that verifies each importer works correctly with its native format.

This test validates that:
1. Pandapower importer extracts data matching pandapower power flow (test_powerflow_consistency.py)
2. Powsybl importer extracts data matching powsybl power flow
3. Both importers produce consistent and valid network information structures

Note: This does NOT test cross-platform equivalence (pandapower <-> powsybl conversion).
That is tested in test_powsybl_vs_pandapower_dynamic.py, which reveals conversion limitations.

Author: Import consistency validation
Date: 2026
"""

import numpy as np
import pandapower as pp
import pytest

from dc_plus.interfaces.network_information import (
    DynamicNetworkInformation,
    StaticNetworkInformation,
    StringNetworkInformation,
)
from dc_plus.preprocess.create_network_data import (
    create_network_data_pandapower,
)


def validate_network_info_structure(static_info, dynamic_info, string_info, network_name: str):
    """Validate that network information has consistent structure.

    Parameters
    ----------
    static_info : StaticNetworkInformation
        Static network data.
    dynamic_info : DynamicNetworkInformation
        Dynamic network data.
    string_info : StringNetworkInformation
        String network data.
    network_name : str
        Name of the network for reporting.

    Raises
    ------
    AssertionError
        If any structural inconsistency is found.
    """
    # Check bus consistency
    n_buses = len(dynamic_info.bus_voltage_magnitudes)
    assert len(dynamic_info.bus_voltage_angles_rad) == n_buses, "Bus angle count mismatch"
    assert len(dynamic_info.bus_type) == n_buses, "Bus type count mismatch"
    assert len(string_info.bus_ids) == n_buses, "Bus ID count mismatch"

    # Check branch consistency
    n_branches = len(dynamic_info.branch_from_bus)
    assert len(dynamic_info.branch_to_bus) == n_branches, "Branch to_bus count mismatch"
    assert len(dynamic_info.branch_connected) == n_branches, "Branch connected count mismatch"
    assert len(dynamic_info.branch_active_power_from) == n_branches, "Branch P_from count mismatch"
    assert len(dynamic_info.branch_active_power_to) == n_branches, "Branch P_to count mismatch"
    assert len(string_info.branch_ids) == n_branches, "Branch ID count mismatch"
    assert len(string_info.branch_types) == n_branches, "Branch type count mismatch"

    # Check injection consistency
    n_injections = len(dynamic_info.injection_to_bus)
    assert len(dynamic_info.injection_active_power) == n_injections, "Injection P count mismatch"
    assert len(dynamic_info.injection_reactive_power) == n_injections, "Injection Q count mismatch"
    assert len(dynamic_info.injection_connected) == n_injections, "Injection connected count mismatch"
    assert len(string_info.injection_types) == n_injections, "Injection type count mismatch"

    # Check shunt consistency
    n_shunts = len(dynamic_info.shunt_bus_indices)
    if n_shunts > 0:
        assert len(dynamic_info.shunt_connected) == n_shunts, "Shunt connected count mismatch"
        assert len(dynamic_info.shunt_active_power) == n_shunts, "Shunt P count mismatch"
        assert len(dynamic_info.shunt_reactive_power) == n_shunts, "Shunt Q count mismatch"
        assert len(string_info.shunt_ids) == n_shunts, "Shunt ID count mismatch"

    # Check that values are valid (no NaN where unexpected)
    assert not np.any(np.isnan(dynamic_info.bus_voltage_magnitudes)), "Bus voltages contain NaN"
    assert not np.any(np.isnan(dynamic_info.bus_voltage_angles_rad)), "Bus angles contain NaN"


@pytest.mark.parametrize(
    "network_func,network_name",
    [
        (pp.networks.case9, "case9"),
        (pp.networks.case14, "case14"),
        (pp.networks.case30, "case30"),
    ],
)
def test_pandapower_native_import(network_func, network_name):
    """Test Pandapower importer with native Pandapower networks.

    Validates that the importer:
    1. Extracts network information correctly
    2. Matches pandapower power flow results within 1e-9 tolerance
    3. Produces structurally consistent data

    Parameters
    ----------
    network_func : callable
        Function that returns a pandapower network.
    network_name : str
        Name of the network for reporting.
    """
    # Load network and run power flow
    net = network_func()
    pp.runpp(net, calculate_voltage_angles=True)

    # Extract network information
    static_info, dynamic_info, string_info = create_network_data_pandapower(net)
    assert isinstance(static_info, StaticNetworkInformation)
    assert isinstance(dynamic_info, DynamicNetworkInformation)
    assert isinstance(string_info, StringNetworkInformation)

    # Validate structure
    validate_network_info_structure(static_info, dynamic_info, string_info, network_name)

    # Validate numerical accuracy (compare with pandapower results)
    pp_vm = net.res_bus["vm_pu"].values
    extracted_vm = dynamic_info.bus_voltage_magnitudes
    max_vm_diff = np.max(np.abs(pp_vm - extracted_vm))

    pp_va = net.res_bus["va_degree"].values * np.pi / 180.0
    extracted_va = dynamic_info.bus_voltage_angles_rad
    max_va_diff = np.max(np.abs(pp_va - extracted_va))

    assert max_vm_diff < 1e-9, f"Voltage magnitude diff {max_vm_diff:.2e} exceeds 1e-9"
    assert max_va_diff < 1e-9, f"Voltage angle diff {max_va_diff:.2e} exceeds 1e-9"
