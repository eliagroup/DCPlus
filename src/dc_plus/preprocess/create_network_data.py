# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Functions to create the network data from different input formats."""

import pypowsybl
from pandapower.auxiliary import pandapowerNet

from dc_plus.importing.import_helpers import (
    _get_admittance_branches,
    _get_bus_active_power_injections,
    _get_bus_admittance_shunts,
    _get_bus_reactive_power_injections,
    _remove_isolated_branches,
    _remove_isolated_buses,
    _remove_isolated_buses_injections,
)
from dc_plus.importing.import_schema import (
    BranchParamSchema,
    BusParamSchema,
    InjectionParamSchema,
    LimitParamSchema,
    ShuntParamSchema,
)
from dc_plus.importing.pandapower.import_helpers import _get_slack_bus_id
from dc_plus.importing.pandapower.pandapower_import import (
    _get_branches_parameter_pandapower,
    _get_buses_pandapower,
    _get_injections_pandapower,
    _get_limits_parameter_pandapower,
    _get_shunts_pandapower,
)
from dc_plus.importing.powsybl.powsybl_import import (
    _get_branches_parameter_powsybl,
    _get_buses_powsybl,
    _get_injections_powsybl,
    _get_limits_parameter_powsybl,
    _get_shunts_powsybl,
)
from dc_plus.interfaces.network_information import (
    BusType,
    DynamicNetworkInformation,
    StaticNetworkInformation,
    StringNetworkInformation,
    _check_network_data_consistency,
)
from dc_plus.preprocess.helper_functions import _is_branch_symmetric, _is_connected_to_slack


def _create_network_data(
    buses: BusParamSchema,
    branches: BranchParamSchema,
    injections: InjectionParamSchema,
    limits: LimitParamSchema,
    shunts: ShuntParamSchema,
) -> tuple[StaticNetworkInformation, DynamicNetworkInformation, StringNetworkInformation]:
    """Create the network data from a Powsybl network.

    Creates the central network data structures used in DCplus from a Powsybl network.

    Parameters
    ----------
    buses : BusParamSchema
        The bus parameters of the network.
    branches : BranchParamSchema
        The branch parameters of the network.
    injections : InjectionParamSchema
        The injection parameters of the network.
    limits : LimitParamSchema
        The limit parameters of the network.
    shunts : ShuntParamSchema
        The shunt parameters of the network.

    Returns
    -------
    tuple[StaticNetworkInformation, DynamicNetworkInformation, StringNetworkInformation]
        The static, dynamic and string network information.
    """
    # get only main grid
    buses = _remove_isolated_buses(buses)
    injections = _remove_isolated_buses_injections(buses, injections)
    shunts = _remove_isolated_buses_injections(buses, shunts)
    branches = _remove_isolated_branches(buses, branches)

    y_ff, y_ft, y_tf, y_tt, y_series, y_charging_symmetric = _get_admittance_branches(branches=branches)
    y_shunts = _get_bus_admittance_shunts(shunts=shunts)

    bus_active_power = _get_bus_active_power_injections(injections=injections, n_buses=len(buses))
    bus_reactive_power = _get_bus_reactive_power_injections(injections=injections, n_buses=len(buses))
    is_branch_symmetric = _is_branch_symmetric(
        y_ff=y_ff,
        y_ft=y_ft,
        y_tf=y_tf,
        y_tt=y_tt,
    )

    slack_bus_ids = buses[buses["bus_type"] == BusType.SLACK]["id_int"].values
    is_connected_to_slack = _is_connected_to_slack(
        branch_from_nodes=branches["from_bus_index"].values,
        branch_to_nodes=branches["to_bus_index"].values,
        slack_bus_indices=slack_bus_ids,
    )

    static_info = StaticNetworkInformation(
        injection_limits=None,
        shunt_section_info=None,
        n_limits=None,
        branch_current_limits=None,
        has_phase_shifting_transformer=None,
        has_ratio_changing_transformer=None,
        phase_shift_info=None,
        ratio_shift_info=None,
    )
    dynamic_info = DynamicNetworkInformation(
        branch_from_bus=branches["from_bus_index"].values.astype(int),
        branch_to_bus=branches["to_bus_index"].values.astype(int),
        branch_active_power_from=branches["p1"].values,
        branch_active_power_to=branches["p2"].values,
        branch_reactive_power_from=branches["q1"].values,
        branch_reactive_power_to=branches["q2"].values,
        branch_current_magnitude_from=branches["i1"].values,
        branch_current_magnitude_to=branches["i2"].values,
        branch_ratio_tap_positions=branches["rho"].values,
        branch_phase_tap_positions=branches["alpha"].values,
        branch_effective_admittance_from_to=y_ft,
        branch_effective_admittance_from_from=y_ff,
        branch_effective_admittance_to_to=y_tt,
        branch_effective_admittance_to_from=y_tf,
        branch_effective_admittance_series=y_series,
        branch_effective_admittance_charging_symmetric=y_charging_symmetric,
        branch_connected=branches["connected"].values,
        is_branch_symmetric=is_branch_symmetric,
        is_connected_to_slack=is_connected_to_slack,
        bus_voltage_magnitudes=buses["voltage_magnitude"].values,
        bus_voltage_angles_rad=buses["voltage_angle"].values,
        bus_active_power=bus_active_power,
        bus_reactive_power=bus_reactive_power,
        bus_type=buses["bus_type"].values.astype(int),
        injection_to_bus=injections["bus_index"].values.astype(int),
        injection_active_power=injections["p"].values,
        injection_reactive_power=injections["q"].values,
        injection_connected=injections["connected"].values,
        shunt_bus_indices=shunts["bus_index"].values.astype(int),
        shunt_active_power=shunts["p"].values,
        shunt_reactive_power=shunts["q"].values,
        shunt_section_count=shunts["section_count"].values.astype(int),
        shunt_effective_bus_admittance=y_shunts,
        shunt_connected=shunts["connected"].values,
    )
    string_info = StringNetworkInformation(
        bus_ids=buses["id_str"].values,
        shunt_ids=shunts["id_str"].values,
        branch_types=branches["branch_type"].values,
        branch_ids=branches["id_str"].values,
        limit_names=limits["name"].values,
        injection_types=injections["injection_type"].values,
    )

    _check_network_data_consistency(dynamic_network_data=dynamic_info, string_network_data=string_info)
    return static_info, dynamic_info, string_info


def create_network_data(
    network: pypowsybl.network.Network,
) -> tuple[StaticNetworkInformation, DynamicNetworkInformation, StringNetworkInformation]:
    """Create the network data from a Powsybl network.

    Creates the central network data structures used in DCplus from a Powsybl network.

    Parameters
    ----------
    network : pypowsybl.network.Network
        The Powsybl network.

    Returns
    -------
    tuple[StaticNetworkInformation, DynamicNetworkInformation, StringNetworkInformation]
        The static, dynamic and string network information.
    """
    network.per_unit = True

    branches = _get_branches_parameter_powsybl(network, split_trafo_charging=True)
    injections = _get_injections_powsybl(network)
    shunts = _get_shunts_powsybl(network)
    slack_id = network.get_extensions("slackTerminal")["bus_id"].values[0]
    buses = _get_buses_powsybl(net=network, slack_id=slack_id, injections=injections)
    limits = _get_limits_parameter_powsybl(network)

    return _create_network_data(
        buses=buses,
        branches=branches,
        injections=injections,
        limits=limits,
        shunts=shunts,
    )


# Backwards compatibility alias for existing imports.
create_network_data_pypowsbl = create_network_data


def create_network_data_pandapower(
    network: pandapowerNet,
) -> tuple[StaticNetworkInformation, DynamicNetworkInformation, StringNetworkInformation]:
    """Create the network data from a Pandapower network.

    Creates the central network data structures used in DCplus from a Pandapower network.

    Parameters
    ----------
    network : pandapowerNet
        The Pandapower network.

    Returns
    -------
    tuple[StaticNetworkInformation, DynamicNetworkInformation, StringNetworkInformation]
        The static, dynamic and string network information.
    """
    branches = _get_branches_parameter_pandapower(network, split_trafo_charging=True)
    injections = _get_injections_pandapower(network)
    shunts = _get_shunts_pandapower(network)
    slack_id = _get_slack_bus_id(network)
    buses = _get_buses_pandapower(net=network, slack_id=slack_id)
    limits = _get_limits_parameter_pandapower(network)

    return _create_network_data(
        buses=buses,
        branches=branches,
        injections=injections,
        limits=limits,
        shunts=shunts,
    )
