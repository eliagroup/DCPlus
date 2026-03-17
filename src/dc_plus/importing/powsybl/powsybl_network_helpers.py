# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Powsybl network helper functions."""

import logging
import tempfile
from typing import Optional

import numpy as np
import pandapower
import pandas as pd
import pypowsybl
import pypowsybl.loadflow
import pytest
from pypowsybl._pypowsybl import LoadFlowComponentStatus
from pypowsybl.loadflow import Parameters as LoadFlowParameters
from pypowsybl.network import Network
from pypowsybl.security.impl.parameters import Parameters as SecurityParameters
from pypowsybl.security.impl.security_analysis_result import SecurityAnalysisResult

from dc_plus.importing.powsybl.powsybl_import_helpers import select_a_generator_as_slack_and_run_loadflow
from dc_plus.importing.powsybl.powsybl_loadflow_parameter import get_powsybl_loadflow_parameter
from dc_plus.interfaces.jacobian_interface import JacobianInterface
from dc_plus.interfaces.jacobian_network_data import _get_jacobian_data_from_network_data
from dc_plus.interfaces.network_information import (
    DynamicNetworkInformation,
    StaticNetworkInformation,
    StringNetworkInformation,
)
from dc_plus.preprocess.create_network_data import create_network_data_pypowsbl

logger = logging.getLogger(__name__)


def powsybl_n1_analysis(
    net: Network,
    outage_grid_ids: Optional[list[str]] = None,
    monitored_voltage_level_ids: Optional[list[str]] = None,
    monitored_branch_ids: Optional[list[str]] = None,
    run_ac: bool = True,
    loadflow_parameter: Optional[LoadFlowParameters] = None,
) -> SecurityAnalysisResult:
    """Run a powsybl N-1 analysis on the network.

    This function runs a security analysis on the network, considering the specified outages and monitored elements.

    Parameters
    ----------
    net : Network
        The powsybl network to analyze.
    outage_grid_ids : Optional[list[str]], optional
        A list of IDs of the grid elements to be considered as outages. If None, all branches will be considered.
        Defaults to None.
    monitored_voltage_level_ids : Optional[list[str]], optional
        A list of IDs of voltage levels to be monitored during the analysis. If None, all voltage levels will be monitored.
        Defaults to None.
    monitored_branch_ids : Optional[list[str]], optional
        A list of IDs of branches to be monitored during the analysis. If None, all branches will be monitored.
        Defaults to None.
    run_ac : bool, optional
        If True, the analysis will be run using AC load flow. If False, it will be run using DC load flow.
        Defaults to True.
    loadflow_parameter : Optional[LoadFlowParameters], optional
        The load flow parameters to use for the analysis. If None, default parameters will be used.
        Defaults to None.
    """
    if loadflow_parameter is None:
        loadflow_parameter = get_powsybl_loadflow_parameter("real")

    sa_param = SecurityParameters(
        load_flow_parameters=loadflow_parameter,
        provider_parameters={"threadCount": "20", "contingencyPropagation": "false"},
    )

    security_analysis = pypowsybl.security.create_analysis()
    if outage_grid_ids is None:
        outage_grid_ids = list(net.get_branches(attributes=[]).index)
    if monitored_voltage_level_ids is None:
        monitored_voltage_level_ids = list(net.get_voltage_levels(attributes=[]).index)
    if monitored_branch_ids is None:
        monitored_branch_ids = list(net.get_branches(attributes=[]).index)

    security_analysis.add_single_element_contingencies(outage_grid_ids)

    security_analysis.add_monitored_elements(voltage_level_ids=monitored_voltage_level_ids, branch_ids=monitored_branch_ids)

    if run_ac:
        results = security_analysis.run_ac(net, parameters=sa_param)
    else:
        results = security_analysis.run_dc(net, parameters=sa_param)

    return results


def get_bus_branch_ids_for_n1_results(net: Network, security_analysis_result: SecurityAnalysisResult) -> pd.DataFrame:
    """Get the bus_branch IDs for the N-1 results.

    The open loadflow results returns a mixture of bus ids. We want the bus_id refering to the Bus-Branch model.

    Parameters
    ----------
    net : Network
        The powsybl network to analyze.
    security_analysis_result : SecurityAnalysisResult
        The results of the security analysis.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the bus IDs for the bus-branch model.
        security_analysis_result.bus_results now contains the "bus_id_bus_branch" column.
    """
    vl = net.get_voltage_levels(attributes=["nominal_v"])
    sa_bus_results = security_analysis_result.bus_results.reset_index()
    sa_bus_results = sa_bus_results.merge(vl, left_on="voltage_level_id", right_index=True, how="left")
    sa_bus_results["v_mag_pu"] = sa_bus_results["v_mag"] / sa_bus_results["nominal_v"]
    sa_bus_results["v_angle_rad"] = sa_bus_results["v_angle"] * np.pi / 180.0

    buses = net.get_bus_breaker_view_buses(attributes=["bus_id"])
    sa_bus_results = sa_bus_results.merge(
        buses, left_on="bus_id", right_index=True, suffixes=("_bus_breaker_view", ""), how="left"
    )
    sa_bus_results.sort_values(by=["bus_id"], inplace=True)
    # drop duplicates that can appear due to bus breaker view
    # drop duplictated bus_ids per unique contingency_id
    duplicates = sa_bus_results.duplicated(subset=["contingency_id", "bus_id"], keep="first")
    sa_bus_results = sa_bus_results[~duplicates]
    sa_bus_results = sa_bus_results.set_index("contingency_id")

    return sa_bus_results


def _load_test_grid(
    get_net: callable,
) -> tuple[Network, StaticNetworkInformation, DynamicNetworkInformation, StringNetworkInformation, JacobianInterface]:
    """Load a test pandapower or powsybl grid and prepare the network data and Jacobian."""
    is_powsybl_net = isinstance(get_net(), pypowsybl.network.Network)
    if is_powsybl_net:
        net = get_net()  # type: pypowsybl.network.Network
    else:
        net_pandapower = get_net()
        try:
            net = load_pandapower_net_for_powsybl(net_pandapower)
        except Exception as e:
            pytest.skip(f"Pandapower loading failed: {e}")
    try:
        select_a_generator_as_slack_and_run_loadflow(net)
    except Exception as e:
        pytest.skip(f"Slack selection failed: {e}")
    pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers(net)

    net.per_unit = True
    loadflow_parameter = get_powsybl_loadflow_parameter("hotstart_test")
    loadflow_parameter.provider_parameters["newtonRaphsonConvEpsPerEq"] = "1e-12"
    loadflow_res_n0 = pypowsybl.loadflow.run_ac(net, parameters=loadflow_parameter)[0]
    if loadflow_res_n0.status != pypowsybl._pypowsybl.LoadFlowComponentStatus.CONVERGED:
        raise ValueError(
            f"Load flow did not converge. Status: {loadflow_res_n0.status}, "
            f"Status text: {loadflow_res_n0.status_text}, "
            f"Reference bus ID: {loadflow_res_n0.reference_bus_id}"
        )
    static_info, dynamic_info, string_info = create_network_data_pypowsbl(net)
    jacobian_data = _get_jacobian_data_from_network_data(dynamic_info)

    return net, static_info, dynamic_info, string_info, jacobian_data


def load_pandapower_net_for_powsybl(net: pandapower.pandapowerNet) -> pypowsybl.network.Network:
    """Load a pandapower network and convert it to a pypowsybl network.

    Known pandapower test grids that fail to convert:
    (This list is a logical AND of convert_from_pandapower and grid2opt conversion methods
    + the logical OR of check_powsybl_import errors)
    'example_multivoltage'      -> Generator minimum reactive power is not set
    'simple_four_bus_system'    -> Generator minimum reactive power is not set
    'simple_mv_open_ring_net'   -> 2 windings transformer '0_1_6': b is invalid
    'create_cigre_network_hv'   -> Generator minimum reactive power is not set
    'case145'                   -> Transformer with negative resistance

    Parameters
    ----------
    net : pandapower.pandapowerNet
        The pandapower network to convert.

    Returns
    -------
    pypowsybl.network.Network
        The converted pypowsybl network.

    """
    try:
        pypowsybl_network = load_pandapower_net_for_powsybl_with_convert_from_pandapower(net)
        check_powsybl_import(pypowsybl_network)
    except (pypowsybl.PyPowsyblError, ValueError) as e:
        try:
            pypowsybl_network = load_pandapower_net_via_grid2opt_for_powsybl(net)
            check_powsybl_import(pypowsybl_network)
        except Exception as e2:
            raise ValueError(
                f"Failed to convert pandapower net to pypowsybl network. "
                f"pypowsybl.network.convert_from_pandapower: {e}. Conversion via grid2opt failed with error: {e2}"
            ) from e2

    return pypowsybl_network


def load_pandapower_net_for_powsybl_with_convert_from_pandapower(net: pandapower.pandapowerNet) -> pypowsybl.network.Network:
    """Load a pandapower network and convert it to a pypowsybl network using convert_from_pandapower.

    Known pandapower test grids that fail to convert:
    'example_simple'            -> Generator minimum reactive power is not set
    'example_multivoltage'      -> Generator minimum reactive power is not set
    'simple_four_bus_system'    -> Generator minimum reactive power is not set
    'simple_mv_open_ring_net'   -> 2 windings transformer '0_1_6': b is invalid
    'create_cigre_network_hv'   -> Generator minimum reactive power is not set

    Parameters
    ----------
    net : pandapower.pandapowerNet
        The pandapower network to convert.

    Returns
    -------
    pypowsybl.network.Network
        The converted pypowsybl network.
    """
    pypowsybl_network = pypowsybl.network.convert_from_pandapower(net)
    return pypowsybl_network


def load_pandapower_net_via_grid2opt_for_powsybl(
    net: pandapower.pandapowerNet,
) -> pypowsybl.network.Network:
    """Load a pandapower network and convert it to a pypowsybl network using grid2opt.

    Known pandapower test grids that fail to convert:
    'example_multivoltage'      -> Transformer with negative resistance
    'create_cigre_network_hv'   -> Line with different voltage levels -> failed transformer conversion
    'case14'                    -> Line with different voltage levels -> failed transformer conversion
    'case_ieee30'               -> Line with different voltage levels -> failed transformer conversion
    'case57'                    -> Line with different voltage levels -> failed transformer conversion
    'case89pegase'              -> Line with different voltage levels -> failed transformer conversion
    'case118'                   -> Line with different voltage levels -> failed transformer conversion
    'case145'                   -> Transformer with negative resistance
    'case_illinois200'          -> Line with different voltage levels -> failed transformer conversion
    'case300'                   -> Line with different voltage levels -> failed transformer conversion

    Parameters
    ----------
    net : pandapower.pandapowerNet
        The pandapower network to convert.

    Returns
    -------
    pypowsybl.network.Network
        The converted pypowsybl network.
    """
    pandapower.runpp(net)
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=True) as tmpfile:
        _ = pandapower.converter.to_mpc(net, tmpfile.name)
        loading_params = {
            "matpower.import.ignore-base-voltage": "false",  # change the voltage from per unit to Kv
        }
        pypowsybl_network = pypowsybl.network.load(tmpfile.name, loading_params)
    return pypowsybl_network


def check_powsybl_import(pypowsybl_network: pypowsybl.network.Network) -> None:
    """Check the import of a pypowsybl network.

    Parameters
    ----------
    pypowsybl_network : pypowsybl.network.Network
        The pypowsybl network to test.

    Raises
    ------
    ValueError
        If a transformer with negative resistance is found in the converted network.
        If a line with different voltage levels is found in the converted network.
        If the load flow does not converge.
    """
    # importing pn.example_multivoltage -> one transformer has negative resistance
    transformers = pypowsybl_network.get_2_windings_transformers()
    if len(transformers[transformers["r"] < 0]) > 0:
        raise ValueError("A Transformer in the converted pandapower net has a negive resistance")

    # test if lines have the same voltage level
    line_voltage = pypowsybl_network.get_lines(attributes=["voltage_level1_id", "voltage_level2_id"])
    line_voltage = line_voltage.merge(
        pypowsybl_network.get_voltage_levels(attributes=["nominal_v"]), left_on="voltage_level1_id", right_index=True
    )
    line_voltage = line_voltage.merge(
        pypowsybl_network.get_voltage_levels(attributes=["nominal_v"]),
        left_on="voltage_level2_id",
        right_index=True,
        suffixes=("_vl1", "_vl2"),
    )
    if not all(line_voltage["nominal_v_vl1"] == line_voltage["nominal_v_vl2"]):
        raise ValueError("A Line in the converted pandapower net has two different voltage levels")

    powsybl_loadflow_param = get_powsybl_loadflow_parameter("academic")
    loadflow_res = pypowsybl.loadflow.run_ac(pypowsybl_network, parameters=powsybl_loadflow_param)[0]
    if loadflow_res.status != LoadFlowComponentStatus.CONVERGED:
        raise ValueError(f"Load flow failed: {loadflow_res.status_text}")
