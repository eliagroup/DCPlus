# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Import data from powsybl network."""

import logging

import numpy as np
import pandas as pd
from pypowsybl.network import Network

from dc_plus.importing.import_schema import (
    BranchParamSchema,
    BusParamSchema,
    InjectionParamSchema,
    LimitParamSchema,
    ShuntParamSchema,
)

logger = logging.getLogger(__name__)

DANGLING_BUS_STRING_SUFFIX = "_dangling_bus"


def _get_unpaired_dangling_lines(net: Network) -> pd.DataFrame:
    """Get the unpaired dangling lines of the network.

    Gets the unpaired dangling lines from the network.

    Parameters
    ----------
    net : Network
        The powsybl network.

    Returns
    -------
    pd.DataFrame
        The unpaired dangling lines of the network.
    """
    available_attributes = ["paired"]
    dangling_lines = net.get_dangling_lines(attributes=available_attributes)
    # get all non-paired dangling lines
    dangling_lines = dangling_lines[~dangling_lines["paired"]]
    dangling_lines.drop(columns=["paired"], inplace=True)
    return dangling_lines


def _get_branches_id_int(net: Network) -> pd.Series:
    """Get the branches id_int of the network.

    Gets the branches id_int from the network.
    Note: the index is sorted by branch id_str -> ensure consistent ordering.

    Parameters
    ----------
    net : Network
        The powsybl network.

    Returns
    -------
    pd.Series
        The branches id_int of the network.
    """
    branches = net.get_branches(attributes=[])
    dangling_lines = _get_unpaired_dangling_lines(net)
    branches = pd.concat([branches, dangling_lines], ignore_index=False).sort_index()
    branches = branches.reset_index(drop=False)
    branches.rename(columns={"id": "id_str"}, inplace=True)
    branches["id_int"] = np.arange(len(branches))
    branches.set_index("id_str", inplace=True)
    return branches


def _get_dangling_bus_ids(net: Network) -> pd.DataFrame:
    """Get the dangling bus ids of the network.

    Gets the dangling bus ids from the network.

    Parameters
    ----------
    net : Network
        The powsybl network.

    Returns
    -------
    pd.DataFrame
        The dangling bus ids of the network.
    """
    dangling_lines = _get_unpaired_dangling_lines(net).sort_index()
    dangling_lines["dangling_bus_id"] = dangling_lines.index + DANGLING_BUS_STRING_SUFFIX
    return dangling_lines


def _get_bus_ids_with_dangling_buses(net: Network) -> pd.Series:
    """Get the bus ids including dangling buses of the network.

    Gets the bus ids including dangling buses from the network.
    Note: the index is sorted by bus id_str -> ensure consistent ordering.

    Parameters
    ----------
    net : Network
        The powsybl network.

    Returns
    -------
    pd.Series
        The bus ids including dangling buses of the network.
    """
    bus = net.get_buses(attributes=[])
    dangling_bus_ids = _get_dangling_bus_ids(net)
    dangling_bus_ids.index = dangling_bus_ids["dangling_bus_id"]
    dangling_bus_ids.drop(columns=["dangling_bus_id"], inplace=True)
    bus = pd.concat([bus, dangling_bus_ids], ignore_index=False).sort_index()
    bus = bus.reset_index(drop=False)
    if "index" in bus.columns:
        bus.rename(columns={"index": "id_str"}, inplace=True)
    else:
        bus.rename(columns={"id": "id_str"}, inplace=True)
    bus.reset_index(drop=False, inplace=True)
    bus.rename(columns={"index": "id_int"}, inplace=True)
    return bus


def _get_branches_with_bus_index(net: Network, branches: pd.DataFrame) -> pd.DataFrame:
    """Get the branches with bus index.

    Gets the branches with bus index from the network.
    Note: the bus index is the index of the bus dataframe sorted by bus id.

    Parameters
    ----------
    net : pypowsybl.network
        The network to get the branches with bus index from.
    branches : pd.DataFrame
        The branches to get the bus index for.
        expected columns: ["bus1_id", "bus2_id"]

    Returns
    -------
    pd.DataFrame
        The branches with bus index of the network.
    """
    bus = _get_bus_ids_with_dangling_buses(net)
    branches["from_bus_index"] = branches["bus1_id"].map(dict(zip(bus["id_str"], bus.index, strict=False)))
    branches["to_bus_index"] = branches["bus2_id"].map(dict(zip(bus["id_str"], bus.index, strict=False)))
    return branches


def _get_injection_with_bus_index(
    net: Network, injections: pd.DataFrame, source_column: str = "bus_id", target_column: str = "bus_index"
) -> pd.DataFrame:
    """Get the injections with bus index.

    Gets the injections with bus index from the network.
    Note: the bus index is the index of the bus dataframe sorted by bus id.

    Parameters
    ----------
    net : pypowsybl.network
        The network to get the injections with bus index from.
    injections : pd.DataFrame
        The injections to get the bus index for.
        expected columns: source_column
    source_column : str
        The source column name in the injections dataframe.
    target_column : str
        The target column name to create in the injections dataframe.


    Returns
    -------
    pd.DataFrame
        The injections with bus index of the network.
    """
    bus = _get_bus_ids_with_dangling_buses(net)
    injections[target_column] = injections[source_column].map(dict(zip(bus["id_str"], bus.index, strict=False)))
    return injections


def _get_tie_line_parameter(net: Network) -> BranchParamSchema:
    """Get the tie lines with ["r", "x", "g", "b"] values of the network.

    Gets the Tie lines and merges them with the dangling lines to get the values of the network.

    Parameters
    ----------
    net : Network
        The powsybl network.

    Returns
    -------
    BranchParamSchema
        The tie lines with ["r", "x", "g", "b"] values of the network.
    """
    tie_lines = net.get_tie_lines()
    if tie_lines.empty:
        tie_lines = pd.DataFrame(columns=list(BranchParamSchema.__annotations__.keys()))
        tie_lines = BranchParamSchema.validate(tie_lines)
        return tie_lines
    available_attributes = ["name", "connected", "r", "x", "g", "b", "p", "q", "i", "bus_id"]
    dangling_lines = net.get_dangling_lines(attributes=available_attributes)
    tie_lines = tie_lines.merge(dangling_lines, how="left", left_on=["dangling_line1_id"], right_index=True)
    tie_lines = tie_lines.merge(
        dangling_lines, how="left", left_on=["dangling_line2_id"], right_index=True, suffixes=("1", "2")
    )
    tie_lines["r"] = tie_lines["r1"] + tie_lines["r2"]
    tie_lines["x"] = tie_lines["x1"] + tie_lines["x2"]
    tie_lines["connected"] = tie_lines["connected1"] & tie_lines["connected2"]
    tie_lines.rename(columns={"bus_id1": "bus1_id"}, inplace=True)
    tie_lines.rename(columns={"bus_id2": "bus2_id"}, inplace=True)
    tie_lines = _get_branches_with_bus_index(net, tie_lines)

    available_attributes = [
        "name",
        "connected",
        "r",
        "x",
        "g1",
        "b1",
        "p1",
        "q1",
        "i1",
        "g2",
        "b2",
        "p2",
        "q2",
        "i2",
        "from_bus_index",
        "to_bus_index",
    ]
    tie_lines = tie_lines[available_attributes]
    tie_lines["rho"] = 1.0
    tie_lines["alpha"] = 0.0
    tie_lines.reset_index(drop=False, inplace=True)
    tie_lines.rename(columns={"id": "id_str"}, inplace=True)

    branches_int_id = _get_branches_id_int(net)
    tie_lines = tie_lines.merge(branches_int_id[["id_int"]], how="left", left_on=["id_str"], right_index=True)
    tie_lines["branch_type"] = "TIE_LINE"
    tie_lines = BranchParamSchema.validate(tie_lines)
    return tie_lines


def _get_dangling_line_branch_parameter(net: Network) -> BranchParamSchema:
    """Get the dangling lines parameters of the network.

    Gets the dangling lines parameters from the network.
    Note: unpaired dangling lines are modelled to create a new bus where the dangling gen/load is connected.

    Parameters
    ----------
    net : Network
        The powsybl network.

    Returns
    -------
    BranchParamSchema
        The dangling lines parameters of the network.
    """
    available_attributes = [
        "name",
        "connected",
        "r",
        "x",
        "g",
        "b",
        "p",
        "q",
        "i",
        "boundary_p",
        "boundary_q",
        "boundary_i",
        "bus_id",
        "paired",
    ]
    dangling_lines = net.get_dangling_lines(attributes=available_attributes).sort_index()
    if "paired" in dangling_lines:
        dangling_lines = dangling_lines[~dangling_lines["paired"]]
        dangling_lines.drop(columns=["paired"], inplace=True)
    if dangling_lines.empty:
        dangling_lines = pd.DataFrame(columns=list(BranchParamSchema.__annotations__.keys()))
        dangling_lines = BranchParamSchema.validate(dangling_lines)
        return dangling_lines
    dangling_lines.rename(columns={"bus_id": "bus1_id"}, inplace=True)
    dangling_lines["bus2_id"] = _get_dangling_bus_ids(net)["dangling_bus_id"].values
    dangling_lines = _get_branches_with_bus_index(net, dangling_lines)
    dangling_lines.drop(columns=["bus1_id", "bus2_id"], inplace=True)

    dangling_lines.reset_index(drop=False, inplace=True)
    dangling_lines.rename(columns={"id": "id_str"}, inplace=True)

    branches_int_id = _get_branches_id_int(net)
    dangling_lines = dangling_lines.merge(branches_int_id[["id_int"]], how="left", left_on=["id_str"], right_index=True)
    dangling_lines["branch_type"] = "DANGLING_LINE"
    dangling_lines.rename(
        columns={
            "g": "g1",
            "b": "b1",
            "p": "p1",
            "q": "q1",
            "i": "i1",
            "boundary_p": "p2",
            "boundary_q": "q2",
            "boundary_i": "i2",
        },
        inplace=True,
    )
    dangling_lines["g2"] = 0.0
    dangling_lines["b2"] = 0.0
    dangling_lines["rho"] = 1.0
    dangling_lines["alpha"] = 0.0

    dangling_lines = BranchParamSchema.validate(dangling_lines)
    return dangling_lines


def _get_line_parameter(net: Network) -> BranchParamSchema:
    """Get the line parameters of the network.

    Gets the line parameters from the network.

    Parameters
    ----------
    net : Network
        The powsybl network.

    Returns
    -------
    BranchParamSchema
        The line parameters of the network.
    """
    available_attributes = [
        "name",
        "connected1",
        "connected2",
        "r",
        "x",
        "g1",
        "b1",
        "p1",
        "q1",
        "i1",
        "g2",
        "b2",
        "p2",
        "q2",
        "i2",
        "bus1_id",
        "bus2_id",
    ]
    lines = net.get_lines(attributes=available_attributes)
    if lines.empty:
        lines = pd.DataFrame(columns=list(BranchParamSchema.__annotations__.keys()))
        lines = BranchParamSchema.validate(lines)
        return lines
    lines["connected"] = lines["connected1"] & lines["connected2"]
    lines.drop(columns=["connected1", "connected2"], inplace=True)

    lines = _get_branches_with_bus_index(net, lines)
    lines.drop(columns=["bus1_id", "bus2_id"], inplace=True)

    lines["rho"] = 1.0
    lines["alpha"] = 0.0
    lines.reset_index(drop=False, inplace=True)
    lines.rename(columns={"id": "id_str"}, inplace=True)

    branches_int_id = _get_branches_id_int(net)
    lines = lines.merge(branches_int_id[["id_int"]], how="left", left_on=["id_str"], right_index=True)
    lines["branch_type"] = "LINE"

    BranchParamSchema.validate(lines)
    return lines


def _get_trafo_parameter(net: Network, split_trafo_charging: bool = True) -> BranchParamSchema:
    """Get the transformer parameters of the network.

    Gets the transformer parameters from the network.

    Parameters
    ----------
    net : Network
        The powsybl network.
    split_trafo_charging : bool
        Whether to split the transformer charging admittance into the series and shunt admittance.
        Powsybl default is False, DCplus uses True.

    Returns
    -------
    BranchParamSchema
        The transformer parameters of the network.
    """
    available_attributes = [
        "name",
        "connected1",
        "connected2",
        "p1",
        "q1",
        "i1",
        "p2",
        "q2",
        "i2",
        "bus1_id",
        "bus2_id",
        "r_at_current_tap",
        "x_at_current_tap",
        "g_at_current_tap",
        "b_at_current_tap",
        "rho",
        "alpha",
    ]
    transformers = net.get_2_windings_transformers(attributes=available_attributes)
    if transformers.empty:
        transformers = pd.DataFrame(columns=list(BranchParamSchema.__annotations__.keys()))
        transformers = BranchParamSchema.validate(transformers)
        return transformers

    transformers.rename(
        columns={
            "r_at_current_tap": "r",
            "x_at_current_tap": "x",
            "g_at_current_tap": "g2",
            "b_at_current_tap": "b2",
        },
        inplace=True,
    )
    transformers["connected"] = transformers["connected1"] & transformers["connected2"]
    transformers.drop(columns=["connected1", "connected2"], inplace=True)

    transformers = _get_branches_with_bus_index(net, transformers)
    transformers.drop(columns=["bus1_id", "bus2_id"], inplace=True)
    transformers.reset_index(drop=False, inplace=True)
    transformers.rename(columns={"id": "id_str"}, inplace=True)
    branches_int_id = _get_branches_id_int(net)
    transformers = transformers.merge(branches_int_id[["id_int"]], how="left", left_on=["id_str"], right_index=True)
    transformers["branch_type"] = "TWO_WINDINGS_TRANSFORMER"
    if split_trafo_charging:
        # split the transformer charging admittance into the series and shunt admittance
        # DCplus uses the split Pi model
        # g2, b2 -> shunt admittance on the to side
        # g1, b1 -> shunt admittance on the from side
        # series admittance is 0
        transformers["b1"] = transformers["b2"] / 2
        transformers["g1"] = transformers["g2"] / 2
        transformers["b2"] = transformers["b2"] / 2
        transformers["g2"] = transformers["g2"] / 2
    else:
        # vanilla powsybl implementation
        # set the from side shunt admittance to 0
        transformers["b1"] = 0.0
        transformers["g1"] = 0.0

    # rho has a different sign convention in powsybl
    transformers["rho"] = 1 / transformers["rho"]
    # alpha has a different sign convention in powsybl
    transformers["alpha"] = -transformers["alpha"]

    assert net.get_3_windings_transformers().empty, "3 windings transformers are not supported"
    BranchParamSchema.validate(transformers)
    return transformers


def _get_branches_parameter_powsybl(net: Network, split_trafo_charging: bool = True) -> BranchParamSchema:
    """Get the branches parameters of the network.

    Gets the branches parameters from the network.

    Parameters
    ----------
    net : Network
        The powsybl network.
    split_trafo_charging : bool
        Whether to split the transformer charging admittance into the series and shunt admittance.
        Powsybl default is False, DCplus uses True.

    Returns
    -------
    BranchParamSchema
        The branches parameters of the network.
    """
    trafos = _get_trafo_parameter(net, split_trafo_charging=split_trafo_charging)
    lines = _get_line_parameter(net)
    tie_lines = _get_tie_line_parameter(net)
    dangling_lines = _get_dangling_line_branch_parameter(net)
    branches = pd.concat([trafos, lines, tie_lines, dangling_lines], ignore_index=True)
    branches = branches.sort_values(by=["id_str"]).reset_index(drop=True)
    BranchParamSchema.validate(branches)
    return branches


def _get_limits_parameter_powsybl(net: Network) -> LimitParamSchema:
    """Get the limits parameters of the network.

    Gets the limits parameters from the network.

    Parameters
    ----------
    net : Network
        The powsybl network.

    Returns
    -------
    LimitParamSchema
        The limits parameters of the network.
    """
    operational_limits = net.get_operational_limits(all_attributes=True)
    operational_limits.reset_index(drop=False, inplace=True)
    operational_limits.rename(columns={"element_id": "element_id_str"}, inplace=True)
    operational_limits.reset_index(drop=False, inplace=True)
    operational_limits.rename(columns={"index": "id_int"}, inplace=True)
    operational_limits.rename(
        columns={
            "type": "limit_type",
        },
        inplace=True,
    )
    operational_limits = operational_limits[
        ["id_int", "element_id_str", "limit_type", "element_type", "acceptable_duration", "side", "name", "value"]
    ]
    operational_limits = LimitParamSchema.validate(operational_limits)
    return operational_limits


# ################ Injections ########################


def _get_injection_id_int(net: Network) -> pd.Series:
    """Get the injection id_int of the network.

    Gets the injection id_int from the network.

    Parameters
    ----------
    net : Network
        The powsybl network.

    Returns
    -------
    pd.Series
        The injection id_int of the network.
    """
    injections = net.get_injections(attributes=[])
    injections = injections.reset_index(drop=False)
    injections.rename(columns={"id": "id_str"}, inplace=True)
    injections["id_int"] = np.arange(len(injections))
    injections.set_index("id_str", inplace=True)
    return injections


def _get_dangling_line_generators(net: Network) -> pd.DataFrame:
    available_attributes = ["p0", "q0", "i", "connected", "paired"]
    dangling_lines = net.get_dangling_lines(attributes=available_attributes)
    # get all non-paired dangling lines -> model as injections
    dangling_gen = dangling_lines[~dangling_lines["paired"]]
    dangling_gen.reset_index(drop=False, inplace=True)
    dangling_gen.rename(columns={"id": "id_str"}, inplace=True)
    dangling_bus_ids = _get_dangling_bus_ids(net)
    dangling_gen = dangling_gen.merge(dangling_bus_ids[["dangling_bus_id"]], how="left", left_on="id_str", right_index=True)
    dangling_gen.rename(columns={"dangling_bus_id": "bus_id"}, inplace=True)
    injection_id_int = _get_injection_id_int(net)
    dangling_gen = dangling_gen.merge(injection_id_int[["id_int"]], how="left", left_on=["id_str"], right_index=True)
    dangling_gen.rename(columns={"p0": "setpoint_p", "q0": "setpoint_q"}, inplace=True)
    dangling_gen = _get_injection_with_bus_index(net, dangling_gen)
    dangling_gen.drop(columns=["bus_id", "paired"], inplace=True)

    dangling_gen["p"] = dangling_gen["setpoint_p"]
    dangling_gen["q"] = dangling_gen["setpoint_q"]
    dangling_gen["injection_type"] = "GENERATOR"
    dangling_gen.loc[dangling_gen["p"] > 0, "injection_type"] = "LOAD"
    dangling_gen["min_p"] = np.nan
    dangling_gen["max_p"] = np.nan
    dangling_gen["min_q"] = np.nan
    dangling_gen["max_q"] = np.nan
    dangling_gen["voltage_regulation"] = False
    dangling_gen["regulated_bus_id_str"] = ""
    dangling_gen["regulated_bus_id_int"] = -1
    dangling_gen = InjectionParamSchema.validate(dangling_gen)
    return dangling_gen


def _get_generators(net: Network) -> pd.DataFrame:
    """Get all generators that are connected to a node in _get_nodes()"""
    available_attributes = [
        "target_p",
        "target_q",
        "p",
        "q",
        "i",
        "max_q",
        "min_q",
        "connected",
        "max_p",
        "min_p",
        "bus_id",
        "voltage_regulator_on",
        "regulated_bus_id",
    ]
    gens = net.get_generators(attributes=available_attributes)
    gens.reset_index(drop=False, inplace=True)
    gens.rename(columns={"id": "id_str"}, inplace=True)
    injection_id_int = _get_injection_id_int(net)
    gens = gens.merge(injection_id_int[["id_int"]], how="left", left_on=["id_str"], right_index=True)
    gens.rename(columns={"target_p": "setpoint_p", "target_q": "setpoint_q"}, inplace=True)
    gens = _get_injection_with_bus_index(net, gens)
    gens.drop(columns=["bus_id"], inplace=True)
    gens.rename(
        columns={"voltage_regulator_on": "voltage_regulation", "regulated_bus_id": "regulated_bus_id_str"}, inplace=True
    )
    gens = _get_injection_with_bus_index(
        net, gens, source_column="regulated_bus_id_str", target_column="regulated_bus_id_int"
    )

    gens["injection_type"] = "GENERATOR"
    gens = InjectionParamSchema.validate(gens)

    return gens


def _get_battery(net: Network) -> pd.DataFrame:
    """Get all batteries that are connected to a node in _get_nodes()"""
    available_attributes = [
        "target_p",
        "target_q",
        "p",
        "q",
        "i",
        "max_q",
        "min_q",
        "connected",
        "max_p",
        "min_p",
        "bus_id",
    ]
    batteries = net.get_batteries(attributes=available_attributes)
    batteries.reset_index(drop=False, inplace=True)
    batteries.rename(columns={"id": "id_str"}, inplace=True)
    injection_id_int = _get_injection_id_int(net)
    batteries = batteries.merge(injection_id_int[["id_int"]], how="left", left_on=["id_str"], right_index=True)
    batteries.rename(columns={"target_p": "setpoint_p", "target_q": "setpoint_q"}, inplace=True)
    batteries = _get_injection_with_bus_index(net, batteries)
    batteries.drop(columns=["bus_id"], inplace=True)

    batteries["injection_type"] = "GENERATOR"
    batteries.loc[batteries["p"] > 0, "injection_type"] = "LOAD"
    batteries["voltage_regulation"] = False
    batteries["regulated_bus_id_str"] = ""
    batteries["regulated_bus_id_int"] = -1

    batteries = InjectionParamSchema.validate(batteries)
    return batteries


def _get_hvdc_lcc(net: Network) -> pd.DataFrame:
    """Get all lcc converter stations that are connected to a node in _get_nodes()"""
    available_attributes = [
        "p",
        "q",
        "i",
        "connected",
        "bus_id",
    ]
    lcc = net.get_lcc_converter_stations(attributes=available_attributes)
    lcc.reset_index(drop=False, inplace=True)
    lcc.rename(columns={"id": "id_str"}, inplace=True)
    injection_id_int = _get_injection_id_int(net)
    lcc = lcc.merge(injection_id_int[["id_int"]], how="left", left_on=["id_str"], right_index=True)
    lcc = _get_injection_with_bus_index(net, lcc)
    lcc.drop(columns=["bus_id"], inplace=True)

    lcc["injection_type"] = "GENERATOR"
    lcc.loc[lcc["p"] > 0, "injection_type"] = "LOAD"
    lcc["setpoint_p"] = np.nan
    lcc["setpoint_q"] = np.nan
    lcc["min_q"] = np.nan
    lcc["max_q"] = np.nan
    lcc["min_p"] = np.nan
    lcc["max_p"] = np.nan
    lcc["voltage_regulation"] = False
    lcc["regulated_bus_id_str"] = ""
    lcc["regulated_bus_id_int"] = -1

    lcc = InjectionParamSchema.validate(lcc)
    return lcc


def _get_hvdc_vsc(net: Network) -> pd.DataFrame:
    """Get all vsc converter stations that are connected to a node in _get_nodes()"""
    # TODO: not sure if correct implemented
    available_attributes = [
        "target_q",
        "p",
        "q",
        "i",
        "max_q",
        "min_q",
        "connected",
        "bus_id",
        "voltage_regulator_on",
        "regulated_element_id",
    ]
    vsc = net.get_vsc_converter_stations(attributes=available_attributes)
    vsc.reset_index(drop=False, inplace=True)
    vsc.rename(columns={"id": "id_str"}, inplace=True)
    injection_id_int = _get_injection_id_int(net)
    vsc = vsc.merge(injection_id_int[["id_int"]], how="left", left_on=["id_str"], right_index=True)
    vsc.rename(columns={"target_q": "setpoint_q"}, inplace=True)
    vsc = _get_injection_with_bus_index(net, vsc)
    # if regulating element points to the vsc -> choose bus id of the vsc
    vsc.loc[vsc["regulated_element_id"] == vsc["id_str"], "regulated_element_id"] = vsc.loc[
        vsc["regulated_element_id"] == vsc["id_str"], "bus_id"
    ]
    vsc.drop(columns=["bus_id"], inplace=True)
    vsc.rename(
        columns={"voltage_regulator_on": "voltage_regulation", "regulated_element_id": "regulated_bus_id_str"}, inplace=True
    )
    vsc = _get_injection_with_bus_index(net, vsc, source_column="regulated_bus_id_str", target_column="regulated_bus_id_int")

    vsc["injection_type"] = "GENERATOR"
    vsc.loc[vsc["p"] > 0, "injection_type"] = "LOAD"
    vsc["setpoint_p"] = np.nan
    vsc["min_p"] = np.nan
    vsc["max_p"] = np.nan
    vsc = InjectionParamSchema.validate(vsc)

    return vsc


def _get_loads(net: Network) -> pd.DataFrame:
    """Get all loads that are connected to a node in _get_nodes()"""
    available_attributes = [
        "p0",
        "q0",
        "p",
        "q",
        "i",
        "connected",
        "bus_id",
    ]
    loads = net.get_loads(attributes=available_attributes)
    loads.reset_index(drop=False, inplace=True)
    loads.rename(columns={"id": "id_str"}, inplace=True)
    injection_id_int = _get_injection_id_int(net)
    loads = loads.merge(injection_id_int[["id_int"]], how="left", left_on=["id_str"], right_index=True)
    loads.rename(columns={"p0": "setpoint_p", "q0": "setpoint_q"}, inplace=True)
    loads = _get_injection_with_bus_index(net, loads)
    loads.drop(columns=["bus_id"], inplace=True)

    loads["injection_type"] = "LOAD"
    loads["min_q"] = np.nan
    loads["max_q"] = np.nan
    loads["min_p"] = np.nan
    loads["max_p"] = np.nan
    loads["voltage_regulation"] = False
    loads["regulated_bus_id_str"] = ""
    loads["regulated_bus_id_int"] = -1

    loads = InjectionParamSchema.validate(loads)
    return loads


def _get_static_var_compensators(net: Network) -> pd.DataFrame:
    """Get all static var compensators that are connected to a node in _get_nodes()"""
    available_attributes = [
        "target_q",
        "p",
        "q",
        "i",
        "connected",
        "bus_id",
        "regulating",
        "regulation_mode",
        "regulated_element_id",
    ]
    svc = net.get_static_var_compensators(attributes=available_attributes)
    svc.reset_index(drop=False, inplace=True)
    svc.rename(columns={"id": "id_str"}, inplace=True)
    injection_id_int = _get_injection_id_int(net)
    svc = svc.merge(injection_id_int[["id_int"]], how="left", left_on=["id_str"], right_index=True)
    svc.rename(columns={"target_q": "setpoint_q"}, inplace=True)
    svc = _get_injection_with_bus_index(net, svc)
    # if regulating element points to the svc -> choose bus id of the svc
    svc.loc[svc["regulated_element_id"] == svc["id_str"], "regulated_element_id"] = svc.loc[
        svc["regulated_element_id"] == svc["id_str"], "bus_id"
    ]
    svc.drop(columns=["bus_id"], inplace=True)

    svc["voltage_regulation"] = svc["regulating"] & (svc["regulation_mode"] == "VOLTAGE")
    svc.drop(columns=["regulating", "regulation_mode"], inplace=True)
    svc.rename(columns={"regulated_element_id": "regulated_bus_id_str"}, inplace=True)
    svc = _get_injection_with_bus_index(net, svc, source_column="regulated_bus_id_str", target_column="regulated_bus_id_int")

    svc["injection_type"] = "GENERATOR"
    svc["setpoint_p"] = np.nan
    svc["min_p"] = np.nan
    svc["max_p"] = np.nan
    svc["min_q"] = np.nan
    svc["max_q"] = np.nan
    svc = InjectionParamSchema.validate(svc)

    return svc


def _get_injections_powsybl(net: Network) -> pd.DataFrame:
    """Merge information from generators, loads and dangling lines into the injections dataframe."""
    injections = pd.concat(
        [
            _get_generators(net),
            _get_loads(net),
            _get_dangling_line_generators(net),
            _get_battery(net),
            _get_hvdc_lcc(net),
            _get_hvdc_vsc(net),
            _get_static_var_compensators(net),
        ]
    )

    injections = InjectionParamSchema.validate(injections)
    return injections


# ################ Shunts ########################


def _get_shunts_powsybl(net: Network) -> ShuntParamSchema:
    """Get the shunt parameters of the network.

    Gets the shunt parameters from the network.

    Parameters
    ----------
    net : Network
        The powsybl network.

    Returns
    -------
    ShuntParamSchema
        The shunt parameters of the network.
    """
    available_attributes = [
        "name",
        "connected",
        "g",
        "b",
        "p",
        "q",
        "i",
        "bus_id",
        "section_count",
        "max_section_count",
        "voltage_regulation_on",
        "regulating_bus_id",
    ]
    shunts = net.get_shunt_compensators(attributes=available_attributes)
    if shunts.empty:
        shunts = pd.DataFrame(columns=list(ShuntParamSchema.__annotations__.keys()))
        shunts = ShuntParamSchema.validate(shunts)
        return shunts
    shunts.reset_index(drop=False, inplace=True)
    shunts.rename(columns={"id": "id_str"}, inplace=True)
    injection_id_int = _get_injection_id_int(net)
    shunts = shunts.merge(injection_id_int[["id_int"]], how="left", left_on=["id_str"], right_index=True)
    shunts = _get_injection_with_bus_index(net, shunts)
    shunts.drop(columns=["bus_id"], inplace=True)
    shunts.rename(
        columns={"voltage_regulation_on": "voltage_regulation", "regulating_bus_id": "regulated_bus_id_str"}, inplace=True
    )
    shunts = _get_injection_with_bus_index(
        net, shunts, source_column="regulated_bus_id_str", target_column="regulated_bus_id_int"
    )

    shunts = ShuntParamSchema.validate(shunts)
    return shunts


# ################ Buses ########################


def _get_dangling_buses(net: Network) -> BusParamSchema:
    """Get the dangling buses of the network.

    A new Dangling bus is created for each unpaired dangling line.

    Parameters
    ----------
    net : Network
        The powsybl network.

    Returns
    -------
    BusParamSchema
        The dangling buses of the network.
    """
    dangling_lines = _get_unpaired_dangling_lines(net)
    available_attributes = ["boundary_v_mag", "boundary_v_angle"]
    dangling_lines = net.get_dangling_lines(attributes=available_attributes).loc[dangling_lines.index]
    dangling_lines.rename(columns={"boundary_v_mag": "voltage_magnitude", "boundary_v_angle": "voltage_angle"}, inplace=True)

    dangling_buses = _get_dangling_bus_ids(net)
    dangling_buses.rename(columns={"dangling_bus_id": "id_str"}, inplace=True)
    dangling_buses["name"] = dangling_buses["id_str"]
    dangling_buses = dangling_buses.merge(dangling_lines, how="left", left_index=True, right_index=True)
    # TODO: get values depending on the connected state and the bus_id of the dangling line
    dangling_buses["grid_island_id"] = 0
    dangling_buses["bus_type"] = 2  # PQ bus
    bus_order_int_ids = _get_bus_ids_with_dangling_buses(net)
    dangling_buses = dangling_buses.merge(bus_order_int_ids, how="left", on=["id_str"])

    dangling_buses = dangling_buses[
        ["id_str", "id_int", "name", "voltage_magnitude", "voltage_angle", "bus_type", "grid_island_id"]
    ]
    dangling_buses = BusParamSchema.validate(dangling_buses)
    return dangling_buses


def _get_buses_powsybl(net: Network, slack_id: str, injections: InjectionParamSchema) -> BusParamSchema:
    """Get the bus parameters of the network.

    Gets the bus parameters from the network.

    Parameters
    ----------
    net : Network
        The powsybl network.
    slack_id : str
        The id of the slack bus.
    injections : InjectionParamSchema
        The injections of the network.

    Returns
    -------
    BusParamSchema
        The bus parameters of the network.
    """
    per_unit_mode = bool(getattr(net, "per_unit", False))

    buses = net.get_buses(all_attributes=True).sort_index()
    buses.reset_index(drop=False, inplace=True)
    buses.rename(columns={"id": "id_str"}, inplace=True)
    buses["bus_type"] = 2  # PQ bus is default

    buses.rename(columns={"v_mag": "voltage_magnitude", "v_angle": "voltage_angle"}, inplace=True)

    if not per_unit_mode:
        # Convert voltage magnitudes from kV to per unit using the nominal voltage of the voltage level
        voltage_levels = net.get_voltage_levels(attributes=["nominal_v"])
        voltage_levels.rename(columns={"nominal_v": "nominal_voltage"}, inplace=True)
        buses = buses.merge(voltage_levels, left_on="voltage_level_id", right_index=True, how="left")
        base_voltage = buses["nominal_voltage"].to_numpy(dtype=float)
        if np.any(np.isnan(base_voltage)):
            logger.warning("Missing nominal voltage for some buses; defaulting to 1.0 p.u. for those entries")
        # Avoid division by zero by defaulting to 1.0 p.u. for invalid base voltages
        base_voltage = np.where((base_voltage == 0.0) | np.isnan(base_voltage), 1.0, base_voltage)
        buses["voltage_magnitude"] = buses["voltage_magnitude"] / base_voltage
        buses["voltage_angle"] = np.deg2rad(buses["voltage_angle"])
    else:
        buses["voltage_magnitude"] = buses["voltage_magnitude"].astype(float)
        buses["voltage_angle"] = buses["voltage_angle"].astype(float)

    buses["grid_island_id"] = _get_grid_island_ids(buses)
    bus_order_int_ids = _get_bus_ids_with_dangling_buses(net)
    buses = buses.merge(bus_order_int_ids, how="left", on=["id_str"])

    # now the main grid "0" has only connected_component == 0 and synchronous_component == 0
    buses = buses[["id_str", "id_int", "name", "voltage_magnitude", "voltage_angle", "bus_type", "grid_island_id"]]

    dangling_buses = _get_dangling_buses(net)
    buses = pd.concat([buses, dangling_buses], ignore_index=True)
    buses.sort_values(by=["id_int"], inplace=True)
    # set bus types
    pv_buses = injections[(injections["voltage_regulation"])]["regulated_bus_id_int"].unique()
    buses.loc[buses["id_int"].isin(pv_buses), "bus_type"] = 1  # PV bus
    buses.loc[buses["id_str"] == slack_id, "bus_type"] = 0  # slack bus

    buses = BusParamSchema.validate(buses)
    return buses


def _get_grid_island_ids(buses: pd.DataFrame) -> pd.Series:
    """Get the grid island ids of the buses.

    Parameters
    ----------
    buses : pd.DataFrame
        The buses dataframe.
        expects all columns "connected_component", "synchronous_component"

    Returns
    -------
    pd.Series
        The grid island ids of the buses.
    """
    grid_island_id = buses["connected_component"]
    # change grid_island_id where synchronous_component != 0 to the node index of that bus
    max_grid_island_id = grid_island_id.max() + 1
    for island_id in range(grid_island_id.max() + 1):
        for sync_component in range(1, buses["synchronous_component"].max() + 1):
            sync_buses = buses[
                (buses["connected_component"] == island_id) & (buses["synchronous_component"] == sync_component)
            ]
            if not sync_buses.empty:
                # modify grid_island_id
                grid_island_id.loc[
                    (buses["connected_component"] == island_id) & (buses["synchronous_component"] == sync_component)
                ] = max_grid_island_id
                max_grid_island_id += 1
    return grid_island_id
