# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import py
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pypowsybl
import pytest

from dc_plus.example_grids.pypowsbl.example_grids import create_complex_grid_battery_hvdc_svc_3w_trafo
from dc_plus.importing.import_helpers import _remove_isolated_buses_injections
from dc_plus.importing.import_schema import (
    BranchParamSchema,
    BusParamSchema,
    InjectionParamSchema,
    LimitParamSchema,
    ShuntParamSchema,
    TapChangerParamSchema,
    TapPositionParamSchema,
)
from dc_plus.importing.powsybl.powsybl_import import (
    _get_battery,
    _get_branches_parameter_powsybl,
    _get_buses_powsybl,
    _get_dangling_bus_ids,
    _get_dangling_line_generators,
    _get_generators,
    _get_grid_island_ids,
    _get_hvdc_lcc,
    _get_hvdc_vsc,
    _get_injections_powsybl,
    _get_limits_parameter_powsybl,
    _get_line_parameter,
    _get_loads,
    _get_shunts_powsybl,
    _get_static_var_compensators,
    _get_tie_line_parameter,
    _get_trafo_parameter,
    _get_tap_steps_parameter_powsybl,
    _get_tap_changer_parameter_powsybl,
)
from dc_plus.importing.powsybl.powsybl_import_helpers import select_a_generator_as_slack_and_run_loadflow
from dc_plus.preprocess.create_network_data import create_network_data

from dc_plus.interfaces.network_information import BusType


def test_get_tie_line_parameter():
    net = pypowsybl.network.create_eurostag_tutorial_example1_with_tie_lines_and_areas()
    tie_lines = _get_tie_line_parameter(net)
    BranchParamSchema.validate(tie_lines)
    assert len(tie_lines) != 0

    net = pypowsybl.network.create_ieee9()
    tie_lines = _get_tie_line_parameter(net)
    BranchParamSchema.validate(tie_lines)
    assert len(tie_lines) == 0


def test_get_line_parameter():
    net = pypowsybl.network.create_eurostag_tutorial_example1_with_tie_lines_and_areas()
    lines = _get_line_parameter(net)
    BranchParamSchema.validate(lines)
    assert len(lines) == 0

    net = pypowsybl.network.create_ieee9()
    lines = _get_line_parameter(net)
    BranchParamSchema.validate(lines)
    assert len(lines) != 0


def test_get_trafo_parameter():
    net = pypowsybl.network.create_micro_grid_nl_network()
    trafos = _get_trafo_parameter(net)
    BranchParamSchema.validate(trafos)
    assert len(trafos) != 0

    trafos = _get_trafo_parameter(net, split_trafo_charging=False)
    BranchParamSchema.validate(trafos)
    assert len(trafos) != 0
    assert trafos["g1"].sum() == 0
    assert trafos["b1"].sum() == 0
    assert trafos["g2"].sum() != 0
    assert trafos["b2"].sum() != 0

    for trafo in trafos["id_str"]:
        net.remove_elements(trafo)

    trafos = _get_trafo_parameter(net)
    BranchParamSchema.validate(trafos)
    assert len(trafos) == 0


def test_get_tap_steps_parameter_powsybl(micro_grid_be_network_with_replaced_transformers: pypowsybl.network.Network):
    net = micro_grid_be_network_with_replaced_transformers
    ratio_tap_changer_steps = _get_tap_steps_parameter_powsybl(net.get_ratio_tap_changer_steps())
    assert len(TapPositionParamSchema.validate(ratio_tap_changer_steps)) != 0
    phase_tap_changer_steps = _get_tap_steps_parameter_powsybl(net.get_phase_tap_changer_steps())
    assert len(TapPositionParamSchema.validate(phase_tap_changer_steps)) != 0


def test_get_tap_changer_parameter_powsybl(micro_grid_be_network_with_replaced_transformers: pypowsybl.network.Network):
    net = micro_grid_be_network_with_replaced_transformers
    tap_changers = net.get_phase_tap_changers()
    transformers = net.get_2_windings_transformers(attributes=["r", "x", "g", "b"])
    phase_changer_params = _get_tap_changer_parameter_powsybl(tap_changers=tap_changers, transformers=transformers)
    assert len(TapChangerParamSchema.validate(phase_changer_params)) != 0
    ratio_changers = net.get_ratio_tap_changers()
    ratio_changers_params = _get_tap_changer_parameter_powsybl(tap_changers=ratio_changers, transformers=transformers)
    assert len(TapChangerParamSchema.validate(ratio_changers_params)) != 0


def test_create_network_data_populates_transformer_tap_information(
    micro_grid_be_network_with_replaced_transformers: pypowsybl.network.Network,
):
    net = micro_grid_be_network_with_replaced_transformers
    static_info, dynamic_info, string_info = create_network_data(net)

    branch_index_by_id = {str(branch_id): idx for idx, branch_id in enumerate(string_info.branch_ids)}
    ratio_taps = net.get_ratio_tap_changers()["tap"].to_dict()
    phase_taps = net.get_phase_tap_changers()["tap"].to_dict()

    expected_ratio_indices = {branch_index_by_id[str(branch_id)] for branch_id in ratio_taps}
    expected_phase_indices = {branch_index_by_id[str(branch_id)] for branch_id in phase_taps}

    assert set(np.flatnonzero(static_info.has_ratio_changing_transformer)) == expected_ratio_indices
    assert set(np.flatnonzero(static_info.has_phase_shifting_transformer)) == expected_phase_indices
    assert set(static_info.ratio_shift_info) == expected_ratio_indices
    assert set(static_info.phase_shift_info) == expected_phase_indices

    for branch_idx, branch_id in enumerate(string_info.branch_ids):
        expected_ratio_tap = int(ratio_taps.get(str(branch_id), 0))
        expected_phase_tap = int(phase_taps.get(str(branch_id), 0))
        assert dynamic_info.branch_ratio_tap_positions[branch_idx] == expected_ratio_tap
        assert dynamic_info.branch_phase_tap_positions[branch_idx] == expected_phase_tap

    for tap_info in static_info.ratio_shift_info.values():
        assert tap_info.n_max_tap_positions > 0
        assert tap_info.tap_offset_shift_ratio_rho.shape == (tap_info.n_max_tap_positions,)

    for tap_info in static_info.phase_shift_info.values():
        assert tap_info.n_max_tap_positions > 0
        assert tap_info.tap_offset_shift_angle.shape == (tap_info.n_max_tap_positions,)


def test_get_branches_parameter():
    net = pypowsybl.network.create_eurostag_tutorial_example1_with_tie_lines_and_areas()
    branches = _get_branches_parameter_powsybl(net)
    BranchParamSchema.validate(branches)
    assert len(branches) == len(net.get_branches())


def test_get_limits_parameter():
    net = pypowsybl.network.create_eurostag_tutorial_example1_with_power_limits_network()
    limits = _get_limits_parameter_powsybl(net)
    LimitParamSchema.validate(limits)
    assert len(limits) != 0

    net = pypowsybl.network.create_ieee9()
    limits = _get_limits_parameter_powsybl(net)
    LimitParamSchema.validate(limits)
    assert len(limits) == 0


def test_get_generators():
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    gens = _get_generators(net)
    InjectionParamSchema.validate(gens)
    gen_powsybl = net.get_generators(attributes=[])
    assert len(gen_powsybl) == len(gens)


def test_get_battery():
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    bat = _get_battery(net)
    InjectionParamSchema.validate(bat)
    bat_powsybl = net.get_batteries(attributes=[])
    assert len(bat_powsybl) == len(bat)


def test_get_hvdc_lcc():
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    hvdc_lcc = _get_hvdc_lcc(net)
    InjectionParamSchema.validate(hvdc_lcc)
    hcdc_lcc_powsybl = net.get_lcc_converter_stations(attributes=[])
    assert len(hcdc_lcc_powsybl) == len(hvdc_lcc)


def test_get_hvdc_vsc():
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    hvdc_vsc = _get_hvdc_vsc(net)
    InjectionParamSchema.validate(hvdc_vsc)
    hcdc_vsc_powsybl = net.get_vsc_converter_stations(attributes=[])
    assert len(hcdc_vsc_powsybl) == len(hvdc_vsc)


def test_get_loads():
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    loads = _get_loads(net)
    InjectionParamSchema.validate(loads)
    load_powsybl = net.get_loads(attributes=[])
    assert len(load_powsybl) == len(loads)


def test_get_dangling_lines():
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    dangling_lines = _get_dangling_line_generators(net)
    InjectionParamSchema.validate(dangling_lines)
    dangling_lines_powsybl = net.get_dangling_lines(attributes=[])
    assert len(dangling_lines_powsybl) == len(dangling_lines)


def test_get_static_var_compensators():
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    svc = _get_static_var_compensators(net)
    InjectionParamSchema.validate(svc)
    svc_powsybl = net.get_static_var_compensators(attributes=[])
    assert len(svc_powsybl) == len(svc)


def test_get_injections():
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    injections = _get_injections_powsybl(net)
    InjectionParamSchema.validate(injections)
    inj_powsybl = net.get_injections(attributes=["type"])
    inj_powsybl = inj_powsybl[(inj_powsybl["type"] != "BUSBAR_SECTION") & (inj_powsybl["type"] != "SHUNT_COMPENSATOR")]
    assert len(inj_powsybl) == len(injections)


def test_get_shunt_compensators():
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    shunts = _get_shunts_powsybl(net)
    ShuntParamSchema.validate(shunts)
    shunt_powsybl = net.get_shunt_compensators(attributes=[])
    assert len(shunt_powsybl) == len(shunts)

    net = pypowsybl.network.create_ieee9()
    shunts = _get_shunts_powsybl(net)
    ShuntParamSchema.validate(shunts)
    shunt_powsybl = net.get_shunt_compensators(attributes=[])
    assert len(shunt_powsybl) == 0


def test_get_buses():
    net = create_complex_grid_battery_hvdc_svc_3w_trafo()
    injections = _get_injections_powsybl(net)
    slack_id = net.get_extensions("slackTerminal")["bus_id"].values[0]
    buses = _get_buses_powsybl(net=net, slack_id=slack_id, injections=injections, reference_id=slack_id)
    BusParamSchema.validate(buses)
    bus_powsybl = net.get_buses(attributes=[])
    dangling_count = len(_get_dangling_bus_ids(net))
    assert len(bus_powsybl) + dangling_count == len(buses)
    assert BusType.SLACK in list(buses["bus_type"].values)  # slack bus
    assert BusType.PV in list(buses["bus_type"].values)  # PV bus
    assert BusType.PQ in list(buses["bus_type"].values)  # PQ bus
    assert np.sum(buses["bus_type"] == BusType.SLACK) == 1  # only one slack bus
    assert np.sum(buses["is_angle_reference"]) == 1  # only one reference bus
    assert np.sum(buses["bus_type"] == BusType.PV) == 2  # there are two PV buses
    assert np.sum(buses["bus_type"] == BusType.PQ) >= 1  # at least one PQ bus


def test_no_synchronous_components_returns_connected_ids():
    df = pd.DataFrame(
        {
            "connected_component": [0, 0, 1, 1],
            "synchronous_component": [0, 0, 0, 0],
        }
    )
    res = _get_grid_island_ids(df.copy())
    assert res.tolist() == [0, 0, 1, 1]

    df = pd.DataFrame(
        {
            "connected_component": [0, 0, 1, 1],
            "synchronous_component": [1, 0, 0, 0],
        }
    )
    res = _get_grid_island_ids(df.copy())
    # original max connected_component is 1 -> first new id should be 2
    assert res.tolist() == [2, 0, 1, 1]

    df = pd.DataFrame(
        {
            "connected_component": [0, 0, 1, 1, 2],
            "synchronous_component": [1, 0, 1, 0, 0],
        }
    )
    # original max connected_component is 2 -> new ids start at 3
    # island 0 -> becomes 3, island 1 -> becomes 4, island 2 stays 2
    res = _get_grid_island_ids(df.copy())
    assert res.tolist() == [3, 0, 4, 1, 2]


def test_remove_isolated_buses_injections_basic():
    buses = pd.DataFrame({"id_int": [0, 1, 2, 3], "grid_island_id": [0, 1, 0, 2]})
    injections = pd.DataFrame(
        {
            "id_int": [10, 11, 12, 13],
            "bus_index": [0, 1, 2, 3],
            "value": [100, 200, 300, 400],
        }
    )

    res = _remove_isolated_buses_injections(buses=buses, injections=injections)
    expected = injections[injections["bus_index"].isin(buses[buses["grid_island_id"] == 0]["id_int"])]

    pdt.assert_frame_equal(res.reset_index(drop=True), expected.reset_index(drop=True))

    buses = pd.DataFrame({"id_int": [0, 1], "grid_island_id": [1, 2]})
    injections = pd.DataFrame({"bus_index": [0, 1, 2], "value": [1, 2, 3]})

    res = _remove_isolated_buses_injections(buses=buses, injections=injections)

    # No bus has grid_island_id == 0, so result should be empty
    assert res.empty
