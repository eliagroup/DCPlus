# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import math

import numpy as np
import pandas as pd

from dc_plus.importing import import_helpers
from dc_plus.importing.import_schema import BranchParamSchema, BusParamSchema, InjectionParamSchema, ShuntParamSchema


def test_remove_isolated_buses_injections():
    buses = pd.DataFrame(
        {
            "id_int": [0, 1, 2],
            "id_str": ["bus_0", "bus_1", "bus_2"],
            "name": ["Bus 0", "Bus 1", "Bus 2"],
            "voltage_magnitude": [1.0, 0.98, 1.02],
            "voltage_angle": [0.0, -0.05, 0.03],
            "bus_type": [0, 1, 2],
            "grid_island_id": [0, 0, 1],
        }
    )
    injections = pd.DataFrame(
        {
            "id_int": [100, 101, 102],
            "id_str": ["inj_0", "inj_1", "inj_2"],
            "injection_type": ["GENERATOR", "LOAD", "GENERATOR"],
            "p": [50.0, -20.0, 15.0],
            "q": [5.0, -2.0, 1.5],
            "i": [1.0, 0.5, 0.3],
            "setpoint_p": [50.0, np.nan, 15.0],
            "setpoint_q": [5.0, np.nan, 1.5],
            "min_q": [-10.0, -5.0, -5.0],
            "max_q": [10.0, 5.0, 5.0],
            "min_p": [0.0, -30.0, 0.0],
            "max_p": [60.0, 0.0, 20.0],
            "bus_index": [0, 1, 2],
            "connected": [True, True, False],
            "voltage_regulation": [True, False, False],
            "regulated_bus_id_str": ["bus_0", "", ""],
            "regulated_bus_id_int": [0, -1, -1],
        }
    )

    BusParamSchema.validate(buses)
    InjectionParamSchema.validate(injections)

    filtered_injections = import_helpers._remove_isolated_buses_injections(buses, injections)
    assert list(filtered_injections["id_int"]) == [100, 101]
    assert set(filtered_injections["bus_index"]) == {0, 1}

    no_main_grid = buses.assign(grid_island_id=[1, 1, 1])
    filtered_empty = import_helpers._remove_isolated_buses_injections(no_main_grid, injections)
    assert filtered_empty.empty

    all_main_grid = buses.assign(grid_island_id=[0, 0, 0])
    filtered_all = import_helpers._remove_isolated_buses_injections(all_main_grid, injections)
    assert list(filtered_all["id_int"]) == [100, 101, 102]


def test_remove_isolated_branches():
    buses = pd.DataFrame(
        {
            "id_int": [0, 1, 2],
            "id_str": ["bus_0", "bus_1", "bus_2"],
            "name": ["Bus 0", "Bus 1", "Bus 2"],
            "voltage_magnitude": [1.0, 1.0, 1.0],
            "voltage_angle": [0.0, 0.0, 0.0],
            "bus_type": [0, 1, 2],
            "grid_island_id": [0, 0, 1],
        }
    )
    branches = pd.DataFrame(
        {
            "id_int": [10, 11, 12],
            "id_str": ["br_0", "br_1", "br_2"],
            "name": ["Branch 0", "Branch 1", "Branch 2"],
            "connected": [True, True, True],
            "r": [0.01, 0.02, 0.03],
            "x": [0.05, 0.04, 0.06],
            "g1": [0.0, 0.0, 0.0],
            "b1": [0.02, 0.01, 0.02],
            "g2": [0.0, 0.0, 0.0],
            "b2": [0.02, 0.01, 0.02],
            "p1": [np.nan, np.nan, np.nan],
            "q1": [np.nan, np.nan, np.nan],
            "i1": [np.nan, np.nan, np.nan],
            "p2": [np.nan, np.nan, np.nan],
            "q2": [np.nan, np.nan, np.nan],
            "i2": [np.nan, np.nan, np.nan],
            "rho": [1.0, 1.0, 1.0],
            "alpha": [0.0, 0.0, 0.0],
            "from_bus_index": [0, 0, 2],
            "to_bus_index": [1, 2, 1],
            "branch_type": ["LINE", "LINE", "LINE"],
        }
    )

    BusParamSchema.validate(buses)
    BranchParamSchema.validate(branches)

    filtered_branches = import_helpers._remove_isolated_branches(buses, branches)
    assert list(filtered_branches["id_int"]) == [10]

    all_isolated_buses = buses.assign(grid_island_id=[1, 1, 1])
    assert import_helpers._remove_isolated_branches(all_isolated_buses, branches).empty

    all_main_grid = buses.assign(grid_island_id=[0, 0, 0])
    filtered_all = import_helpers._remove_isolated_branches(all_main_grid, branches)
    assert list(filtered_all["id_int"]) == [10, 11, 12]


def test_remove_isolated_buses():
    buses = pd.DataFrame(
        {
            "id_int": [0, 1, 2, 3],
            "id_str": ["bus_0", "bus_1", "bus_2", "bus_3"],
            "name": ["Bus 0", "Bus 1", "Bus 2", "Bus 3"],
            "voltage_magnitude": [1.0, 0.97, 1.01, 0.95],
            "voltage_angle": [0.0, -0.03, 0.02, -0.05],
            "bus_type": [0, 1, 2, 2],
            "grid_island_id": [0, 1, 0, 2],
        }
    )
    BusParamSchema.validate(buses)

    filtered_buses = import_helpers._remove_isolated_buses(buses)
    assert list(filtered_buses["id_int"]) == [0, 2]
    assert (filtered_buses["grid_island_id"] == 0).all()

    no_main_grid = buses.assign(grid_island_id=[1, 1, 2, 2])
    filtered_empty = import_helpers._remove_isolated_buses(no_main_grid)
    assert filtered_empty.empty

    all_main_grid = buses.assign(grid_island_id=[0, 0, 0, 0])
    filtered_all = import_helpers._remove_isolated_buses(all_main_grid)
    assert list(filtered_all["id_int"]) == [0, 1, 2, 3]


def test_get_admittance_branches():
    branches = pd.DataFrame(
        {
            "id_int": [0, 1],
            "id_str": ["br_0", "br_1"],
            "name": ["Branch 0", "Branch 1"],
            "connected": [True, True],
            "r": [0.01, 0.02],
            "x": [0.05, 0.06],
            "g1": [0.0, 0.001],
            "b1": [0.02, -0.01],
            "g2": [0.0, 0.0015],
            "b2": [0.02, 0.0],
            "p1": [np.nan, np.nan],
            "q1": [np.nan, np.nan],
            "i1": [np.nan, np.nan],
            "p2": [np.nan, np.nan],
            "q2": [np.nan, np.nan],
            "i2": [np.nan, np.nan],
            "rho": [1.0, 1.1],
            "alpha": [0.0, math.radians(10.0)],
            "from_bus_index": [0, 1],
            "to_bus_index": [1, 2],
            "branch_type": ["LINE", "TRANSFORMER"],
        }
    )
    BranchParamSchema.validate(branches)

    y_series = 1 / (branches["r"] + 1j * branches["x"])
    rho_alpha = branches["rho"] * np.exp(1j * branches["alpha"])
    y_charging_from = branches["g1"] + 1j * branches["b1"]
    y_charging_to = branches["g2"] + 1j * branches["b2"]

    expected_Yff = (y_series + y_charging_from) / (rho_alpha * np.conj(rho_alpha))
    expected_Yft = -y_series / np.conj(rho_alpha)
    expected_Ytf = -y_series / rho_alpha
    expected_Ytt = y_series + y_charging_to

    Yff, Yft, Ytf, Ytt, y_series_res, y_charging_symmetric_res = import_helpers._get_admittance_branches(branches)

    np.testing.assert_allclose(Yff, expected_Yff)
    np.testing.assert_allclose(Yft, expected_Yft)
    np.testing.assert_allclose(Ytf, expected_Ytf)
    np.testing.assert_allclose(Ytt, expected_Ytt)
    np.testing.assert_allclose(y_series, y_series_res)
    expected_charging_symmetric = (y_charging_from + y_charging_to) / 2
    np.testing.assert_allclose(y_charging_symmetric_res, expected_charging_symmetric)


def test_get_bus_admittance_shunts():
    shunts = pd.DataFrame(
        {
            "id_int": [0, 1, 2],
            "id_str": ["sh_0", "sh_1", "sh_2"],
            "name": ["Shunt 0", "Shunt 1", "Shunt 2"],
            "connected": [True, True, False],
            "g": [0.001, 0.0025, -0.0005],
            "b": [0.01, -0.02, 0.0],
            "p": [np.nan, np.nan, np.nan],
            "q": [np.nan, np.nan, np.nan],
            "i": [np.nan, np.nan, np.nan],
            "bus_index": [0, 1, 2],
            "section_count": [1, 2, 1],
            "max_section_count": [2, 3, 1],
            "voltage_regulation": [False, False, False],
            "regulated_bus_id_str": ["bus_0", "", ""],
            "regulated_bus_id_int": [0, -1, -1],
        }
    )
    ShuntParamSchema.validate(shunts)

    y_shunt = import_helpers._get_bus_admittance_shunts(shunts)
    expected = shunts["g"] + 1j * shunts["b"]
    np.testing.assert_allclose(y_shunt, expected)


def test_get_bus_active_power_injections():
    injections = pd.DataFrame(
        {
            "id_int": [0, 1, 2, 3],
            "id_str": ["inj_0", "inj_1", "inj_2", "inj_3"],
            "injection_type": ["GENERATOR", "LOAD", "GENERATOR", "LOAD"],
            "p": [10.0, -2.5, np.nan, 5.0],
            "q": [0.0, 0.0, 0.0, 0.0],
            "i": [np.nan, np.nan, np.nan, np.nan],
            "setpoint_p": [np.nan, np.nan, np.nan, np.nan],
            "setpoint_q": [np.nan, np.nan, np.nan, np.nan],
            "min_q": [np.nan, np.nan, np.nan, np.nan],
            "max_q": [np.nan, np.nan, np.nan, np.nan],
            "min_p": [np.nan, np.nan, np.nan, np.nan],
            "max_p": [np.nan, np.nan, np.nan, np.nan],
            "bus_index": [0, 0, 1, 2],
            "connected": [True, True, True, True],
            "voltage_regulation": [False, False, False, False],
            "regulated_bus_id_str": ["bus_0", "bus_0", "", "bus_2"],
            "regulated_bus_id_int": [0, 0, -1, 2],
        }
    )
    InjectionParamSchema.validate(injections)

    result = import_helpers._get_bus_active_power_injections(injections, n_buses=4)
    assert np.isclose(result[0], 7.5)
    assert np.isnan(result[1])
    assert np.isclose(result[2], 5.0)
    assert np.isclose(result[3], 0.0)


def test_get_bus_reactive_power_injections():
    injections = pd.DataFrame(
        {
            "id_int": [0, 1, 2, 3],
            "id_str": ["inj_0", "inj_1", "inj_2", "inj_3"],
            "injection_type": ["GENERATOR", "GENERATOR", "LOAD", "GENERATOR"],
            "p": [0.0, 0.0, 0.0, 0.0],
            "q": [3.0, 1.5, -0.5, np.nan],
            "i": [np.nan, np.nan, np.nan, np.nan],
            "setpoint_p": [np.nan, np.nan, np.nan, np.nan],
            "setpoint_q": [np.nan, np.nan, np.nan, np.nan],
            "min_q": [np.nan, np.nan, np.nan, np.nan],
            "max_q": [np.nan, np.nan, np.nan, np.nan],
            "min_p": [np.nan, np.nan, np.nan, np.nan],
            "max_p": [np.nan, np.nan, np.nan, np.nan],
            "bus_index": [0, 1, 1, 3],
            "connected": [True, True, True, True],
            "voltage_regulation": [False, False, False, False],
            "regulated_bus_id_str": ["bus_0", "bus_1", "bus_1", ""],
            "regulated_bus_id_int": [0, 1, 1, -1],
        }
    )
    InjectionParamSchema.validate(injections)

    result = import_helpers._get_bus_reactive_power_injections(injections, n_buses=4)
    assert np.isclose(result[0], 3.0)
    assert np.isclose(result[1], 1.0)
    assert np.isclose(result[2], 0.0)
    assert np.isnan(result[3])
