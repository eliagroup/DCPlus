# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Ensure the Pandapower importer reproduces native AC power-flow results."""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pytest

from dc_plus.preprocess.create_network_data import create_network_data_pandapower


def _max_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size == 0:
        return 0.0
    diff = np.abs(lhs - rhs)
    diff = diff[~np.isnan(diff)]
    return float(diff.max()) if diff.size else 0.0


def _assert_close(name: str, lhs: np.ndarray, rhs: np.ndarray, tol: float) -> None:
    assert lhs.shape == rhs.shape, f"{name} shape mismatch: {lhs.shape} vs {rhs.shape}"
    if not np.allclose(lhs, rhs, rtol=tol, atol=tol, equal_nan=True):
        raise AssertionError(f"{name} mismatch (max diff {_max_abs_diff(lhs, rhs):.2e}) exceeds {tol:.1e}")


def assert_import_matches_pandapower(net: "pp.pandapowerNet", tol: float) -> None:
    _, dynamic, string = create_network_data_pandapower(net)

    _assert_close(
        "bus voltage magnitude",
        dynamic.bus_voltage_magnitudes,
        net.res_bus["vm_pu"].to_numpy(),
        tol,
    )
    _assert_close(
        "bus voltage angle",
        dynamic.bus_voltage_angles_rad,
        np.deg2rad(net.res_bus["va_degree"].to_numpy()),
        tol,
    )

    n_lines = len(net.line)
    if n_lines:
        _assert_close(
            "line P from",
            dynamic.branch_active_power_from[:n_lines],
            net.res_line["p_from_mw"].to_numpy() / net.sn_mva,
            tol,
        )
        _assert_close(
            "line P to",
            dynamic.branch_active_power_to[:n_lines],
            net.res_line["p_to_mw"].to_numpy() / net.sn_mva,
            tol,
        )
        _assert_close(
            "line Q from",
            dynamic.branch_reactive_power_from[:n_lines],
            net.res_line["q_from_mvar"].to_numpy() / net.sn_mva,
            tol,
        )
        _assert_close(
            "line Q to",
            dynamic.branch_reactive_power_to[:n_lines],
            net.res_line["q_to_mvar"].to_numpy() / net.sn_mva,
            tol,
        )

    if len(net.trafo):
        start = n_lines
        stop = start + len(net.trafo)
        _assert_close(
            "trafo P from",
            dynamic.branch_active_power_from[start:stop],
            net.res_trafo["p_hv_mw"].to_numpy() / net.sn_mva,
            tol,
        )
        _assert_close(
            "trafo P to",
            dynamic.branch_active_power_to[start:stop],
            net.res_trafo["p_lv_mw"].to_numpy() / net.sn_mva,
            tol,
        )
        _assert_close(
            "trafo Q from",
            dynamic.branch_reactive_power_from[start:stop],
            net.res_trafo["q_hv_mvar"].to_numpy() / net.sn_mva,
            tol,
        )
        _assert_close(
            "trafo Q to",
            dynamic.branch_reactive_power_to[start:stop],
            net.res_trafo["q_lv_mvar"].to_numpy() / net.sn_mva,
            tol,
        )

    if len(net.gen):
        mask = string.injection_types == "GENERATOR"
        assert mask.sum() >= len(net.gen), "Generator injections missing from import"
        _assert_close(
            "generator P",
            dynamic.injection_active_power[mask][: len(net.gen)],
            net.res_gen["p_mw"].to_numpy() / net.sn_mva,
            tol,
        )
        _assert_close(
            "generator Q",
            dynamic.injection_reactive_power[mask][: len(net.gen)],
            net.res_gen["q_mvar"].to_numpy() / net.sn_mva,
            tol,
        )

    if len(net.load):
        mask = string.injection_types == "LOAD"
        assert mask.sum() >= len(net.load), "Load injections missing from import"
        _assert_close(
            "load P",
            -dynamic.injection_active_power[mask][: len(net.load)],
            net.res_load["p_mw"].to_numpy() / net.sn_mva,
            tol,
        )
        _assert_close(
            "load Q",
            -dynamic.injection_reactive_power[mask][: len(net.load)],
            net.res_load["q_mvar"].to_numpy() / net.sn_mva,
            tol,
        )


@pytest.mark.parametrize(
    "network_func",
    [
        pp.networks.case9,
        pp.networks.case14,
        pp.networks.case30,
        pp.networks.case118,
        pp.networks.case1354pegase,
    ],
)
def test_pandapower_import_consistency(network_func) -> None:
    net = network_func()
    pp.runpp(net, calculate_voltage_angles=True)
    assert_import_matches_pandapower(net, tol=1e-9)
