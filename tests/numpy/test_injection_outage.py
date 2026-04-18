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

from dc_plus.example_grids.pypowsbl.example_grids import PANDAPOWER_NETWORKS_FOR_POWSYBL, POWSYBL_NETWORKS
from dc_plus.importing.powsybl.powsybl_import import DANGLING_BUS_STRING_SUFFIX, _get_injections_powsybl
from dc_plus.importing.powsybl.powsybl_loadflow_parameter import get_powsybl_loadflow_parameter
from dc_plus.importing.powsybl.powsybl_network_helpers import _load_test_grid
from dc_plus.numpy.injection_outage import non_voltage_regulating_injection_outage_dx
from dc_plus.preprocess.create_network_data import create_network_data

powsybl_networks = POWSYBL_NETWORKS
pandapower_networks = PANDAPOWER_NETWORKS_FOR_POWSYBL
TOL = 5e-10


def _connected_injection_indices(injections):
    candidates = np.flatnonzero(injections["connected"].to_numpy(dtype=bool))
    if candidates.size == 0:
        pytest.skip("No connected injections available for an injection outage test")
    return candidates


def _expected_bus_delta(dynamic_info, outage_injection_indices, outage_p, outage_q):
    delta_p = np.zeros(dynamic_info.n_buses, dtype=float)
    delta_q = np.zeros(dynamic_info.n_buses, dtype=float)
    outage_indices = np.asarray(outage_injection_indices, dtype=np.int64)
    valid_mask = outage_indices >= 0
    if not np.any(valid_mask):
        return delta_p, delta_q

    valid_outages = outage_indices[valid_mask]
    buses = dynamic_info.injection_to_bus[valid_outages]
    np.add.at(delta_p, buses, -outage_p[valid_outages])
    np.add.at(delta_q, buses, -outage_q[valid_outages])
    return delta_p, delta_q


def _pad_outage_rows(outage_rows: list[np.ndarray]) -> np.ndarray:
    if not outage_rows:
        return np.empty((0, 0), dtype=np.int64)

    max_outages = max(outage_row.size for outage_row in outage_rows)
    padded = np.full((len(outage_rows), max_outages), -1, dtype=np.int64)
    for row_idx, outage_row in enumerate(outage_rows):
        padded[row_idx, : outage_row.size] = outage_row
    return padded


@pytest.mark.parametrize("get_net", powsybl_networks + pandapower_networks)
def test_non_voltage_regulating_injection_outage_numpy(get_net) -> None:
    """Test non voltage regulating injection outages with a one-step update against powsybl load flow results.

    This does not test HVDC converter and dangling line outages.
    It does test voltage regulatingGenerator outages, as the powsybl one step does also not switch from PV to PQ
    node. This therefore does not account for the voltage drop of the disconnected generator.

    In DC+ dangling lines are modelled as an additional bus with a fixed injection p0 and q0
    a one sided line is added between the original bus and the new bus to account for the line losses
    of the dangling line. Here numerical errors occur.
    The test still verifies the voltages of the bus where the dangling line is attached to with the
    full precision.

    Parameters
    ----------
    get_net : Callable[[], pypowsybl.network.Network]
        A fixture that returns a powsybl network to test on.
    """
    net, _, dynamic_info, string_info, jacobian_data = _load_test_grid(get_net)
    injections = _get_injections_powsybl(net).reset_index(drop=True)
    injection_powsybl_type = net.get_injections(attributes=["type"])
    outage_p = injections["setpoint_p"].fillna(injections["p"]).to_numpy(dtype=float)
    outage_q = injections["setpoint_q"].fillna(injections["q"]).to_numpy(dtype=float)
    dangling_bus_mask = np.char.endswith(string_info.bus_ids.astype(str), DANGLING_BUS_STRING_SUFFIX)

    assert len(injections) == dynamic_info.n_injections

    accepted_cases: list[tuple[str, np.ndarray, object]] = []
    for injection_idx in _connected_injection_indices(injections):
        injection_id = injections.loc[injection_idx, "id_str"]
        if (injection_powsybl_type.loc[injection_id, "type"]) in (["HVDC_CONVERTER_STATION", "DANGLING_LINE"]):
            # - HVDC would need a multi outage
            # - Dangling line removal would change the network for DC+
            continue

        net_n1 = deepcopy(net)
        net_n1.remove_elements(injection_id)

        loadflow_parameter = get_powsybl_loadflow_parameter("one_step")
        loadflow_res = pypowsybl.loadflow.run_ac(net_n1, parameters=loadflow_parameter)[0]
        assert loadflow_res.iteration_count <= 1
        if loadflow_res.status != pypowsybl._pypowsybl.LoadFlowComponentStatus.CONVERGED:
            continue

        _, dynamic_info_n1, string_info_n1 = create_network_data(net_n1)
        if not np.array_equal(string_info.bus_ids, string_info_n1.bus_ids):
            continue

        same_angle_structure = np.array_equal(
            dynamic_info.pvpq_buses_indices_pvpq_order,
            dynamic_info_n1.pvpq_buses_indices_pvpq_order,
        )
        same_magnitude_structure = np.array_equal(
            dynamic_info.pq_buses_indices,
            dynamic_info_n1.pq_buses_indices,
        )
        if not (same_angle_structure and same_magnitude_structure):
            continue

        outage_row = np.array([injection_idx], dtype=np.int64)
        expected_delta_p, expected_delta_q = _expected_bus_delta(dynamic_info, outage_row, outage_p, outage_q)
        actual_delta_p = dynamic_info_n1.bus_active_power - dynamic_info.bus_active_power
        actual_delta_q = dynamic_info_n1.bus_reactive_power - dynamic_info.bus_reactive_power
        p_buses = dynamic_info.pvpq_buses_indices_pvpq_order
        q_buses = dynamic_info.pq_buses_indices
        if not (
            np.allclose(actual_delta_p[p_buses], expected_delta_p[p_buses], rtol=TOL, atol=TOL)
            and np.allclose(actual_delta_q[q_buses], expected_delta_q[q_buses], rtol=TOL, atol=TOL)
        ):
            continue

        accepted_cases.append((injection_id, outage_row, dynamic_info_n1))

    assert accepted_cases, f"No injection outages matched the fixed-injection assumptions for {get_net.__name__}"

    outage_batch = _pad_outage_rows([outage_row for _, outage_row, _ in accepted_cases])
    dx_batch = non_voltage_regulating_injection_outage_dx(
        jacobian_inv=jacobian_data.inverse_jacobian,
        outage_injection_indices=outage_batch,
        injection_to_bus=dynamic_info.injection_to_bus,
        injection_active_power=outage_p,
        injection_reactive_power=outage_q,
        angle_component_indices=jacobian_data.angle_component_indices,
        magnitude_component_indices=jacobian_data.magnitude_component_indices,
    )

    assert dx_batch.shape == (len(accepted_cases), jacobian_data.inverse_jacobian.shape[0])

    for case_idx, (_, _, dynamic_info_n1) in enumerate(accepted_cases):
        dx = dx_batch[case_idx]
        theta_updated = dynamic_info.bus_voltage_angles_rad.copy()
        vm_updated = dynamic_info.bus_voltage_magnitudes.copy()
        theta_updated[dynamic_info.pvpq_buses_indices_pvpq_order] += dx[jacobian_data.is_angle_component]
        vm_updated[dynamic_info.pq_buses_indices] += dx[jacobian_data.is_magnitude_component]
        np.testing.assert_allclose(
            theta_updated[~dangling_bus_mask],
            dynamic_info_n1.bus_voltage_angles_rad[~dangling_bus_mask],
            rtol=TOL,
            atol=TOL,
        )
        np.testing.assert_allclose(
            vm_updated[~dangling_bus_mask],
            dynamic_info_n1.bus_voltage_magnitudes[~dangling_bus_mask],
            rtol=TOL,
            atol=TOL,
        )
