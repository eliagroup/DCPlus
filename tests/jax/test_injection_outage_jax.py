# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from dc_plus.example_grids.pypowsbl.example_grids import PANDAPOWER_NETWORKS_FOR_POWSYBL, POWSYBL_NETWORKS
from dc_plus.importing.powsybl.powsybl_import import _get_injections_powsybl
from dc_plus.importing.powsybl.powsybl_network_helpers import _load_test_grid
from dc_plus.jax.injection_outage import (
    non_voltage_regulating_injection_outage_monitor_buses,
    non_voltage_regulating_injection_outage_monitored,
)
from dc_plus.jax.injection_outage import non_voltage_regulating_injection_outage_dx as jax_injection_outage_dx
from dc_plus.numpy.injection_outage import non_voltage_regulating_injection_outage_dx as numpy_injection_outage_dx

powsybl_networks = POWSYBL_NETWORKS
pandapower_networks = PANDAPOWER_NETWORKS_FOR_POWSYBL
TOL = 1e-10


def _connected_injection_indices(injections):
    candidates = np.flatnonzero(injections["connected"].to_numpy(dtype=bool))
    if candidates.size == 0:
        pytest.skip("No connected injections available for an injection outage test")
    return candidates


def _supported_injection_outage_rows(net, injections, max_rows: int | None = None) -> list[np.ndarray]:
    injection_powsybl_type = net.get_injections(attributes=["type"])
    outage_rows: list[np.ndarray] = []
    for injection_idx in _connected_injection_indices(injections):
        injection_id = injections.loc[injection_idx, "id_str"]
        if (injection_powsybl_type.loc[injection_id, "type"]) in (["HVDC_CONVERTER_STATION", "DANGLING_LINE"]):
            continue
        outage_rows.append(np.array([injection_idx], dtype=np.int64))
        if max_rows is not None and len(outage_rows) >= max_rows:
            break
    return outage_rows


def _pad_outage_rows(outage_rows: list[np.ndarray]) -> np.ndarray:
    if not outage_rows:
        return np.empty((0, 0), dtype=np.int64)

    max_outages = max(outage_row.size for outage_row in outage_rows)
    padded = np.full((len(outage_rows), max_outages), -1, dtype=np.int64)
    for row_idx, outage_row in enumerate(outage_rows):
        padded[row_idx, : outage_row.size] = outage_row
    return padded


def test_non_voltage_regulating_injection_outage_jax_batch_multi_outage() -> None:
    jacobian_inv = np.eye(4, dtype=float)
    outage_batch = np.array([[0, 2], [1, -1], [-1, -1]], dtype=np.int64)
    injection_to_bus = np.array([0, 1, 0], dtype=np.int64)
    injection_active_power = np.array([1.0, 2.0, 3.0], dtype=float)
    injection_reactive_power = np.array([10.0, 20.0, 30.0], dtype=float)
    angle_component_indices = np.array([0, 1], dtype=np.int64)
    magnitude_component_indices = np.array([2, 3], dtype=np.int64)

    dx_batch_jax = np.asarray(
        jax_injection_outage_dx(
            jacobian_inv_transposed=jnp.asarray(jacobian_inv.T),
            outage_injection_indices=jnp.asarray(outage_batch),
            injection_to_bus=jnp.asarray(injection_to_bus),
            injection_active_power=jnp.asarray(injection_active_power),
            injection_reactive_power=jnp.asarray(injection_reactive_power),
            angle_component_indices=jnp.asarray(angle_component_indices),
            magnitude_component_indices=jnp.asarray(magnitude_component_indices),
        )
    )
    dx_batch_numpy = numpy_injection_outage_dx(
        jacobian_inv=jacobian_inv,
        outage_injection_indices=outage_batch,
        injection_to_bus=injection_to_bus,
        injection_active_power=injection_active_power,
        injection_reactive_power=injection_reactive_power,
        angle_component_indices=angle_component_indices,
        magnitude_component_indices=magnitude_component_indices,
    )

    np.testing.assert_allclose(dx_batch_jax, dx_batch_numpy, rtol=TOL, atol=TOL)


@pytest.mark.parametrize("get_net", powsybl_networks + pandapower_networks)
def test_non_voltage_regulating_injection_outage_jax(get_net) -> None:
    net, _, dynamic_info, _, jacobian_data = _load_test_grid(get_net)
    injections = _get_injections_powsybl(net).reset_index(drop=True)
    outage_p = injections["setpoint_p"].fillna(injections["p"]).to_numpy(dtype=float)
    outage_q = injections["setpoint_q"].fillna(injections["q"]).to_numpy(dtype=float)

    assert len(injections) == dynamic_info.n_injections

    outage_rows = _supported_injection_outage_rows(net, injections)
    assert outage_rows, f"No supported injection outages found for {get_net.__name__}"

    outage_batch = _pad_outage_rows(outage_rows)
    dx_batch_jax = np.asarray(
        jax_injection_outage_dx(
            jacobian_inv_transposed=jnp.asarray(jacobian_data.inverse_jacobian.T),
            outage_injection_indices=jnp.asarray(outage_batch),
            injection_to_bus=jnp.asarray(dynamic_info.injection_to_bus),
            injection_active_power=jnp.asarray(outage_p),
            injection_reactive_power=jnp.asarray(outage_q),
            angle_component_indices=jnp.asarray(jacobian_data.angle_component_indices),
            magnitude_component_indices=jnp.asarray(jacobian_data.magnitude_component_indices),
        )
    )
    dx_batch_numpy = numpy_injection_outage_dx(
        jacobian_inv=jacobian_data.inverse_jacobian,
        outage_injection_indices=outage_batch,
        injection_to_bus=dynamic_info.injection_to_bus,
        injection_active_power=outage_p,
        injection_reactive_power=outage_q,
        angle_component_indices=jacobian_data.angle_component_indices,
        magnitude_component_indices=jacobian_data.magnitude_component_indices,
    )

    assert dx_batch_jax.shape == (len(outage_rows), jacobian_data.inverse_jacobian.shape[0])
    # the numpy code is tested against powsybl -> we do not need to repeat the test against powsybl
    np.testing.assert_allclose(dx_batch_jax, dx_batch_numpy, rtol=TOL, atol=TOL)


@pytest.mark.parametrize("get_net", powsybl_networks + pandapower_networks)
def test_non_voltage_regulating_injection_outage_monitored_jax(get_net) -> None:
    net, _, dynamic_info, _, jacobian_data = _load_test_grid(get_net)
    injections = _get_injections_powsybl(net).reset_index(drop=True)
    outage_p = injections["setpoint_p"].fillna(injections["p"]).to_numpy(dtype=float)
    outage_q = injections["setpoint_q"].fillna(injections["q"]).to_numpy(dtype=float)
    outage_rows = _supported_injection_outage_rows(net, injections, max_rows=5)
    assert outage_rows, f"No supported injection outages found for {get_net.__name__}"
    outage_batch = _pad_outage_rows(outage_rows)

    branch_connected = np.asarray(dynamic_info.branch_connected, dtype=bool)
    monitor_branch_indices = np.flatnonzero(branch_connected)[: min(5, int(branch_connected.sum()))]
    if monitor_branch_indices.size == 0:
        pytest.skip("No connected monitored branches available")

    branch_from = np.asarray(dynamic_info.branch_from_bus, dtype=np.int64)
    branch_to = np.asarray(dynamic_info.branch_to_bus, dtype=np.int64)
    monitor_bus_indices = np.unique(
        np.concatenate([branch_from[monitor_branch_indices], branch_to[monitor_branch_indices]])
    ).astype(np.int64)
    bus_to_mon_index = np.full(dynamic_info.n_buses, -1, dtype=np.int64)
    bus_to_mon_index[monitor_bus_indices] = np.arange(monitor_bus_indices.size, dtype=np.int64)

    monitor_theta_jax, monitor_vm_jax = non_voltage_regulating_injection_outage_monitor_buses(
        jacobian_inv_transposed=jnp.asarray(jacobian_data.inverse_jacobian.T),
        outage_injection_indices=jnp.asarray(outage_batch),
        injection_to_bus=jnp.asarray(dynamic_info.injection_to_bus),
        injection_active_power=jnp.asarray(outage_p),
        injection_reactive_power=jnp.asarray(outage_q),
        angle_component_indices=jnp.asarray(jacobian_data.angle_component_indices),
        magnitude_component_indices=jnp.asarray(jacobian_data.magnitude_component_indices),
        monitor_bus_indices=jnp.asarray(monitor_bus_indices),
        v_mag_hat=jnp.asarray(dynamic_info.bus_voltage_magnitudes),
        theta_hat=jnp.asarray(dynamic_info.bus_voltage_angles_rad),
    )
    monitored_results = non_voltage_regulating_injection_outage_monitored(
        jacobian_inv_transposed=jnp.asarray(jacobian_data.inverse_jacobian.T),
        outage_injection_indices=jnp.asarray(outage_batch),
        injection_to_bus=jnp.asarray(dynamic_info.injection_to_bus),
        injection_active_power=jnp.asarray(outage_p),
        injection_reactive_power=jnp.asarray(outage_q),
        angle_component_indices=jnp.asarray(jacobian_data.angle_component_indices),
        magnitude_component_indices=jnp.asarray(jacobian_data.magnitude_component_indices),
        monitor_bus_indices=jnp.asarray(monitor_bus_indices),
        v_mag_hat=jnp.asarray(dynamic_info.bus_voltage_magnitudes),
        theta_hat=jnp.asarray(dynamic_info.bus_voltage_angles_rad),
        branch_from=jnp.asarray(branch_from),
        branch_to=jnp.asarray(branch_to),
        y_ff=jnp.asarray(dynamic_info.branch_effective_admittance_from_from),
        y_ft=jnp.asarray(dynamic_info.branch_effective_admittance_from_to),
        y_tf=jnp.asarray(dynamic_info.branch_effective_admittance_to_from),
        y_tt=jnp.asarray(dynamic_info.branch_effective_admittance_to_to),
        monitor_branch_indices=jnp.asarray(monitor_branch_indices),
        bus_to_mon_index=jnp.asarray(bus_to_mon_index),
    )
    dx_batch_numpy = numpy_injection_outage_dx(
        jacobian_inv=jacobian_data.inverse_jacobian,
        outage_injection_indices=outage_batch,
        injection_to_bus=dynamic_info.injection_to_bus,
        injection_active_power=outage_p,
        injection_reactive_power=outage_q,
        angle_component_indices=jacobian_data.angle_component_indices,
        magnitude_component_indices=jacobian_data.magnitude_component_indices,
    )

    expected_theta = np.broadcast_to(
        np.asarray(dynamic_info.bus_voltage_angles_rad, dtype=float)[monitor_bus_indices],
        (len(outage_rows), monitor_bus_indices.size),
    ).copy()
    expected_vm = np.broadcast_to(
        np.asarray(dynamic_info.bus_voltage_magnitudes, dtype=float)[monitor_bus_indices],
        (len(outage_rows), monitor_bus_indices.size),
    ).copy()
    theta_component_indices = np.asarray(jacobian_data.angle_component_indices, dtype=np.int64)[monitor_bus_indices]
    vm_component_indices = np.asarray(jacobian_data.magnitude_component_indices, dtype=np.int64)[monitor_bus_indices]
    theta_mask = theta_component_indices >= 0
    vm_mask = vm_component_indices >= 0
    expected_theta[:, theta_mask] += dx_batch_numpy[:, theta_component_indices[theta_mask]]
    expected_vm[:, vm_mask] += dx_batch_numpy[:, vm_component_indices[vm_mask]]

    np.testing.assert_allclose(np.asarray(monitor_theta_jax), expected_theta, rtol=TOL, atol=TOL)
    np.testing.assert_allclose(np.asarray(monitor_vm_jax), expected_vm, rtol=TOL, atol=TOL)
    np.testing.assert_allclose(np.asarray(monitored_results.n_1_theta), expected_theta, rtol=TOL, atol=TOL)
    np.testing.assert_allclose(np.asarray(monitored_results.n_1_voltage), expected_vm, rtol=TOL, atol=TOL)

    f_pos = bus_to_mon_index[branch_from[monitor_branch_indices]]
    t_pos = bus_to_mon_index[branch_to[monitor_branch_indices]]
    assert np.all(f_pos >= 0)
    assert np.all(t_pos >= 0)

    v_post = expected_vm * np.exp(1j * expected_theta)
    y_ff_mon = np.asarray(dynamic_info.branch_effective_admittance_from_from)[monitor_branch_indices]
    y_ft_mon = np.asarray(dynamic_info.branch_effective_admittance_from_to)[monitor_branch_indices]
    y_tf_mon = np.asarray(dynamic_info.branch_effective_admittance_to_from)[monitor_branch_indices]
    y_tt_mon = np.asarray(dynamic_info.branch_effective_admittance_to_to)[monitor_branch_indices]
    v_from = v_post[:, f_pos]
    v_to = v_post[:, t_pos]
    i_from = y_ff_mon[None, :] * v_from + y_ft_mon[None, :] * v_to
    i_to = y_tf_mon[None, :] * v_from + y_tt_mon[None, :] * v_to
    s_from = v_from * np.conj(i_from)
    s_to = v_to * np.conj(i_to)

    np.testing.assert_allclose(np.asarray(monitored_results.n_1_i_from), i_from, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(monitored_results.n_1_i_to), i_to, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(monitored_results.n_1_p_from), s_from.real, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(monitored_results.n_1_p_to), s_to.real, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(monitored_results.n_1_q_from), s_from.imag, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(monitored_results.n_1_q_to), s_to.imag, rtol=1e-9, atol=1e-9)
