# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Branch flow helpers building on the LODF voltage solver."""

from typing import Tuple

import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Bool, Complex128, Float, Int

from ..interfaces.network_inputs import (
    JacobianComponentInputs,
    NetworkAdmittanceInputs,
    NetworkTopologyInputs,
    VoltageStateInputs,
)
from .lodf_voltages import line_outage_post_contingency_voltages
from .network_state_helper import _calculate_branch_complex_power

# ruff: noqa: PLR0913


@pytree_dataclass
class SolverLoadflowResults:
    """One-step post-contingency results for monitored elements (SoA)."""

    n_1_theta: Float[Array, "... n_outages n_buses_monitored"]
    n_1_voltage: Float[Array, "... n_outages n_buses_monitored"]

    n_1_p_from: Float[Array, "... n_outages n_branches_monitored"]
    n_1_p_to: Float[Array, "... n_outages n_branches_monitored"]
    n_1_q_from: Float[Array, "... n_outages n_branches_monitored"]
    n_1_q_to: Float[Array, "... n_outages n_branches_monitored"]
    n_1_i_from: Float[Array, "... n_outages n_branches_monitored"]
    n_1_i_to: Float[Array, "... n_outages n_branches_monitored"]


def _prepare_monitored_branch_pack(
    network_topology: NetworkTopologyInputs,
    network_admittance: NetworkAdmittanceInputs,
    monitor_branch_indices: Int[jnp.ndarray, " n_mon_br"],
    bus_to_mon_index: Int[jnp.ndarray, " n_buses"],
    dtype: jnp.dtype,
) -> Tuple[
    Int[jnp.ndarray, " n_mon_br"],
    Complex128[jnp.ndarray, " n_mon_br"],
    Complex128[jnp.ndarray, " n_mon_br"],
    Complex128[jnp.ndarray, " n_mon_br"],
    Complex128[jnp.ndarray, " n_mon_br"],
    Int[jnp.ndarray, " n_mon_br"],
    Int[jnp.ndarray, " n_mon_br"],
    Float[jnp.ndarray, " n_mon_br"],
]:
    """Gather monitored branch admittances and prepare safe bus indices.

    Parameters
    ----------
    network_topology : NetworkTopologyInputs
        Network topology inputs for the base operating point.
    network_admittance : NetworkAdmittanceInputs
        Network admittance inputs for the base operating point.
    monitor_branch_indices: Int[jnp.ndarray, " n_mon_br"]
        Indices of monitored branches.
    bus_to_mon_index: Int[jnp.ndarray, " n_buses"]
        Mapping from bus indices to monitored bus positions (-1 if not monitored).
    dtype: jnp.dtype
        Data type for the end mask.

    Returns
    -------
    Tuple containing:
    - mon_br: Int[jnp.ndarray, " n_mon_br"]
        The indices of the monitored branches.
    - y_ff_mon, y_ft_mon, y_tf_mon, y_tt_mon: Complex128[jnp.ndarray, " n_mon_br"]
        The admittance components for the monitored branches.
    - f_pos_safe, t_pos_safe: Int[jnp.ndarray, " n_mon_br"]
        Safe bus position indices for the from and to buses of monitored branches.
    - end_mask: Float[jnp.ndarray, " n_mon_br"]
        Mask indicating which monitored branches have both endpoints monitored.
    """
    mon_br = monitor_branch_indices
    y_ff_mon = jnp.take(network_admittance.y_ff, mon_br, axis=0)
    y_ft_mon = jnp.take(network_admittance.y_ft, mon_br, axis=0)
    y_tf_mon = jnp.take(network_admittance.y_tf, mon_br, axis=0)
    y_tt_mon = jnp.take(network_admittance.y_tt, mon_br, axis=0)

    f_bus = jnp.take(network_topology.branch_from, mon_br, axis=0)
    t_bus = jnp.take(network_topology.branch_to, mon_br, axis=0)
    f_pos = jnp.take(bus_to_mon_index, f_bus, axis=0)
    t_pos = jnp.take(bus_to_mon_index, t_bus, axis=0)

    f_ok = f_pos >= 0
    t_ok = t_pos >= 0
    f_pos_safe = jnp.where(f_ok, f_pos, 0).astype(jnp.int32)
    t_pos_safe = jnp.where(t_ok, t_pos, 0).astype(jnp.int32)
    end_mask = (f_ok & t_ok).astype(dtype)

    return (
        mon_br,
        y_ff_mon,
        y_ft_mon,
        y_tf_mon,
        y_tt_mon,
        f_pos_safe,
        t_pos_safe,
        end_mask,
    )


def _compute_monitored_branch_currents(
    theta_all: Float[jnp.ndarray, " n_outages n_mon_bus"],
    vm_all: Float[jnp.ndarray, " n_outages n_mon_bus"],
    f_pos_safe: Int[jnp.ndarray, " n_mon_br"],
    t_pos_safe: Int[jnp.ndarray, " n_mon_br"],
    y_ff_mon: Complex128[jnp.ndarray, " n_mon_br"],
    y_ft_mon: Complex128[jnp.ndarray, " n_mon_br"],
    y_tf_mon: Complex128[jnp.ndarray, " n_mon_br"],
    y_tt_mon: Complex128[jnp.ndarray, " n_mon_br"],
    end_mask: Float[jnp.ndarray, " n_mon_br"],
) -> Tuple[
    Complex128[jnp.ndarray, " n_outages n_mon_br"],
    Complex128[jnp.ndarray, " n_outages n_mon_br"],
    Complex128[jnp.ndarray, " n_outages n_mon_br"],
    Complex128[jnp.ndarray, " n_outages n_mon_br"],
]:
    """Compute monitored branch endpoint voltages and currents for all outages.

    Parameters
    ----------
    theta_all: Float[jnp.ndarray, " n_outages n_mon_bus"]
        Post-contingency voltage angles for monitored buses across all outages.
    vm_all: Float[jnp.ndarray, " n_outages n_mon_bus"]
        Post-contingency voltage magnitudes for monitored buses across all outages.
    f_pos_safe: Int[jnp.ndarray, " n_mon_br"]
        Safe from bus position indices for monitored branches.
    t_pos_safe: Int[jnp.ndarray, " n_mon_br"]
        Safe to bus position indices for monitored branches.
    y_ff_mon, y_ft_mon, y_tf_mon, y_tt_mon: Complex128[jnp.ndarray, " n_mon_br"]
        Admittance components for monitored branches.
    end_mask: Float[jnp.ndarray, " n_mon_br"]
        Mask for monitored branches whose endpoints are available in ``theta_all``/``vm_all``.

    Returns
    -------
    Tuple containing:
    - v_from_all: Complex128[jnp.ndarray, " n_outages n_mon_br"]
        Voltages at the from buses of monitored branches for all outages.
    - v_to_all: Complex128[jnp.ndarray, " n_outages n_mon_br"]
        Voltages at the to buses of monitored branches for all outages.
    - i_from_all: Complex128[jnp.ndarray, " n_outages n_mon_br"]
        Currents at the from buses of monitored branches for all outages.
    - i_to_all: Complex128[jnp.ndarray, " n_outages n_mon_br"]
        Currents at the to buses of monitored branches for all outages.
    """
    complex_dtype = jnp.result_type(theta_all, vm_all, y_ff_mon, y_ft_mon, y_tf_mon, y_tt_mon, 1j)
    one_j = jnp.asarray(1j, dtype=complex_dtype)
    voltage_all = vm_all.astype(complex_dtype) * (jnp.cos(theta_all) + one_j * jnp.sin(theta_all))

    v_from_all = jnp.take(voltage_all, f_pos_safe, axis=1)
    v_to_all = jnp.take(voltage_all, t_pos_safe, axis=1)

    mask_complex = end_mask.astype(complex_dtype)[None, :]
    i_from_all = (v_from_all * y_ff_mon[None, :] + v_to_all * y_ft_mon[None, :]) * mask_complex
    i_to_all = (v_from_all * y_tf_mon[None, :] + v_to_all * y_tt_mon[None, :]) * mask_complex
    return v_from_all, v_to_all, i_from_all, i_to_all


def line_outage_post_contingency_voltages_current(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    outage_branch_idx: Int[jnp.ndarray, " n_outages"],
    jacobian_components: JacobianComponentInputs,
    network_topology: NetworkTopologyInputs,
    voltage_state: VoltageStateInputs,
    network_admittance: NetworkAdmittanceInputs,
    monitor_bus_indices: Int[jnp.ndarray, " n_mon_bus"],
    branch_pq_base: Float[jnp.ndarray, " n_branches 4"],
    lower_residual_uses_reactive_mismatch: Bool[jnp.ndarray, " n_buses"],
    monitor_branch_indices: Int[jnp.ndarray, " n_mon_br"],
    bus_to_mon_index: Int[jnp.ndarray, " n_buses"],
) -> Tuple[
    Float[jnp.ndarray, " n_outages n_mon_bus"],
    Float[jnp.ndarray, " n_outages n_mon_bus"],
    Complex128[jnp.ndarray, " n_outages n_mon_br"],
    Complex128[jnp.ndarray, " n_outages n_mon_br"],
]:
    """Compute post-contingency monitored bus voltages and branch currents."""
    dtype = jacobian_inv_transposed.dtype

    theta_all, vm_all = line_outage_post_contingency_voltages(
        jacobian_inv_transposed=jacobian_inv_transposed,
        outage_branch_idx=outage_branch_idx,
        jacobian_components=jacobian_components,
        network_topology=network_topology,
        voltage_state=voltage_state,
        network_admittance=network_admittance,
        monitor_bus_indices=monitor_bus_indices,
        branch_pq_base=branch_pq_base,
        lower_residual_uses_reactive_mismatch=lower_residual_uses_reactive_mismatch,
    )

    _, y_ff_mon, y_ft_mon, y_tf_mon, y_tt_mon, f_pos_safe, t_pos_safe, end_mask = _prepare_monitored_branch_pack(
        network_topology=network_topology,
        network_admittance=network_admittance,
        monitor_branch_indices=monitor_branch_indices,
        bus_to_mon_index=bus_to_mon_index,
        dtype=dtype,
    )

    _, _, i_from_all, i_to_all = _compute_monitored_branch_currents(
        theta_all=theta_all,
        vm_all=vm_all,
        f_pos_safe=f_pos_safe,
        t_pos_safe=t_pos_safe,
        y_ff_mon=y_ff_mon,
        y_ft_mon=y_ft_mon,
        y_tf_mon=y_tf_mon,
        y_tt_mon=y_tt_mon,
        end_mask=end_mask,
    )

    return theta_all, vm_all, i_from_all, i_to_all


def line_outage_post_contingency_monitored(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    outage_branch_idx: Int[jnp.ndarray, " n_outages"],
    jacobian_components: JacobianComponentInputs,
    network_topology: NetworkTopologyInputs,
    voltage_state: VoltageStateInputs,
    network_admittance: NetworkAdmittanceInputs,
    monitor_bus_indices: Int[jnp.ndarray, " n_mon_bus"],
    branch_pq_base: Float[jnp.ndarray, " n_branches 4"],
    lower_residual_uses_reactive_mismatch: Bool[jnp.ndarray, " n_buses"],
    monitor_branch_indices: Int[jnp.ndarray, " n_mon_br"],
    bus_to_mon_index: Int[jnp.ndarray, " n_buses"],
) -> SolverLoadflowResults:
    """Compute post-contingency bus states and monitored branch powers."""
    dtype = jacobian_inv_transposed.dtype

    theta_all, vm_all = line_outage_post_contingency_voltages(
        jacobian_inv_transposed=jacobian_inv_transposed,
        outage_branch_idx=outage_branch_idx,
        jacobian_components=jacobian_components,
        network_topology=network_topology,
        voltage_state=voltage_state,
        network_admittance=network_admittance,
        monitor_bus_indices=monitor_bus_indices,
        branch_pq_base=branch_pq_base,
        lower_residual_uses_reactive_mismatch=lower_residual_uses_reactive_mismatch,
    )

    (
        mon_br,
        y_ff_mon,
        y_ft_mon,
        y_tf_mon,
        y_tt_mon,
        f_pos_safe,
        t_pos_safe,
        end_mask,
    ) = _prepare_monitored_branch_pack(
        network_topology=network_topology,
        network_admittance=network_admittance,
        monitor_branch_indices=monitor_branch_indices,
        bus_to_mon_index=bus_to_mon_index,
        dtype=dtype,
    )

    v_from_all, v_to_all, i_from_all, i_to_all = _compute_monitored_branch_currents(
        theta_all=theta_all,
        vm_all=vm_all,
        f_pos_safe=f_pos_safe,
        t_pos_safe=t_pos_safe,
        y_ff_mon=y_ff_mon,
        y_ft_mon=y_ft_mon,
        y_tf_mon=y_tf_mon,
        y_tt_mon=y_tt_mon,
        end_mask=end_mask,
    )

    end_mask_real = end_mask.astype(dtype)[None, :]

    s_from_all, s_to_all = _calculate_branch_complex_power(
        v_from=v_from_all,
        v_to=v_to_all,
        current_from=i_from_all,
        current_to=i_to_all,
        branch_mask=end_mask,
    )

    is_outaged = mon_br[None, :] == outage_branch_idx[:, None]
    zeros_complex = jnp.zeros_like(s_from_all)
    s_from_all = jnp.where(is_outaged, zeros_complex, s_from_all)
    s_to_all = jnp.where(is_outaged, zeros_complex, s_to_all)

    p_from_all = s_from_all.real.astype(dtype) * end_mask_real
    p_to_all = s_to_all.real.astype(dtype) * end_mask_real
    q_from_all = s_from_all.imag.astype(dtype) * end_mask_real
    q_to_all = s_to_all.imag.astype(dtype) * end_mask_real

    return SolverLoadflowResults(
        n_1_theta=theta_all,
        n_1_voltage=vm_all,
        n_1_p_from=p_from_all,
        n_1_p_to=p_to_all,
        n_1_q_from=q_from_all,
        n_1_q_to=q_to_all,
        n_1_i_from=i_from_all,
        n_1_i_to=i_to_all,
    )
