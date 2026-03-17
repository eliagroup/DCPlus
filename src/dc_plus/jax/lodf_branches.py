# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Branch flow helpers building on the LODF voltage solver."""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Complex128, Float, Int

from .lodf_voltages import line_outage_post_contingency_voltages

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


def _prepare_voltages_for_currents_not_linearized(
    theta_post: Float[jnp.ndarray, " n_mon_bus"],
    vm_post: Float[jnp.ndarray, " n_mon_bus"],
    dtype: jnp.dtype,
    complex_dtype: jnp.dtype,
) -> Complex128[jnp.ndarray, " n_mon_bus"]:
    """Build complex monitored-bus voltages from NR state updates.

    Parameters    ----------
    theta_post: Float[jnp.ndarray, " n_mon_bus"]
        Post-contingency voltage angle updates for monitored buses.
    vm_post: Float[jnp.ndarray, " n_mon_bus"]
        Post-contingency voltage magnitude updates for monitored buses.
    dtype: jnp.dtype
        Data type for intermediate computations.
    complex_dtype: jnp.dtype
        Complex data type for the output voltages.

    Returns
    -------
    Complex128[jnp.ndarray, " n_mon_bus"]
        Complex voltage values for monitored buses, ready for current calculations.
    """
    one_j = jnp.asarray(1j, dtype=complex_dtype)
    theta_post_real = theta_post.astype(dtype)
    vm_post_real = vm_post.astype(dtype)
    return vm_post_real * (jnp.cos(theta_post_real) + one_j * jnp.sin(theta_post_real))


def _calculate_branch_currents(
    v_post: Complex128[jnp.ndarray, " n_mon_bus"],
    f_pos_safe: Int[jnp.ndarray, " n_mon_br"],
    t_pos_safe: Int[jnp.ndarray, " n_mon_br"],
    y_ff_mon: Complex128[jnp.ndarray, " n_mon_br"],
    y_ft_mon: Complex128[jnp.ndarray, " n_mon_br"],
    y_tf_mon: Complex128[jnp.ndarray, " n_mon_br"],
    y_tt_mon: Complex128[jnp.ndarray, " n_mon_br"],
) -> Tuple[
    Complex128[jnp.ndarray, " n_mon_br"],
    Complex128[jnp.ndarray, " n_mon_br"],
    Complex128[jnp.ndarray, " n_mon_br"],
    Complex128[jnp.ndarray, " n_mon_br"],
]:
    """Gather branch endpoint voltages and compute monitored currents.

    Parameters
    ----------
    v_post: Complex128[jnp.ndarray, " n_mon_bus"]
        Complex voltage values for monitored buses.
    f_pos_safe: Int[jnp.ndarray, " n_mon_br"]
        Safe from bus position indices for monitored branches.
    t_pos_safe: Int[jnp.ndarray, " n_mon_br"]
        Safe to bus position indices for monitored branches.
    y_ff_mon: Complex128[jnp.ndarray, " n_mon_br"]
        Monitored branch admittance from-from components.
    y_ft_mon: Complex128[jnp.ndarray, " n_mon_br"]
        Monitored branch admittance from-to components.
    y_tf_mon: Complex128[jnp.ndarray, " n_mon_br"]
        Monitored branch admittance to-from components.
    y_tt_mon: Complex128[jnp.ndarray, " n_mon_br"]
        Monitored branch admittance to-to components.

    Returns
    -------
    Tuple containing:


    """
    v_from = v_post[f_pos_safe]
    v_to = v_post[t_pos_safe]

    current_from = y_ff_mon * v_from + y_ft_mon * v_to
    current_to = y_tf_mon * v_from + y_tt_mon * v_to

    return v_from, v_to, current_from, current_to


def _compute_complex_branch_powers(
    v_from: Complex128[jnp.ndarray, " n_mon_br"],
    v_to: Complex128[jnp.ndarray, " n_mon_br"],
    current_from: Complex128[jnp.ndarray, " n_mon_br"],
    current_to: Complex128[jnp.ndarray, " n_mon_br"],
    end_mask: Float[jnp.ndarray, " n_mon_br"],
    mon_br: Int[jnp.ndarray, " n_mon_br"],
    outage_idx: Int[jnp.ndarray, ""],
) -> Tuple[
    Complex128[jnp.ndarray, " n_mon_br"],
    Complex128[jnp.ndarray, " n_mon_br"],
]:
    """Compute complex power flow endpoints for monitored branches."""
    end_mask_complex = end_mask.astype(v_from.dtype)

    s_from = v_from * jnp.conj(current_from) * end_mask_complex
    s_to = v_to * jnp.conj(current_to) * end_mask_complex

    is_outaged = mon_br == outage_idx
    zeros = jnp.zeros_like(s_from)

    s_from = jnp.where(is_outaged, zeros, s_from)
    s_to = jnp.where(is_outaged, zeros, s_to)

    return s_from, s_to


def _prepare_monitored_branch_pack(
    branch_from: Int[jnp.ndarray, " n_branches"],
    branch_to: Int[jnp.ndarray, " n_branches"],
    y_ff: Complex128[jnp.ndarray, " n_branches"],
    y_ft: Complex128[jnp.ndarray, " n_branches"],
    y_tf: Complex128[jnp.ndarray, " n_branches"],
    y_tt: Complex128[jnp.ndarray, " n_branches"],
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
    branch_from: Int[jnp.ndarray, " n_branches"]
        From bus indices for all branches.
    branch_to: Int[jnp.ndarray, " n_branches"]
        To bus indices for all branches.
    y_ff, y_ft, y_tf, y_tt: Complex128[jnp.ndarray, " n_branches"]
        Pi-model admittance components for all branches.
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
    mon_br = jnp.asarray(monitor_branch_indices, dtype=jnp.int32)
    y_ff_mon = jnp.take(y_ff, mon_br, axis=0)
    y_ft_mon = jnp.take(y_ft, mon_br, axis=0)
    y_tf_mon = jnp.take(y_tf, mon_br, axis=0)
    y_tt_mon = jnp.take(y_tt, mon_br, axis=0)

    f_bus = jnp.take(branch_from, mon_br, axis=0)
    t_bus = jnp.take(branch_to, mon_br, axis=0)
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
    dtype: jnp.dtype,
) -> Tuple[
    Complex128[jnp.ndarray, " n_outages n_mon_br"],
    Complex128[jnp.ndarray, " n_outages n_mon_br"],
    Complex128[jnp.ndarray, " n_outages n_mon_br"],
    Complex128[jnp.ndarray, " n_outages n_mon_br"],
]:
    """Vectorized monitored branch current computation for all outages.

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
    dtype: jnp.dtype
        Data type for intermediate computations.

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
    complex_dtype = y_ff_mon.dtype

    def _per_outage(
        theta_post: jnp.ndarray, vm_post: jnp.ndarray
    ) -> Tuple[
        Complex128[jnp.ndarray, " n_mon_br"],
        Complex128[jnp.ndarray, " n_mon_br"],
        Complex128[jnp.ndarray, " n_mon_br"],
        Complex128[jnp.ndarray, " n_mon_br"],
    ]:
        v_post = _prepare_voltages_for_currents_not_linearized(
            theta_post=theta_post,
            vm_post=vm_post,
            dtype=dtype,
            complex_dtype=complex_dtype,
        )
        return _calculate_branch_currents(
            v_post=v_post,
            f_pos_safe=f_pos_safe,
            t_pos_safe=t_pos_safe,
            y_ff_mon=y_ff_mon,
            y_ft_mon=y_ft_mon,
            y_tf_mon=y_tf_mon,
            y_tt_mon=y_tt_mon,
        )

    return jax.vmap(_per_outage)(theta_all, vm_all)


def line_outage_post_contingency_voltages_current(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    outage_branch_idx: Int[jnp.ndarray, " n_outages"],
    branch_from: Int[jnp.ndarray, " n_branches"],
    branch_to: Int[jnp.ndarray, " n_branches"],
    v_mag_hat: Float[jnp.ndarray, " n_buses"],
    theta_hat: Float[jnp.ndarray, " n_buses"],
    angle_component_indices: Int[jnp.ndarray, " n_eq_jacobian"],
    magnitude_component_indices: Int[jnp.ndarray, " n_eq_jacobian"],
    y_ff: Complex128[jnp.ndarray, " n_branches"],
    y_ft: Complex128[jnp.ndarray, " n_branches"],
    y_tf: Complex128[jnp.ndarray, " n_branches"],
    y_tt: Complex128[jnp.ndarray, " n_branches"],
    monitor_bus_indices: Int[jnp.ndarray, " n_mon_bus"],
    branch_pq_base: Float[jnp.ndarray, " n_branches 4"],
    monitor_branch_indices: Int[jnp.ndarray, " n_mon_br"],
    bus_to_mon_index: Int[jnp.ndarray, " n_buses"],
) -> Tuple[
    Float[jnp.ndarray, " n_outages n_mon_bus"],
    Float[jnp.ndarray, " n_outages n_mon_bus"],
    Complex128[jnp.ndarray, " n_outages n_mon_br"],
    Complex128[jnp.ndarray, " n_outages n_mon_br"],
]:
    """Compute post-contingency monitored bus voltages and branch currents."""
    jacobian_inv_transposed = jnp.asarray(jacobian_inv_transposed)
    dtype = jacobian_inv_transposed.dtype

    outage_branch_idx = jnp.asarray(outage_branch_idx, dtype=jnp.int32)
    monitor_bus_indices = jnp.asarray(monitor_bus_indices, dtype=jnp.int32)

    branch_from = jnp.asarray(branch_from, dtype=jnp.int32)
    branch_to = jnp.asarray(branch_to, dtype=jnp.int32)
    v_mag_hat = jnp.asarray(v_mag_hat, dtype=dtype)
    theta_hat = jnp.asarray(theta_hat, dtype=dtype)
    angle_component_indices = jnp.asarray(angle_component_indices, dtype=jnp.int32)
    magnitude_component_indices = jnp.asarray(magnitude_component_indices, dtype=jnp.int32)
    y_ff = jnp.asarray(y_ff)
    y_ft = jnp.asarray(y_ft)
    y_tf = jnp.asarray(y_tf)
    y_tt = jnp.asarray(y_tt)
    branch_pq_base = jnp.asarray(branch_pq_base, dtype=dtype)
    monitor_branch_indices = jnp.asarray(monitor_branch_indices, dtype=jnp.int32)
    bus_to_mon_index = jnp.asarray(bus_to_mon_index, dtype=jnp.int32)

    theta_all, vm_all = line_outage_post_contingency_voltages(
        jacobian_inv_transposed=jacobian_inv_transposed,
        outage_branch_idx=outage_branch_idx,
        branch_from=branch_from,
        branch_to=branch_to,
        v_mag_hat=v_mag_hat,
        theta_hat=theta_hat,
        angle_component_indices=angle_component_indices,
        magnitude_component_indices=magnitude_component_indices,
        y_ff=y_ff,
        y_ft=y_ft,
        y_tf=y_tf,
        y_tt=y_tt,
        monitor_bus_indices=monitor_bus_indices,
        branch_pq_base=branch_pq_base,
    )

    _, y_ff_mon, y_ft_mon, y_tf_mon, y_tt_mon, f_pos_safe, t_pos_safe, _ = _prepare_monitored_branch_pack(
        branch_from=branch_from,
        branch_to=branch_to,
        y_ff=y_ff,
        y_ft=y_ft,
        y_tf=y_tf,
        y_tt=y_tt,
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
        dtype=dtype,
    )

    return theta_all, vm_all, i_from_all, i_to_all


def line_outage_post_contingency_monitored(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    outage_branch_idx: Int[jnp.ndarray, " n_outages"],
    branch_from: Int[jnp.ndarray, " n_branches"],
    branch_to: Int[jnp.ndarray, " n_branches"],
    v_mag_hat: Float[jnp.ndarray, " n_buses"],
    theta_hat: Float[jnp.ndarray, " n_buses"],
    angle_component_indices: Int[jnp.ndarray, " n_eq_jacobian"],
    magnitude_component_indices: Int[jnp.ndarray, " n_eq_jacobian"],
    y_ff: Complex128[jnp.ndarray, " n_branches"],
    y_ft: Complex128[jnp.ndarray, " n_branches"],
    y_tf: Complex128[jnp.ndarray, " n_branches"],
    y_tt: Complex128[jnp.ndarray, " n_branches"],
    monitor_bus_indices: Int[jnp.ndarray, " n_mon_bus"],
    branch_pq_base: Float[jnp.ndarray, " n_branches 4"],
    monitor_branch_indices: Int[jnp.ndarray, " n_mon_br"],
    bus_to_mon_index: Int[jnp.ndarray, " n_buses"],
) -> SolverLoadflowResults:
    """Compute post-contingency bus states and monitored branch powers."""
    jacobian_inv_transposed = jnp.asarray(jacobian_inv_transposed)
    dtype = jacobian_inv_transposed.dtype

    outage_branch_idx = jnp.asarray(outage_branch_idx, dtype=jnp.int32)
    monitor_bus_indices = jnp.asarray(monitor_bus_indices, dtype=jnp.int32)

    branch_from = jnp.asarray(branch_from, dtype=jnp.int32)
    branch_to = jnp.asarray(branch_to, dtype=jnp.int32)
    v_mag_hat = jnp.asarray(v_mag_hat, dtype=dtype)
    theta_hat = jnp.asarray(theta_hat, dtype=dtype)
    angle_component_indices = jnp.asarray(angle_component_indices, dtype=jnp.int32)
    magnitude_component_indices = jnp.asarray(magnitude_component_indices, dtype=jnp.int32)
    y_ff = jnp.asarray(y_ff)
    y_ft = jnp.asarray(y_ft)
    y_tf = jnp.asarray(y_tf)
    y_tt = jnp.asarray(y_tt)
    branch_pq_base = jnp.asarray(branch_pq_base, dtype=dtype)
    monitor_branch_indices = jnp.asarray(monitor_branch_indices, dtype=jnp.int32)
    bus_to_mon_index = jnp.asarray(bus_to_mon_index, dtype=jnp.int32)

    theta_all, vm_all = line_outage_post_contingency_voltages(
        jacobian_inv_transposed=jacobian_inv_transposed,
        outage_branch_idx=outage_branch_idx,
        branch_from=branch_from,
        branch_to=branch_to,
        v_mag_hat=v_mag_hat,
        theta_hat=theta_hat,
        angle_component_indices=angle_component_indices,
        magnitude_component_indices=magnitude_component_indices,
        y_ff=y_ff,
        y_ft=y_ft,
        y_tf=y_tf,
        y_tt=y_tt,
        monitor_bus_indices=monitor_bus_indices,
        branch_pq_base=branch_pq_base,
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
        branch_from=branch_from,
        branch_to=branch_to,
        y_ff=y_ff,
        y_ft=y_ft,
        y_tf=y_tf,
        y_tt=y_tt,
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
        dtype=dtype,
    )

    complex_dtype = y_ff_mon.dtype
    end_mask_complex = end_mask.astype(complex_dtype)[None, :]
    end_mask_real = end_mask.astype(dtype)[None, :]

    s_from_all = v_from_all * jnp.conj(i_from_all) * end_mask_complex
    s_to_all = v_to_all * jnp.conj(i_to_all) * end_mask_complex

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
