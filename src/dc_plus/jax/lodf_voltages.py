# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Voltage-only helpers for Jacobian-based N-1 screening."""

from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float, Int

from ..interfaces.network_inputs import (
    JacobianComponentInputs,
    NetworkAdmittanceInputs,
    NetworkTopologyInputs,
    VoltageStateInputs,
)
from .low_rank_helper import _prepare_low_rank_factors_from_admittance

# ruff: noqa: PLR0913


def _dot4_unrolled(
    g0: jnp.ndarray,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    g3: jnp.ndarray,
    w0: jnp.ndarray,
    w1: jnp.ndarray,
    w2: jnp.ndarray,
    w3: jnp.ndarray,
) -> jnp.ndarray:
    """Unrolled 4-term dot to avoid reduction kernels."""
    return g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3


@jax.jit
def build_monitor_rows(
    jacobian_components: JacobianComponentInputs,
    monitor_bus_indices: Int[jnp.ndarray, " n_bus_mon"],
) -> Tuple[
    Int[jnp.ndarray, " n_bus_mon"],
    Int[jnp.ndarray, " n_bus_mon"],
    Float[jnp.ndarray, " n_bus_mon"],
    Float[jnp.ndarray, " n_bus_mon"],
]:
    """Precompute safe Jacobian indices and masks for monitored buses."""
    theta_idx = jacobian_components.angle_component_indices[monitor_bus_indices]
    vm_idx = jacobian_components.magnitude_component_indices[monitor_bus_indices]

    theta_ok = theta_idx >= 0
    vm_ok = vm_idx >= 0

    theta_rows = jnp.where(theta_ok, theta_idx, 0).astype(jnp.int32)
    vm_rows = jnp.where(vm_ok, vm_idx, 0).astype(jnp.int32)

    theta_mask = theta_ok.astype(jnp.float64)
    vm_mask = vm_ok.astype(jnp.float64)

    return theta_rows, vm_rows, theta_mask, vm_mask


@jax.jit
def _compute_post_contingency_states(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    outage_idx: Int[jnp.ndarray, ""],
    mismatch_vec: Float[jnp.ndarray, "4"],
    network_topology: NetworkTopologyInputs,
    voltage_state: VoltageStateInputs,
    jacobian_components: JacobianComponentInputs,
    network_admittance: NetworkAdmittanceInputs,
    base_theta0: Float[jnp.ndarray, " n_mon_bus"],
    base_vm0: Float[jnp.ndarray, " n_mon_bus"],
    theta_rows: Int[jnp.ndarray, " n_mon_bus"],
    vm_rows: Int[jnp.ndarray, " n_mon_bus"],
    theta_mask: Float[jnp.ndarray, " n_mon_bus"],
    vm_mask: Float[jnp.ndarray, " n_mon_bus"],
) -> Tuple[
    Float[jnp.ndarray, " n_mon_bus"],
    Float[jnp.ndarray, " n_mon_bus"],
]:
    """Solve the post-contingency monitored bus states for a single outage."""
    dtype = jacobian_inv_transposed.dtype

    theta_mask_d = theta_mask.astype(dtype)
    vm_mask_d = vm_mask.astype(dtype)

    d_mat, branch_indices, branch_valid_mask = _prepare_low_rank_factors_from_admittance(
        branch_idx=outage_idx,
        network_topology=network_topology,
        voltage_state=voltage_state,
        network_admittance=network_admittance,
        jacobian_components=jacobian_components,
    )
    d_mat = d_mat.astype(dtype)
    branch_indices = branch_indices.astype(jnp.int32)
    branch_mask = branch_valid_mask.astype(dtype)

    mismatch = mismatch_vec.astype(dtype) * branch_mask

    a_sub_t = jacobian_inv_transposed[branch_indices[:, None], branch_indices[None, :]]
    a_sub = a_sub_t.T * branch_mask[:, None] * branch_mask[None, :]

    d_masked = d_mat * branch_mask[:, None] * branch_mask[None, :]

    y_sub = a_sub @ mismatch

    k_mat = jnp.eye(4, dtype=dtype) + (d_masked @ a_sub)
    rhs = d_masked @ y_sub
    corr_factor = jnp.linalg.solve(k_mat, rhs) * branch_mask

    g_th = jacobian_inv_transposed[branch_indices[:, None], theta_rows[None, :]] * branch_mask[:, None]
    g_vm = jacobian_inv_transposed[branch_indices[:, None], vm_rows[None, :]] * branch_mask[:, None]

    theta_base = _dot4_unrolled(
        g_th[0],
        g_th[1],
        g_th[2],
        g_th[3],
        mismatch[0],
        mismatch[1],
        mismatch[2],
        mismatch[3],
    )
    vm_base = _dot4_unrolled(
        g_vm[0],
        g_vm[1],
        g_vm[2],
        g_vm[3],
        mismatch[0],
        mismatch[1],
        mismatch[2],
        mismatch[3],
    )

    base_theta_dx = (-theta_base) * theta_mask_d
    base_vm_dx = (-vm_base) * vm_mask_d

    theta_corr = _dot4_unrolled(
        g_th[0],
        g_th[1],
        g_th[2],
        g_th[3],
        corr_factor[0],
        corr_factor[1],
        corr_factor[2],
        corr_factor[3],
    )
    vm_corr = _dot4_unrolled(
        g_vm[0],
        g_vm[1],
        g_vm[2],
        g_vm[3],
        corr_factor[0],
        corr_factor[1],
        corr_factor[2],
        corr_factor[3],
    )

    dtheta = base_theta_dx + theta_corr * theta_mask_d
    dvm = base_vm_dx + vm_corr * vm_mask_d

    theta_post = base_theta0 + dtheta
    vm_post = base_vm0 + dvm

    return theta_post, vm_post


def _solve_outage_voltages(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    outage_branch_idx: Int[jnp.ndarray, " n_outages"],
    jacobian_components: JacobianComponentInputs,
    network_topology: NetworkTopologyInputs,
    voltage_state: VoltageStateInputs,
    network_admittance: NetworkAdmittanceInputs,
    branch_pq_base: Float[jnp.ndarray, "n_branches 4"],
    lower_residual_uses_reactive_mismatch: Bool[jnp.ndarray, " n_buses"],
    base_theta0: Float[jnp.ndarray, " n_mon_bus"],
    base_vm0: Float[jnp.ndarray, " n_mon_bus"],
    theta_rows: Int[jnp.ndarray, " n_mon_bus"],
    vm_rows: Int[jnp.ndarray, " n_mon_bus"],
    theta_mask: Float[jnp.ndarray, " n_mon_bus"],
    vm_mask: Float[jnp.ndarray, " n_mon_bus"],
) -> Tuple[
    Float[jnp.ndarray, "n_outages n_mon_bus"],
    Float[jnp.ndarray, "n_outages n_mon_bus"],
]:
    """Vectorized post-contingency solve for monitored bus states."""
    dtype = jacobian_inv_transposed.dtype

    def _solve_single(out_idx: jnp.ndarray) -> Tuple[Float[jnp.ndarray, " n_mon_bus"], Float[jnp.ndarray, " n_mon_bus"]]:
        mismatch_vec = -jnp.take(branch_pq_base, out_idx, axis=0).astype(dtype)
        from_bus = jnp.take(network_topology.branch_from, out_idx, axis=0)
        to_bus = jnp.take(network_topology.branch_to, out_idx, axis=0)
        mismatch_vec = mismatch_vec.at[2].set(
            jnp.where(jnp.take(lower_residual_uses_reactive_mismatch, from_bus, axis=0), mismatch_vec[2], 0.0)
        )
        mismatch_vec = mismatch_vec.at[3].set(
            jnp.where(jnp.take(lower_residual_uses_reactive_mismatch, to_bus, axis=0), mismatch_vec[3], 0.0)
        )
        return _compute_post_contingency_states(
            jacobian_inv_transposed=jacobian_inv_transposed,
            outage_idx=out_idx,
            mismatch_vec=mismatch_vec,
            network_topology=network_topology,
            voltage_state=voltage_state,
            jacobian_components=jacobian_components,
            network_admittance=network_admittance,
            base_theta0=base_theta0,
            base_vm0=base_vm0,
            theta_rows=theta_rows,
            vm_rows=vm_rows,
            theta_mask=theta_mask,
            vm_mask=vm_mask,
        )

    return jax.vmap(_solve_single)(outage_branch_idx)


def line_outage_post_contingency_voltages(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    outage_branch_idx: Int[jnp.ndarray, " n_outages"],
    jacobian_components: JacobianComponentInputs,
    network_topology: NetworkTopologyInputs,
    voltage_state: VoltageStateInputs,
    network_admittance: NetworkAdmittanceInputs,
    monitor_bus_indices: Int[jnp.ndarray, " n_mon_bus"],
    branch_pq_base: Float[jnp.ndarray, "n_branches 4"],
    lower_residual_uses_reactive_mismatch: Bool[jnp.ndarray, " n_buses"],
) -> Tuple[
    Float[jnp.ndarray, "n_outages n_mon_bus"],
    Float[jnp.ndarray, "n_outages n_mon_bus"],
]:
    """Compute post-contingency monitored bus voltages (θ, Vm) only."""
    theta_rows, vm_rows, theta_mask, vm_mask = build_monitor_rows(
        jacobian_components=jacobian_components,
        monitor_bus_indices=monitor_bus_indices,
    )

    base_theta0 = jnp.take(voltage_state.bus_voltage_angles_rad, monitor_bus_indices, axis=0)
    base_vm0 = jnp.take(voltage_state.bus_voltage_magnitudes, monitor_bus_indices, axis=0)

    return _solve_outage_voltages(
        jacobian_inv_transposed=jacobian_inv_transposed,
        outage_branch_idx=outage_branch_idx,
        jacobian_components=jacobian_components,
        network_topology=network_topology,
        voltage_state=voltage_state,
        network_admittance=network_admittance,
        branch_pq_base=branch_pq_base,
        lower_residual_uses_reactive_mismatch=lower_residual_uses_reactive_mismatch,
        base_theta0=base_theta0,
        base_vm0=base_vm0,
        theta_rows=theta_rows,
        vm_rows=vm_rows,
        theta_mask=theta_mask,
        vm_mask=vm_mask,
    )
