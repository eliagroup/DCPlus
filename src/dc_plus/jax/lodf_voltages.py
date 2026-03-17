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
from jaxtyping import Complex128, Float, Int

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
    angle_component_indices: Int[jnp.ndarray, " n_eq_jacobian"],
    magnitude_component_indices: Int[jnp.ndarray, " n_eq_jacobian"],
    monitor_bus_indices: Int[jnp.ndarray, " n_bus_mon"],
) -> Tuple[
    Int[jnp.ndarray, " n_bus_mon"],
    Int[jnp.ndarray, " n_bus_mon"],
    Float[jnp.ndarray, " n_bus_mon"],
    Float[jnp.ndarray, " n_bus_mon"],
]:
    """Precompute safe Jacobian indices and masks for monitored buses."""
    theta_idx = angle_component_indices[monitor_bus_indices]
    vm_idx = magnitude_component_indices[monitor_bus_indices]

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
        branch_from=branch_from,
        branch_to=branch_to,
        v_mag_hat=v_mag_hat,
        theta_hat=theta_hat,
        y_ff=y_ff,
        y_ft=y_ft,
        y_tf=y_tf,
        y_tt=y_tt,
        angle_component_indices=angle_component_indices,
        magnitude_component_indices=magnitude_component_indices,
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
    branch_pq_base: Float[jnp.ndarray, "n_branches 4"],
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
        return _compute_post_contingency_states(
            jacobian_inv_transposed=jacobian_inv_transposed,
            outage_idx=out_idx,
            mismatch_vec=mismatch_vec,
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
    branch_pq_base: Float[jnp.ndarray, "n_branches 4"],
) -> Tuple[
    Float[jnp.ndarray, "n_outages n_mon_bus"],
    Float[jnp.ndarray, "n_outages n_mon_bus"],
]:
    """Compute post-contingency monitored bus voltages (θ, Vm) only."""
    jacobian_inv_transposed = jnp.asarray(jacobian_inv_transposed)
    dtype = jacobian_inv_transposed.dtype

    outage_branch_idx = jnp.asarray(outage_branch_idx, dtype=jnp.int32)
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
    monitor_bus_indices = jnp.asarray(monitor_bus_indices, dtype=jnp.int32)
    branch_pq_base = jnp.asarray(branch_pq_base, dtype=dtype)

    theta_rows, vm_rows, theta_mask, vm_mask = build_monitor_rows(
        angle_component_indices=angle_component_indices,
        magnitude_component_indices=magnitude_component_indices,
        monitor_bus_indices=monitor_bus_indices,
    )

    base_theta0 = jnp.take(theta_hat, monitor_bus_indices, axis=0)
    base_vm0 = jnp.take(v_mag_hat, monitor_bus_indices, axis=0)

    return _solve_outage_voltages(
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
        branch_pq_base=branch_pq_base,
        base_theta0=base_theta0,
        base_vm0=base_vm0,
        theta_rows=theta_rows,
        vm_rows=vm_rows,
        theta_mask=theta_mask,
        vm_mask=vm_mask,
    )
