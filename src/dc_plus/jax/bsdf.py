# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Bus-splitting distribution factor (BSDF) kernels.

This module provides two layers:

1. **Rank-4 collectors** that express a BSDF (bus split with branch
   re-attachments) as a padded batch of rank-4 Jacobian updates and mismatch
   contributions.

2. A legacy dense-inverse updater (kept for backwards-compatibility) that
   materializes the updated Jacobian inverse transpose.


Assumptions
-----------
- The split is a *pure re-attachment*: branch parameters (series admittance,
  taps, phase shifts, shunts, etc.) do not change.
- The placeholder new bus voltage is initialized to the split bus voltage
  (as done in your profiling harness), so the initial mismatch is dominated by
  moving branch injection contributions from the original bus to the new bus.

"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Complex128, Float, Int

from .low_rank_helper import _compute_branch_delta_submatrix_from_admittance

# ruff: noqa: ARG001, PLR0913, PLR0915


def _full_rank_lodf(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    delta_block: Float[jnp.ndarray, " k k"],
    idx_list: Int[jnp.ndarray, " k"],
    idx_mask: jnp.ndarray,
) -> Float[jnp.ndarray, " n_eq n_eq"]:
    """Legacy full-rank update helper."""
    dtype = jacobian_inv_transposed.dtype
    safe_indices = jnp.where(idx_mask, idx_list, 0)
    rows_transposed = jnp.take(jacobian_inv_transposed, safe_indices, axis=1)
    cols_transposed = jnp.take(jacobian_inv_transposed, safe_indices, axis=0)
    mask_vec = idx_mask.astype(dtype)
    rows_masked_transposed = rows_transposed * mask_vec[None, :]
    cols_masked_transposed = cols_transposed * mask_vec[:, None]
    sub_matrix_transposed = jnp.take(rows_masked_transposed, safe_indices, axis=0)
    sub_matrix_transposed = sub_matrix_transposed * mask_vec[None, :]
    delta_masked = delta_block * mask_vec[:, None] * mask_vec[None, :]
    eye = jnp.eye(delta_block.shape[0], dtype=dtype)
    k_mat = eye + jnp.einsum("ij,lj->il", delta_masked, sub_matrix_transposed)
    rhs = jnp.einsum("ij,mj->im", delta_masked, rows_masked_transposed)
    corr = jnp.linalg.solve(k_mat, rhs)
    lodf_transposed = jnp.einsum("jm,ji->mi", corr, cols_masked_transposed)
    return lodf_transposed


@jax.jit
def _compute_bsdf_update_impl(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    bus_to_split: int,
    new_bus_b_index: int,
    branches_connected_to_bus_b: Int[jnp.ndarray, " n_branches_B"],
    branch_from: Int[jnp.ndarray, " n_branches"],
    branch_to: Int[jnp.ndarray, " n_branches"],
    v_mag_hat: Float[jnp.ndarray, " n_buses"],
    theta_hat: Float[jnp.ndarray, " n_buses"],
    y_ff: Complex128[jnp.ndarray, " n_branches"],
    y_ft: Complex128[jnp.ndarray, " n_branches"],
    y_tf: Complex128[jnp.ndarray, " n_branches"],
    y_tt: Complex128[jnp.ndarray, " n_branches"],
    angle_component_indices: Int[jnp.ndarray, " n_eq_jacobian"],
    magnitude_component_indices: Int[jnp.ndarray, " n_eq_jacobian"],
) -> Float[jnp.ndarray, " n_eq n_eq"]:
    """Legacy implementation: materialize updated inverse transpose."""
    dtype = jacobian_inv_transposed.dtype
    n_eq = jacobian_inv_transposed.shape[0]

    branch_from_old = jnp.take(branch_from, branches_connected_to_bus_b, axis=0)
    branch_to_old = jnp.take(branch_to, branches_connected_to_bus_b, axis=0)

    branch_from_new = jnp.where(branch_from_old == bus_to_split, new_bus_b_index, branch_from_old)
    branch_to_new = jnp.where(branch_to_old == bus_to_split, new_bus_b_index, branch_to_old)

    base_bus_indices = jnp.array([bus_to_split, new_bus_b_index], dtype=jnp.int32)
    bus_candidates = jnp.concatenate(
        [
            base_bus_indices,
            branch_from_old,
            branch_to_old,
            branch_from_new,
            branch_to_new,
        ],
        axis=0,
    )

    theta_candidates = jnp.take(angle_component_indices, bus_candidates, axis=0)
    mag_candidates = jnp.take(magnitude_component_indices, bus_candidates, axis=0)

    comp_candidates = jnp.concatenate([theta_candidates, mag_candidates], axis=0)
    k_max = min(int(comp_candidates.shape[0]), n_eq)

    safe_theta = jnp.where(theta_candidates >= 0, theta_candidates, 0)
    safe_mag = jnp.where(mag_candidates >= 0, mag_candidates, 0)
    theta_valid = (theta_candidates >= 0).astype(jnp.int32)
    mag_valid = (mag_candidates >= 0).astype(jnp.int32)

    mask_counts = jnp.zeros((n_eq,), dtype=jnp.int32)
    mask_counts = mask_counts.at[safe_theta].add(theta_valid)
    mask_counts = mask_counts.at[safe_mag].add(mag_valid)

    top_vals, idx_list = jax.lax.top_k(mask_counts, k_max)
    idx_mask = top_vals > 0

    positions = -jnp.ones((n_eq,), dtype=jnp.int32)
    position_updates = jnp.arange(k_max, dtype=jnp.int32)
    positions = positions.at[idx_list].set(jnp.where(idx_mask, position_updates, -1))

    def _gather_positions(indices: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        valid = indices >= 0
        safe_indices = jnp.where(valid, indices, 0)
        pos = jnp.take(positions, safe_indices, axis=0)
        pos = jnp.where(valid, pos, -1)
        return pos, valid

    theta_from_old_idx = jnp.take(angle_component_indices, branch_from_old, axis=0)
    theta_to_old_idx = jnp.take(angle_component_indices, branch_to_old, axis=0)
    mag_from_old_idx = jnp.take(magnitude_component_indices, branch_from_old, axis=0)
    mag_to_old_idx = jnp.take(magnitude_component_indices, branch_to_old, axis=0)

    theta_from_new_idx = jnp.take(angle_component_indices, branch_from_new, axis=0)
    theta_to_new_idx = jnp.take(angle_component_indices, branch_to_new, axis=0)
    mag_from_new_idx = jnp.take(magnitude_component_indices, branch_from_new, axis=0)
    mag_to_new_idx = jnp.take(magnitude_component_indices, branch_to_new, axis=0)

    old_indices = jnp.stack(
        [
            theta_from_old_idx,
            theta_to_old_idx,
            mag_from_old_idx,
            mag_to_old_idx,
        ],
        axis=1,
    )

    new_indices = jnp.stack(
        [
            theta_from_new_idx,
            theta_to_new_idx,
            mag_from_new_idx,
            mag_to_new_idx,
        ],
        axis=1,
    )

    old_pos, old_valid = _gather_positions(old_indices)
    new_pos, new_valid = _gather_positions(new_indices)

    vm_from_old = jnp.take(v_mag_hat, branch_from_old, axis=0)
    vm_to_old = jnp.take(v_mag_hat, branch_to_old, axis=0)
    th_from_old = jnp.take(theta_hat, branch_from_old, axis=0)
    th_to_old = jnp.take(theta_hat, branch_to_old, axis=0)

    vm_from_new = jnp.take(v_mag_hat, branch_from_new, axis=0)
    vm_to_new = jnp.take(v_mag_hat, branch_to_new, axis=0)
    th_from_new = jnp.take(theta_hat, branch_from_new, axis=0)
    th_to_new = jnp.take(theta_hat, branch_to_new, axis=0)

    y_ff_sel = jnp.take(y_ff, branches_connected_to_bus_b, axis=0)
    y_ft_sel = jnp.take(y_ft, branches_connected_to_bus_b, axis=0)
    y_tf_sel = jnp.take(y_tf, branches_connected_to_bus_b, axis=0)
    y_tt_sel = jnp.take(y_tt, branches_connected_to_bus_b, axis=0)

    compute_delta = jax.vmap(_compute_branch_delta_submatrix_from_admittance)
    delta_old = compute_delta(
        vm_from_old,
        vm_to_old,
        th_from_old,
        th_to_old,
        y_ff_sel,
        y_ft_sel,
        y_tf_sel,
        y_tt_sel,
    )
    delta_new = compute_delta(
        vm_from_new,
        vm_to_new,
        th_from_new,
        th_to_new,
        y_ff_sel,
        y_ft_sel,
        y_tf_sel,
        y_tt_sel,
    )

    k_shape = (k_max, k_max)
    delta_block = jnp.zeros(k_shape, dtype=dtype)

    def _accumulate(
        delta: jnp.ndarray, pos: jnp.ndarray, valid: jnp.ndarray, weight: float, target: jnp.ndarray
    ) -> jnp.ndarray:
        safe_pos = jnp.where(valid, pos, 0)
        pair_mask = valid[:, :, None] & valid[:, None, :]
        updates = delta * pair_mask.astype(delta.dtype) * weight
        row_idx = jnp.broadcast_to(safe_pos[:, :, None], pair_mask.shape)
        col_idx = jnp.broadcast_to(safe_pos[:, None, :], pair_mask.shape)
        flat_rows = row_idx.reshape(-1)
        flat_cols = col_idx.reshape(-1)
        flat_updates = updates.reshape(-1)
        flat_mask = pair_mask.reshape(-1)
        safe_rows = jnp.where(flat_mask, flat_rows, 0)
        safe_cols = jnp.where(flat_mask, flat_cols, 0)
        safe_vals = jnp.where(flat_mask, flat_updates, 0.0)
        return target.at[safe_rows, safe_cols].add(safe_vals)

    delta_block = _accumulate(delta_old, old_pos, old_valid, 1.0, delta_block)
    delta_block = _accumulate(delta_new, new_pos, new_valid, -1.0, delta_block)

    theta_new_idx = angle_component_indices[new_bus_b_index]
    mag_new_idx = magnitude_component_indices[new_bus_b_index]

    theta_new_pos = jnp.take(positions, jnp.where(theta_new_idx >= 0, theta_new_idx, 0), axis=0)
    mag_new_pos = jnp.take(positions, jnp.where(mag_new_idx >= 0, mag_new_idx, 0), axis=0)

    theta_mask = (theta_new_idx >= 0) & (theta_new_pos >= 0)
    mag_mask = (mag_new_idx >= 0) & (mag_new_pos >= 0)

    theta_row = jnp.where(theta_mask, theta_new_pos, 0)
    mag_row = jnp.where(mag_mask, mag_new_pos, 0)

    minus_one = jnp.asarray(-1.0, dtype=dtype)
    zero_val = jnp.asarray(0.0, dtype=dtype)
    theta_update = jnp.where(theta_mask, minus_one, zero_val)
    mag_update = jnp.where(mag_mask, minus_one, zero_val)
    delta_block = delta_block.at[theta_row, theta_row].add(theta_update)
    delta_block = delta_block.at[mag_row, mag_row].add(mag_update)

    lodf_matrix_transposed = _full_rank_lodf(
        jacobian_inv_transposed=jacobian_inv_transposed,
        delta_block=delta_block,
        idx_list=idx_list,
        idx_mask=idx_mask,
    )

    updated_inverse_transposed = jacobian_inv_transposed - lodf_matrix_transposed
    return updated_inverse_transposed


def compute_bsdf_update(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    bus_to_split: int,
    new_bus_b_index: int,
    new_bus_type: int,
    branches_connected_to_bus_b: Int[jnp.ndarray, " n_branches_B"],
    shunt_connected_to_bus_b: Int[jnp.ndarray, " n_shunts_B"],
    branch_from: Int[jnp.ndarray, " n_branches"],
    branch_to: Int[jnp.ndarray, " n_branches"],
    shunt_to_bus: Int[jnp.ndarray, " n_shunts"],
    v_mag_hat: Float[jnp.ndarray, " n_buses"],
    theta_hat: Float[jnp.ndarray, " n_buses"],
    y_ff: Complex128[jnp.ndarray, " n_branches"],
    y_ft: Complex128[jnp.ndarray, " n_branches"],
    y_tf: Complex128[jnp.ndarray, " n_branches"],
    y_tt: Complex128[jnp.ndarray, " n_branches"],
    y_shunt: Complex128[jnp.ndarray, " n_buses"],
    angle_component_indices: Int[jnp.ndarray, " n_eq_jacobian"],
    magnitude_component_indices: Int[jnp.ndarray, " n_eq_jacobian"],
) -> Float[jnp.ndarray, " n_eq n_eq"]:
    """Legacy dense inverse transpose update for BSDF.

    This function computes the updated Jacobian inverse transpose after a bus split
    with branch re-attachments, using a full-rank update approach.
    It is intended for reference and testing purposes, and is not optimized for performance.

    Note: currently not supported:
        - Shunt reassignments
        - Changes in branch parameters (e.g., series admittance, taps, phase shifts)
          (This would involve computing the different delta_blocks)
        - Changes in the type of the new bus (e.g., PQ, PV, slack)
        - injection reassignments

    Parameters
    ----------
    jacobian_inv_transposed : Float[jnp.ndarray, " n_eq n_eq"]
        Current Jacobian inverse transpose, shape (n_eq, n_eq).
    bus_to_split : int
        Index of the bus being split.
    new_bus_b_index : int
        Index of the new bus being created.
    new_bus_type : int
        Type of the new bus (e.g., PQ, PV, slack).
    branches_connected_to_bus_b : Int[jnp.ndarray, " n_branches_B"]
        Indices of branches connected to the bus being split.
    shunt_connected_to_bus_b : Int[jnp.ndarray, " n_shunts_B"]
        Indices of shunts connected to the bus being split.
    branch_from : Int[jnp.ndarray, " n_branches"]
        "From" bus indices for all branches.
    branch_to : Int[jnp.ndarray, " n_branches"]
        "To" bus indices for all branches.
    shunt_to_bus : Int[jnp.ndarray, " n_shunts"]
        Bus indices for all shunts.
    v_mag_hat : Float[jnp.ndarray, " n_buses"]
        Voltage magnitude estimates for all buses.
    theta_hat : Float[jnp.ndarray, " n_buses"]
        Voltage angle estimates for all buses.
    y_ff : Complex128[jnp.ndarray, " n_branches"]
        "From-From" admittance for all branches.
    y_ft : Complex128[jnp.ndarray, " n_branches"]
        "From-To" admittance for all branches.
    y_tf : Complex128[jnp.ndarray, " n_branches"]
        "To-From" admittance for all branches.
    y_tt : Complex128[jnp.ndarray, " n_branches"]
        "To-To" admittance for all branches.
    y_shunt : Complex128[jnp.ndarray, " n_buses"]
        Shunt admittance for all buses.
    angle_component_indices : Int[jnp.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to angle component indices in the Jacobian.
    magnitude_component_indices : Int[jnp.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to magnitude component indices in the Jacobian.

    Returns
    -------
    Float[jnp.ndarray, " n_eq n_eq"]
        Updated Jacobian inverse transpose after the bus split and branch re-attachments.
    """
    jacobian_arr_transposed = jnp.asarray(jacobian_inv_transposed)
    real_dtype = jacobian_arr_transposed.dtype

    return _compute_bsdf_update_impl(
        jacobian_inv_transposed=jacobian_arr_transposed,
        bus_to_split=int(bus_to_split),
        new_bus_b_index=int(new_bus_b_index),
        branches_connected_to_bus_b=jnp.asarray(branches_connected_to_bus_b, dtype=jnp.int32),
        branch_from=jnp.asarray(branch_from, dtype=jnp.int32),
        branch_to=jnp.asarray(branch_to, dtype=jnp.int32),
        v_mag_hat=jnp.asarray(v_mag_hat, dtype=real_dtype),
        theta_hat=jnp.asarray(theta_hat, dtype=real_dtype),
        y_ff=jnp.asarray(y_ff, dtype=jnp.complex128),
        y_ft=jnp.asarray(y_ft, dtype=jnp.complex128),
        y_tf=jnp.asarray(y_tf, dtype=jnp.complex128),
        y_tt=jnp.asarray(y_tt, dtype=jnp.complex128),
        angle_component_indices=jnp.asarray(angle_component_indices, dtype=jnp.int32),
        magnitude_component_indices=jnp.asarray(magnitude_component_indices, dtype=jnp.int32),
    )
