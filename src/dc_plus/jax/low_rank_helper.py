# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Low rank helper functions implemented in JAX.

These helpers mirror the NumPy reference implementation but operate on
``jax.Array`` objects so that the downstream LODF routines can be compiled for
accelerators. Shapes are kept static (max four states per branch) to keep JIT
compilations fast and avoid unnecessary device buffer materialisation.
"""

from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Complex128, Float, Int

# ruff: noqa: PLR0913


def _branch_state_indices(
    branch_idx: Int[jnp.ndarray, ""],
    branch_from: Int[jnp.ndarray, " n_branches"],
    branch_to: Int[jnp.ndarray, " n_branches"],
    angle_component_indices: Int[jnp.ndarray, " n_nodes"],
    magnitude_component_indices: Int[jnp.ndarray, " n_nodes"],
) -> tuple[Int[jnp.ndarray, "4"], jnp.ndarray]:
    """Return mapping from branch end states to Jacobian indices.

    The output retains a static length of four entries (theta_from, theta_to,
    u_from, u_to) and a boolean mask indicating which of these states map into
    the system of equations. Invalid entries have their mask bit cleared.
    """
    idx_theta_f = angle_component_indices[branch_from[branch_idx]]
    idx_theta_t = angle_component_indices[branch_to[branch_idx]]
    idx_u_f = magnitude_component_indices[branch_from[branch_idx]]
    idx_u_t = magnitude_component_indices[branch_to[branch_idx]]

    idx_arr = jnp.stack([idx_theta_f, idx_theta_t, idx_u_f, idx_u_t], dtype=jnp.int32)
    valid_mask = idx_arr >= 0
    safe_idx = jnp.where(valid_mask, idx_arr, 0)
    return safe_idx, valid_mask


def _extract_rows_from_matrix(
    matrix: Float[jnp.ndarray, " n_eq n_eq"],
    row_indices: Int[jnp.ndarray, " n_rows"],
) -> Float[jnp.ndarray, " n_rows n_eq"]:
    """Return the selected rows using a device-side gather."""
    return jnp.take(matrix, row_indices, axis=0)


def _extract_columns_from_matrix(
    matrix: Float[jnp.ndarray, " n_eq n_eq"],
    col_indices: Int[jnp.ndarray, " n_cols"],
) -> Float[jnp.ndarray, " n_eq n_cols"]:
    """Return the selected columns using a device-side gather."""
    return jnp.take(matrix, col_indices, axis=1)


def _extract_submatrix(
    matrix: Float[jnp.ndarray, " n_eq n_eq"],
    row_indices: Int[jnp.ndarray, " n_rows"],
    col_indices: Int[jnp.ndarray, " n_cols"],
) -> Float[jnp.ndarray, " n_rows n_cols"]:
    """Extract the dense submatrix addressed by ``row_indices`` and ``col_indices``."""
    # Extract rows first to keep the gather stride contiguous, then columns.
    sub_rows = jnp.take(matrix, row_indices, axis=0)
    return jnp.take(sub_rows, col_indices, axis=1)


def _compute_branch_delta_submatrix_from_admittance(
    v_mag_from: Float[jnp.ndarray, ""],
    v_mag_to: Float[jnp.ndarray, ""],
    theta_from: Float[jnp.ndarray, ""],
    theta_to: Float[jnp.ndarray, ""],
    y_ff: Complex128[jnp.ndarray, ""],
    y_ft: Complex128[jnp.ndarray, ""],
    y_tf: Complex128[jnp.ndarray, ""],
    y_tt: Complex128[jnp.ndarray, ""],
) -> Float[jnp.ndarray, "4 4"]:
    """Return the 4x4 Jacobian contribution for a branch using admittance terms.

    Notes
    -----
    The negative sign is applied so this delta can be used directly as a Jacobian
    modification in the low-rank update.
    """
    theta_diff_ft = theta_from - theta_to
    cos_ft = jnp.cos(theta_diff_ft)
    sin_ft = jnp.sin(theta_diff_ft)

    cos_tf = cos_ft
    sin_tf = -sin_ft

    v_f = v_mag_from
    v_t = v_mag_to

    g_ff = jnp.real(y_ff)
    b_ff = jnp.imag(y_ff)
    g_ft = jnp.real(y_ft)
    b_ft = jnp.imag(y_ft)
    g_tf = jnp.real(y_tf)
    b_tf = jnp.imag(y_tf)
    g_tt = jnp.real(y_tt)
    b_tt = jnp.imag(y_tt)

    dpf_dthf = v_f * v_t * (-g_ft * sin_ft + b_ft * cos_ft)
    dpf_dtht = -dpf_dthf
    dpf_dvf = 2.0 * v_f * g_ff + v_t * (g_ft * cos_ft + b_ft * sin_ft)
    dpf_dvt = v_f * (g_ft * cos_ft + b_ft * sin_ft)

    dqf_dthf = v_f * v_t * (g_ft * cos_ft + b_ft * sin_ft)
    dqf_dtht = -dqf_dthf
    dqf_dvf = -2.0 * v_f * b_ff + v_t * (g_ft * sin_ft - b_ft * cos_ft)
    dqf_dvt = v_f * (g_ft * sin_ft - b_ft * cos_ft)

    dpt_dtht = v_t * v_f * (-g_tf * sin_tf + b_tf * cos_tf)
    dpt_dthf = -dpt_dtht
    dpt_dvt = 2.0 * v_t * g_tt + v_f * (g_tf * cos_tf + b_tf * sin_tf)
    dpt_dvf = v_t * (g_tf * cos_tf + b_tf * sin_tf)

    dqt_dtht = v_t * v_f * (g_tf * cos_tf + b_tf * sin_tf)
    dqt_dthf = -dqt_dtht
    dqt_dvt = -2.0 * v_t * b_tt + v_f * (g_tf * sin_tf - b_tf * cos_tf)
    dqt_dvf = v_t * (g_tf * sin_tf - b_tf * cos_tf)

    dtype = jnp.result_type(v_mag_from, v_mag_to, theta_from, theta_to, y_ff)

    delta = jnp.array(
        [
            [-dpf_dthf, -dpf_dtht, -dpf_dvf, -dpf_dvt],
            [-dpt_dthf, -dpt_dtht, -dpt_dvf, -dpt_dvt],
            [-dqf_dthf, -dqf_dtht, -dqf_dvf, -dqf_dvt],
            [-dqt_dthf, -dqt_dtht, -dqt_dvf, -dqt_dvt],
        ],
        dtype=dtype,
    )
    return delta


def _prepare_low_rank_factors_from_admittance(
    branch_idx: Int[jnp.ndarray, ""],
    branch_from: Int[jnp.ndarray, " n_branches"],
    branch_to: Int[jnp.ndarray, " n_branches"],
    v_mag_hat: Float[jnp.ndarray, " n_buses"],
    theta_hat: Float[jnp.ndarray, " n_buses"],
    y_ff: Complex128[jnp.ndarray, " n_branches"],
    y_ft: Complex128[jnp.ndarray, " n_branches"],
    y_tf: Complex128[jnp.ndarray, " n_branches"],
    y_tt: Complex128[jnp.ndarray, " n_branches"],
    angle_component_indices: Int[jnp.ndarray, " n_buses"],
    magnitude_component_indices: Int[jnp.ndarray, " n_buses"],
) -> Tuple[Float[jnp.ndarray, "4 4"], Int[jnp.ndarray, "4"], jnp.ndarray]:
    """Build low-rank factors for a line outage from pi-model admittances."""
    safe_idx, valid_mask = _branch_state_indices(
        branch_idx=branch_idx,
        branch_from=branch_from,
        branch_to=branch_to,
        angle_component_indices=angle_component_indices,
        magnitude_component_indices=magnitude_component_indices,
    )

    f = branch_from[branch_idx]
    t = branch_to[branch_idx]
    v_from = v_mag_hat[f]
    v_to = v_mag_hat[t]
    theta_from = theta_hat[f]
    theta_to = theta_hat[t]

    delta_full = _compute_branch_delta_submatrix_from_admittance(
        v_mag_from=v_from,
        v_mag_to=v_to,
        theta_from=theta_from,
        theta_to=theta_to,
        y_ff=y_ff[branch_idx],
        y_ft=y_ft[branch_idx],
        y_tf=y_tf[branch_idx],
        y_tt=y_tt[branch_idx],
    )

    mask = valid_mask.astype(delta_full.dtype)
    d_mat = delta_full * mask[:, None] * mask[None, :]
    return d_mat, safe_idx.astype(jnp.int32), valid_mask
