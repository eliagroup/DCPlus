# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions for low-rank updates of the power flow Jacobian."""

from typing import Tuple

import numpy as np
from jaxtyping import Complex128, Float, Int

# ruff: noqa: PLR0913


def _branch_state_indices(
    branch_idx: int,
    branch_from: Int[np.ndarray, " n_branches"],
    branch_to: Int[np.ndarray, " n_branches"],
    angle_component_indices: Int[np.ndarray, " n_eq_jacobian"],
    magnitude_component_indices: Int[np.ndarray, " n_eq_jacobian"],
) -> tuple[Int[np.ndarray, "4"], np.ndarray]:
    """Return mapping from branch end states to Jacobian indices.

    Parameters
    ----------
    branch_idx : int
        Index of the branch.
    branch_from : Int[np.ndarray, " n_branches"]
        From bus indices of branches.
    branch_to : Int[np.ndarray, " n_branches"]
        To bus indices of branches.
    angle_component_indices : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from node index to theta equation index in Jacobian.
        Entries are -1 for nodes without theta equation.
    magnitude_component_indices : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from node index to voltage magnitude equation index in Jacobian.
        Entries are -1 for nodes without voltage magnitude equation.

    Returns
    -------
    safe_idx : Int[np.ndarray, "4"]
        Array with the indices in the Jacobian for [theta_from, theta_to, u_from, u_to].
        Invalid entries are set to zero so the array can be consumed safely.
    valid_mask : np.ndarray
        Boolean mask indicating which positions in safe_idx map to valid Jacobian rows/columns.
    """
    idx_theta_f = angle_component_indices[branch_from[branch_idx]]
    idx_theta_t = angle_component_indices[branch_to[branch_idx]]
    idx_u_f = magnitude_component_indices[branch_from[branch_idx]]
    idx_u_t = magnitude_component_indices[branch_to[branch_idx]]

    idx_arr = np.array([idx_theta_f, idx_theta_t, idx_u_f, idx_u_t], dtype=np.int32)
    valid_mask = idx_arr >= 0
    safe_idx = np.where(valid_mask, idx_arr, 0)
    return safe_idx, valid_mask


def _extract_rows_from_matrix(
    matrix: Float[np.ndarray, " n_eq n_eq"], row_indices: Int[np.ndarray, " n_rows"]
) -> Float[np.ndarray, " n_rows n_eq"]:
    """Extract selected rows as a contiguous ndarray.

    n_eq: number of equations in the Jacobian matrix
    n_rows: number of rows to extract

    Parameters
    ----------
    matrix : Float[np.ndarray, " n_eq n_eq"]
        The input matrix from which to extract rows.
    row_indices : Int[np.ndarray, " n_rows"]
        The indices of the rows to extract.

    Returns
    -------
    subset : Float[np.ndarray, " n_rows n_eq"]
        The extracted rows as a contiguous ndarray.
    """
    subset = matrix[row_indices, :]
    return subset


def _extract_columns_from_matrix(
    matrix: Float[np.ndarray, " n_eq n_eq"], col_indices: Int[np.ndarray, " n_cols"]
) -> Float[np.ndarray, " n_eq n_cols"]:
    """Extract selected columns as a contiguous ndarray.

    n_eq: number of equations in the Jacobian matrix
    n_cols: number of columns to extract

    Parameters
    ----------
    matrix : Float[np.ndarray, " n_eq n_eq"]
        The input matrix from which to extract columns.
    col_indices : Int[np.ndarray, " n_cols"]
        The indices of the columns to extract.

    Returns
    -------
    subset : Float[np.ndarray, " n_eq n_cols"]
        The extracted columns as a contiguous ndarray.
    """
    subset = matrix[:, col_indices]
    return subset


def _extract_submatrix(
    matrix: Float[np.ndarray, " n_eq n_eq"],
    row_indices: Int[np.ndarray, " n_rows"],
    col_indices: Int[np.ndarray, " n_cols"],
) -> Float[np.ndarray, " n_rows n_cols"]:
    """Extract a dense sub matrix for the supplied indices.

    n_eq: number of equations in the Jacobian matrix
    n_rows: number of rows to extract
    n_cols: number of columns to extract

    Parameters
    ----------
    matrix : Float[np.ndarray, " n_eq n_eq"]
        The input matrix from which to extract the sub matrix.
    row_indices : Int[np.ndarray, " n_rows"]
        The indices of the rows to extract.
    col_indices : Int[np.ndarray, " n_cols"]
        The indices of the columns to extract.

    Returns
    -------
    sub_matrix : Float[np.ndarray, " n_rows n_cols"]
        The extracted submatrix as a contiguous ndarray.
    """
    return matrix[np.ix_(row_indices, col_indices)]


def _compute_branch_delta_submatrix_from_admittance(
    v_mag_from: Float[np.ndarray, ""],
    v_mag_to: Float[np.ndarray, ""],
    theta_from: Float[np.ndarray, ""],
    theta_to: Float[np.ndarray, ""],
    y_ff: Complex128[np.ndarray, ""],
    y_ft: Complex128[np.ndarray, ""],
    y_tf: Complex128[np.ndarray, ""],
    y_tt: Complex128[np.ndarray, ""],
) -> Float[np.ndarray, "4 4"]:
    """Return the 4x4 Jacobian contribution for a branch using admittance terms.

    Parameters
    ----------
    v_mag_from : Float
        Voltage magnitude in p.u. at the "from" bus.
    v_mag_to : Float
        Voltage magnitude in p.u. at the "to" bus.
    theta_from : Float
        Voltage angle (rad) at the "from" bus.
    theta_to : Float
        Voltage angle (rad) at the "to" bus.
    y_ff : Complex128
        Self admittance at the "from" bus.
    y_ft : Complex128
        Mutual admittance from "from" to "to" bus.
    y_tf : Complex128
        Mutual admittance from "to" to "from" bus.
    y_tt : Complex128
        Self admittance at the "to" bus.

    Returns
    -------
    delta : Float[np.ndarray, "4 4"]
        The 4x4 branch Jacobian submatrix contribution.
    """
    theta_diff_ft = theta_from - theta_to
    cos_ft = np.cos(theta_diff_ft)
    sin_ft = np.sin(theta_diff_ft)

    cos_tf = cos_ft
    sin_tf = -sin_ft

    v_f = v_mag_from
    v_t = v_mag_to

    g_ff = np.real(y_ff)
    b_ff = np.imag(y_ff)
    g_ft = np.real(y_ft)
    b_ft = np.imag(y_ft)
    g_tf = np.real(y_tf)
    b_tf = np.imag(y_tf)
    g_tt = np.real(y_tt)
    b_tt = np.imag(y_tt)

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

    dtype = np.result_type(v_mag_from, v_mag_to, theta_from, theta_to, y_ff)

    delta = np.array(
        [
            [dpf_dthf, dpf_dtht, dpf_dvf, dpf_dvt],
            [dpt_dthf, dpt_dtht, dpt_dvf, dpt_dvt],
            [dqf_dthf, dqf_dtht, dqf_dvf, dqf_dvt],
            [dqt_dthf, dqt_dtht, dqt_dvf, dqt_dvt],
        ],
        dtype=dtype,
    )
    # add the sign subtracting the Jacobian contribution
    delta = delta * -1
    return delta


def _prepare_low_rank_factors_from_admittance(
    branch_idx: Int[np.ndarray, ""],
    branch_from: Int[np.ndarray, " n_branches"],
    branch_to: Int[np.ndarray, " n_branches"],
    v_mag_hat: Float[np.ndarray, " n_buses"],
    theta_hat: Float[np.ndarray, " n_buses"],
    y_ff: Complex128[np.ndarray, " n_branches"],
    y_ft: Complex128[np.ndarray, " n_branches"],
    y_tf: Complex128[np.ndarray, " n_branches"],
    y_tt: Complex128[np.ndarray, " n_branches"],
    angle_component_indices: Int[np.ndarray, " n_eq_jacobian"],
    magnitude_component_indices: Int[np.ndarray, " n_eq_jacobian"],
) -> Tuple[Float[np.ndarray, "4 4"], Int[np.ndarray, "4"]]:
    """Build low-rank factors for a line outage from pi-model admittances.

    Parameters
    ----------
    branch_idx : Int[np.ndarray, ""]
        Index of the branch.
    branch_from : Int[np.ndarray, " n_branches"]
        From bus indices of branches.
    branch_to : Int[np.ndarray, " n_branches"]
        To bus indices of branches.
    v_mag_hat : Float[np.ndarray, " n_buses"]
        Hot-start voltage magnitudes |V| in p.u. at all buses.
    theta_hat : Float[np.ndarray, " n_buses"]
        Hot-start voltage angles (rad) at all buses.
    y_ff : Complex128[np.ndarray, " n_branches"]
        Self admittance at the "from" bus.
    y_ft : Complex128[np.ndarray, " n_branches"]
        Mutual admittance from "from" to "to" bus.
    y_tf : Complex128[np.ndarray, " n_branches"]
        Mutual admittance from "to" to "from" bus.
    y_tt : Complex128[np.ndarray, " n_branches"]
        Self admittance at the "to" bus.
    angle_component_indices : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from node index to theta equation index in Jacobian.
        Entries are -1 for nodes without theta equation.
    magnitude_component_indices : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from node index to voltage magnitude equation index in Jacobian.
        Entries are -1 for nodes without voltage magnitude equation.

    Returns
    -------
    d_mat : Float[np.ndarray, "4 4"]
        Dense branch delta restricted to valid Jacobian rows/columns.
    safe_idx : Int[np.ndarray, "4"]
        Array with the indices in the Jacobian for [theta_from, theta_to, u_from, u_to].
        Invalid entries are set to zero, matching the mask used to build d_mat.
    """
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

    mask_numeric = valid_mask.astype(delta_full.dtype)
    d_mat = delta_full * mask_numeric[:, None] * mask_numeric[None, :]
    return d_mat, safe_idx.astype(np.int32)
