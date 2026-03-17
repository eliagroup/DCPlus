# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Low-rank updates to power flow Jacobian for line outages.

This module provides functions to compute and apply low-rank updates to the
inverse power flow Jacobian matrix in response to line outages. The updates
are derived from branch admittance parameters and allow recalculation
of system states without full matrix inversion.

This implementation is currently not optimized for speed and is intended
for reference and testing purposes only.
"""

import numpy as np
from jaxtyping import Complex128, Float, Int

from dc_plus.numpy.low_rank_helper import (
    _extract_columns_from_matrix,
    _extract_rows_from_matrix,
    _extract_submatrix,
    _prepare_low_rank_factors_from_admittance,
)

# ruff: noqa: PLR0913


def full_rank_delta_inv_jacobian(
    jacobian_inv: Float[np.ndarray, " n_eq n_eq"],
    jacobian_delta_submatrix: Float[np.ndarray, " k k"],
    idx_list: Int[np.ndarray, " k"],
) -> Float[np.ndarray, " n_eq n_eq"]:
    """Compute the full-rank delta_inv_jacobian matrix from low-rank factors.

    A full rank update to the inverse Jacobian is computed as:
        delta_J_inverse = C @ inv(I + D @ B) @ D @ R
    where C and R are the columns and rows of J_inverse corresponding to the branch states.

    Note: this is a reference implementation and is not optimized for performance.

    Parameters
    ----------
    jacobian_inv : Float[np.ndarray, " n_eq n_eq"]
        Current inverse Jacobian J_inverse, shape (n_eq, n_eq).
    jacobian_delta_submatrix : Float[np.ndarray, " k k"]
        Middle low-rank factor, shape (k, k).
        "D" in the Woodbury identity formulation.
    idx_list : Int[np.ndarray, " k"]
        Indices in the Jacobian corresponding to the branch states.

    Returns
    -------
    delta_inv_jacobian : Float[np.ndarray, " n_eq n_eq"]
        Full delta_inv_jacobian matrix, shape (n_eq, n_eq).
    """
    rows = _extract_rows_from_matrix(jacobian_inv, idx_list)
    sub = rows[:, idx_list]
    woodbury_matrix_k = np.eye(jacobian_delta_submatrix.shape[0], dtype=float) + (jacobian_delta_submatrix @ sub)
    rhs = np.linalg.solve(woodbury_matrix_k, jacobian_delta_submatrix @ rows)
    cols = _extract_columns_from_matrix(jacobian_inv, idx_list)
    delta_inv_jacobian = cols @ rhs
    return delta_inv_jacobian


def branch_outage_monitored_bus_dx(
    jacobian_inv: Float[np.ndarray, " n_eq n_eq"],
    j_at_mismatch: Float[np.ndarray, " n_eq"],
    branch_idx: int,
    branch_from: Int[np.ndarray, " n_branches"],
    branch_to: Int[np.ndarray, " n_branches"],
    v_mag_hat: Float[np.ndarray, " n_buses"],
    theta_hat: Float[np.ndarray, " n_buses"],
    angle_component_indices: Int[np.ndarray, " n_eq_jacobian"],
    magnitude_component_indices: Int[np.ndarray, " n_eq_jacobian"],
    monitor_bus: int,
    y_ff: Complex128[np.ndarray, " n_branches"],
    y_ft: Complex128[np.ndarray, " n_branches"],
    y_tf: Complex128[np.ndarray, " n_branches"],
    y_tt: Complex128[np.ndarray, " n_branches"],
) -> Float[np.ndarray, "2"]:
    """Return the updated state increments for a monitored bus.

    Parameters
    ----------
    jacobian_inv : Float[np.ndarray, " n_eq n_eq"]
        Current inverse Jacobian J_inverse, shape (neq, neq).
    j_at_mismatch : Float[np.ndarray, " n_eq"]
        Product of J_inverse and the mismatch vector at N-1. Named `j_at_mismatch`.
    branch_idx : int
        Index of the branch to outage.
    branch_from : Int[np.ndarray, " n_branches"]
        From-bus indices for all branches.
    branch_to : Int[np.ndarray, " n_branches"]
        To-bus indices for all branches.
    v_mag_hat : Float[np.ndarray, " n_buses"]
        Hot-start voltage magnitudes |V| in p.u. at all buses.
    theta_hat : Float[np.ndarray, " n_buses"]
        Hot-start voltage angles (rad) at all buses.
    angle_component_indices : Int[np.ndarray, " n_eq_jacobian"]
        Bus -> θ/P position map.
    magnitude_component_indices : Int[np.ndarray, " n_eq_jacobian"]
        Bus -> u/Q position map.
    monitor_bus : int
        The bus at which the state update is monitored.
    y_ff : Complex128[np.ndarray, " n_branches"]
        Self-admittance at the "from" bus for all branches.
    y_ft : Complex128[np.ndarray, " n_branches"]
        Mutual admittance from "from" to "to" bus for all branches.
    y_tf : Complex128[np.ndarray, " n_branches"]
        Mutual admittance from "to" to "from" bus for all branches.
    y_tt : Complex128[np.ndarray, " n_branches"]
        Self-admittance at the "to" bus for all branches.

    Returns
    -------
    updates : Float[np.ndarray, "2"]
        The updated state increments [delta_theta, delta_u] at the monitored bus.
    """
    theta_idx = angle_component_indices[monitor_bus]
    u_idx = magnitude_component_indices[monitor_bus]

    jacobian_delta_submatrix, branch_indices = _prepare_low_rank_factors_from_admittance(
        branch_idx=branch_idx,
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

    branch_sub = _extract_submatrix(jacobian_inv, branch_indices, branch_indices)
    woodbury_matrix_k = np.eye(jacobian_delta_submatrix.shape[0], dtype=jacobian_delta_submatrix.dtype) + (
        jacobian_delta_submatrix @ branch_sub
    )
    rhs = jacobian_delta_submatrix @ j_at_mismatch[branch_indices]
    corr_factor = np.linalg.solve(woodbury_matrix_k, rhs)

    state_indices = np.array([theta_idx, u_idx], dtype=int)
    mask = state_indices >= 0
    updates = np.zeros(2, dtype=jacobian_inv.dtype)

    monitor_indices = state_indices[mask]
    monitor_rows = _extract_rows_from_matrix(jacobian_inv, monitor_indices)
    base_update = -(j_at_mismatch[monitor_indices])
    monitor_cols = monitor_rows[:, branch_indices]
    corr_update = monitor_cols @ corr_factor
    updates[mask] = base_update + corr_update

    return updates


def branch_outage_update_inverse(
    jacobian_inv: Float[np.ndarray, " n_eq n_eq"],
    outage_branches_indices: Int[np.ndarray, " n_branches_outage"],  # could also be a Bool[np.ndarray, " n_branches"]
    branch_from: Float[np.ndarray, " n_branches"],
    branch_to: Float[np.ndarray, " n_branches"],
    v_mag_hat: Float[np.ndarray, " n_buses"],
    theta_hat: Float[np.ndarray, " n_buses"],
    y_ff: Complex128[np.complex128, " n_branches"],
    y_ft: Complex128[np.complex128, " n_branches"],
    y_tf: Complex128[np.complex128, " n_branches"],
    y_tt: Complex128[np.complex128, " n_branches"],
    angle_component_indices: Int[np.ndarray, " n_eq_jacobian"],
    magnitude_component_indices: Int[np.ndarray, " n_eq_jacobian"],
) -> Float[np.ndarray, " n_eq n_eq"]:
    """Apply multiple line outages as a full rank update to J_inverse.

    A full rank update to the inverse Jacobian is computed as:
        J_inverse_updated = J_inverse - sum_over_outages(delta_J_inverse)
    where each delta_J_inverse is computed from low-rank factors as:
        delta_J_inverse = C @ inv(I + D @ B) @ D @ R
    where C and R are the columns and rows of J_inverse corresponding to the branch states.

    Note: this is a reference implementation and is not optimized for performance.

    Parameters
    ----------
    jacobian_inv : Float[np.ndarray, " n_eq n_eq"]
        Current inverse Jacobian J_inverse, shape (n_eq, n_eq).
    outage_branches_indices : Int[np.ndarray, " n_branches_outage"]
        Indices of branches to outage.
    branch_from : Float[np.ndarray, " n_branches"]
        From-bus indices for all branches.
    branch_to : Float[np.ndarray, " n_branches"]
        To-bus indices for all branches.
    v_mag_hat : Float[np.ndarray, " n_buses"]
        Converged N-0 voltage magnitudes |V| at all buses.
    theta_hat : Float[np.ndarray, " n_buses"]
        Converged N-0 voltage angles (rad) at all buses.
    y_ff : Complex128[np.ndarray, " n_branches"]
        Self-admittance at the "from" bus for all branches.
    y_ft : Complex128[np.ndarray, " n_branches"]
        Mutual admittance from "from" to "to" bus for all branches.
    y_tf : Complex128[np.ndarray, " n_branches"]
        Mutual admittance from "to" to "from" bus for all branches.
    y_tt : Complex128[np.ndarray, " n_branches"]
        Self-admittance at the "to" bus for all branches.
    angle_component_indices : Int[np.ndarray, " n_eq_jacobian"]
        Bus -> θ/P position map in the Jacobian.
    magnitude_component_indices : Int[np.ndarray, " n_eq_jacobian"]
        Bus -> u/Q position map in the Jacobian.

    Returns
    -------
    updated_inv : Float[np.ndarray, " n_eq n_eq"]
        The updated inverse Jacobian after applying outages.
    """
    jacobian_inv_arr = np.asarray(jacobian_inv)
    if not np.issubdtype(jacobian_inv_arr.dtype, np.floating):
        jacobian_inv_arr = jacobian_inv_arr.astype(float)

    updated_inv = jacobian_inv_arr.copy()

    v_mag = np.asarray(v_mag_hat).reshape(-1)
    theta = np.asarray(theta_hat).reshape(-1)

    y_ff_arr = np.asarray(y_ff)
    y_ft_arr = np.asarray(y_ft)
    y_tf_arr = np.asarray(y_tf)
    y_tt_arr = np.asarray(y_tt)

    # Note: this is a reference implementation and is not optimized for performance.
    # In particular, multiple outages could be processed together for better efficiency.
    for branch in outage_branches_indices:
        jacobian_delta_submatrix, idx_list = _prepare_low_rank_factors_from_admittance(
            branch_idx=branch,
            branch_from=branch_from,
            branch_to=branch_to,
            v_mag_hat=v_mag,
            theta_hat=theta,
            y_ff=y_ff_arr,
            y_ft=y_ft_arr,
            y_tf=y_tf_arr,
            y_tt=y_tt_arr,
            angle_component_indices=angle_component_indices,
            magnitude_component_indices=magnitude_component_indices,
        )

        if idx_list.size == 0 or jacobian_delta_submatrix.size == 0:
            continue

        delta_inv_jacobian = full_rank_delta_inv_jacobian(
            jacobian_inv=updated_inv,
            jacobian_delta_submatrix=jacobian_delta_submatrix,
            idx_list=idx_list,
        )
        updated_inv = updated_inv - delta_inv_jacobian

    return updated_inv
