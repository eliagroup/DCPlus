# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""BSDF update implementation for the full-rank case."""

import numpy as np
from jaxtyping import Complex128, Float, Int

from dc_plus.numpy.lodf import (
    full_rank_delta_inv_jacobian,
)
from dc_plus.numpy.low_rank_helper import (
    _compute_branch_delta_submatrix_from_admittance,
)

# ruff: noqa: ARG001, C901, PLR0913, PLR0915


def _apply_full_rank_update(
    jacobian_inv: Float[np.ndarray, " n_eq n_eq"],
    jacobian_delta_submatrix: Float[np.ndarray, " k k"],
    idx_list: Int[np.ndarray, " k"],
) -> Float[np.ndarray, " n_eq n_eq"]:
    """Apply the full-rank update to the Jacobian inverse.

    Parameters
    ----------
    jacobian_inv : Float[np.ndarray, " n_eq n_eq"]
        The original Jacobian inverse.
    jacobian_delta_submatrix : Float[np.ndarray, " k k"]
        The submatrix of the Jacobian delta corresponding to the affected indices.
        "D" in the Woodbury formula.
    idx_list : Int[np.ndarray, " k"]
        The list of indices corresponding to the rows and columns of the Jacobian delta submatrix.

    Returns
    -------
    Float[np.ndarray, " n_eq n_eq"]
        The updated Jacobian inverse after applying the full-rank update.
    """
    if idx_list.size == 0 or jacobian_delta_submatrix.size == 0:
        return jacobian_inv.copy()

    delta_inv_jacobian = full_rank_delta_inv_jacobian(
        jacobian_inv=jacobian_inv,
        jacobian_delta_submatrix=jacobian_delta_submatrix,
        idx_list=idx_list,
    )
    return jacobian_inv - delta_inv_jacobian


# ruff: noqa: C901, PLR0915


def compute_bsdf_update(
    jacobian_inv: Float[np.ndarray, " n_eq n_eq"],
    bus_to_split: int,
    new_bus_b_index: int,
    new_bus_type: int,
    branches_connected_to_bus_b: Int[np.ndarray, " n_branches_B"],
    shunt_connected_to_bus_b: Int[np.ndarray, " n_shunts_B"],
    branch_from: Float[np.ndarray, " n_branches"],
    branch_to: Float[np.ndarray, " n_branches"],
    shunt_to_bus: Float[np.ndarray, " n_shunts"],
    v_mag_hat: Float[np.ndarray, " n_buses"],
    theta_hat: Float[np.ndarray, " n_buses"],
    y_ff: Complex128[np.ndarray, " n_branches"],
    y_ft: Complex128[np.ndarray, " n_branches"],
    y_tf: Complex128[np.ndarray, " n_branches"],
    y_tt: Complex128[np.ndarray, " n_branches"],
    y_shunt: Complex128[np.ndarray, " n_buses"],
    angle_component_indices: Int[np.ndarray, " n_eq_jacobian"],
    magnitude_component_indices: Int[np.ndarray, " n_eq_jacobian"],
) -> Float[np.ndarray, " n_eq n_eq"]:
    """Compute the BSDF update for a bus split with the full-rank update approach.

    Note: Injection and shunt changes are currently not supported, and the function assumes a PQ split.
    Only branch reassignments are handled.

    Parameters
    ----------
    jacobian_inv : Float[np.ndarray, " n_eq n_eq"]
        The original Jacobian inverse before the bus split.
    bus_to_split : int
        The index of the bus that is being split.
    new_bus_b_index : int
        The index of the new bus B that is created from the split.
    new_bus_type : int
        The type of the new bus B (e.g., PQ, PV, Slack).
    branches_connected_to_bus_b : Int[np.ndarray, " n_branches_B"]
        The indices of the branches that are connected to the new bus B after the split.
    shunt_connected_to_bus_b : Int[np.ndarray, " n_shunts_B"]
        The indices of the shunts that are connected to the new bus B after the split.
    branch_from : Float[np.ndarray, " n_branches"]
        The "from" bus indices for all branches in the original system.
    branch_to : Float[np.ndarray, " n_branches"]
        The "to" bus indices for all branches in the original system.
    shunt_to_bus : Float[np.ndarray, " n_shunts"]
        The bus indices for all shunts in the original system.
    v_mag_hat : Float[np.ndarray, " n_buses"]
        The voltage magnitudes at the buses in the original system.
    theta_hat : Float[np.ndarray, " n_buses"]
        The voltage angles at the buses in the original system.
    y_ff : Complex128[np.ndarray, " n_branches"]
        The "from-from" admittance values for all branches in the original system.
    y_ft : Complex128[np.ndarray, " n_branches"]
        The "from-to" admittance values for all branches in the original system.
    y_tf : Complex128[np.ndarray, " n_branches"]
        The "to-from" admittance values for all branches in the original system.
    y_tt : Complex128[np.ndarray, " n_branches"]
        The "to-to" admittance values for all branches in the original system.
    y_shunt : Complex128[np.ndarray, " n_buses"]
        The shunt admittance values for all buses in the original system.
    angle_component_indices : Int[np.ndarray, " n_eq_jacobian"]
        The mapping from bus indices to angle component indices in the Jacobian.
    magnitude_component_indices : Int[np.ndarray, " n_eq_jacobian"]
        The mapping from bus indices to magnitude component indices in the Jacobian.

    Returns
    -------
    Float[np.ndarray, " n_eq n_eq"]
        The updated Jacobian inverse after applying the BSDF update for the bus split.
    """
    if new_bus_type != 2:
        raise NotImplementedError("Only PQ splits are supported")

    if np.asarray(shunt_connected_to_bus_b).size:
        raise NotImplementedError("Shunt reassignment is not supported")

    if branches_connected_to_bus_b.size == 0:
        return np.asarray(jacobian_inv, dtype=float).copy()

    base_inverse = np.asarray(jacobian_inv, dtype=float)
    updated_inverse = base_inverse.copy()

    branch_from_arr = np.asarray(branch_from, dtype=int).reshape(-1)
    branch_to_arr = np.asarray(branch_to, dtype=int).reshape(-1)

    angle_idx_map = np.asarray(angle_component_indices, dtype=int).reshape(-1)
    magnitude_idx_map = np.asarray(magnitude_component_indices, dtype=int).reshape(-1)

    if new_bus_b_index >= angle_idx_map.size or new_bus_b_index >= magnitude_idx_map.size:
        raise IndexError("New bus index is out of bounds for component index arrays")

    v_mag_vec = np.asarray(v_mag_hat, dtype=float).reshape(-1)
    theta_vec = np.asarray(theta_hat, dtype=float).reshape(-1)

    y_ff_vec = np.asarray(y_ff, dtype=np.complex128).reshape(-1)
    y_ft_vec = np.asarray(y_ft, dtype=np.complex128).reshape(-1)
    y_tf_vec = np.asarray(y_tf, dtype=np.complex128).reshape(-1)
    y_tt_vec = np.asarray(y_tt, dtype=np.complex128).reshape(-1)

    def _add_component_indices(container: set[int], bus_idx: int) -> None:
        if 0 <= bus_idx < angle_idx_map.size:
            theta_idx = int(angle_idx_map[bus_idx])
            if theta_idx >= 0:
                container.add(theta_idx)
        if 0 <= bus_idx < magnitude_idx_map.size:
            mag_idx = int(magnitude_idx_map[bus_idx])
            if mag_idx >= 0:
                container.add(mag_idx)

    targeted_indices: set[int] = set()

    _add_component_indices(targeted_indices, bus_to_split)
    _add_component_indices(targeted_indices, new_bus_b_index)

    # remove components from bus_to_split
    for branch_idx in branches_connected_to_bus_b:
        if branch_idx < 0 or branch_idx >= branch_from_arr.size:
            raise IndexError("Branch index assigned to bus B is out of bounds")

        from_bus_old = int(branch_from_arr[branch_idx])
        to_bus_old = int(branch_to_arr[branch_idx])

        _add_component_indices(targeted_indices, from_bus_old)
        _add_component_indices(targeted_indices, to_bus_old)

        from_bus_new = new_bus_b_index if from_bus_old == bus_to_split else from_bus_old
        to_bus_new = new_bus_b_index if to_bus_old == bus_to_split else to_bus_old

        _add_component_indices(targeted_indices, from_bus_new)
        _add_component_indices(targeted_indices, to_bus_new)

    if not targeted_indices:
        return updated_inverse

    idx_list = np.array(sorted(targeted_indices), dtype=int)
    position_lookup = {idx: pos for pos, idx in enumerate(idx_list.tolist())}

    delta_block = np.zeros((idx_list.size, idx_list.size), dtype=updated_inverse.dtype)

    def _accumulate_branch_delta(
        delta_matrix: Float[np.ndarray, "4 4"],
        bus_from: int,
        bus_to: int,
        weight: float,
    ) -> None:
        component_indices = np.array(
            [
                angle_idx_map[bus_from],
                angle_idx_map[bus_to],
                magnitude_idx_map[bus_from],
                magnitude_idx_map[bus_to],
            ],
            dtype=int,
        )
        valid_positions = np.flatnonzero(component_indices >= 0)
        if valid_positions.size == 0:
            return

        local_indices = component_indices[valid_positions]
        mapped_positions = [position_lookup[idx] for idx in local_indices]
        sub_delta = delta_matrix[np.ix_(valid_positions, valid_positions)]

        for row_offset, pos_row in enumerate(mapped_positions):
            for col_offset, pos_col in enumerate(mapped_positions):
                delta_block[pos_row, pos_col] += weight * float(sub_delta[row_offset, col_offset])

    # Accumulate the contributions from the branches that are reassigned to bus B
    for branch_idx in branches_connected_to_bus_b:
        from_bus_old = int(branch_from_arr[branch_idx])
        to_bus_old = int(branch_to_arr[branch_idx])

        delta_old = _compute_branch_delta_submatrix_from_admittance(
            v_mag_from=v_mag_vec[from_bus_old],
            v_mag_to=v_mag_vec[to_bus_old],
            theta_from=theta_vec[from_bus_old],
            theta_to=theta_vec[to_bus_old],
            y_ff=y_ff_vec[branch_idx],
            y_ft=y_ft_vec[branch_idx],
            y_tf=y_tf_vec[branch_idx],
            y_tt=y_tt_vec[branch_idx],
        )
        _accumulate_branch_delta(delta_old, from_bus_old, to_bus_old, weight=1.0)

        from_bus_new = new_bus_b_index if from_bus_old == bus_to_split else from_bus_old
        to_bus_new = new_bus_b_index if to_bus_old == bus_to_split else to_bus_old

        delta_new = _compute_branch_delta_submatrix_from_admittance(
            v_mag_from=v_mag_vec[from_bus_new],
            v_mag_to=v_mag_vec[to_bus_new],
            theta_from=theta_vec[from_bus_new],
            theta_to=theta_vec[to_bus_new],
            y_ff=y_ff_vec[branch_idx],
            y_ft=y_ft_vec[branch_idx],
            y_tf=y_tf_vec[branch_idx],
            y_tt=y_tt_vec[branch_idx],
        )
        _accumulate_branch_delta(delta_new, from_bus_new, to_bus_new, weight=-1.0)

    theta_idx_new = int(angle_idx_map[new_bus_b_index])
    if theta_idx_new >= 0 and theta_idx_new in position_lookup:
        pos = position_lookup[theta_idx_new]
        delta_block[pos, pos] -= 1.0

    mag_idx_new = int(magnitude_idx_map[new_bus_b_index])
    if mag_idx_new >= 0 and mag_idx_new in position_lookup:
        pos = position_lookup[mag_idx_new]
        delta_block[pos, pos] -= 1.0

    if not np.any(delta_block):
        return updated_inverse

    updated_inverse = _apply_full_rank_update(
        jacobian_inv=base_inverse,
        jacobian_delta_submatrix=delta_block,
        idx_list=idx_list,
    )

    return updated_inverse
