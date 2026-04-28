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

# ruff: noqa: PLR0913


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


def _compute_shunt_delta_submatrix_from_admittance(
    v_mag: Float[np.ndarray, ""],
    y_shunt: Complex128[np.ndarray, ""],
) -> Float[np.ndarray, "2 2"]:
    """Return the 2x2 Jacobian contribution for a shunt admittance at one bus."""
    conductance = np.real(y_shunt)
    susceptance = np.imag(y_shunt)
    dtype = np.result_type(v_mag, y_shunt)

    delta = np.array(
        [
            [0.0, 2.0 * v_mag * conductance],
            [0.0, -2.0 * v_mag * susceptance],
        ],
        dtype=dtype,
    )
    return delta * -1


def _add_bus_component_indices(
    container: set[int],
    bus_idx: int,
    angle_idx_map: Int[np.ndarray, " n_eq_jacobian"],
    magnitude_idx_map: Int[np.ndarray, " n_eq_jacobian"],
) -> None:
    """Add valid Jacobian component indices for one bus into a set.

    Parameters
    ----------
    container : set[int]
        Mutable set collecting Jacobian indices affected by the split update.
    bus_idx : int
        Bus whose angle and magnitude component indices should be added.
    angle_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to angle-equation indices.
    magnitude_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to magnitude-equation indices.
    """
    if 0 <= bus_idx < angle_idx_map.size:
        theta_idx = int(angle_idx_map[bus_idx])
        if theta_idx >= 0:
            container.add(theta_idx)
    if 0 <= bus_idx < magnitude_idx_map.size:
        mag_idx = int(magnitude_idx_map[bus_idx])
        if mag_idx >= 0:
            container.add(mag_idx)


def _collect_targeted_indices(
    bus_to_split: int,
    new_bus_b_index: int,
    branches_connected_to_bus_b: Int[np.ndarray, " n_branches_B"],
    shunt_connected_to_bus_b: Int[np.ndarray, " n_shunts_B"],
    branch_from: Int[np.ndarray, " n_branches"],
    branch_to: Int[np.ndarray, " n_branches"],
    shunt_to_bus: Int[np.ndarray, " n_shunts"],
    angle_idx_map: Int[np.ndarray, " n_eq_jacobian"],
    magnitude_idx_map: Int[np.ndarray, " n_eq_jacobian"],
) -> Int[np.ndarray, " k"]:
    """Collect all Jacobian indices affected by branch and shunt reassignment.

    Parameters
    ----------
    bus_to_split : int
        Index of the original bus being split.
    new_bus_b_index : int
        Index of the new bus created by the split.
    branches_connected_to_bus_b : Int[np.ndarray, " n_branches_B"]
        Indices of branches reassigned to the new bus.
    shunt_connected_to_bus_b : Int[np.ndarray, " n_shunts_B"]
        Indices of shunts reassigned to the new bus.
    branch_from : Int[np.ndarray, " n_branches"]
        Branch ``from`` bus indices.
    branch_to : Int[np.ndarray, " n_branches"]
        Branch ``to`` bus indices.
    shunt_to_bus : Int[np.ndarray, " n_shunts"]
        Shunt bus indices.
    angle_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to Jacobian angle-equation indices.
    magnitude_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to Jacobian magnitude-equation indices.

    Returns
    -------
    Int[np.ndarray, " k"]
        Sorted Jacobian indices that appear in the local BSDF update block.

    Raises
    ------
    IndexError
        Raised when a reassigned branch or shunt index is outside the available arrays.
    """
    targeted_indices: set[int] = set()
    _add_bus_component_indices(targeted_indices, bus_to_split, angle_idx_map, magnitude_idx_map)
    _add_bus_component_indices(targeted_indices, new_bus_b_index, angle_idx_map, magnitude_idx_map)

    for branch_idx in branches_connected_to_bus_b:
        if branch_idx < 0 or branch_idx >= branch_from.size:
            raise IndexError("Branch index assigned to bus B is out of bounds")

        from_bus_old = int(branch_from[branch_idx])
        to_bus_old = int(branch_to[branch_idx])
        from_bus_new = new_bus_b_index if from_bus_old == bus_to_split else from_bus_old
        to_bus_new = new_bus_b_index if to_bus_old == bus_to_split else to_bus_old

        _add_bus_component_indices(targeted_indices, from_bus_old, angle_idx_map, magnitude_idx_map)
        _add_bus_component_indices(targeted_indices, to_bus_old, angle_idx_map, magnitude_idx_map)
        _add_bus_component_indices(targeted_indices, from_bus_new, angle_idx_map, magnitude_idx_map)
        _add_bus_component_indices(targeted_indices, to_bus_new, angle_idx_map, magnitude_idx_map)

    for shunt_idx in shunt_connected_to_bus_b:
        if shunt_idx < 0 or shunt_idx >= shunt_to_bus.size:
            raise IndexError("Shunt index assigned to bus B is out of bounds")

        shunt_bus_old = int(shunt_to_bus[shunt_idx])
        shunt_bus_new = new_bus_b_index if shunt_bus_old == bus_to_split else shunt_bus_old

        _add_bus_component_indices(targeted_indices, shunt_bus_old, angle_idx_map, magnitude_idx_map)
        _add_bus_component_indices(targeted_indices, shunt_bus_new, angle_idx_map, magnitude_idx_map)

    return np.array(sorted(targeted_indices), dtype=int)


def _get_branch_component_indices(
    bus_from: int,
    bus_to: int,
    angle_idx_map: Int[np.ndarray, " n_eq_jacobian"],
    magnitude_idx_map: Int[np.ndarray, " n_eq_jacobian"],
) -> Int[np.ndarray, "4"]:
    """Return the Jacobian component indices touched by one branch contribution.

    Parameters
    ----------
    bus_from : int
        ``From`` bus of the branch contribution.
    bus_to : int
        ``To`` bus of the branch contribution.
    angle_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to angle-equation indices.
    magnitude_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to magnitude-equation indices.

    Returns
    -------
    Int[np.ndarray, "4"]
        Component indices ordered as angle-from, angle-to, magnitude-from,
        magnitude-to.
    """
    return np.array(
        [
            angle_idx_map[bus_from],
            angle_idx_map[bus_to],
            magnitude_idx_map[bus_from],
            magnitude_idx_map[bus_to],
        ],
        dtype=int,
    )


def _get_shunt_component_indices(
    bus_idx: int,
    angle_idx_map: Int[np.ndarray, " n_eq_jacobian"],
    magnitude_idx_map: Int[np.ndarray, " n_eq_jacobian"],
) -> Int[np.ndarray, "2"]:
    """Return the Jacobian component indices touched by one shunt contribution.

    Parameters
    ----------
    bus_idx : int
        Bus receiving the shunt contribution.
    angle_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to angle-equation indices.
    magnitude_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to magnitude-equation indices.

    Returns
    -------
    Int[np.ndarray, "2"]
        Component indices ordered as angle, magnitude.
    """
    return np.array([angle_idx_map[bus_idx], magnitude_idx_map[bus_idx]], dtype=int)


def _accumulate_sub_delta(
    delta_block: Float[np.ndarray, " k k"],
    delta_matrix: Float[np.ndarray, " m m"],
    component_indices: Int[np.ndarray, " m"],
    position_lookup: dict[int, int],
    weight: float,
) -> None:
    """Add a local branch or shunt Jacobian delta into the global update block.

    Parameters
    ----------
    delta_block : Float[np.ndarray, " k k"]
        Global Jacobian update block being assembled for the Woodbury update.
    delta_matrix : Float[np.ndarray, " m m"]
        Local Jacobian contribution for one branch or one shunt.
    component_indices : Int[np.ndarray, " m"]
        Jacobian indices touched by the local contribution.
    position_lookup : dict[int, int]
        Mapping from global Jacobian index to local row/column position in
        ``delta_block``.
    weight : float
        Scaling factor used to add or subtract the contribution.
    """
    valid_positions = np.flatnonzero(component_indices >= 0)
    if valid_positions.size == 0:
        return

    local_indices = component_indices[valid_positions]
    mapped_positions = [position_lookup[int(idx)] for idx in local_indices]
    sub_delta = delta_matrix[np.ix_(valid_positions, valid_positions)]

    for row_offset, pos_row in enumerate(mapped_positions):
        for col_offset, pos_col in enumerate(mapped_positions):
            delta_block[pos_row, pos_col] += weight * float(sub_delta[row_offset, col_offset])


def _accumulate_branch_reassignment_delta(
    delta_block: Float[np.ndarray, " k k"],
    position_lookup: dict[int, int],
    branches_connected_to_bus_b: Int[np.ndarray, " n_branches_B"],
    bus_to_split: int,
    new_bus_b_index: int,
    branch_from: Int[np.ndarray, " n_branches"],
    branch_to: Int[np.ndarray, " n_branches"],
    v_mag_hat: Float[np.ndarray, " n_buses"],
    theta_hat: Float[np.ndarray, " n_buses"],
    y_ff: Complex128[np.ndarray, " n_branches"],
    y_ft: Complex128[np.ndarray, " n_branches"],
    y_tf: Complex128[np.ndarray, " n_branches"],
    y_tt: Complex128[np.ndarray, " n_branches"],
    angle_idx_map: Int[np.ndarray, " n_eq_jacobian"],
    magnitude_idx_map: Int[np.ndarray, " n_eq_jacobian"],
) -> None:
    """Assemble the branch part of the BSDF Jacobian update block.

    Parameters
    ----------
    delta_block : Float[np.ndarray, " k k"]
        Global Jacobian update block being assembled.
    position_lookup : dict[int, int]
        Mapping from global Jacobian index to local row/column position.
    branches_connected_to_bus_b : Int[np.ndarray, " n_branches_B"]
        Branch indices reassigned to the new bus.
    bus_to_split : int
        Index of the original split bus.
    new_bus_b_index : int
        Index of the new bus.
    branch_from : Int[np.ndarray, " n_branches"]
        Branch ``from`` bus indices.
    branch_to : Int[np.ndarray, " n_branches"]
        Branch ``to`` bus indices.
    v_mag_hat : Float[np.ndarray, " n_buses"]
        Voltage magnitudes used for the local Jacobian contributions.
    theta_hat : Float[np.ndarray, " n_buses"]
        Voltage angles used for the local Jacobian contributions.
    y_ff : Complex128[np.ndarray, " n_branches"]
        Branch self-admittance at the ``from`` side.
    y_ft : Complex128[np.ndarray, " n_branches"]
        Branch mutual admittance from ``from`` to ``to``.
    y_tf : Complex128[np.ndarray, " n_branches"]
        Branch mutual admittance from ``to`` to ``from``.
    y_tt : Complex128[np.ndarray, " n_branches"]
        Branch self-admittance at the ``to`` side.
    angle_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to angle-equation indices.
    magnitude_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to magnitude-equation indices.
    """
    for branch_idx in branches_connected_to_bus_b:
        from_bus_old = int(branch_from[branch_idx])
        to_bus_old = int(branch_to[branch_idx])

        delta_old = _compute_branch_delta_submatrix_from_admittance(
            v_mag_from=v_mag_hat[from_bus_old],
            v_mag_to=v_mag_hat[to_bus_old],
            theta_from=theta_hat[from_bus_old],
            theta_to=theta_hat[to_bus_old],
            y_ff=y_ff[branch_idx],
            y_ft=y_ft[branch_idx],
            y_tf=y_tf[branch_idx],
            y_tt=y_tt[branch_idx],
        )
        _accumulate_sub_delta(
            delta_block=delta_block,
            delta_matrix=delta_old,
            component_indices=_get_branch_component_indices(
                from_bus_old,
                to_bus_old,
                angle_idx_map,
                magnitude_idx_map,
            ),
            position_lookup=position_lookup,
            weight=1.0,
        )

        from_bus_new = new_bus_b_index if from_bus_old == bus_to_split else from_bus_old
        to_bus_new = new_bus_b_index if to_bus_old == bus_to_split else to_bus_old
        delta_new = _compute_branch_delta_submatrix_from_admittance(
            v_mag_from=v_mag_hat[from_bus_new],
            v_mag_to=v_mag_hat[to_bus_new],
            theta_from=theta_hat[from_bus_new],
            theta_to=theta_hat[to_bus_new],
            y_ff=y_ff[branch_idx],
            y_ft=y_ft[branch_idx],
            y_tf=y_tf[branch_idx],
            y_tt=y_tt[branch_idx],
        )
        _accumulate_sub_delta(
            delta_block=delta_block,
            delta_matrix=delta_new,
            component_indices=_get_branch_component_indices(
                from_bus_new,
                to_bus_new,
                angle_idx_map,
                magnitude_idx_map,
            ),
            position_lookup=position_lookup,
            weight=-1.0,
        )


def _accumulate_shunt_reassignment_delta(
    delta_block: Float[np.ndarray, " k k"],
    position_lookup: dict[int, int],
    shunt_connected_to_bus_b: Int[np.ndarray, " n_shunts_B"],
    bus_to_split: int,
    new_bus_b_index: int,
    shunt_to_bus: Int[np.ndarray, " n_shunts"],
    v_mag_hat: Float[np.ndarray, " n_buses"],
    y_shunt: Complex128[np.ndarray, " n_shunts"],
    angle_idx_map: Int[np.ndarray, " n_eq_jacobian"],
    magnitude_idx_map: Int[np.ndarray, " n_eq_jacobian"],
) -> None:
    """Assemble the shunt part of the BSDF Jacobian update block.

    Parameters
    ----------
    delta_block : Float[np.ndarray, " k k"]
        Global Jacobian update block being assembled.
    position_lookup : dict[int, int]
        Mapping from global Jacobian index to local row/column position.
    shunt_connected_to_bus_b : Int[np.ndarray, " n_shunts_B"]
        Shunt indices reassigned to the new bus.
    bus_to_split : int
        Index of the original split bus.
    new_bus_b_index : int
        Index of the new bus.
    shunt_to_bus : Int[np.ndarray, " n_shunts"]
        Shunt bus indices.
    v_mag_hat : Float[np.ndarray, " n_buses"]
        Voltage magnitudes used for the local Jacobian contributions.
    y_shunt : Complex128[np.ndarray, " n_shunts"]
        Effective shunt admittances.
    angle_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to angle-equation indices.
    magnitude_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to magnitude-equation indices.
    """
    for shunt_idx in shunt_connected_to_bus_b:
        shunt_bus_old = int(shunt_to_bus[shunt_idx])
        shunt_bus_new = new_bus_b_index if shunt_bus_old == bus_to_split else shunt_bus_old

        delta_old = _compute_shunt_delta_submatrix_from_admittance(
            v_mag=v_mag_hat[shunt_bus_old],
            y_shunt=y_shunt[shunt_idx],
        )
        _accumulate_sub_delta(
            delta_block=delta_block,
            delta_matrix=delta_old,
            component_indices=_get_shunt_component_indices(
                shunt_bus_old,
                angle_idx_map,
                magnitude_idx_map,
            ),
            position_lookup=position_lookup,
            weight=1.0,
        )

        delta_new = _compute_shunt_delta_submatrix_from_admittance(
            v_mag=v_mag_hat[shunt_bus_new],
            y_shunt=y_shunt[shunt_idx],
        )
        _accumulate_sub_delta(
            delta_block=delta_block,
            delta_matrix=delta_new,
            component_indices=_get_shunt_component_indices(
                shunt_bus_new,
                angle_idx_map,
                magnitude_idx_map,
            ),
            position_lookup=position_lookup,
            weight=-1.0,
        )


def _apply_new_bus_diagonal_adjustment(
    delta_block: Float[np.ndarray, " k k"],
    position_lookup: dict[int, int],
    new_bus_b_index: int,
    angle_idx_map: Int[np.ndarray, " n_eq_jacobian"],
    magnitude_idx_map: Int[np.ndarray, " n_eq_jacobian"],
) -> None:
    """Apply the diagonal terms introduced by the additional PQ bus equations.

    Parameters
    ----------
    delta_block : Float[np.ndarray, " k k"]
        Global Jacobian update block being assembled.
    position_lookup : dict[int, int]
        Mapping from global Jacobian index to local row/column position.
    new_bus_b_index : int
        Index of the new bus.
    angle_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to angle-equation indices.
    magnitude_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to magnitude-equation indices.
    """
    theta_idx_new = int(angle_idx_map[new_bus_b_index])
    if theta_idx_new >= 0 and theta_idx_new in position_lookup:
        delta_block[position_lookup[theta_idx_new], position_lookup[theta_idx_new]] -= 1.0

    mag_idx_new = int(magnitude_idx_map[new_bus_b_index])
    if mag_idx_new >= 0 and mag_idx_new in position_lookup:
        delta_block[position_lookup[mag_idx_new], position_lookup[mag_idx_new]] -= 1.0


def _build_delta_block(
    idx_list: Int[np.ndarray, " k"],
    dtype: np.dtype,
    branches_connected_to_bus_b: Int[np.ndarray, " n_branches_B"],
    shunt_connected_to_bus_b: Int[np.ndarray, " n_shunts_B"],
    bus_to_split: int,
    new_bus_b_index: int,
    branch_from: Int[np.ndarray, " n_branches"],
    branch_to: Int[np.ndarray, " n_branches"],
    shunt_to_bus: Int[np.ndarray, " n_shunts"],
    v_mag_hat: Float[np.ndarray, " n_buses"],
    theta_hat: Float[np.ndarray, " n_buses"],
    y_ff: Complex128[np.ndarray, " n_branches"],
    y_ft: Complex128[np.ndarray, " n_branches"],
    y_tf: Complex128[np.ndarray, " n_branches"],
    y_tt: Complex128[np.ndarray, " n_branches"],
    y_shunt: Complex128[np.ndarray, " n_shunts"],
    angle_idx_map: Int[np.ndarray, " n_eq_jacobian"],
    magnitude_idx_map: Int[np.ndarray, " n_eq_jacobian"],
) -> Float[np.ndarray, " k k"]:
    """Build the local Jacobian update block used in the Woodbury correction.

    Parameters
    ----------
    idx_list : Int[np.ndarray, " k"]
        Sorted Jacobian indices present in the update block.
    dtype : np.dtype
        Data type used to allocate the update block.
    branches_connected_to_bus_b : Int[np.ndarray, " n_branches_B"]
        Branch indices reassigned to the new bus.
    shunt_connected_to_bus_b : Int[np.ndarray, " n_shunts_B"]
        Shunt indices reassigned to the new bus.
    bus_to_split : int
        Index of the original split bus.
    new_bus_b_index : int
        Index of the new bus.
    branch_from : Int[np.ndarray, " n_branches"]
        Branch ``from`` bus indices.
    branch_to : Int[np.ndarray, " n_branches"]
        Branch ``to`` bus indices.
    shunt_to_bus : Int[np.ndarray, " n_shunts"]
        Shunt bus indices.
    v_mag_hat : Float[np.ndarray, " n_buses"]
        Voltage magnitudes of the base state.
    theta_hat : Float[np.ndarray, " n_buses"]
        Voltage angles of the base state.
    y_ff : Complex128[np.ndarray, " n_branches"]
        Branch self-admittance at the ``from`` side.
    y_ft : Complex128[np.ndarray, " n_branches"]
        Branch mutual admittance from ``from`` to ``to``.
    y_tf : Complex128[np.ndarray, " n_branches"]
        Branch mutual admittance from ``to`` to ``from``.
    y_tt : Complex128[np.ndarray, " n_branches"]
        Branch self-admittance at the ``to`` side.
    y_shunt : Complex128[np.ndarray, " n_shunts"]
        Effective shunt admittances.
    angle_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to angle-equation indices.
    magnitude_idx_map : Int[np.ndarray, " n_eq_jacobian"]
        Mapping from bus indices to magnitude-equation indices.

    Returns
    -------
    Float[np.ndarray, " k k"]
        Dense local Jacobian update block corresponding to ``idx_list``.
    """
    position_lookup = {int(idx): pos for pos, idx in enumerate(idx_list.tolist())}
    delta_block = np.zeros((idx_list.size, idx_list.size), dtype=dtype)

    _accumulate_branch_reassignment_delta(
        delta_block=delta_block,
        position_lookup=position_lookup,
        branches_connected_to_bus_b=branches_connected_to_bus_b,
        bus_to_split=bus_to_split,
        new_bus_b_index=new_bus_b_index,
        branch_from=branch_from,
        branch_to=branch_to,
        v_mag_hat=v_mag_hat,
        theta_hat=theta_hat,
        y_ff=y_ff,
        y_ft=y_ft,
        y_tf=y_tf,
        y_tt=y_tt,
        angle_idx_map=angle_idx_map,
        magnitude_idx_map=magnitude_idx_map,
    )
    _accumulate_shunt_reassignment_delta(
        delta_block=delta_block,
        position_lookup=position_lookup,
        shunt_connected_to_bus_b=shunt_connected_to_bus_b,
        bus_to_split=bus_to_split,
        new_bus_b_index=new_bus_b_index,
        shunt_to_bus=shunt_to_bus,
        v_mag_hat=v_mag_hat,
        y_shunt=y_shunt,
        angle_idx_map=angle_idx_map,
        magnitude_idx_map=magnitude_idx_map,
    )
    _apply_new_bus_diagonal_adjustment(
        delta_block=delta_block,
        position_lookup=position_lookup,
        new_bus_b_index=new_bus_b_index,
        angle_idx_map=angle_idx_map,
        magnitude_idx_map=magnitude_idx_map,
    )
    return delta_block


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
    y_shunt: Complex128[np.ndarray, " n_shunts"],
    angle_component_indices: Int[np.ndarray, " n_eq_jacobian"],
    magnitude_component_indices: Int[np.ndarray, " n_eq_jacobian"],
) -> Float[np.ndarray, " n_eq n_eq"]:
    """Compute the BSDF update for a bus split with the full-rank update approach.

    Note: injection reassignments do not affect the Jacobian and therefore do not enter this update.
    The function currently assumes a PQ split and supports branch and shunt reassignments.

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
    y_shunt : Complex128[np.ndarray, " n_shunts"]
        The shunt admittance values for all shunts in the original system.
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

    if branches_connected_to_bus_b.size == 0 and shunt_connected_to_bus_b.size == 0:
        return jacobian_inv.copy()
    if new_bus_b_index >= angle_component_indices.size or new_bus_b_index >= magnitude_component_indices.size:
        raise IndexError("New bus index is out of bounds for component index arrays")
    idx_list = _collect_targeted_indices(
        bus_to_split=bus_to_split,
        new_bus_b_index=new_bus_b_index,
        branches_connected_to_bus_b=branches_connected_to_bus_b,
        shunt_connected_to_bus_b=shunt_connected_to_bus_b,
        branch_from=branch_from,
        branch_to=branch_to,
        shunt_to_bus=shunt_to_bus,
        angle_idx_map=angle_component_indices,
        magnitude_idx_map=magnitude_component_indices,
    )
    if idx_list.size == 0:
        return jacobian_inv.copy()

    delta_block = _build_delta_block(
        idx_list=idx_list,
        dtype=jacobian_inv.dtype,
        branches_connected_to_bus_b=branches_connected_to_bus_b,
        shunt_connected_to_bus_b=shunt_connected_to_bus_b,
        bus_to_split=bus_to_split,
        new_bus_b_index=new_bus_b_index,
        branch_from=branch_from,
        branch_to=branch_to,
        shunt_to_bus=shunt_to_bus,
        v_mag_hat=v_mag_hat,
        theta_hat=theta_hat,
        y_ff=y_ff,
        y_ft=y_ft,
        y_tf=y_tf,
        y_tt=y_tt,
        y_shunt=y_shunt,
        angle_idx_map=angle_component_indices,
        magnitude_idx_map=magnitude_component_indices,
    )
    if not np.any(delta_block):
        return jacobian_inv.copy()

    return _apply_full_rank_update(
        jacobian_inv=jacobian_inv,
        jacobian_delta_submatrix=delta_block,
        idx_list=idx_list,
    )
