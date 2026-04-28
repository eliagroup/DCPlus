# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""One-step state updates for injection outages."""

import numpy as np
from jaxtyping import Float, Int


def _collect_injection_outage_mismatch(
    jacobian_size: int,
    outage_injection_indices: Int[np.ndarray, " n_contingencies n_outages"],
    injection_to_bus: Int[np.ndarray, " n_injections"],
    injection_active_power: Float[np.ndarray, " n_injections"],
    injection_reactive_power: Float[np.ndarray, " n_injections"],
    angle_component_indices: Int[np.ndarray, " n_buses"],
    magnitude_component_indices: Int[np.ndarray, " n_buses"],
    dtype: np.dtype,
) -> Float[np.ndarray, " n_contingencies n_eq"]:
    """Assemble outage-induced mismatch vectors in Jacobian ordering.

    Parameters
    ----------
    jacobian_size : int
        Number of equations in the Jacobian system.
    outage_injection_indices : Int[np.ndarray, " n_contingencies n_outages"]
        Batched injection outage indices. Each row represents one contingency and
        may include negative padding entries.
    injection_to_bus : Int[np.ndarray, " n_injections"]
        Map from injection index to hosting bus index.
    injection_active_power : Float[np.ndarray, " n_injections"]
        Active power associated with each injection.
    injection_reactive_power : Float[np.ndarray, " n_injections"]
        Reactive power associated with each injection.
    angle_component_indices : Int[np.ndarray, " n_buses"]
        Map from bus index to active-power mismatch component index.
    magnitude_component_indices : Int[np.ndarray, " n_buses"]
        Map from bus index to reactive-power mismatch component index.
    dtype : np.dtype
        Target dtype of the assembled mismatch matrix.

    Returns
    -------
    Float[np.ndarray, " n_contingencies n_eq"]
        Batched mismatch vectors in Jacobian component ordering.
    """
    outage_indices = outage_injection_indices
    mismatch = np.zeros((outage_indices.shape[0], jacobian_size), dtype=dtype)
    if outage_indices.size == 0:
        return mismatch

    valid_outage_mask = outage_indices >= 0
    if not np.any(valid_outage_mask):
        return mismatch

    valid_rows, valid_cols = np.nonzero(valid_outage_mask)
    valid_outages = outage_indices[valid_rows, valid_cols]
    if np.any(valid_outages >= injection_to_bus.size):
        raise IndexError("Injection outage index is out of bounds")

    buses = injection_to_bus[valid_outages]

    p_idx = angle_component_indices[buses]
    valid_p = p_idx >= 0
    if np.any(valid_p):
        np.add.at(
            mismatch,
            (valid_rows[valid_p], p_idx[valid_p]),
            -injection_active_power[valid_outages][valid_p],
        )

    q_idx = magnitude_component_indices[buses]
    valid_q = q_idx >= 0
    if np.any(valid_q):
        np.add.at(
            mismatch,
            (valid_rows[valid_q], q_idx[valid_q]),
            -injection_reactive_power[valid_outages][valid_q],
        )

    return mismatch


def _calculate_injection_outage_dx(
    jacobian_inv: Float[np.ndarray, " n_eq n_eq"],
    mismatch: Float[np.ndarray, " n_contingencies n_eq"],
) -> Float[np.ndarray, " n_contingencies n_eq"]:
    """Map outage mismatch vectors to one-step state increments.

    Parameters
    ----------
    jacobian_inv : Float[np.ndarray, " n_eq n_eq"]
        Inverse Jacobian at the base operating point.
    mismatch : Float[np.ndarray, " n_contingencies n_eq"]
        Batched outage-induced mismatch vectors.

    Returns
    -------
    Float[np.ndarray, " n_contingencies n_eq"]
        Batched one-step state increments in Jacobian ordering.
    """
    return -(mismatch @ jacobian_inv.T)


def non_voltage_regulating_injection_outage_dx(
    jacobian_inv: Float[np.ndarray, " n_eq n_eq"],
    outage_injection_indices: Int[np.ndarray, " n_contingencies n_outages"],
    injection_to_bus: Int[np.ndarray, " n_injections"],
    injection_active_power: Float[np.ndarray, " n_injections"],
    injection_reactive_power: Float[np.ndarray, " n_injections"],
    angle_component_indices: Int[np.ndarray, " n_buses"],
    magnitude_component_indices: Int[np.ndarray, " n_buses"],
) -> Float[np.ndarray, " n_contingencies n_eq"]:
    """One-step state increment caused by disconnecting non voltage regulating injections.

    The Jacobian is unchanged for a pure injection outage because network
    topology and admittances remain fixed. The outage only changes the nodal
    power mismatch at the buses hosting the disconnected injections.

    This helper assumes the bus type pattern used by the Jacobian remains
    unchanged across the outage. That is appropriate for load outages and other
    contingencies that do not alter voltage-control structure.

    Note: this function still works for voltage regulating injection outages,
    but the PV->PQ bus type change is not accounted for.

    Parameters
    ----------
    jacobian_inv : Float[np.ndarray, " n_eq n_eq"]
        Base inverse Jacobian at the hot-start operating point.
    outage_injection_indices : Int[np.ndarray, " n_contingencies n_outages"]
        Batched indices of injections to disconnect. Each row is one
        contingency and may contain multiple outages. Negative entries are
        ignored and can be used as padding for rows with different outage
        counts.
    injection_to_bus : Int[np.ndarray, " n_injections"]
        Mapping from injection index to bus index.
    injection_active_power : Float[np.ndarray, " n_injections"]
        Active power removed when each injection is disconnected.
        In practice this is typically the injection setpoint, or the solved
        injection power when no separate setpoint is available.
    injection_reactive_power : Float[np.ndarray, " n_injections"]
        Reactive power removed when each injection is disconnected.
        In practice this is typically the injection setpoint, or the solved
        injection power when no separate setpoint is available.
    angle_component_indices : Int[np.ndarray, " n_buses"]
        Bus to active-power Jacobian row mapping.
    magnitude_component_indices : Int[np.ndarray, " n_buses"]
        Bus to reactive-power Jacobian row mapping.

    Returns
    -------
    Float[np.ndarray, " n_contingencies n_eq"]
        Batched one-step update vectors in Jacobian ordering.
    """
    mismatch = _collect_injection_outage_mismatch(
        jacobian_size=jacobian_inv.shape[0],
        outage_injection_indices=outage_injection_indices,
        injection_to_bus=injection_to_bus,
        injection_active_power=injection_active_power,
        injection_reactive_power=injection_reactive_power,
        angle_component_indices=angle_component_indices,
        magnitude_component_indices=magnitude_component_indices,
        dtype=jacobian_inv.dtype,
    )
    return _calculate_injection_outage_dx(jacobian_inv=jacobian_inv, mismatch=mismatch)
