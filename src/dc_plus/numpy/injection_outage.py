# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""One-step state updates for injection outages."""

import numpy as np
from jaxtyping import Float, Int


def injection_outage_dx(
    jacobian_inv: Float[np.ndarray, " n_eq n_eq"],
    outage_injection_indices: Int[np.ndarray, " n_outages"],
    injection_to_bus: Int[np.ndarray, " n_injections"],
    injection_active_power: Float[np.ndarray, " n_injections"],
    injection_reactive_power: Float[np.ndarray, " n_injections"],
    angle_component_indices: Int[np.ndarray, " n_buses"],
    magnitude_component_indices: Int[np.ndarray, " n_buses"],
) -> Float[np.ndarray, " n_eq"]:
    """Return the one-step state increment caused by disconnecting injections.

    The Jacobian is unchanged for a pure injection outage because network
    topology and admittances remain fixed. The outage only changes the nodal
    power mismatch at the buses hosting the disconnected injections.

    This helper assumes the bus type pattern used by the Jacobian remains
    unchanged across the outage. That is appropriate for load outages and other
    contingencies that do not alter voltage-control structure.

    Parameters
    ----------
    jacobian_inv : Float[np.ndarray, " n_eq n_eq"]
        Base inverse Jacobian at the hot-start operating point.
    outage_injection_indices : Int[np.ndarray, " n_outages"]
        Indices of injections to disconnect.
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
    Float[np.ndarray, " n_eq"]
        One-step update vector in Jacobian ordering.
    """
    outage_indices = outage_injection_indices
    mismatch = np.zeros(jacobian_inv.shape[0], dtype=jacobian_inv.dtype)
    if outage_indices.size == 0:
        return mismatch

    if np.any((outage_indices < 0) | (outage_indices >= injection_to_bus.size)):
        raise IndexError("Injection outage index is out of bounds")

    buses = injection_to_bus[outage_indices]

    p_idx = angle_component_indices[buses]
    valid_p = p_idx >= 0
    if np.any(valid_p):
        np.add.at(mismatch, p_idx[valid_p], -injection_active_power[outage_indices][valid_p])

    q_idx = magnitude_component_indices[buses]
    valid_q = q_idx >= 0
    if np.any(valid_q):
        np.add.at(mismatch, q_idx[valid_q], -injection_reactive_power[outage_indices][valid_q])

    return -jacobian_inv @ mismatch
