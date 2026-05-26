# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Shared NumPy helpers for injection-based operating-point updates."""

import numpy as np
from jaxtyping import Bool, Float, Int


def _aggregate_bus_injections(
    injection_to_bus: Int[np.ndarray, " n_injections"] | Int[np.ndarray, " n_injections n_timesteps"],
    injection_active_power: Float[np.ndarray, " n_injections"] | Float[np.ndarray, " n_injections n_timesteps"],
    injection_reactive_power: Float[np.ndarray, " n_injections"] | Float[np.ndarray, " n_injections n_timesteps"],
    injection_connected: Bool[np.ndarray, " n_injections"] | Bool[np.ndarray, " n_injections n_timesteps"],
    n_buses: int,
) -> tuple[
    Float[np.ndarray, " n_buses"] | Float[np.ndarray, " n_buses n_timesteps"],
    Float[np.ndarray, " n_buses"] | Float[np.ndarray, " n_buses n_timesteps"],
]:
    """Aggregate per-injection powers back to bus totals."""
    injection_to_bus_arr = injection_to_bus
    injection_active_power_arr = injection_active_power
    injection_reactive_power_arr = injection_reactive_power
    injection_connected_arr = injection_connected

    if injection_to_bus_arr.ndim == 1:
        bus_active_power = np.zeros(n_buses, dtype=injection_active_power_arr.dtype)
        bus_reactive_power = np.zeros(n_buses, dtype=injection_reactive_power_arr.dtype)
        connected_mask = injection_connected_arr.astype(bool)
        np.add.at(bus_active_power, injection_to_bus_arr[connected_mask], injection_active_power_arr[connected_mask])
        np.add.at(bus_reactive_power, injection_to_bus_arr[connected_mask], injection_reactive_power_arr[connected_mask])
        return bus_active_power, bus_reactive_power

    n_timesteps = injection_to_bus_arr.shape[1]
    bus_active_power = np.zeros((n_buses, n_timesteps), dtype=injection_active_power_arr.dtype)
    bus_reactive_power = np.zeros((n_buses, n_timesteps), dtype=injection_reactive_power_arr.dtype)

    for timestep in range(n_timesteps):
        connected_mask = injection_connected_arr[:, timestep].astype(bool)
        np.add.at(
            bus_active_power[:, timestep],
            injection_to_bus_arr[connected_mask, timestep],
            injection_active_power_arr[connected_mask, timestep],
        )
        np.add.at(
            bus_reactive_power[:, timestep],
            injection_to_bus_arr[connected_mask, timestep],
            injection_reactive_power_arr[connected_mask, timestep],
        )

    return bus_active_power, bus_reactive_power
