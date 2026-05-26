# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helpers for arbitrary injection-power changes without topology changes."""

import numpy as np
from jaxtyping import Bool, Float, Int

from dc_plus.interfaces.network_information import DynamicNetworkInformation, replace_network_data

from .injection_helpers import _aggregate_bus_injections


def with_updated_injection_powers(
    dynamic_network_data: DynamicNetworkInformation,
    injection_active_power: Float[np.ndarray, " n_injections"] | Float[np.ndarray, " n_injections n_timesteps"],
    injection_reactive_power: Float[np.ndarray, " n_injections"] | Float[np.ndarray, " n_injections n_timesteps"],
    injection_connected: Bool[np.ndarray, " n_injections"] | Bool[np.ndarray, " n_injections n_timesteps"] | None = None,
    injection_to_bus: Int[np.ndarray, " n_injections"] | Int[np.ndarray, " n_injections n_timesteps"] | None = None,
) -> DynamicNetworkInformation:
    """Return a copy of the network data with updated injection powers.

    This helper is meant for operating-point changes such as time-series load and
    generator scaling where topology and branch admittances stay unchanged but the
    net bus injections must be re-aggregated before running a voltage update.
    """
    updated_injection_active_power = injection_active_power
    updated_injection_reactive_power = injection_reactive_power

    if updated_injection_active_power.shape != dynamic_network_data.injection_active_power.shape:
        raise ValueError("Active-power array shape must match dynamic_network_data.injection_active_power.")
    if updated_injection_reactive_power.shape != dynamic_network_data.injection_reactive_power.shape:
        raise ValueError("Reactive-power array shape must match dynamic_network_data.injection_reactive_power.")

    updated_injection_connected = (
        dynamic_network_data.injection_connected if injection_connected is None else injection_connected
    )
    updated_injection_to_bus = dynamic_network_data.injection_to_bus if injection_to_bus is None else injection_to_bus

    if updated_injection_connected.shape != dynamic_network_data.injection_connected.shape:
        raise ValueError("Connected-status array shape must match dynamic_network_data.injection_connected.")
    if updated_injection_to_bus.shape != dynamic_network_data.injection_to_bus.shape:
        raise ValueError("Injection-to-bus array shape must match dynamic_network_data.injection_to_bus.")

    updated_bus_active_power, updated_bus_reactive_power = _aggregate_bus_injections(
        injection_to_bus=updated_injection_to_bus,
        injection_active_power=updated_injection_active_power,
        injection_reactive_power=updated_injection_reactive_power,
        injection_connected=updated_injection_connected,
        n_buses=dynamic_network_data.n_buses,
    )

    return replace_network_data(
        dynamic_network_data,
        injection_to_bus=updated_injection_to_bus,
        injection_active_power=updated_injection_active_power,
        injection_reactive_power=updated_injection_reactive_power,
        injection_connected=updated_injection_connected,
        bus_active_power=updated_bus_active_power,
        bus_reactive_power=updated_bus_reactive_power,
    )
