# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Shared JAX pytree inputs for solver topology, indices, and state data."""

import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Bool, Complex128, Float, Int


@pytree_dataclass
class JacobianComponentInputs:
    """Bus-to-Jacobian component mappings shared across JAX contingency kernels."""

    angle_component_indices: Int[Array, " n_buses"]
    magnitude_component_indices: Int[Array, " n_buses"]


@pytree_dataclass
class VoltageStateInputs:
    """Voltage-only bus state inputs shared across JAX contingency kernels."""

    bus_voltage_magnitudes: Float[Array, "*batch n_buses"]
    bus_voltage_angles_rad: Float[Array, "*batch n_buses"]


@pytree_dataclass
class BusStateInputs:
    """Bus-level power, type, and voltage state inputs shared across JAX solvers."""

    bus_active_power: Float[Array, "*batch n_buses"]
    bus_reactive_power: Float[Array, "*batch n_buses"]
    bus_type: Int[Array, "*batch n_buses"]
    bus_voltage_magnitude_setpoint: Float[Array, "*batch n_buses"]
    bus_voltage_magnitudes: Float[Array, "*batch n_buses"]
    bus_voltage_angles_rad: Float[Array, "*batch n_buses"]


@pytree_dataclass
class NetworkTopologyInputs:
    """Branch and shunt connectivity inputs shared across JAX solvers."""

    branch_from: Int[jnp.ndarray, " n_branches"]
    branch_to: Int[jnp.ndarray, " n_branches"]
    branch_connected: Bool[jnp.ndarray, " *batch n_branches"]
    shunt_to_bus: Int[jnp.ndarray, " n_shunts"]
    shunt_connected: Bool[jnp.ndarray, " *batch n_shunts"]


@pytree_dataclass
class NetworkAdmittanceInputs:
    """Branch and shunt admittance inputs shared across JAX solvers."""

    y_ff: Complex128[jnp.ndarray, " n_branches"]
    y_ft: Complex128[jnp.ndarray, " n_branches"]
    y_tf: Complex128[jnp.ndarray, " n_branches"]
    y_tt: Complex128[jnp.ndarray, " n_branches"]
    y_shunt: Complex128[jnp.ndarray, " n_shunts"]
