# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""One-step fixed-Jacobian state updates for injection outages and injection changes using JAX."""

import jax
import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Complex128, Float, Int

from ..interfaces.network_inputs import (
    JacobianComponentInputs,
    NetworkAdmittanceInputs,
    NetworkTopologyInputs,
    VoltageStateInputs,
)
from .lodf_branches import _compute_monitored_branch_currents, _prepare_monitored_branch_pack
from .lodf_voltages import build_monitor_rows
from .network_state_helper import _calculate_branch_complex_power

# ruff: noqa: PLR0913


@pytree_dataclass
class InjectionOutageMonitoredResults:
    """One-step post-contingency results restricted to monitored elements."""

    n_1_theta: Float[Array, "... n_contingencies n_buses_monitored"]
    n_1_voltage: Float[Array, "... n_contingencies n_buses_monitored"]

    n_1_p_from: Float[Array, "... n_contingencies n_branches_monitored"]
    n_1_p_to: Float[Array, "... n_contingencies n_branches_monitored"]
    n_1_q_from: Float[Array, "... n_contingencies n_branches_monitored"]
    n_1_q_to: Float[Array, "... n_contingencies n_branches_monitored"]
    n_1_i_from: Complex128[Array, "... n_contingencies n_branches_monitored"]
    n_1_i_to: Complex128[Array, "... n_contingencies n_branches_monitored"]


def _collect_single_injection_outage_mismatch(
    outage_injection_indices: Int[jnp.ndarray, " n_outages"],
    jacobian_size: int,
    injection_to_bus: Int[jnp.ndarray, " n_injections"],
    injection_active_power: Float[jnp.ndarray, " n_injections"],
    injection_reactive_power: Float[jnp.ndarray, " n_injections"],
    jacobian_components: JacobianComponentInputs,
    dtype: jnp.dtype,
) -> Float[jnp.ndarray, " n_eq"]:
    """Assemble the outage-induced mismatch vector for one contingency.

    Parameters
    ----------
    outage_injection_indices : Int[jnp.ndarray, " n_outages"]
            Indices of injections to disconnect for one contingency. Negative entries
            are ignored and can be used as padding.
    jacobian_size : int
            Number of equations in the Jacobian system.
    injection_to_bus : Int[jnp.ndarray, " n_injections"]
            Map from injection index to hosting bus index.
    injection_active_power : Float[jnp.ndarray, " n_injections"]
            Active power associated with each injection.
    injection_reactive_power : Float[jnp.ndarray, " n_injections"]
            Reactive power associated with each injection.
    jacobian_components : JacobianComponentInputs
            Bus-to-Jacobian component mappings for the base operating point.
    dtype : jnp.dtype
            Target dtype of the assembled mismatch vector.

    Returns
    -------
    Float[jnp.ndarray, " n_eq"]
            Mismatch vector in Jacobian component ordering.
    """
    valid_outage_mask = outage_injection_indices >= 0
    safe_outages = jnp.where(valid_outage_mask, outage_injection_indices, 0)
    buses = injection_to_bus[safe_outages]

    mismatch = jnp.zeros((jacobian_size,), dtype=dtype)

    p_idx = jacobian_components.angle_component_indices[buses]
    valid_p = valid_outage_mask & (p_idx >= 0)
    safe_p_idx = jnp.where(valid_p, p_idx, 0)
    p_values = jnp.where(valid_p, -injection_active_power[safe_outages], 0.0)
    mismatch = mismatch.at[safe_p_idx].add(p_values)

    q_idx = jacobian_components.magnitude_component_indices[buses]
    valid_q = valid_outage_mask & (q_idx >= 0)
    safe_q_idx = jnp.where(valid_q, q_idx, 0)
    q_values = jnp.where(valid_q, -injection_reactive_power[safe_outages], 0.0)
    mismatch = mismatch.at[safe_q_idx].add(q_values)

    return mismatch


def _collect_injection_outage_mismatch(
    jacobian_size: int,
    outage_injection_indices: Int[jnp.ndarray, " n_contingencies n_outages"],
    injection_to_bus: Int[jnp.ndarray, " n_injections"],
    injection_active_power: Float[jnp.ndarray, " n_injections"],
    injection_reactive_power: Float[jnp.ndarray, " n_injections"],
    jacobian_components: JacobianComponentInputs,
    dtype: jnp.dtype,
) -> Float[jnp.ndarray, " n_contingencies n_eq"]:
    """Assemble outage-induced mismatch vectors in Jacobian ordering.

    Parameters
    ----------
    jacobian_size : int
            Number of equations in the Jacobian system.
    outage_injection_indices : Int[jnp.ndarray, " n_contingencies n_outages"]
            Batched injection outage indices. Each row represents one contingency and
            may include negative padding entries.
    injection_to_bus : Int[jnp.ndarray, " n_injections"]
            Map from injection index to hosting bus index.
    injection_active_power : Float[jnp.ndarray, " n_injections"]
            Active power associated with each injection.
    injection_reactive_power : Float[jnp.ndarray, " n_injections"]
            Reactive power associated with each injection.
    jacobian_components : JacobianComponentInputs
            Bus-to-Jacobian component mappings for the base operating point.
    dtype : jnp.dtype
            Target dtype of the assembled mismatch matrix.

    Returns
    -------
    Float[jnp.ndarray, " n_contingencies n_eq"]
            Batched mismatch vectors in Jacobian component ordering.
    """

    def collect_single(outage_row: Int[jnp.ndarray, " n_outages"]) -> Float[jnp.ndarray, " n_eq"]:
        """Assemble the mismatch vector for a single contingency."""
        return _collect_single_injection_outage_mismatch(
            outage_injection_indices=outage_row,
            jacobian_size=jacobian_size,
            injection_to_bus=injection_to_bus,
            injection_active_power=injection_active_power,
            injection_reactive_power=injection_reactive_power,
            jacobian_components=jacobian_components,
            dtype=dtype,
        )

    return jax.vmap(collect_single)(outage_injection_indices)


@jax.jit
def _calculate_fixed_jacobian_dx(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    mismatch: Float[jnp.ndarray, " ... n_eq"],
) -> Float[jnp.ndarray, " ... n_eq"]:
    """Map one or more mismatch vectors to fixed-Jacobian state increments.

    Parameters
    ----------
    jacobian_inv_transposed : Float[jnp.ndarray, " n_eq n_eq"]
            Transposed inverse Jacobian at the base operating point.
    mismatch : Float[jnp.ndarray, " ... n_eq"]
            One or more mismatch vectors in Jacobian ordering.

    Returns
    -------
    Float[jnp.ndarray, " ... n_eq"]
            Fixed-Jacobian state increments with the same leading batch shape as ``mismatch``.
    """
    return -(mismatch @ jacobian_inv_transposed)


def _collect_single_injection_change_mismatch(
    injection_active_power_change: Float[jnp.ndarray, " n_injections"],
    injection_reactive_power_change: Float[jnp.ndarray, " n_injections"],
    jacobian_size: int,
    injection_to_bus: Int[jnp.ndarray, " n_injections"],
    jacobian_components: JacobianComponentInputs,
    dtype: jnp.dtype,
) -> Float[jnp.ndarray, " n_eq"]:
    """Assemble one arbitrary injection-change mismatch vector."""
    buses = injection_to_bus
    mismatch = jnp.zeros((jacobian_size,), dtype=dtype)

    p_idx = jacobian_components.angle_component_indices[buses]
    valid_p = p_idx >= 0
    safe_p_idx = jnp.where(valid_p, p_idx, 0)
    p_values = jnp.where(valid_p, injection_active_power_change, 0.0)
    mismatch = mismatch.at[safe_p_idx].add(p_values)

    q_idx = jacobian_components.magnitude_component_indices[buses]
    valid_q = q_idx >= 0
    safe_q_idx = jnp.where(valid_q, q_idx, 0)
    q_values = jnp.where(valid_q, injection_reactive_power_change, 0.0)
    mismatch = mismatch.at[safe_q_idx].add(q_values)

    return mismatch


def _collect_injection_change_mismatch(
    jacobian_size: int,
    injection_active_power_changes: Float[jnp.ndarray, " n_timesteps n_injections"],
    injection_reactive_power_changes: Float[jnp.ndarray, " n_timesteps n_injections"],
    injection_to_bus: Int[jnp.ndarray, " n_injections"],
    jacobian_components: JacobianComponentInputs,
    dtype: jnp.dtype,
) -> Float[jnp.ndarray, " n_timesteps n_eq"]:
    """Assemble arbitrary injection-change mismatch vectors for a full time series."""

    def collect_single(
        injection_active_power_change: Float[jnp.ndarray, " n_injections"],
        injection_reactive_power_change: Float[jnp.ndarray, " n_injections"],
    ) -> Float[jnp.ndarray, " n_eq"]:
        """Assemble the mismatch vector for a single time step."""
        return _collect_single_injection_change_mismatch(
            injection_active_power_change=injection_active_power_change,
            injection_reactive_power_change=injection_reactive_power_change,
            jacobian_size=jacobian_size,
            injection_to_bus=injection_to_bus,
            jacobian_components=jacobian_components,
            dtype=dtype,
        )

    return jax.vmap(collect_single)(injection_active_power_changes, injection_reactive_power_changes)


@jax.jit
def _calculate_monitor_bus_state_updates(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    mismatch: Float[jnp.ndarray, " n_contingencies n_eq"],
    theta_rows: Int[jnp.ndarray, " n_mon_bus"],
    vm_rows: Int[jnp.ndarray, " n_mon_bus"],
    theta_mask: Float[jnp.ndarray, " n_mon_bus"],
    vm_mask: Float[jnp.ndarray, " n_mon_bus"],
) -> tuple[
    Float[jnp.ndarray, " n_contingencies n_mon_bus"],
    Float[jnp.ndarray, " n_contingencies n_mon_bus"],
]:
    """Map outage mismatch vectors to monitored bus state increments only.

    Parameters
    ----------
    jacobian_inv_transposed : Float[jnp.ndarray, " n_eq n_eq"]
            Transposed inverse Jacobian at the base operating point.
    mismatch : Float[jnp.ndarray, " n_contingencies n_eq"]
            Batched outage-induced mismatch vectors.
    theta_rows : Int[jnp.ndarray, " n_mon_bus"]
            Safe Jacobian row indices for monitored bus angles.
    vm_rows : Int[jnp.ndarray, " n_mon_bus"]
            Safe Jacobian row indices for monitored bus magnitudes.
    theta_mask : Float[jnp.ndarray, " n_mon_bus"]
            Mask for monitored buses with angle states in the Jacobian.
    vm_mask : Float[jnp.ndarray, " n_mon_bus"]
            Mask for monitored buses with magnitude states in the Jacobian.

    Returns
    -------
    tuple[Float[jnp.ndarray, " n_contingencies n_mon_bus"], Float[jnp.ndarray, " n_contingencies n_mon_bus"]]
            Monitored angle and magnitude increments for each contingency.
    """
    dtype = jacobian_inv_transposed.dtype
    theta_mask_d = theta_mask.astype(dtype)
    vm_mask_d = vm_mask.astype(dtype)
    theta_dx = (
        _calculate_fixed_jacobian_dx(
            jacobian_inv_transposed=jacobian_inv_transposed[:, theta_rows],
            mismatch=mismatch,
        )
        * theta_mask_d[None, :]
    )
    vm_dx = (
        _calculate_fixed_jacobian_dx(
            jacobian_inv_transposed=jacobian_inv_transposed[:, vm_rows],
            mismatch=mismatch,
        )
        * vm_mask_d[None, :]
    )
    return theta_dx, vm_dx


def non_voltage_regulating_injection_outage_monitor_buses(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    outage_injection_indices: Int[jnp.ndarray, " n_contingencies n_outages"],
    injection_to_bus: Int[jnp.ndarray, " n_injections"],
    injection_active_power: Float[jnp.ndarray, " n_injections"],
    injection_reactive_power: Float[jnp.ndarray, " n_injections"],
    jacobian_components: JacobianComponentInputs,
    monitor_bus_indices: Int[jnp.ndarray, " n_mon_bus"],
    voltage_state: VoltageStateInputs,
) -> tuple[
    Float[jnp.ndarray, " n_contingencies n_mon_bus"],
    Float[jnp.ndarray, " n_contingencies n_mon_bus"],
]:
    """Compute monitored post-contingency bus voltages for injection outages only.

    Parameters
    ----------
    jacobian_inv_transposed : Float[jnp.ndarray, " n_eq n_eq"]
            Base transposed inverse Jacobian at the hot-start operating point.
    outage_injection_indices : Int[jnp.ndarray, " n_contingencies n_outages"]
            Batched indices of injections to disconnect. Each row is one contingency.
    injection_to_bus : Int[jnp.ndarray, " n_injections"]
            Map from injection index to hosting bus index.
    injection_active_power : Float[jnp.ndarray, " n_injections"]
            Active power of each injection at the hot-start operating point.
    injection_reactive_power : Float[jnp.ndarray, " n_injections"]
            Reactive power of each injection at the hot-start operating point.
    jacobian_components : JacobianComponentInputs
            Bus-to-Jacobian component mappings for the base operating point.
    monitor_bus_indices : Int[jnp.ndarray, " n_mon_bus"]
            Indices of buses whose post-contingency voltages should be returned.
    voltage_state : VoltageStateInputs
            Base-case voltage magnitudes and angles for all buses.

    Returns
    -------
    tuple[Float[jnp.ndarray, " n_contingencies n_mon_bus"], Float[jnp.ndarray, " n_contingencies n_mon_bus"]]
            Post-contingency monitored voltage angles and magnitudes.
    """
    dtype = jacobian_inv_transposed.dtype

    mismatch = _collect_injection_outage_mismatch(
        jacobian_size=jacobian_inv_transposed.shape[0],
        outage_injection_indices=outage_injection_indices,
        injection_to_bus=injection_to_bus,
        injection_active_power=injection_active_power,
        injection_reactive_power=injection_reactive_power,
        jacobian_components=jacobian_components,
        dtype=dtype,
    )
    theta_rows, vm_rows, theta_mask, vm_mask = build_monitor_rows(
        jacobian_components=jacobian_components,
        monitor_bus_indices=monitor_bus_indices,
    )
    theta_dx, vm_dx = _calculate_monitor_bus_state_updates(
        jacobian_inv_transposed=jacobian_inv_transposed,
        mismatch=mismatch,
        theta_rows=theta_rows,
        vm_rows=vm_rows,
        theta_mask=theta_mask,
        vm_mask=vm_mask,
    )
    base_theta0 = jnp.take(voltage_state.bus_voltage_angles_rad, monitor_bus_indices, axis=0).astype(dtype)
    base_vm0 = jnp.take(voltage_state.bus_voltage_magnitudes, monitor_bus_indices, axis=0).astype(dtype)
    return base_theta0[None, :] + theta_dx, base_vm0[None, :] + vm_dx


def non_voltage_regulating_injection_outage_monitored(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    outage_injection_indices: Int[jnp.ndarray, " n_contingencies n_outages"],
    injection_to_bus: Int[jnp.ndarray, " n_injections"],
    injection_active_power: Float[jnp.ndarray, " n_injections"],
    injection_reactive_power: Float[jnp.ndarray, " n_injections"],
    jacobian_components: JacobianComponentInputs,
    monitor_bus_indices: Int[jnp.ndarray, " n_mon_bus"],
    voltage_state: VoltageStateInputs,
    network_topology: NetworkTopologyInputs,
    network_admittance: NetworkAdmittanceInputs,
    monitor_branch_indices: Int[jnp.ndarray, " n_mon_br"],
    bus_to_mon_index: Int[jnp.ndarray, " n_buses"],
) -> InjectionOutageMonitoredResults:
    """Compute monitored bus states and branch flows for injection outages.

    Parameters
    ----------
    jacobian_inv_transposed : Float[jnp.ndarray, " n_eq n_eq"]
            Base transposed inverse Jacobian at the hot-start operating point.
    outage_injection_indices : Int[jnp.ndarray, " n_contingencies n_outages"]
            Batched indices of injections to disconnect. Each row is one contingency.
    injection_to_bus : Int[jnp.ndarray, " n_injections"]
            Map from injection index to hosting bus index.
    injection_active_power : Float[jnp.ndarray, " n_injections"]
            Active power of each injection at the hot-start operating point.
    injection_reactive_power : Float[jnp.ndarray, " n_injections"]
            Reactive power of each injection at the hot-start operating point.
    jacobian_components : JacobianComponentInputs
            Bus-to-Jacobian component mappings for the base operating point.
    monitor_bus_indices : Int[jnp.ndarray, " n_mon_bus"]
            Indices of buses whose post-contingency voltages should be returned.
    voltage_state : VoltageStateInputs
            Base-case voltage magnitudes and angles for all buses.
    network_topology : NetworkTopologyInputs
            Network topology inputs for the base operating point.
    network_admittance : NetworkAdmittanceInputs
            Network admittance inputs for the base operating point.
    monitor_branch_indices : Int[jnp.ndarray, " n_mon_br"]
            Indices of branches whose post-contingency flows should be returned.
    bus_to_mon_index : Int[jnp.ndarray, " n_buses"]
            Map from bus index to monitored bus index. -1 if the bus is not monitored.

    Returns
    -------
    InjectionOutageMonitoredResults
            Post-contingency monitored bus states and monitored branch flows.
    """
    dtype = jacobian_inv_transposed.dtype

    theta_all, vm_all = non_voltage_regulating_injection_outage_monitor_buses(
        jacobian_inv_transposed=jacobian_inv_transposed,
        outage_injection_indices=outage_injection_indices,
        injection_to_bus=injection_to_bus,
        injection_active_power=injection_active_power,
        injection_reactive_power=injection_reactive_power,
        jacobian_components=jacobian_components,
        monitor_bus_indices=monitor_bus_indices,
        voltage_state=voltage_state,
    )
    (
        _,
        y_ff_mon,
        y_ft_mon,
        y_tf_mon,
        y_tt_mon,
        f_pos_safe,
        t_pos_safe,
        end_mask,
    ) = _prepare_monitored_branch_pack(
        network_topology=network_topology,
        network_admittance=network_admittance,
        monitor_branch_indices=monitor_branch_indices,
        bus_to_mon_index=bus_to_mon_index,
        dtype=dtype,
    )
    v_from_all, v_to_all, i_from_all, i_to_all = _compute_monitored_branch_currents(
        theta_all=theta_all,
        vm_all=vm_all,
        f_pos_safe=f_pos_safe,
        t_pos_safe=t_pos_safe,
        y_ff_mon=y_ff_mon,
        y_ft_mon=y_ft_mon,
        y_tf_mon=y_tf_mon,
        y_tt_mon=y_tt_mon,
        end_mask=end_mask,
    )
    complex_dtype = y_ff_mon.dtype
    end_mask_complex = end_mask.astype(complex_dtype)[None, :]
    end_mask_real = end_mask.astype(dtype)[None, :]
    s_from_all, s_to_all = _calculate_branch_complex_power(
        v_from=v_from_all,
        v_to=v_to_all,
        current_from=i_from_all,
        current_to=i_to_all,
        branch_mask=end_mask,
    )
    return InjectionOutageMonitoredResults(
        n_1_theta=theta_all,
        n_1_voltage=vm_all,
        n_1_p_from=s_from_all.real.astype(dtype) * end_mask_real,
        n_1_p_to=s_to_all.real.astype(dtype) * end_mask_real,
        n_1_q_from=s_from_all.imag.astype(dtype) * end_mask_real,
        n_1_q_to=s_to_all.imag.astype(dtype) * end_mask_real,
        n_1_i_from=i_from_all * end_mask_complex,
        n_1_i_to=i_to_all * end_mask_complex,
    )


def non_voltage_regulating_injection_outage_dx(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    outage_injection_indices: Int[jnp.ndarray, " n_contingencies n_outages"],
    injection_to_bus: Int[jnp.ndarray, " n_injections"],
    injection_active_power: Float[jnp.ndarray, " n_injections"],
    injection_reactive_power: Float[jnp.ndarray, " n_injections"],
    jacobian_components: JacobianComponentInputs,
) -> Float[jnp.ndarray, " n_contingencies n_eq"]:
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
    jacobian_inv_transposed : Float[jnp.ndarray, " n_eq n_eq"]
            Base transposed inverse Jacobian at the hot-start operating point.
    outage_injection_indices : Int[jnp.ndarray, " n_contingencies n_outages"]
            Batched indices of injections to disconnect. Each row is one contingency.
    injection_to_bus : Int[jnp.ndarray, " n_injections"]
            Map from injection index to hosting bus index.
    injection_active_power : Float[jnp.ndarray, " n_injections"]
            Active power of each injection at the hot-start operating point.
    injection_reactive_power : Float[jnp.ndarray, " n_injections"]
            Reactive power of each injection at the hot-start operating point.
    jacobian_components : JacobianComponentInputs
            Bus-to-Jacobian component mappings for the base operating point.

    Returns
    -------
    Float[jnp.ndarray, " n_contingencies n_eq"]
            One-step state increment for each outage in Jacobian ordering.
    """
    dtype = jacobian_inv_transposed.dtype
    mismatch = _collect_injection_outage_mismatch(
        jacobian_size=jacobian_inv_transposed.shape[0],
        outage_injection_indices=outage_injection_indices,
        injection_to_bus=injection_to_bus,
        injection_active_power=injection_active_power,
        injection_reactive_power=injection_reactive_power,
        jacobian_components=jacobian_components,
        dtype=dtype,
    )
    return _calculate_fixed_jacobian_dx(jacobian_inv_transposed=jacobian_inv_transposed, mismatch=mismatch)


def non_voltage_regulating_injection_changes_dx(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    injection_active_power_changes: Float[jnp.ndarray, " n_timesteps n_injections"],
    injection_reactive_power_changes: Float[jnp.ndarray, " n_timesteps n_injections"],
    injection_to_bus: Int[jnp.ndarray, " n_injections"],
    jacobian_components: JacobianComponentInputs,
) -> Float[jnp.ndarray, " n_timesteps n_eq"]:
    """One-step fixed-Jacobian state increments for batched injection changes.

    Each row in ``injection_active_power_changes`` and ``injection_reactive_power_changes``
    represents one time step or scenario. All rows are assembled into mismatch vectors and
    solved in parallel through the same fixed inverse Jacobian:

        ``dx = -(mismatch @ jacobian_inv_transposed)``

    The injection deltas must use the same sign convention as the imported
    dynamic network data: loads are positive and generators are negative.

    Parameters
    ----------
    jacobian_inv_transposed : Float[jnp.ndarray, " n_eq n_eq"]
            Base transposed inverse Jacobian at the hot-start operating point.
    injection_active_power_changes : Float[jnp.ndarray, " n_timesteps n_injections"]
            Batched active-power deltas for each injection.
    injection_reactive_power_changes : Float[jnp.ndarray, " n_timesteps n_injections"]
            Batched reactive-power deltas for each injection.
    injection_to_bus : Int[jnp.ndarray, " n_injections"]
            Map from injection index to hosting bus index.
    jacobian_components : JacobianComponentInputs
            Bus-to-Jacobian component mappings for the base operating point.

    Returns
    -------
    Float[jnp.ndarray, " n_timesteps n_eq"]
            Batched one-step state increments in Jacobian ordering.
    """
    dtype = jacobian_inv_transposed.dtype
    mismatch = _collect_injection_change_mismatch(
        jacobian_size=jacobian_inv_transposed.shape[0],
        injection_active_power_changes=injection_active_power_changes,
        injection_reactive_power_changes=injection_reactive_power_changes,
        injection_to_bus=injection_to_bus,
        jacobian_components=jacobian_components,
        dtype=dtype,
    )
    return _calculate_fixed_jacobian_dx(jacobian_inv_transposed=jacobian_inv_transposed, mismatch=mismatch)
