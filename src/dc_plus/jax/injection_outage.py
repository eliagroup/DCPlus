# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""One-step state updates for injection outages using JAX."""

import jax
import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Complex128, Float, Int

from .lodf_branches import _compute_monitored_branch_currents, _prepare_monitored_branch_pack
from .lodf_voltages import build_monitor_rows

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
    angle_component_indices: Int[jnp.ndarray, " n_buses"],
    magnitude_component_indices: Int[jnp.ndarray, " n_buses"],
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
    angle_component_indices : Int[jnp.ndarray, " n_buses"]
            Map from bus index to active-power mismatch component index.
    magnitude_component_indices : Int[jnp.ndarray, " n_buses"]
            Map from bus index to reactive-power mismatch component index.
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

    p_idx = angle_component_indices[buses]
    valid_p = valid_outage_mask & (p_idx >= 0)
    safe_p_idx = jnp.where(valid_p, p_idx, 0)
    p_values = jnp.where(valid_p, -injection_active_power[safe_outages], 0.0)
    mismatch = mismatch.at[safe_p_idx].add(p_values)

    q_idx = magnitude_component_indices[buses]
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
    angle_component_indices: Int[jnp.ndarray, " n_buses"],
    magnitude_component_indices: Int[jnp.ndarray, " n_buses"],
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
    angle_component_indices : Int[jnp.ndarray, " n_buses"]
            Map from bus index to active-power mismatch component index.
    magnitude_component_indices : Int[jnp.ndarray, " n_buses"]
            Map from bus index to reactive-power mismatch component index.
    dtype : jnp.dtype
            Target dtype of the assembled mismatch matrix.

    Returns
    -------
    Float[jnp.ndarray, " n_contingencies n_eq"]
            Batched mismatch vectors in Jacobian component ordering.
    """

    def collect_single(outage_row: Int[jnp.ndarray, " n_outages"]) -> Float[jnp.ndarray, " n_eq"]:
        return _collect_single_injection_outage_mismatch(
            outage_injection_indices=outage_row,
            jacobian_size=jacobian_size,
            injection_to_bus=injection_to_bus,
            injection_active_power=injection_active_power,
            injection_reactive_power=injection_reactive_power,
            angle_component_indices=angle_component_indices,
            magnitude_component_indices=magnitude_component_indices,
            dtype=dtype,
        )

    return jax.vmap(collect_single)(outage_injection_indices)


@jax.jit
def _calculate_injection_outage_dx(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    mismatch: Float[jnp.ndarray, " n_contingencies n_eq"],
) -> Float[jnp.ndarray, " n_contingencies n_eq"]:
    """Map outage mismatch vectors to one-step state increments.

    Parameters
    ----------
    jacobian_inv_transposed : Float[jnp.ndarray, " n_eq n_eq"]
            Transposed inverse Jacobian at the base operating point.
    mismatch : Float[jnp.ndarray, " n_contingencies n_eq"]
            Batched outage-induced mismatch vectors.

    Returns
    -------
    Float[jnp.ndarray, " n_contingencies n_eq"]
            Batched one-step state increments in Jacobian ordering.
    """
    return -(mismatch @ jacobian_inv_transposed)


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
    theta_dx = -(mismatch @ jacobian_inv_transposed[:, theta_rows]) * theta_mask_d[None, :]
    vm_dx = -(mismatch @ jacobian_inv_transposed[:, vm_rows]) * vm_mask_d[None, :]
    return theta_dx, vm_dx


def non_voltage_regulating_injection_outage_monitor_buses(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    outage_injection_indices: Int[jnp.ndarray, " n_contingencies n_outages"],
    injection_to_bus: Int[jnp.ndarray, " n_injections"],
    injection_active_power: Float[jnp.ndarray, " n_injections"],
    injection_reactive_power: Float[jnp.ndarray, " n_injections"],
    angle_component_indices: Int[jnp.ndarray, " n_buses"],
    magnitude_component_indices: Int[jnp.ndarray, " n_buses"],
    monitor_bus_indices: Int[jnp.ndarray, " n_mon_bus"],
    v_mag_hat: Float[jnp.ndarray, " n_buses"],
    theta_hat: Float[jnp.ndarray, " n_buses"],
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
    angle_component_indices : Int[jnp.ndarray, " n_buses"]
            Map from bus index to angle component index in the Jacobian. -1 if not present.
    magnitude_component_indices : Int[jnp.ndarray, " n_buses"]
            Map from bus index to magnitude component index in the Jacobian. -1 if not present.
    monitor_bus_indices : Int[jnp.ndarray, " n_mon_bus"]
            Indices of buses whose post-contingency voltages should be returned.
    v_mag_hat : Float[jnp.ndarray, " n_buses"]
            Base-case voltage magnitudes.
    theta_hat : Float[jnp.ndarray, " n_buses"]
            Base-case voltage angles.

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
        angle_component_indices=angle_component_indices,
        magnitude_component_indices=magnitude_component_indices,
        dtype=dtype,
    )
    theta_rows, vm_rows, theta_mask, vm_mask = build_monitor_rows(
        angle_component_indices=angle_component_indices,
        magnitude_component_indices=magnitude_component_indices,
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
    base_theta0 = jnp.take(theta_hat, monitor_bus_indices, axis=0).astype(dtype)
    base_vm0 = jnp.take(v_mag_hat, monitor_bus_indices, axis=0).astype(dtype)
    return base_theta0[None, :] + theta_dx, base_vm0[None, :] + vm_dx


def non_voltage_regulating_injection_outage_monitored(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    outage_injection_indices: Int[jnp.ndarray, " n_contingencies n_outages"],
    injection_to_bus: Int[jnp.ndarray, " n_injections"],
    injection_active_power: Float[jnp.ndarray, " n_injections"],
    injection_reactive_power: Float[jnp.ndarray, " n_injections"],
    angle_component_indices: Int[jnp.ndarray, " n_buses"],
    magnitude_component_indices: Int[jnp.ndarray, " n_buses"],
    monitor_bus_indices: Int[jnp.ndarray, " n_mon_bus"],
    v_mag_hat: Float[jnp.ndarray, " n_buses"],
    theta_hat: Float[jnp.ndarray, " n_buses"],
    branch_from: Int[jnp.ndarray, " n_branches"],
    branch_to: Int[jnp.ndarray, " n_branches"],
    y_ff: Complex128[jnp.ndarray, " n_branches"],
    y_ft: Complex128[jnp.ndarray, " n_branches"],
    y_tf: Complex128[jnp.ndarray, " n_branches"],
    y_tt: Complex128[jnp.ndarray, " n_branches"],
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
    angle_component_indices : Int[jnp.ndarray, " n_buses"]
            Map from bus index to angle component index in the Jacobian. -1 if not present.
    magnitude_component_indices : Int[jnp.ndarray, " n_buses"]
            Map from bus index to magnitude component index in the Jacobian. -1 if not present.
    monitor_bus_indices : Int[jnp.ndarray, " n_mon_bus"]
            Indices of buses whose post-contingency voltages should be returned.
    v_mag_hat : Float[jnp.ndarray, " n_buses"]
            Base-case voltage magnitudes.
    theta_hat : Float[jnp.ndarray, " n_buses"]
            Base-case voltage angles.
    branch_from : Int[jnp.ndarray, " n_branches"]
            From bus index of each branch.
    branch_to : Int[jnp.ndarray, " n_branches"]
            To bus index of each branch.
    y_ff : Complex128[jnp.ndarray, " n_branches"]
            From-from branch admittances.
    y_ft : Complex128[jnp.ndarray, " n_branches"]
            From-to branch admittances.
    y_tf : Complex128[jnp.ndarray, " n_branches"]
            To-from branch admittances.
    y_tt : Complex128[jnp.ndarray, " n_branches"]
            To-to branch admittances.
    monitor_branch_indices : Int[jnp.ndarray, " n_mon_br"]
            Indices of monitored branches.
    bus_to_mon_index : Int[jnp.ndarray, " n_buses"]
            Mapping from global bus index to monitored-bus position. Use -1 for unmonitored buses.

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
        angle_component_indices=angle_component_indices,
        magnitude_component_indices=magnitude_component_indices,
        monitor_bus_indices=monitor_bus_indices,
        v_mag_hat=v_mag_hat,
        theta_hat=theta_hat,
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
        branch_from=branch_from,
        branch_to=branch_to,
        y_ff=y_ff,
        y_ft=y_ft,
        y_tf=y_tf,
        y_tt=y_tt,
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
        dtype=dtype,
    )
    complex_dtype = y_ff_mon.dtype
    end_mask_complex = end_mask.astype(complex_dtype)[None, :]
    end_mask_real = end_mask.astype(dtype)[None, :]
    s_from_all = v_from_all * jnp.conj(i_from_all) * end_mask_complex
    s_to_all = v_to_all * jnp.conj(i_to_all) * end_mask_complex
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
    angle_component_indices: Int[jnp.ndarray, " n_buses"],
    magnitude_component_indices: Int[jnp.ndarray, " n_buses"],
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
    angle_component_indices : Int[jnp.ndarray, " n_buses"]
            Map from bus index to angle component index in the Jacobian. -1 if not present.
    magnitude_component_indices : Int[jnp.ndarray, " n_buses"]
            Map from bus index to magnitude component index in the Jacobian. -1 if not present.

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
        angle_component_indices=angle_component_indices,
        magnitude_component_indices=magnitude_component_indices,
        dtype=dtype,
    )
    return _calculate_injection_outage_dx(jacobian_inv_transposed=jacobian_inv_transposed, mismatch=mismatch)
