# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Quasi-Newton voltage updates with lazy inverse-Broyden corrections in JAX."""

from functools import partial
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax import lax
from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Bool, Complex128, Float, Int

# ruff: noqa: PLR0913

InverseUpdateResult: TypeAlias = tuple[
    Float[jnp.ndarray, " n_eq n_iterations"],
    Float[jnp.ndarray, " n_iterations n_eq"],
    Float[jnp.ndarray, " n_eq"],
]

QuasiNewtonCarry: TypeAlias = tuple[
    Float[jnp.ndarray, " n_buses"],
    Float[jnp.ndarray, " n_buses"],
    Float[jnp.ndarray, " n_eq"],
    Float[jnp.ndarray, " n_eq"],
    Float[jnp.ndarray, " n_eq n_iterations"],
    Float[jnp.ndarray, " n_iterations n_eq"],
    Float[jnp.ndarray, " n_iterations"],
]


@pytree_dataclass
class QuasiNewtonResults:
    """Quasi-Newton voltage update results."""

    jacobian_inv_transposed: Float[Array, " n_eq n_eq"]
    bus_voltage_angles_rad: Float[Array, " n_buses"]
    bus_voltage_magnitudes: Float[Array, " n_buses"]
    mismatch_history: Float[Array, " n_iterations"]


def _apply_inverse_approximation(
    jacobian_inv: Float[jnp.ndarray, " n_eq n_eq"],
    a_factors: Float[jnp.ndarray, " n_eq n_updates"],
    b_factors: Float[jnp.ndarray, " n_updates n_eq"],
    vector: Float[jnp.ndarray, " n_eq"],
) -> Float[jnp.ndarray, " n_eq"]:
    """Apply the current inverse-Jacobian approximation to one vector."""
    result = jacobian_inv @ vector
    correction_projection = b_factors @ vector
    return result + a_factors @ correction_projection


def _calculate_nodal_mismatch_impl(
    branch_from: Int[jnp.ndarray, " n_branches"],
    branch_to: Int[jnp.ndarray, " n_branches"],
    shunt_to_bus: Int[jnp.ndarray, " n_shunts"],
    v_mag_hat: Float[jnp.ndarray, " n_buses"],
    theta_hat: Float[jnp.ndarray, " n_buses"],
    y_ff: Complex128[jnp.ndarray, " n_branches"],
    y_ft: Complex128[jnp.ndarray, " n_branches"],
    y_tf: Complex128[jnp.ndarray, " n_branches"],
    y_tt: Complex128[jnp.ndarray, " n_branches"],
    y_shunt: Complex128[jnp.ndarray, " n_shunts"],
    branch_mask: Complex128[jnp.ndarray, " n_branches"],
    shunt_mask: Complex128[jnp.ndarray, " n_shunts"],
    specified_power: Complex128[jnp.ndarray, " n_buses"],
    pvpq_indices: Int[jnp.ndarray, " n_pvpq"],
    pq_indices: Int[jnp.ndarray, " n_pq"],
) -> Float[jnp.ndarray, " n_eq"]:
    """Assemble the nodal mismatch in Jacobian ordering from branch and shunt data."""
    complex_dtype = branch_mask.dtype
    one_j = jnp.asarray(1j, dtype=complex_dtype)

    v_complex = v_mag_hat.astype(jnp.result_type(v_mag_hat, theta_hat)) * (jnp.cos(theta_hat) + one_j * jnp.sin(theta_hat))

    v_from = v_complex[branch_from]
    v_to = v_complex[branch_to]

    i_from = (y_ff * v_from + y_ft * v_to) * branch_mask
    i_to = (y_tf * v_from + y_tt * v_to) * branch_mask

    bus_current = jnp.zeros((v_complex.shape[0],), dtype=complex_dtype)
    bus_current = bus_current.at[branch_from].add(i_from)
    bus_current = bus_current.at[branch_to].add(i_to)

    shunt_current = y_shunt * v_complex[shunt_to_bus] * shunt_mask
    bus_current = bus_current.at[shunt_to_bus].add(shunt_current)

    mismatch = v_complex * jnp.conj(bus_current) - specified_power

    return jnp.concatenate([mismatch[pvpq_indices].real, mismatch[pq_indices].imag])


def _calculate_nodal_mismatch(
    branch_from: Int[jnp.ndarray, " n_branches"],
    branch_to: Int[jnp.ndarray, " n_branches"],
    branch_connected: Bool[jnp.ndarray, " n_branches"],
    shunt_to_bus: Int[jnp.ndarray, " n_shunts"],
    shunt_connected: Bool[jnp.ndarray, " n_shunts"],
    bus_active_power: Float[jnp.ndarray, " n_buses"],
    bus_reactive_power: Float[jnp.ndarray, " n_buses"],
    v_mag_hat: Float[jnp.ndarray, " n_buses"],
    theta_hat: Float[jnp.ndarray, " n_buses"],
    y_ff: Complex128[jnp.ndarray, " n_branches"],
    y_ft: Complex128[jnp.ndarray, " n_branches"],
    y_tf: Complex128[jnp.ndarray, " n_branches"],
    y_tt: Complex128[jnp.ndarray, " n_branches"],
    y_shunt: Complex128[jnp.ndarray, " n_shunts"],
    pvpq_indices: Int[jnp.ndarray, " n_pvpq"],
    pq_indices: Int[jnp.ndarray, " n_pq"],
) -> Float[jnp.ndarray, " n_eq"]:
    """Assemble the nodal mismatch in Jacobian ordering from branch and shunt data."""
    complex_dtype = jnp.result_type(y_ff, y_ft, y_tf, y_tt, y_shunt, 1j)
    one_j = jnp.asarray(1j, dtype=complex_dtype)
    return _calculate_nodal_mismatch_impl(
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
        branch_mask=branch_connected.astype(complex_dtype),
        shunt_mask=shunt_connected.astype(complex_dtype),
        specified_power=-(bus_active_power + one_j * bus_reactive_power),
        pvpq_indices=pvpq_indices,
        pq_indices=pq_indices,
    )


def _apply_jacobian_dx(
    bus_voltage_angles_rad: Float[jnp.ndarray, " n_buses"],
    bus_voltage_magnitudes: Float[jnp.ndarray, " n_buses"],
    pvpq_indices: Int[jnp.ndarray, " n_pvpq"],
    pq_indices: Int[jnp.ndarray, " n_pq"],
    dx: Float[jnp.ndarray, " n_eq"],
) -> tuple[Float[jnp.ndarray, " n_buses"], Float[jnp.ndarray, " n_buses"]]:
    """Apply one Jacobian-ordered state increment to bus voltages."""
    n_angle = pvpq_indices.shape[0]
    updated_angles = bus_voltage_angles_rad.at[pvpq_indices].add(dx[:n_angle])
    updated_magnitudes = bus_voltage_magnitudes.at[pq_indices].add(dx[n_angle:])
    return updated_angles, updated_magnitudes


@partial(jax.jit, static_argnames=("n_iterations",))
def _run_quasi_newton_updates(
    jacobian_inv: Float[jnp.ndarray, " n_eq n_eq"],
    pvpq_indices: Int[jnp.ndarray, " n_pvpq"],
    pq_indices: Int[jnp.ndarray, " n_pq"],
    branch_from: Int[jnp.ndarray, " n_branches"],
    branch_to: Int[jnp.ndarray, " n_branches"],
    branch_connected: Bool[jnp.ndarray, " n_branches"],
    shunt_to_bus: Int[jnp.ndarray, " n_shunts"],
    shunt_connected: Bool[jnp.ndarray, " n_shunts"],
    bus_active_power: Float[jnp.ndarray, " n_buses"],
    bus_reactive_power: Float[jnp.ndarray, " n_buses"],
    bus_voltage_magnitudes: Float[jnp.ndarray, " n_buses"],
    bus_voltage_angles_rad: Float[jnp.ndarray, " n_buses"],
    y_ff: Complex128[jnp.ndarray, " n_branches"],
    y_ft: Complex128[jnp.ndarray, " n_branches"],
    y_tf: Complex128[jnp.ndarray, " n_branches"],
    y_tt: Complex128[jnp.ndarray, " n_branches"],
    y_shunt: Complex128[jnp.ndarray, " n_shunts"],
    n_iterations: int,
    regularization: float,
) -> QuasiNewtonResults:
    """Jitted lazy inverse-Broyden quasi-Newton loop."""
    dtype = jacobian_inv.dtype
    n_eq = jacobian_inv.shape[0]
    a_factors = jnp.zeros((n_eq, n_iterations), dtype=dtype)
    b_factors = jnp.zeros((n_iterations, n_eq), dtype=dtype)
    mismatch_history = jnp.zeros((n_iterations,), dtype=dtype)
    complex_dtype = jnp.result_type(y_ff, y_ft, y_tf, y_tt, y_shunt, 1j)
    one_j = jnp.asarray(1j, dtype=complex_dtype)
    branch_mask = branch_connected.astype(complex_dtype)
    shunt_mask = shunt_connected.astype(complex_dtype)
    specified_power = -(bus_active_power + one_j * bus_reactive_power)

    mismatch0 = _calculate_nodal_mismatch_impl(
        branch_from=branch_from,
        branch_to=branch_to,
        shunt_to_bus=shunt_to_bus,
        v_mag_hat=bus_voltage_magnitudes,
        theta_hat=bus_voltage_angles_rad,
        y_ff=y_ff,
        y_ft=y_ft,
        y_tf=y_tf,
        y_tt=y_tt,
        y_shunt=y_shunt,
        branch_mask=branch_mask,
        shunt_mask=shunt_mask,
        specified_power=specified_power,
        pvpq_indices=pvpq_indices,
        pq_indices=pq_indices,
    )
    inverse_mismatch0 = _apply_inverse_approximation(
        jacobian_inv=jacobian_inv,
        a_factors=a_factors,
        b_factors=b_factors,
        vector=mismatch0,
    )

    def _body(
        iteration_index: int,
        carry: QuasiNewtonCarry,
    ) -> QuasiNewtonCarry:
        theta_curr, vm_curr, mismatch_curr, inverse_mismatch_curr, a_curr, b_curr, history_curr = carry

        state_step = -inverse_mismatch_curr
        theta_next, vm_next = _apply_jacobian_dx(
            bus_voltage_angles_rad=theta_curr,
            bus_voltage_magnitudes=vm_curr,
            pvpq_indices=pvpq_indices,
            pq_indices=pq_indices,
            dx=state_step,
        )
        mismatch_next = _calculate_nodal_mismatch_impl(
            branch_from=branch_from,
            branch_to=branch_to,
            shunt_to_bus=shunt_to_bus,
            v_mag_hat=vm_next,
            theta_hat=theta_next,
            y_ff=y_ff,
            y_ft=y_ft,
            y_tf=y_tf,
            y_tt=y_tt,
            y_shunt=y_shunt,
            branch_mask=branch_mask,
            shunt_mask=shunt_mask,
            specified_power=specified_power,
            pvpq_indices=pvpq_indices,
            pq_indices=pq_indices,
        )
        history_curr = history_curr.at[iteration_index].set(jnp.max(jnp.abs(mismatch_next)))

        inverse_mismatch_next_old = _apply_inverse_approximation(
            jacobian_inv=jacobian_inv,
            a_factors=a_curr,
            b_factors=b_curr,
            vector=mismatch_next,
        )
        mismatch_delta = mismatch_next - mismatch_curr
        denominator = jnp.dot(mismatch_delta, mismatch_delta)

        def _accept_update(_: None) -> InverseUpdateResult:
            new_a_factor = -inverse_mismatch_next_old
            new_b_factor = mismatch_delta / denominator.astype(dtype)
            updated_a = a_curr.at[:, iteration_index].set(new_a_factor)
            updated_b = b_curr.at[iteration_index, :].set(new_b_factor)
            correction_scale = jnp.dot(new_b_factor, mismatch_next)
            updated_inverse_mismatch = inverse_mismatch_next_old + new_a_factor * correction_scale
            return updated_a, updated_b, updated_inverse_mismatch

        def _skip_update(_: None) -> InverseUpdateResult:
            return a_curr, b_curr, inverse_mismatch_next_old

        updated_a, updated_b, updated_inverse_mismatch = lax.cond(
            denominator > regularization,
            _accept_update,
            _skip_update,
            operand=None,
        )

        return (
            theta_next,
            vm_next,
            mismatch_next,
            updated_inverse_mismatch,
            updated_a,
            updated_b,
            history_curr,
        )

    theta_final, vm_final, _, _, a_final, b_final, history_final = lax.fori_loop(
        0,
        n_iterations,
        _body,
        (
            bus_voltage_angles_rad,
            bus_voltage_magnitudes,
            mismatch0,
            inverse_mismatch0,
            a_factors,
            b_factors,
            mismatch_history,
        ),
    )

    jacobian_inv_final = jacobian_inv + a_final @ b_final

    return QuasiNewtonResults(
        jacobian_inv_transposed=jacobian_inv_final.T,
        bus_voltage_angles_rad=theta_final,
        bus_voltage_magnitudes=vm_final,
        mismatch_history=history_final,
    )


def run_quasi_newton_updates(
    jacobian_inv_transposed: Float[jnp.ndarray, " n_eq n_eq"],
    pvpq_indices: Int[jnp.ndarray, " n_pvpq"],
    pq_indices: Int[jnp.ndarray, " n_pq"],
    branch_from: Int[jnp.ndarray, " n_branches"],
    branch_to: Int[jnp.ndarray, " n_branches"],
    branch_connected: Bool[jnp.ndarray, " n_branches"],
    shunt_to_bus: Int[jnp.ndarray, " n_shunts"],
    shunt_connected: Bool[jnp.ndarray, " n_shunts"],
    bus_active_power: Float[jnp.ndarray, " n_buses"],
    bus_reactive_power: Float[jnp.ndarray, " n_buses"],
    bus_voltage_magnitudes: Float[jnp.ndarray, " n_buses"],
    bus_voltage_angles_rad: Float[jnp.ndarray, " n_buses"],
    y_ff: Complex128[jnp.ndarray, " n_branches"],
    y_ft: Complex128[jnp.ndarray, " n_branches"],
    y_tf: Complex128[jnp.ndarray, " n_branches"],
    y_tt: Complex128[jnp.ndarray, " n_branches"],
    y_shunt: Complex128[jnp.ndarray, " n_shunts"],
    n_iterations: int = 2,
    regularization: float = 1e-12,
) -> QuasiNewtonResults:
    """Run quasi-Newton steps with lazy inverse-Broyden updates in JAX.

    The implementation mirrors the NumPy reference but stays entirely on device.
    It takes the transposed inverse Jacobian to match the surrounding JAX kernels.
    """
    jacobian_inv_arr = jnp.asarray(jacobian_inv_transposed)
    dtype_jac = jacobian_inv_arr.dtype

    return _run_quasi_newton_updates(
        jacobian_inv=jacobian_inv_arr.T,
        pvpq_indices=jnp.asarray(pvpq_indices, dtype=jnp.int32),
        pq_indices=jnp.asarray(pq_indices, dtype=jnp.int32),
        branch_from=jnp.asarray(branch_from, dtype=jnp.int32),
        branch_to=jnp.asarray(branch_to, dtype=jnp.int32),
        branch_connected=jnp.asarray(branch_connected, dtype=bool),
        shunt_to_bus=jnp.asarray(shunt_to_bus, dtype=jnp.int32),
        shunt_connected=jnp.asarray(shunt_connected, dtype=bool),
        bus_active_power=jnp.asarray(bus_active_power, dtype=dtype_jac),
        bus_reactive_power=jnp.asarray(bus_reactive_power, dtype=dtype_jac),
        bus_voltage_magnitudes=jnp.asarray(bus_voltage_magnitudes, dtype=dtype_jac),
        bus_voltage_angles_rad=jnp.asarray(bus_voltage_angles_rad, dtype=dtype_jac),
        y_ff=jnp.asarray(y_ff),
        y_ft=jnp.asarray(y_ft),
        y_tf=jnp.asarray(y_tf),
        y_tt=jnp.asarray(y_tt),
        y_shunt=jnp.asarray(y_shunt),
        n_iterations=int(n_iterations),
        regularization=float(regularization),
    )
