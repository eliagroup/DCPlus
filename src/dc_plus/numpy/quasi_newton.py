# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Quasi-Newton voltage updates with inverse Broyden corrections.

Based on:
A Class of Methods for Solving Nonlinear Simultaneous Equations
Author: C. G. Broyden
Source: Mathematics of Computation, Vol. 19, No. 92 (Oct., 1965), pp. 577-593
"""

import numpy as np
from jaxtyping import Float
from scipy import sparse

from dc_plus.interfaces.jacobian_interface import JacobianInterface
from dc_plus.interfaces.jacobian_network_data import (
    _apply_jacobian_dx_to_network_data,
    _get_admittance_matrix_from_network_data,
    calculate_nodal_mismatch_network_data,
)
from dc_plus.interfaces.network_information import DynamicNetworkInformation

InverseJacobianApproximation = Float[np.ndarray, " n_eq_jacobian n_eq_jacobian"]


def run_quasi_newton_updates(
    jacobian_data: JacobianInterface,
    dynamic_network_data: DynamicNetworkInformation,
    n_iterations: int = 2,
    y_matrix: sparse.sparray | None = None,
    regularization: float = 1e-12,
    damping_factor: float = 1.0,
) -> tuple[
    DynamicNetworkInformation,
    list[float],
    Float[np.ndarray, " n_eq_jacobian n_eq_jacobian"],
]:
    """Run quasi-Newton steps with fast lazy inverse Broyden updates.

    This starts from a fixed inverse Jacobian approximation and applies
    inverse Broyden rank-one secant corrections lazily. Instead of updating
    the dense inverse Jacobian matrix directly, the method stores low-rank
    correction factors ``A`` and ``B`` such that the current inverse
    approximation is

        H_k = H_0 + A_k B_k.T

    Applying the current inverse approximation to a vector ``v`` is therefore

        H_k v = H_0 v + A_k (B_k.T v).

    The implementation uses the identity

        a_k = s_k - H_k y_k = -H_k f_{k+1}

    where

        s_k = -H_k f_k
        y_k = f_{k+1} - f_k

    to avoid one additional inverse application per iteration.

    Parameters
    ----------
    jacobian_data : JacobianInterface
        Initial Jacobian layout and inverse approximation.
    dynamic_network_data : DynamicNetworkInformation
        Target system state providing the topology/admittance data and initial
        voltage iterate.
    n_iterations : int, optional
        Number of quasi-Newton correction steps to perform.
    y_matrix : sparse.sparray | None, optional
        Admittance matrix matching ``dynamic_network_data``. If omitted it is
        built once from ``dynamic_network_data`` and then reused.
    regularization : float, optional
        Minimum denominator magnitude accepted for the inverse Broyden update.
    damping_factor : float, optional
        Scalar multiplier applied to the quasi-Newton state step, i.e.
        x_{k+1} = x_k + alpha * Delta x_k.

    Returns
    -------
    tuple[DynamicNetworkInformation, list[float], Float[np.ndarray, " n_eq_jacobian n_eq_jacobian"]]
        Updated network state, mismatch infinity-norm history after each
        quasi-Newton step, and the final materialized inverse Jacobian
        approximation.

    Raises
    ------
    ValueError
        If ``n_iterations`` is negative.
    """
    if n_iterations < 0:
        raise ValueError("Number of quasi-Newton iterations must be non-negative.")

    expected_size = jacobian_data.n_angle_components + jacobian_data.n_magnitude_components

    if y_matrix is None:
        y_matrix = _get_admittance_matrix_from_network_data(dynamic_network_data)

    base_inverse_jacobian = jacobian_data.inverse_jacobian
    updated_network_data = dynamic_network_data

    # Preallocate lazy Broyden factors.
    #
    # After k accepted updates:
    #     H_k = H_0 + A[:, :k] @ B[:, :k].T
    #
    # Each inverse Broyden update is:
    #     H_{k+1} = H_k + a_k b_k.T
    #
    # The second inverse-Broyden factors are:
    #     a_k = s_k - H_k y_k
    #     b_k = y_k / (y_k.T @ y_k)
    #
    # Since s_k = -H_k f_k and y_k = f_{k+1} - f_k:
    #     a_k = -H_k f_{k+1}
    #
    # This avoids explicitly applying H_k to y_k.
    a_factors = np.zeros((expected_size, n_iterations), dtype=float)
    b_factors = np.zeros((expected_size, n_iterations), dtype=float)
    n_active_factors = 0

    def apply_inverse_approximation(
        vector: Float[np.ndarray, " n_eq_jacobian"],
    ) -> Float[np.ndarray, " n_eq_jacobian"]:
        """Apply the current lazy inverse-Jacobian approximation.

        Parameters
        ----------
        vector : Float[np.ndarray, " n_eq_jacobian"]
            Vector to multiply by the current inverse approximation.

        Returns
        -------
        Float[np.ndarray, " n_eq_jacobian"]
            Product of the current inverse approximation and ``vector``.
        """
        result = base_inverse_jacobian @ vector

        if n_active_factors == 0:
            return result

        active_a = a_factors[:, :n_active_factors]
        active_b = b_factors[:, :n_active_factors]
        return result + active_a @ (active_b.T @ vector)

    mismatch = calculate_nodal_mismatch_network_data(
        dynamic_network_data=updated_network_data,
        y_matrix=y_matrix,
        jacobian_data=jacobian_data,
    )
    inverse_mismatch = apply_inverse_approximation(mismatch)
    mismatch_history: list[float] = []

    for _ in range(n_iterations):
        state_step = -damping_factor * inverse_mismatch

        next_network_data = _apply_jacobian_dx_to_network_data(
            dynamic_network_data=updated_network_data,
            dx=state_step,
            jacobian_data=jacobian_data,
        )
        next_mismatch = calculate_nodal_mismatch_network_data(
            dynamic_network_data=next_network_data,
            y_matrix=y_matrix,
            jacobian_data=jacobian_data,
        )

        mismatch_history.append(float(np.max(np.abs(next_mismatch))))

        # This is H_k f_{k+1}. It is needed both for the Broyden factor and
        # for the next step, so keep it instead of recomputing H_k y_k.
        next_inverse_mismatch_old = apply_inverse_approximation(next_mismatch)

        mismatch_delta = next_mismatch - mismatch
        denominator = float(mismatch_delta @ mismatch_delta)

        if denominator > regularization:
            factor_index = n_active_factors
            new_a_factor = -next_inverse_mismatch_old
            new_b_factor = mismatch_delta / denominator

            a_factors[:, factor_index] = new_a_factor
            b_factors[:, factor_index] = new_b_factor
            n_active_factors += 1

            # Update H_{k+1} f_{k+1} from H_k f_{k+1} using the newly
            # appended rank-one factor:
            #
            #     H_{k+1} f_{k+1}
            #       = H_k f_{k+1} + a_k (b_k.T f_{k+1})
            correction_scale = float(new_b_factor @ next_mismatch)
            inverse_mismatch = next_inverse_mismatch_old + new_a_factor * correction_scale
        else:
            inverse_mismatch = next_inverse_mismatch_old

        updated_network_data = next_network_data
        mismatch = next_mismatch

    updated_inverse_jacobian = base_inverse_jacobian + a_factors[:, :n_active_factors] @ b_factors[:, :n_active_factors].T

    return updated_network_data, mismatch_history, updated_inverse_jacobian
