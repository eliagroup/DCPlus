# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Functions to perform fixed-Jacobian iterations."""

import numpy as np
from jaxtyping import Float
from scipy import sparse

from dc_plus.interfaces.jacobian_interface import JacobianInterface
from dc_plus.interfaces.jacobian_network_data import (
    _apply_jacobian_dx_to_network_data,
    calculate_nodal_mismatch_network_data,
)
from dc_plus.interfaces.network_information import DynamicNetworkInformation

FixedJacobianInverse = Float[np.ndarray, " n_eq_jacobian n_eq_jacobian"]


def _run_single_fixed_jacobian_iteration(
    jacobian_data: JacobianInterface,
    dynamic_network_data: DynamicNetworkInformation,
    y_matrix: sparse.sparray,
) -> DynamicNetworkInformation:
    """Apply one fixed-Jacobian correction step."""
    mismatch = calculate_nodal_mismatch_network_data(
        dynamic_network_data=dynamic_network_data,
        y_matrix=y_matrix,
        jacobian_data=jacobian_data,
    )
    dx = -jacobian_data.inverse_jacobian @ mismatch
    return _apply_jacobian_dx_to_network_data(
        dynamic_network_data=dynamic_network_data,
        dx=dx,
        jacobian_data=jacobian_data,
    )


def run_fixed_jacobian_iterations(
    jacobian_data: JacobianInterface,
    dynamic_network_data: DynamicNetworkInformation,
    y_matrix: sparse.sparray,
    n_iterations: int = 2,
) -> DynamicNetworkInformation:
    """Apply modified-Newton iterations with a fixed Jacobian inverse.

    This performs repeated mismatch evaluations while reusing the same inverse
    Jacobian. It is useful when the Jacobian update is cheap and accurate but a
    single linear step is not sufficient for larger nonlinear changes such as a
    far transformer tap move.

    Parameters
    ----------
    jacobian_data : JacobianInterface
        Fixed Jacobian layout and inverse used for every iteration.
    dynamic_network_data : DynamicNetworkInformation
        Dynamic network data at the target topology/admittance state and current
        voltage iterate.
    y_matrix : sparse.sparray
        Admittance matrix matching ``dynamic_network_data``.
    n_iterations : int, optional
        Number of fixed-Jacobian correction steps to apply.

    Returns
    -------
    DynamicNetworkInformation
        Dynamic network data after the requested number of fixed-Jacobian
        iterations.
    """
    updated_network_data = dynamic_network_data
    for _ in range(n_iterations):
        updated_network_data = _run_single_fixed_jacobian_iteration(
            jacobian_data=jacobian_data,
            dynamic_network_data=updated_network_data,
            y_matrix=y_matrix,
        )

    return updated_network_data
