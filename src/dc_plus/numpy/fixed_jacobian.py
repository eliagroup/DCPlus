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

from dc_plus.interfaces.jacobian_network_data import (
    _apply_jacobian_dx_to_network_data,
    _get_admittance_matrix_from_network_data,
    calculate_nodal_mismatch_network_data,
)
from dc_plus.interfaces.network_information import DynamicNetworkInformation


def run_fixed_jacobian_iterations(
    jacobian_inv: Float[np.ndarray, " n_eq_jacobian n_eq_jacobian"],
    dynamic_network_data: DynamicNetworkInformation,
    n_iterations: int = 2,
    y_matrix: sparse.sparray | None = None,
) -> DynamicNetworkInformation:
    """Apply modified-Newton iterations with a fixed Jacobian inverse.

    This performs repeated mismatch evaluations while reusing the same inverse
    Jacobian. It is useful when the Jacobian update is cheap and accurate but a
    single linear step is not sufficient for larger nonlinear changes such as a
    far transformer tap move.

    Parameters
    ----------
    jacobian_inv : Float[np.ndarray, " n_eq_jacobian n_eq_jacobian"]
        Fixed inverse Jacobian used for every iteration.
    dynamic_network_data : DynamicNetworkInformation
        Dynamic network data at the target topology/admittance state and current
        voltage iterate.
    n_iterations : int, optional
        Number of fixed-Jacobian correction steps to apply.
    y_matrix : sparse.sparray | None, optional
        Admittance matrix matching ``dynamic_network_data``. If omitted it is
        built once from ``dynamic_network_data`` and then reused for all
        iterations.

    Returns
    -------
    DynamicNetworkInformation
        Dynamic network data after the requested number of fixed-Jacobian
        iterations.
    """
    if n_iterations < 0:
        raise ValueError("Number of fixed Jacobian iterations must be non-negative.")

    if y_matrix is None:
        y_matrix = _get_admittance_matrix_from_network_data(dynamic_network_data)

    updated_network_data = dynamic_network_data
    for _ in range(n_iterations):
        mismatch = calculate_nodal_mismatch_network_data(
            dynamic_network_data=updated_network_data,
            y_matrix=y_matrix,
        )
        dx = -jacobian_inv @ mismatch
        updated_network_data = _apply_jacobian_dx_to_network_data(
            dynamic_network_data=updated_network_data,
            dx=dx,
        )

    return updated_network_data
