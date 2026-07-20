# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helpers for sparse network-state calculations without constructing dense Y matrices.

Note: The branch admittances need to be updated when a topology change occurs.
"""

from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Bool, Complex128


def _calculate_branch_complex_power(
    v_from: Complex128[jnp.ndarray, " ... n_branches"],
    v_to: Complex128[jnp.ndarray, " ... n_branches"],
    current_from: Complex128[jnp.ndarray, " ... n_branches"],
    current_to: Complex128[jnp.ndarray, " ... n_branches"],
    branch_mask: Bool[jnp.ndarray, " n_branches"],
) -> Tuple[
    Complex128[jnp.ndarray, " ... n_branches"],
    Complex128[jnp.ndarray, " ... n_branches"],
]:
    """Compute masked complex branch powers from endpoint voltages and currents."""
    branch_mask_cast = branch_mask.astype(v_from.dtype)
    s_from = v_from * jnp.conj(current_from) * branch_mask_cast
    s_to = v_to * jnp.conj(current_to) * branch_mask_cast
    return s_from, s_to
