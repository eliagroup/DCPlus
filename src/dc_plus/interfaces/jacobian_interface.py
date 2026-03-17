# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Jacobian interface for DC+ power flow calculations."""

import numpy as np
import scipy.sparse as sp
from jaxtyping import Bool, Float, Int


class JacobianInterface:
    """Interface class to store Jacobian-related data.

    The core component for the DC+ power flow calculations is the inverse Jacobian matrix.


    """

    bus_is_used: Bool[np.ndarray, " n_bus"]
    """Boolean vector indicating if the bus is used in the system.
    True: the bus is used in the Jacobian and part of the main grid.
    False: the bus us a placeholder to be used when a busbar coupler is opened or a PV bus is converted to PQ bus.
    """

    jacobian_index_in_use: Bool[np.ndarray, " n_eq_jacobian"]
    """Boolean vector indicating which rows/columns of the Jacobian are in use."""

    pointer_to_original_bus: Int[np.ndarray, " n_bus"]
    """Integer vector indicating the original bus.
    pointing to -1: bus is original bus or free bus.
    pointing to another bus: bus used by the bsdf, pointing to the original bus.
    """

    jacobian: Float[np.ndarray, " n_eq_jacobian n_eq_jacobian"]
    """The Jacobian matrix used for power flow calculations."""

    inverse_jacobian: Float[np.ndarray, " n_eq_jacobian n_eq_jacobian"]
    """Inverse of the Jacobian matrix used for power flow calculations."""

    is_angle_component: Bool[np.ndarray, " n_eq_jacobian"]
    """Boolean vector indicating which components of the jacobian correspond to angle variables."""

    is_magnitude_component: Bool[np.ndarray, " n_eq_jacobian"]
    """Boolean vector indicating which components of the jacobian correspond to magnitude variables."""

    angle_component_indices: Int[np.ndarray, " n_eq_jacobian"]
    """Integer vector indicating the indices of angle variables in the Jacobian."""

    magnitude_component_indices: Int[np.ndarray, " n_eq_jacobian"]
    """Integer vector indicating the indices of magnitude variables in the Jacobian."""

    bus_angle_indices: Int[np.ndarray, " n_bus"]
    """Integer vector indicating the indices of angle variables in the Jacobian."""

    bus_magnitude_indices: Int[np.ndarray, " n_bus"]
    """Integer vector indicating the indices of magnitude variables in the Jacobian."""

    def __init__(
        self,
        bus_is_used: Bool[np.ndarray, " n_bus"],
        jacobian_index_in_use: Bool[np.ndarray, " n_eq_jacobian"],
        pointer_to_original_bus: Int[np.ndarray, " n_bus"],
        jacobian: sp.csr_array,
        inverse_jacobian: Float[np.ndarray, " n_eq_jacobian n_eq_jacobian "],
        is_angle_component: Bool[np.ndarray, " n_eq_jacobian"],
        is_magnitude_component: Bool[np.ndarray, " n_eq_jacobian"],
        pvpq_indices: Int[np.ndarray, " n_pvpq_bus "],
        pq_indices: Int[np.ndarray, " n_pq_bus "],
        n_buses: int,
    ) -> "JacobianInterface":
        """Initialize the JacobianInterface.

        Parameters
        ----------
        bus_is_used: Bool[np.ndarray, " n_bus"]
            Boolean vector indicating if the bus is used in the system.
        jacobian_index_in_use: Bool[np.ndarray, " n_eq_jacobian"]
            Boolean vector indicating which rows/columns of the Jacobian are in use.
        pointer_to_original_bus: Int[np.ndarray, " n_bus"]
            Integer vector indicating the original bus.
        jacobian: sp.csr_array
            The Jacobian matrix used for power flow calculations.
        inverse_jacobian: Float[np.ndarray, " n_eq_jacobian n_eq_jacobian "]
            Inverse of the Jacobian matrix used for power flow calculations.
        is_angle_component: Bool[np.ndarray, " n_eq_jacobian"]
            Boolean vector indicating which components of the jacobian correspond to angle variables.
        is_magnitude_component: Bool[np.ndarray, " n_eq_jacobian"]
            Boolean vector indicating which components of the jacobian correspond to magnitude variables.
        pvpq_indices: Int[np.ndarray, " n_pvpq_bus "]
            Integer vector indicating the indices of PV and PQ buses.
        pq_indices: Int[np.ndarray, " n_pq_bus "]
            Integer vector indicating the indices of PQ buses.
        n_buses: int
            Total number of buses in the system.

        Returns
        -------
        JacobianInterface
            An instance of the JacobianInterface class.
        """
        self.bus_is_used = bus_is_used
        self.jacobian_index_in_use = jacobian_index_in_use
        self.pointer_to_original_bus = pointer_to_original_bus
        self.jacobian = jacobian
        # must first be processed by the bsdf pre-processor
        self.inverse_jacobian = inverse_jacobian
        self.is_angle_component = is_angle_component
        self.is_magnitude_component = is_magnitude_component
        self.bus_angle_indices = pvpq_indices
        self.bus_magnitude_indices = pq_indices
        angle_component_indices = np.full(n_buses, -1, dtype=np.int32)
        magnitude_component_indices = np.full(n_buses, -1, dtype=np.int32)
        angle_component_indices[pvpq_indices] = np.flatnonzero(is_angle_component)
        magnitude_component_indices[pq_indices] = np.flatnonzero(is_magnitude_component)
        self.angle_component_indices = angle_component_indices
        self.magnitude_component_indices = magnitude_component_indices

    # ruff: noqa: PLR0911, C901
    def __eq__(self, value: "JacobianInterface") -> bool:
        """Equal definition of JacobianInterface.

        Parameters
        ----------
        self: JacobianInterface
            The JacobianInterface.
        value: JacobianInterface
            The JacobianInterface to compare to

        Returns
        -------
        bool
            True if the JacobianInterfaces are equal, False otherwise.
        """
        if not isinstance(value, JacobianInterface):
            return False

        if not np.array_equal(self.bus_is_used, value.bus_is_used):
            return False
        if not np.array_equal(self.jacobian_index_in_use, value.jacobian_index_in_use):
            return False

        if not np.array_equal(self.is_angle_component, value.is_angle_component):
            return False
        if not np.array_equal(self.is_magnitude_component, value.is_magnitude_component):
            return False

        if not np.array_equal(np.sort(self.bus_angle_indices), np.sort(value.bus_angle_indices)):
            return False

        if not np.array_equal(np.sort(self.bus_magnitude_indices), np.sort(value.bus_magnitude_indices)):
            return False

        if not np.allclose(self.inverse_jacobian, value.inverse_jacobian):
            return False

        if not np.allclose(self.jacobian.toarray(), value.jacobian.toarray()):
            return False

        if not np.array_equal(self.pointer_to_original_bus, value.pointer_to_original_bus):
            return False

        return True
