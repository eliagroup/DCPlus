# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Preprocessing functions for Jacobian-based BSDF computations.

Logic: each possible split gets a new PV and PQ entry in the Jacobian.
It is therefore crucial to check the max split number beforehand.
E.g. if you want to perform 3 splits, make sure the Jacobian has space for 3 PV and 3 PQ buses.
This is needed, as you do not know beforehand how to split and how many PV/PQ buses you will need.
But note, a switch outage can lead to more splits than anticipated.
"""

from dataclasses import replace

import numpy as np
import scipy.sparse as sp

from dc_plus.interfaces.jacobian_interface import JacobianInterface
from dc_plus.interfaces.network_information import BusType, DynamicNetworkInformation
from dc_plus.interfaces.type_hints import PosInt


def _validate_preprocessing_inputs(
    jacobian_data: JacobianInterface,
    max_bus_splits: PosInt,
) -> None:
    """Validate inputs for Jacobian preprocessing.

    Parameters
    ----------
    jacobian_data : JacobianInterface
        The original Jacobian data.
    max_bus_splits : PosInt
        Maximum number of bus splits to accommodate.

    Raises
    ------
    ValueError
        If max_bus_splits is not positive or Jacobian is not in CSR format.
    """
    if max_bus_splits <= 0:
        raise ValueError("max_bus_splits must be a positive integer.")

    original_jacobian = jacobian_data.jacobian
    if not (sp.isspmatrix_csr(original_jacobian) or isinstance(original_jacobian, sp.csr_array)):
        raise ValueError("Input Jacobian must be in CSR format.")


def _calculate_extended_dimensions(
    jacobian_data: JacobianInterface,
    split_count: int,
) -> tuple[int, int, int, int]:
    """Calculate dimensions for the extended Jacobian.

    Parameters
    ----------
    jacobian_data : JacobianInterface
        The original Jacobian data.
    split_count : int
        Number of bus splits to accommodate.

    Returns
    -------
    tuple[int, int, int, int]
        Number of angle equations, voltage equations, original equations, and extended equations.
    """
    n_angle = int(np.count_nonzero(jacobian_data.is_angle_component))
    n_voltage = int(np.count_nonzero(jacobian_data.is_magnitude_component))
    n_eq_original = n_angle + n_voltage
    n_eq_extended = n_eq_original + 2 * split_count
    return n_angle, n_voltage, n_eq_original, n_eq_extended


def _extend_jacobian_matrix(
    original_jacobian: sp.spmatrix,
    n_angle: int,
    n_voltage: int,
    n_eq_original: int,
    n_eq_extended: int,
    split_count: int,
) -> sp.spmatrix:
    """Extend Jacobian matrix with identity padding for additional buses.

    Parameters
    ----------
    original_jacobian : sp.spmatrix
        The original Jacobian matrix in CSR format.
    n_angle : int
        Number of angle equations.
    n_voltage : int
        Number of voltage magnitude equations.
    n_eq_original : int
        Total number of original equations.
    n_eq_extended : int
        Total number of extended equations.
    split_count : int
        Number of bus splits to accommodate.

    Returns
    -------
    sp.spmatrix
        Extended Jacobian matrix in CSR format.
    """
    augmented_jacobian = sp.lil_matrix((n_eq_extended, n_eq_extended), dtype=original_jacobian.dtype)

    angle_slice_old = slice(0, n_angle)
    magnitude_slice_old = slice(n_angle, n_eq_original)

    angle_slice_new = slice(0, n_angle)
    magnitude_slice_new = slice(n_angle + split_count, n_angle + split_count + n_voltage)

    augmented_jacobian[angle_slice_new, angle_slice_new] = original_jacobian[angle_slice_old, angle_slice_old]
    augmented_jacobian[angle_slice_new, magnitude_slice_new] = original_jacobian[angle_slice_old, magnitude_slice_old]
    augmented_jacobian[magnitude_slice_new, angle_slice_new] = original_jacobian[magnitude_slice_old, angle_slice_old]
    augmented_jacobian[magnitude_slice_new, magnitude_slice_new] = original_jacobian[
        magnitude_slice_old, magnitude_slice_old
    ]

    angle_padding_indices = np.arange(n_angle, n_angle + split_count, dtype=np.int32)
    magnitude_padding_indices = np.arange(n_angle + split_count + n_voltage, n_eq_extended, dtype=np.int32)

    augmented_jacobian[angle_padding_indices, angle_padding_indices] = 1.0
    augmented_jacobian[magnitude_padding_indices, magnitude_padding_indices] = 1.0

    if isinstance(original_jacobian, sp.csr_array):
        return sp.csr_array(augmented_jacobian)
    return augmented_jacobian.tocsr()


def _extend_inverse_jacobian(
    inverse_jacobian: np.ndarray,
    n_angle: int,
    n_voltage: int,
    n_eq_original: int,
    n_eq_extended: int,
    split_count: int,
) -> np.ndarray:
    """Extend inverse Jacobian with identity blocks for padded buses.

    Parameters
    ----------
    inverse_jacobian : np.ndarray
        The original inverse Jacobian matrix.
    n_angle : int
        Number of angle equations.
    n_voltage : int
        Number of voltage magnitude equations.
    n_eq_original : int
        Total number of original equations.
    n_eq_extended : int
        Total number of extended equations.
    split_count : int
        Number of bus splits to accommodate.

    Returns
    -------
    np.ndarray
        Extended inverse Jacobian matrix.
    """
    extended_inverse = np.zeros((n_eq_extended, n_eq_extended), dtype=inverse_jacobian.dtype)

    angle_slice_old = slice(0, n_angle)
    magnitude_slice_old = slice(n_angle, n_eq_original)

    angle_slice_new = slice(0, n_angle)
    magnitude_slice_new = slice(n_angle + split_count, n_angle + split_count + n_voltage)

    extended_inverse[angle_slice_new, angle_slice_new] = inverse_jacobian[angle_slice_old, angle_slice_old]
    extended_inverse[angle_slice_new, magnitude_slice_new] = inverse_jacobian[angle_slice_old, magnitude_slice_old]
    extended_inverse[magnitude_slice_new, angle_slice_new] = inverse_jacobian[magnitude_slice_old, angle_slice_old]
    extended_inverse[magnitude_slice_new, magnitude_slice_new] = inverse_jacobian[magnitude_slice_old, magnitude_slice_old]

    angle_padding_indices = np.arange(n_angle, n_angle + split_count, dtype=np.int32)
    magnitude_padding_indices = np.arange(n_angle + split_count + n_voltage, n_eq_extended, dtype=np.int32)

    one_value = np.array(1.0, dtype=inverse_jacobian.dtype)
    extended_inverse[angle_padding_indices, angle_padding_indices] = one_value
    extended_inverse[magnitude_padding_indices, magnitude_padding_indices] = one_value

    return extended_inverse


def _extend_boolean_masks(
    is_angle_component: np.ndarray,
    is_magnitude_component: np.ndarray,
    n_angle: int,
    n_voltage: int,
    n_eq_original: int,
    n_eq_extended: int,
    split_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend boolean masks describing variable types.

    Parameters
    ----------
    is_angle_component : np.ndarray
        Original mask for angle components.
    is_magnitude_component : np.ndarray
        Original mask for magnitude components.
    n_angle : int
        Number of angle equations.
    n_voltage : int
        Number of voltage magnitude equations.
    n_eq_original : int
        Total number of original equations.
    n_eq_extended : int
        Total number of extended equations.
    split_count : int
        Number of bus splits to accommodate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Extended angle and magnitude component masks.
    """
    extended_is_angle = np.zeros(n_eq_extended, dtype=is_angle_component.dtype)
    extended_is_angle[:n_angle] = is_angle_component[:n_angle]

    angle_padding_indices = np.arange(n_angle, n_angle + split_count, dtype=np.int32)
    extended_is_angle[angle_padding_indices] = True

    magnitude_slice_old = slice(n_angle, n_eq_original)
    magnitude_slice_new = slice(n_angle + split_count, n_angle + split_count + n_voltage)

    extended_is_magnitude = np.zeros(n_eq_extended, dtype=is_magnitude_component.dtype)
    extended_is_magnitude[magnitude_slice_new] = is_magnitude_component[magnitude_slice_old]

    magnitude_padding_indices = np.arange(n_angle + split_count + n_voltage, n_eq_extended, dtype=np.int32)
    extended_is_magnitude[magnitude_padding_indices] = True

    return extended_is_angle, extended_is_magnitude


def _extend_bus_bookkeeping(
    bus_is_used: np.ndarray,
    pointer_to_original_bus: np.ndarray,
    split_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend bus bookkeeping information.

    Parameters
    ----------
    bus_is_used : np.ndarray
        Original bus usage flags.
    pointer_to_original_bus : np.ndarray
        Original bus pointer array.
    split_count : int
        Number of bus splits to accommodate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Extended bus usage flags and pointer array.
    """
    n_buses_original = bus_is_used.size
    n_buses_extended = n_buses_original + split_count

    extended_bus_is_used = np.zeros(n_buses_extended, dtype=bus_is_used.dtype)
    extended_bus_is_used[:n_buses_original] = bus_is_used

    extended_pointer_to_original_bus = np.full(n_buses_extended, -1, dtype=pointer_to_original_bus.dtype)
    extended_pointer_to_original_bus[:n_buses_original] = pointer_to_original_bus

    return extended_bus_is_used, extended_pointer_to_original_bus


def _extend_dynamic_network_data(
    dynamic_network_data: DynamicNetworkInformation,
    split_count: int,
) -> DynamicNetworkInformation:
    """Extend dynamic network bus data.

    Parameters
    ----------
    dynamic_network_data : DynamicNetworkInformation
        Original dynamic network information.
    split_count : int
        Number of bus splits to accommodate.

    Returns
    -------
    DynamicNetworkInformation
        Extended dynamic network information.
    """

    def _pad_axis0(array: np.ndarray, value: float) -> np.ndarray:
        pad_shape = (split_count,) + array.shape[1:]
        pad_block = np.full(pad_shape, value, dtype=array.dtype)
        return np.concatenate((array, pad_block), axis=0)

    extended_bus_voltage_magnitudes = _pad_axis0(dynamic_network_data.bus_voltage_magnitudes, 1.0)
    extended_bus_voltage_angles = _pad_axis0(dynamic_network_data.bus_voltage_angles_rad, 0.0)
    extended_bus_active_power = _pad_axis0(dynamic_network_data.bus_active_power, 0.0)
    extended_bus_reactive_power = _pad_axis0(dynamic_network_data.bus_reactive_power, 0.0)
    extended_bus_type = np.concatenate(
        (
            dynamic_network_data.bus_type,
            np.full(split_count, BusType.PQ, dtype=dynamic_network_data.bus_type.dtype),
        )
    )

    return replace(
        dynamic_network_data,
        bus_voltage_magnitudes=extended_bus_voltage_magnitudes,
        bus_voltage_angles_rad=extended_bus_voltage_angles,
        bus_active_power=extended_bus_active_power,
        bus_reactive_power=extended_bus_reactive_power,
        bus_type=extended_bus_type,
    )


def preprocess_jacobian_bsdf(
    jacobian_data: JacobianInterface,
    max_bus_splits: PosInt,
    dynamic_network_data: DynamicNetworkInformation,
) -> tuple[JacobianInterface, DynamicNetworkInformation]:
    """Preprocess Jacobian and network data for BSDF computations involving bus splits.

    Adds additional rows/columns to the Jacobian and extends the dynamic network bus data so that
    both structures remain aligned when future bus splits are activated.

    Parameters
    ----------
    jacobian_data : JacobianInterface
        The original Jacobian data.
    max_bus_splits : PosInt
        Maximum number of bus splits to accommodate. Must be strictly positive.
    dynamic_network_data : DynamicNetworkInformation
        Dynamic network snapshot containing bus voltages, powers and types.

    Returns
    -------
    tuple[JacobianInterface, DynamicNetworkInformation]
        Updated Jacobian data, mask identifying rows/columns belonging to the original Jacobian,
        and the extended dynamic network information.
    """
    # Validate inputs
    _validate_preprocessing_inputs(jacobian_data, max_bus_splits)

    split_count = int(max_bus_splits)

    # Calculate dimensions for extended Jacobian
    n_angle, n_voltage, n_eq_original, n_eq_extended = _calculate_extended_dimensions(jacobian_data, split_count)

    # Extend Jacobian matrix
    extended_jacobian = _extend_jacobian_matrix(
        jacobian_data.jacobian,
        n_angle,
        n_voltage,
        n_eq_original,
        n_eq_extended,
        split_count,
    )

    # Extend inverse Jacobian
    extended_inverse = _extend_inverse_jacobian(
        jacobian_data.inverse_jacobian,
        n_angle,
        n_voltage,
        n_eq_original,
        n_eq_extended,
        split_count,
    )

    # Extend boolean masks
    extended_is_angle, extended_is_magnitude = _extend_boolean_masks(
        jacobian_data.is_angle_component,
        jacobian_data.is_magnitude_component,
        n_angle,
        n_voltage,
        n_eq_original,
        n_eq_extended,
        split_count,
    )

    # Extend bus bookkeeping
    extended_bus_is_used, extended_pointer_to_original_bus = _extend_bus_bookkeeping(
        jacobian_data.bus_is_used,
        jacobian_data.pointer_to_original_bus,
        split_count,
    )

    # Extend dynamic network data
    extended_dynamic_network_data = _extend_dynamic_network_data(
        dynamic_network_data,
        split_count,
    )

    # Calculate valid indices and prepare final data structures
    n_buses_extended = extended_bus_is_used.size

    magnitude_slice_new = slice(n_angle + split_count, n_angle + split_count + n_voltage)
    is_valid_index = np.zeros(n_eq_extended, dtype=bool)
    is_valid_index[:n_angle] = True
    is_valid_index[magnitude_slice_new] = True

    pq_indices = extended_dynamic_network_data.pq_buses_indices
    pvpq_indices = extended_dynamic_network_data.pvpq_buses_indices_pvpq_order

    extended_jacobian_data = JacobianInterface(
        bus_is_used=extended_bus_is_used,
        jacobian_index_in_use=is_valid_index,
        pointer_to_original_bus=extended_pointer_to_original_bus,
        jacobian=extended_jacobian,
        inverse_jacobian=extended_inverse,
        is_angle_component=extended_is_angle,
        is_magnitude_component=extended_is_magnitude,
        pvpq_indices=pvpq_indices,
        pq_indices=pq_indices,
        n_buses=n_buses_extended,
    )

    return extended_jacobian_data, extended_dynamic_network_data
