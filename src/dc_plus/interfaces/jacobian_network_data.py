# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Functions to compute the Jacobian matrix and related data from the dynamic network data."""

import numpy as np
from jaxtyping import Float
from scipy import sparse
from scipy.sparse.linalg import inv as sparse_inv

from dc_plus.interfaces.jacobian_interface import JacobianInterface
from dc_plus.interfaces.network_information import DynamicNetworkInformation


def _get_jacobian_data_from_network_data(dynamic_network_data: DynamicNetworkInformation) -> JacobianInterface:
    """Get the Jacobian data from the dynamic network data.

    Parameters
    ----------
    dynamic_network_data : DynamicNetworkInformation
        The dynamic network data.

    Returns
    -------
    JacobianInterface
        The Jacobian data interface.
    """
    jacobian = get_jacobian_from_network_data(
        dynamic_network_data=dynamic_network_data,
    )

    n_pq = dynamic_network_data.n_pq_buses
    n_pv = dynamic_network_data.n_pv_buses
    is_angle_component = np.zeros(jacobian.shape[0], dtype=bool)
    is_angle_component[: (n_pq + n_pv)] = True
    is_magnitude_component = np.zeros(jacobian.shape[0], dtype=bool)
    is_magnitude_component[(n_pq + n_pv) :] = True
    jacobian_index_in_use = np.ones(jacobian.shape[0], dtype=bool)

    bus_is_used = np.ones(dynamic_network_data.n_buses, dtype=bool)

    pq_indices = dynamic_network_data.pq_buses_indices
    pvpq_indices = dynamic_network_data.pvpq_buses_indices_pvpq_order

    return JacobianInterface(
        bus_is_used=bus_is_used,
        jacobian_index_in_use=jacobian_index_in_use,
        pointer_to_original_bus=np.arange(dynamic_network_data.n_buses, dtype=np.int32),
        jacobian=jacobian,
        inverse_jacobian=sparse_inv(jacobian).toarray(),
        is_angle_component=is_angle_component,
        is_magnitude_component=is_magnitude_component,
        pvpq_indices=pvpq_indices,
        pq_indices=pq_indices,
        n_buses=dynamic_network_data.n_buses,
    )


def get_jacobian_from_network_data(
    dynamic_network_data: DynamicNetworkInformation,
) -> sparse.sparray:
    """Calculate Jacobian.

    Parameters
    ----------
    dynamic_network_data : DynamicNetworkInformation
        The dynamic network data.

    Returns
    -------
    sparse.sparray
        The Jacobian matrix.
    """
    y_bus = _get_admittance_matrix_from_network_data(
        dynamic_network_data=dynamic_network_data,
    )

    voltage_magnitudes = dynamic_network_data.bus_voltage_magnitudes
    voltage_angles = dynamic_network_data.bus_voltage_angles_rad

    pq_indices = dynamic_network_data.pq_buses_indices
    pqpv_indices = dynamic_network_data.pvpq_buses_indices_pvpq_order

    voltage = voltage_magnitudes * np.exp(1.0j * voltage_angles)
    if np.any(voltage == 0.0):
        raise ValueError("Voltage magnitudes must be strictly positive to construct the Jacobian.")

    voltage_norm = voltage / abs(voltage)
    current = y_bus @ voltage

    diag_voltage = sparse.diags(voltage)
    diag_current = sparse.diags(current)
    diag_voltage_norm = sparse.diags(voltage_norm)

    # dS/dV = diag(V) * (Ybus * diag(V/|V|))^* + diag(I)^* * diag(V/|V|)
    power_jacobian_voltage_mag = (
        diag_voltage @ (y_bus @ diag_voltage_norm).conjugate() + diag_current.conjugate() @ diag_voltage_norm
    )
    # dS/dtheta = 1j * diag(V) * (diag(I) - Ybus * diag(V))^*
    power_jacobian_voltage_angle = 1j * diag_voltage @ (diag_current - y_bus @ diag_voltage).conjugate()

    jacobian_active_power_angle = sparse.csr_array(power_jacobian_voltage_angle[np.ix_(pqpv_indices, pqpv_indices)].real)
    jacobian_active_power_voltage = sparse.csr_array(power_jacobian_voltage_mag[np.ix_(pqpv_indices, pq_indices)].real)
    jacobian_reactive_power_angle = sparse.csr_array(power_jacobian_voltage_angle[np.ix_(pq_indices, pqpv_indices)].imag)
    jacobian_reactive_power_voltage = sparse.csr_array(power_jacobian_voltage_mag[np.ix_(pq_indices, pq_indices)].imag)

    jacobian = sparse.vstack(
        [
            sparse.hstack([jacobian_active_power_angle, jacobian_active_power_voltage], format="csr"),
            sparse.hstack([jacobian_reactive_power_angle, jacobian_reactive_power_voltage], format="csr"),
        ],
        format="csr",
    )

    return sparse.csr_array(jacobian)


def _get_admittance_matrix_from_network_data(
    dynamic_network_data: DynamicNetworkInformation,
) -> sparse.sparray:
    """Compute the admittance matrix from the branch admittances.

    Parameters
    ----------
    dynamic_network_data : DynamicNetworkInformation
        The dynamic network data.


    number_buses : int
        Number of buses in the network.

    Returns
    -------
    Ybus : sparse.sparray
        The admittance matrix of the network.
    """
    branch_connected = dynamic_network_data.branch_connected
    shunt_connected = dynamic_network_data.shunt_connected
    number_buses = dynamic_network_data.n_buses

    branch_effective_admittance_from_to = dynamic_network_data.branch_effective_admittance_from_to
    branch_effective_admittance_from_from = dynamic_network_data.branch_effective_admittance_from_from
    branch_effective_admittance_to_to = dynamic_network_data.branch_effective_admittance_to_to
    branch_effective_admittance_to_from = dynamic_network_data.branch_effective_admittance_to_from
    branch_from_nodes = dynamic_network_data.branch_from_bus
    branch_to_nodes = dynamic_network_data.branch_to_bus

    shunt_effective_bus_admittance = dynamic_network_data.shunt_effective_bus_admittance
    shunt_bus_indices = dynamic_network_data.shunt_bus_indices

    f_idx = branch_from_nodes[branch_connected]
    t_idx = branch_to_nodes[branch_connected]

    ff = branch_effective_admittance_from_from[branch_connected]
    tt = branch_effective_admittance_to_to[branch_connected]
    ft = branch_effective_admittance_from_to[branch_connected]
    tf = branch_effective_admittance_to_from[branch_connected]

    rows = np.concatenate((f_idx, t_idx, f_idx, t_idx))
    cols = np.concatenate((f_idx, t_idx, t_idx, f_idx))
    data = np.concatenate((ff, tt, ft, tf))

    connectivity = sparse.coo_array(
        (data, (rows, cols)),
        shape=(number_buses, number_buses),
        dtype=branch_effective_admittance_from_to.dtype,
    )

    if shunt_bus_indices.size:
        shunt_bus_indices = shunt_bus_indices[shunt_connected]
        shunt_values = shunt_effective_bus_admittance[shunt_connected]

        if shunt_bus_indices.size:
            shunt_contribution = sparse.coo_array(
                (shunt_values.astype(connectivity.dtype, copy=False), (shunt_bus_indices, shunt_bus_indices)),
                shape=(number_buses, number_buses),
                dtype=connectivity.dtype,
            )
            connectivity = connectivity + shunt_contribution

    y_bus = connectivity.tocsr()

    return y_bus


def calculate_nodal_mismatch_network_data(
    dynamic_network_data: DynamicNetworkInformation,
    y_matrix: sparse.sparray,
) -> Float[np.ndarray, " n_eq_jacobian"]:
    """Calculate the nodal mismatches nodal mismatches.

    Parameters
    ----------
    dynamic_network_data : DynamicNetworkInformation
        The dynamic network data.
    y_matrix : sparse.sparray
        The admittance matrix.

    Returns
    -------
    Float[np.ndarray, " n_eq_jacobian"]
        The nodal mismatches nodal mismatches.
    """
    # Powsybl exports bus injections with the load convention (loads > 0, generation < 0).
    # The network mismatch requires the injection convention, hence flip the sign.
    s_pu = -(dynamic_network_data.bus_active_power + 1j * dynamic_network_data.bus_reactive_power)

    v_pu = dynamic_network_data.bus_voltage_magnitudes * np.exp(1j * dynamic_network_data.bus_voltage_angles_rad)

    mismatch = v_pu * np.conj(y_matrix @ v_pu) - s_pu

    # Jacobian ordering uses PV buses first then PQ buses for the angle (P) equations
    # and PQ buses for the magnitude (Q) equations. Assemble the mismatch vector
    # in the same order: [P@PV,P@PQ, Q@PQ] -> equivalently [P@(PV+PQ), Q@PQ].
    pvpq_indices = dynamic_network_data.pvpq_buses_indices_pvpq_order
    pq_indices = dynamic_network_data.pq_buses_indices

    return np.r_[mismatch[pvpq_indices].real, mismatch[pq_indices].imag]
