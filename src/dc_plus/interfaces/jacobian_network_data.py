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
from dc_plus.interfaces.network_information import (
    BusType,
    DynamicNetworkInformation,
    StaticNetworkInformation,
    replace_network_data,
)


def _get_local_voltage_regulation_data(
    dynamic_network_data: DynamicNetworkInformation,
    static_network_data: StaticNetworkInformation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate local voltage-regulating reactive limits and fixed Q injections by bus."""
    n_buses = dynamic_network_data.n_buses
    injection_to_bus = dynamic_network_data.injection_to_bus
    injection_regulated_bus = dynamic_network_data.injection_regulated_bus
    injection_connected = dynamic_network_data.injection_connected
    injection_voltage_regulation = dynamic_network_data.injection_voltage_regulation
    injection_reactive_power = dynamic_network_data.injection_reactive_power

    local_regulation_mask = (
        injection_connected
        & injection_voltage_regulation
        & (injection_to_bus == injection_regulated_bus)
        & (injection_regulated_bus >= 0)
    )

    regulating_bus_count = np.zeros(n_buses, dtype=np.int32)
    regulating_q_min = np.zeros(n_buses, dtype=float)
    regulating_q_max = np.zeros(n_buses, dtype=float)
    fixed_bus_reactive_power = np.zeros(n_buses, dtype=float)

    if np.any(local_regulation_mask):
        regulated_buses = injection_to_bus[local_regulation_mask]
        q_min = np.nan_to_num(
            static_network_data.injection_limit_reactive_power_min[local_regulation_mask],
            nan=-np.inf,
        )
        q_max = np.nan_to_num(
            static_network_data.injection_limit_reactive_power_max[local_regulation_mask],
            nan=np.inf,
        )
        np.add.at(regulating_bus_count, regulated_buses, 1)
        np.add.at(regulating_q_min, regulated_buses, q_min)
        np.add.at(regulating_q_max, regulated_buses, q_max)

    fixed_injection_mask = injection_connected & ~local_regulation_mask
    if np.any(fixed_injection_mask):
        np.add.at(
            fixed_bus_reactive_power,
            injection_to_bus[fixed_injection_mask],
            injection_reactive_power[fixed_injection_mask],
        )

    has_local_regulation = regulating_bus_count > 0
    return has_local_regulation, regulating_q_min, regulating_q_max, fixed_bus_reactive_power


def _get_imported_local_regulating_reactive_power_by_bus(
    dynamic_network_data: DynamicNetworkInformation,
) -> np.ndarray:
    """Aggregate imported local regulating reactive power by bus in load convention."""
    n_buses = dynamic_network_data.n_buses
    injection_to_bus = dynamic_network_data.injection_to_bus
    injection_regulated_bus = dynamic_network_data.injection_regulated_bus
    injection_connected = dynamic_network_data.injection_connected
    injection_voltage_regulation = dynamic_network_data.injection_voltage_regulation
    injection_reactive_power = dynamic_network_data.injection_reactive_power

    local_regulation_mask = (
        injection_connected
        & injection_voltage_regulation
        & (injection_to_bus == injection_regulated_bus)
        & (injection_regulated_bus >= 0)
    )
    imported_regulating_reactive_power = np.zeros(n_buses, dtype=float)
    if np.any(local_regulation_mask):
        np.add.at(
            imported_regulating_reactive_power,
            injection_to_bus[local_regulation_mask],
            injection_reactive_power[local_regulation_mask],
        )
    return imported_regulating_reactive_power


def _calculate_lower_residual(
    dynamic_network_data: DynamicNetworkInformation,
    mismatch: Float[np.ndarray, " n_buses"],
    magnitude_bus_indices: np.ndarray,
    static_network_data: StaticNetworkInformation | None,
    q_limit_tolerance: float,
) -> Float[np.ndarray, " n_bus_eq"]:
    """Assemble the mode-dependent lower residual block for the active Jacobian layout."""
    bus_type = dynamic_network_data.bus_type
    lower_residual = mismatch[magnitude_bus_indices].imag.copy()
    pv_mask = bus_type[magnitude_bus_indices] == BusType.PV
    if not np.any(pv_mask):
        return lower_residual

    lower_residual[pv_mask] = (
        dynamic_network_data.bus_voltage_magnitudes[magnitude_bus_indices][pv_mask]
        - dynamic_network_data.bus_voltage_magnitude_setpoint[magnitude_bus_indices][pv_mask]
    )

    if static_network_data is None:
        return lower_residual

    has_local_regulation, regulating_q_min, regulating_q_max, fixed_bus_reactive_power = _get_local_voltage_regulation_data(
        dynamic_network_data=dynamic_network_data,
        static_network_data=static_network_data,
    )
    imported_regulating_reactive_power = _get_imported_local_regulating_reactive_power_by_bus(dynamic_network_data)

    local_regulated_pv_mask = pv_mask & has_local_regulation[magnitude_bus_indices]
    if not np.any(local_regulated_pv_mask):
        return lower_residual

    # ``mismatch`` is assembled in injection convention while the exported bus and
    # injection reactive powers use the load convention. Reconstruct the actual
    # local regulating-generator Q in load convention before checking limits.
    net_bus_reactive_power = dynamic_network_data.bus_reactive_power - mismatch.imag
    regulating_reactive_power = net_bus_reactive_power - fixed_bus_reactive_power
    regulating_reactive_power_pv = regulating_reactive_power[magnitude_bus_indices]
    imported_regulating_reactive_power_pv = imported_regulating_reactive_power[magnitude_bus_indices]
    lower_limit_mask = local_regulated_pv_mask & (
        regulating_reactive_power_pv < (regulating_q_min[magnitude_bus_indices] - q_limit_tolerance)
    )
    upper_limit_mask = local_regulated_pv_mask & (
        regulating_reactive_power_pv > (regulating_q_max[magnitude_bus_indices] + q_limit_tolerance)
    )
    imported_lower_limit_mask = local_regulated_pv_mask & (
        imported_regulating_reactive_power_pv <= (regulating_q_min[magnitude_bus_indices] + q_limit_tolerance)
    )
    imported_upper_limit_mask = local_regulated_pv_mask & (
        imported_regulating_reactive_power_pv >= (regulating_q_max[magnitude_bus_indices] - q_limit_tolerance)
    )
    lower_limit_mask |= imported_lower_limit_mask
    upper_limit_mask = (upper_limit_mask | imported_upper_limit_mask) & ~lower_limit_mask

    lower_residual[lower_limit_mask] = (
        regulating_q_min[magnitude_bus_indices][lower_limit_mask] - regulating_reactive_power_pv[lower_limit_mask]
    )
    lower_residual[upper_limit_mask] = (
        regulating_q_max[magnitude_bus_indices][upper_limit_mask] - regulating_reactive_power_pv[upper_limit_mask]
    )
    return lower_residual


def get_jacobian_data_from_network_data(
    dynamic_network_data: DynamicNetworkInformation,
) -> JacobianInterface:
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
    jacobian = _get_jacobian_from_network_data(
        dynamic_network_data=dynamic_network_data,
    )

    angle_bus_indices = dynamic_network_data.pvpq_buses_indices_pvpq_order
    magnitude_bus_indices = dynamic_network_data.pq_buses_indices
    n_angle_eq = angle_bus_indices.size
    n_magnitude_eq = magnitude_bus_indices.size
    is_angle_component = np.zeros(jacobian.shape[0], dtype=bool)
    is_angle_component[:n_angle_eq] = True
    is_magnitude_component = np.zeros(jacobian.shape[0], dtype=bool)
    is_magnitude_component[n_angle_eq : n_angle_eq + n_magnitude_eq] = True
    jacobian_index_in_use = np.ones(jacobian.shape[0], dtype=bool)

    bus_is_used = np.ones(dynamic_network_data.n_buses, dtype=bool)

    return JacobianInterface(
        bus_is_used=bus_is_used,
        jacobian_index_in_use=jacobian_index_in_use,
        pointer_to_original_bus=np.arange(dynamic_network_data.n_buses, dtype=np.int32),
        jacobian=jacobian,
        inverse_jacobian=sparse_inv(jacobian.tocsc()).toarray(),
        is_angle_component=is_angle_component,
        is_magnitude_component=is_magnitude_component,
        angle_bus_indices=angle_bus_indices,
        magnitude_bus_indices=magnitude_bus_indices,
        n_buses=dynamic_network_data.n_buses,
    )


def _get_sparse_power_jacobian_blocks(
    y_bus: sparse.sparray,
    voltage_magnitudes: np.ndarray,
    voltage_angles: np.ndarray,
    angle_bus_indices: np.ndarray,
    magnitude_bus_indices: np.ndarray,
) -> tuple[sparse.csr_array, sparse.csr_array, sparse.csr_array, sparse.csr_array]:
    """Build sliced AC power-Jacobian blocks from the sparse network admittance."""
    voltage = voltage_magnitudes * np.exp(1.0j * voltage_angles)
    if np.any(np.isclose(voltage, 0.0, atol=1e-10)):
        raise ValueError("Voltage magnitudes must be strictly positive to construct the Jacobian.")

    voltage_norm = voltage / np.abs(voltage)
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

    jacobian_active_power_angle = sparse.csr_array(
        power_jacobian_voltage_angle[np.ix_(angle_bus_indices, angle_bus_indices)].real
    )
    jacobian_active_power_voltage = sparse.csr_array(
        power_jacobian_voltage_mag[np.ix_(angle_bus_indices, magnitude_bus_indices)].real
    )
    jacobian_reactive_power_angle = sparse.csr_array(
        power_jacobian_voltage_angle[np.ix_(magnitude_bus_indices, angle_bus_indices)].imag
    )
    jacobian_reactive_power_voltage = sparse.csr_array(
        power_jacobian_voltage_mag[np.ix_(magnitude_bus_indices, magnitude_bus_indices)].imag
    )

    return (
        jacobian_active_power_angle,
        jacobian_active_power_voltage,
        jacobian_reactive_power_angle,
        jacobian_reactive_power_voltage,
    )


def _get_jacobian_from_network_data(
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

    angle_bus_indices = dynamic_network_data.pvpq_buses_indices_pvpq_order
    magnitude_bus_indices = dynamic_network_data.pq_buses_indices

    (
        jacobian_active_power_angle,
        jacobian_active_power_voltage,
        jacobian_reactive_power_angle,
        jacobian_reactive_power_voltage,
    ) = _get_sparse_power_jacobian_blocks(
        y_bus=y_bus,
        voltage_magnitudes=voltage_magnitudes,
        voltage_angles=voltage_angles,
        angle_bus_indices=angle_bus_indices,
        magnitude_bus_indices=magnitude_bus_indices,
    )

    full_reactive_power_angle = jacobian_reactive_power_angle.toarray()
    full_reactive_power_voltage = jacobian_reactive_power_voltage.toarray()

    bus_type = dynamic_network_data.bus_type[magnitude_bus_indices]
    pv_mask = bus_type == BusType.PV
    jacobian_reactive_power_angle = full_reactive_power_angle.copy()
    jacobian_reactive_power_voltage = full_reactive_power_voltage.copy()
    if np.any(pv_mask):
        jacobian_reactive_power_angle[pv_mask, :] = 0.0
        jacobian_reactive_power_voltage[pv_mask, :] = 0.0
        jacobian_reactive_power_voltage[pv_mask, np.flatnonzero(pv_mask)] = 1.0

    jacobian = sparse.vstack(
        [
            sparse.hstack([jacobian_active_power_angle, jacobian_active_power_voltage], format="csr"),
            sparse.hstack(
                [
                    sparse.csr_array(jacobian_reactive_power_angle),
                    sparse.csr_array(jacobian_reactive_power_voltage),
                ],
                format="csr",
            ),
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
    jacobian_data: JacobianInterface,
    static_network_data: StaticNetworkInformation | None = None,
    q_limit_tolerance: float = 1e-10,
) -> Float[np.ndarray, " n_eq_jacobian"]:
    """Calculate the nodal mismatches nodal mismatches.

    Parameters
    ----------
    dynamic_network_data : DynamicNetworkInformation
        The dynamic network data.
    y_matrix : sparse.sparray
        The admittance matrix.
    jacobian_data : JacobianInterface
        Jacobian layout defining the active angle and magnitude equation order.
    static_network_data : StaticNetworkInformation | None, optional
        Static network information used to apply reactive-power limit residuals
        for locally regulated buses when available.
    q_limit_tolerance : float, optional
        Numerical tolerance used when deciding whether reactive-power limits are
        active.

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

    angle_bus_indices = jacobian_data.bus_angle_indices
    magnitude_bus_indices = jacobian_data.bus_magnitude_indices
    lower_residual = _calculate_lower_residual(
        dynamic_network_data=dynamic_network_data,
        mismatch=mismatch,
        magnitude_bus_indices=magnitude_bus_indices,
        static_network_data=static_network_data,
        q_limit_tolerance=q_limit_tolerance,
    )

    return np.r_[mismatch[angle_bus_indices].real, lower_residual]


def _apply_jacobian_dx_to_network_data(
    dynamic_network_data: DynamicNetworkInformation,
    dx: Float[np.ndarray, " n_eq_jacobian"],
    jacobian_data: JacobianInterface,
) -> DynamicNetworkInformation:
    """Apply one Jacobian-ordered state increment to the network voltages.

    Parameters
    ----------
    dynamic_network_data : DynamicNetworkInformation
        Dynamic network data whose voltage state is updated.
    dx : Float[np.ndarray, " n_eq_jacobian"]
        State increment in the active Jacobian ordering.
    jacobian_data : JacobianInterface
        Jacobian layout defining which buses receive angle and magnitude
        updates from ``dx``.

    Returns
    -------
    DynamicNetworkInformation
        Copy of ``dynamic_network_data`` with updated voltage angles and magnitudes.
    """
    angle_bus_indices = jacobian_data.bus_angle_indices
    magnitude_bus_indices = jacobian_data.bus_magnitude_indices
    n_angle = angle_bus_indices.size
    n_magnitude = magnitude_bus_indices.size
    expected_size = n_angle + n_magnitude

    if dx.shape[0] != expected_size:
        raise ValueError(f"Jacobian increment has size {dx.shape[0]}, expected {expected_size} for the given network data.")

    updated_angles = dynamic_network_data.bus_voltage_angles_rad.copy()
    updated_magnitudes = dynamic_network_data.bus_voltage_magnitudes.copy()
    updated_angles[angle_bus_indices] += dx[:n_angle]
    updated_magnitudes[magnitude_bus_indices] += dx[n_angle : n_angle + n_magnitude]

    return replace_network_data(
        dynamic_network_data,
        bus_voltage_angles_rad=updated_angles,
        bus_voltage_magnitudes=updated_magnitudes,
    )
