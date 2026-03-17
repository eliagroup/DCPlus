# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from dataclasses import dataclass, replace
from typing import Any, Callable, List

import numpy as np
import pypowsybl

from dc_plus.example_grids.pypowsbl.example_grids import basic_node_breaker_network_powsybl
from dc_plus.importing.powsybl.powsybl_network_helpers import _load_test_grid
from dc_plus.importing.powsybl.powsybl_loadflow_parameter import get_powsybl_loadflow_parameter
from dc_plus.interfaces.jacobian_network_data import (
    _get_admittance_matrix_from_network_data,
    _get_jacobian_data_from_network_data,
    calculate_nodal_mismatch_network_data,
)
from dc_plus.interfaces.network_information import DynamicNetworkInformation
from dc_plus.preprocess.create_network_data import create_network_data_pypowsbl
from dc_plus.preprocess.preprocess_jacobian_bsdf import preprocess_jacobian_bsdf


@dataclass
class BsdfTestCase:
    get_net: Callable[..., Any]
    bus_to_split: int
    branches_to_move: np.ndarray
    open_switches: tuple[str, ...]
    close_switches: tuple[str, ...]
    bus_order: np.ndarray


@dataclass
class BsdfTestContext:
    net: Any
    dynamic_info: Any
    dynamic_info_with_placeholders: Any
    dynamic_info_split_manual: Any
    jacobian_data_with_extra_buses: Any
    jacobian_data_split_manual: Any
    new_bus_index: int
    bus_to_split: int
    branches_to_move: np.ndarray
    y_ff: np.ndarray
    y_ft: np.ndarray
    y_tf: np.ndarray
    y_tt: np.ndarray
    branch_from_original: np.ndarray
    branch_to_original: np.ndarray
    v_mag_hat: np.ndarray
    theta_hat: np.ndarray
    mismatch_n1: np.ndarray
    theta_base: np.ndarray
    vm_base: np.ndarray
    pvpq_indices: np.ndarray
    pq_indices: np.ndarray


def get_bsdf_cases() -> List[BsdfTestCase]:
    """Return test cases for BSDF tests."""
    get_net = basic_node_breaker_network_powsybl
    bus_to_split = 2
    branches_to_move = np.array([2, 5], dtype=np.int32)
    open_switches = (
        "VL3_BREAKER",
        "L32_DISCONNECTOR_3_0",
        "load2_DISCONNECTOR_13_1",
        "L72_DISCONNECTOR_7_1",
        "L62_DISCONNECTOR_5_0",
    )
    close_switches = (
        "L32_DISCONNECTOR_3_1",
        "load2_DISCONNECTOR_13_0",
        "L72_DISCONNECTOR_7_0",
        "L62_DISCONNECTOR_5_1",
    )
    bus_order = np.asarray([0, 1, 2, 4, 5, 3], dtype=np.int32)
    return [
        BsdfTestCase(
            get_net=get_net,
            bus_to_split=bus_to_split,
            branches_to_move=branches_to_move,
            open_switches=open_switches,
            close_switches=close_switches,
            bus_order=bus_order,
        )
    ]


def prepare_bsdf_test_context(bsdf_test_case: BsdfTestCase) -> BsdfTestContext:
    """Build split-bus context shared by BSDF tests to avoid duplication."""
    branches_to_move = np.asarray(bsdf_test_case.branches_to_move, dtype=np.int32)
    net, _, dynamic_info, _, jacobian_data = _load_test_grid(bsdf_test_case.get_net)
    jacobian_data_with_extra_buses, dynamic_info_with_placeholders = preprocess_jacobian_bsdf(
        jacobian_data=jacobian_data,
        max_bus_splits=1,
        dynamic_network_data=dynamic_info,
    )
    new_bus_index = dynamic_info_with_placeholders.n_buses - 1
    v_mag_placeholder = dynamic_info_with_placeholders.bus_voltage_magnitudes.copy()
    theta_placeholder = dynamic_info_with_placeholders.bus_voltage_angles_rad.copy()
    v_mag_placeholder[new_bus_index] = dynamic_info.bus_voltage_magnitudes[bsdf_test_case.bus_to_split]
    theta_placeholder[new_bus_index] = dynamic_info.bus_voltage_angles_rad[bsdf_test_case.bus_to_split]
    dynamic_info_with_placeholders = replace(
        dynamic_info_with_placeholders,
        bus_voltage_magnitudes=v_mag_placeholder,
        bus_voltage_angles_rad=theta_placeholder,
    )
    branch_from_split = dynamic_info.branch_from_bus.copy()
    branch_to_split = dynamic_info.branch_to_bus.copy()
    for branch_idx in branches_to_move:
        branch_from_split[branch_idx] = np.where(
            branch_from_split[branch_idx] == bsdf_test_case.bus_to_split,
            new_bus_index,
            branch_from_split[branch_idx],
        )
        branch_to_split[branch_idx] = np.where(
            branch_to_split[branch_idx] == bsdf_test_case.bus_to_split,
            new_bus_index,
            branch_to_split[branch_idx],
        )
    dynamic_info_split_manual = replace(
        dynamic_info_with_placeholders,
        branch_from_bus=branch_from_split,
        branch_to_bus=branch_to_split,
    )
    y_ff = np.asarray(dynamic_info.branch_effective_admittance_from_from, dtype=np.complex128)
    y_ft = np.asarray(dynamic_info.branch_effective_admittance_from_to, dtype=np.complex128)
    y_tf = np.asarray(dynamic_info.branch_effective_admittance_to_from, dtype=np.complex128)
    y_tt = np.asarray(dynamic_info.branch_effective_admittance_to_to, dtype=np.complex128)
    branch_from_original = np.asarray(dynamic_info.branch_from_bus, dtype=np.int32)
    branch_to_original = np.asarray(dynamic_info.branch_to_bus, dtype=np.int32)
    v_mag_hat = np.asarray(dynamic_info_with_placeholders.bus_voltage_magnitudes, dtype=float).flatten()
    theta_hat = np.asarray(dynamic_info_with_placeholders.bus_voltage_angles_rad, dtype=float).flatten()
    y_matrix_n1 = _get_admittance_matrix_from_network_data(dynamic_info_split_manual)
    mismatch_n1 = calculate_nodal_mismatch_network_data(dynamic_network_data=dynamic_info_split_manual, y_matrix=y_matrix_n1)
    theta_base = np.asarray(dynamic_info_split_manual.bus_voltage_angles_rad, dtype=float).flatten()
    vm_base = np.asarray(dynamic_info_split_manual.bus_voltage_magnitudes, dtype=float).flatten()
    jacobian_data_split_manual = _get_jacobian_data_from_network_data(dynamic_info_split_manual)
    pvpq_indices = np.asarray(dynamic_info_split_manual.pvpq_buses_indices_pvpq_order, dtype=int)
    pq_indices = np.asarray(dynamic_info_split_manual.pq_buses_indices, dtype=int)
    return BsdfTestContext(
        net=net,
        dynamic_info=dynamic_info,
        dynamic_info_with_placeholders=dynamic_info_with_placeholders,
        dynamic_info_split_manual=dynamic_info_split_manual,
        jacobian_data_with_extra_buses=jacobian_data_with_extra_buses,
        jacobian_data_split_manual=jacobian_data_split_manual,
        new_bus_index=new_bus_index,
        bus_to_split=bsdf_test_case.bus_to_split,
        branches_to_move=branches_to_move,
        y_ff=y_ff,
        y_ft=y_ft,
        y_tf=y_tf,
        y_tt=y_tt,
        branch_from_original=branch_from_original,
        branch_to_original=branch_to_original,
        v_mag_hat=v_mag_hat,
        theta_hat=theta_hat,
        mismatch_n1=mismatch_n1,
        theta_base=theta_base,
        vm_base=vm_base,
        pvpq_indices=pvpq_indices,
        pq_indices=pq_indices,
    )


def run_reference_one_step(net: Any, bsdf_test_case: BsdfTestCase) -> DynamicNetworkInformation:
    """Apply the reference switch pattern and run one-step AC load-flow for the given case."""
    for switch_id in bsdf_test_case.open_switches:
        net.open_switch(switch_id)
    for switch_id in bsdf_test_case.close_switches:
        net.close_switch(switch_id)

    loadflow_parameter = get_powsybl_loadflow_parameter("one_step")
    pypowsybl.loadflow.run_ac(net, parameters=loadflow_parameter)[0]
    _static_info, dynamic_info_one_step, _string_info = create_network_data_pypowsbl(net)
    return dynamic_info_one_step
