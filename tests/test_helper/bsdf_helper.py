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

from dc_plus.example_grids.pypowsbl.example_grids import (
    basic_node_breaker_network_powsybl,
    create_complex_grid_battery_hvdc_svc_3w_trafo,
)
from dc_plus.importing.powsybl.powsybl_network_helpers import _load_test_grid
from dc_plus.importing.powsybl.powsybl_loadflow_parameter import get_powsybl_loadflow_parameter
from dc_plus.interfaces.jacobian_network_data import (
    _get_admittance_matrix_from_network_data,
    _get_jacobian_data_from_network_data,
    calculate_nodal_mismatch_network_data,
)
from dc_plus.interfaces.network_information import DynamicNetworkInformation, StringNetworkInformation
from dc_plus.preprocess.create_network_data import create_network_data_pypowsybl
from dc_plus.preprocess.preprocess_jacobian_bsdf import preprocess_jacobian_bsdf


_NEW_BUS_PLACEHOLDER_ID = "__new_bus_b__"


@dataclass
class BsdfTestCase:
    get_net: Callable[..., Any]
    """Callable that returns a pypowsybl network instance for the test case."""
    bus_to_split: int
    """Index of the bus to split in the BSDF test case.
    Note: this index number refers to the DC+ internal ordering of buses, which differs
    from the pypowsybl bus numbering. (string index is sorted by name -> internal index)
    """
    branches_connected_to_bus_b_string: list[str]
    """Powsybl/DC+ branch IDs connected to the new bus B after the split."""

    shunt_connected_to_bus_b_string: list[str]
    """Powsybl/DC+ shunt IDs connected to the new bus B after the split."""
    injections_connected_to_bus_b_string: list[str]
    """Powsybl/DC+ injection IDs connected to the new bus B after the split.
    Note: you may only reassign injections which have no PV regulation.
    If you want to reassign a PV injection, you need to change the bus b from default type PQ to PV."""
    open_switches: tuple[str, ...]
    """pypowsybl IDs of switches to open in the reference switch pattern for the test case."""
    close_switches: tuple[str, ...]
    """pypowsybl IDs of switches to close in the reference switch pattern for the test case."""


@dataclass
class BsdfTestContext:
    net: Any
    dynamic_info: Any
    dynamic_info_with_placeholders: Any
    dynamic_info_split_manual: Any
    jacobian_data_with_extra_buses: Any
    jacobian_data_split_manual: Any
    new_bus_index: int
    split_bus_ids: np.ndarray
    bus_to_split: int
    branches_connected_to_bus_b: np.ndarray
    shunt_connected_to_bus_b: np.ndarray
    injections_connected_to_bus_b: np.ndarray
    y_ff: np.ndarray
    y_ft: np.ndarray
    y_tf: np.ndarray
    y_tt: np.ndarray
    branch_from_original: np.ndarray
    branch_to_original: np.ndarray
    v_mag_hat: np.ndarray
    theta_hat: np.ndarray
    mismatch_bsdf_reference: np.ndarray
    theta_base: np.ndarray
    vm_base: np.ndarray
    pvpq_indices: np.ndarray
    pq_indices: np.ndarray


def get_bsdf_cases() -> List[BsdfTestCase]:
    """Return test cases for BSDF tests."""

    # simple_bsdf_test_case
    # based on basic_node_breaker_network_powsybl
    # simple reasignment and
    get_net = basic_node_breaker_network_powsybl
    bus_to_split = 2
    branches_connected_to_bus_b_string = ["L3", "L6"]
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
    simple_bsdf_test_case = BsdfTestCase(
        get_net=get_net,
        bus_to_split=bus_to_split,
        branches_connected_to_bus_b_string=branches_connected_to_bus_b_string,
        open_switches=open_switches,
        close_switches=close_switches,
        shunt_connected_to_bus_b_string=[],
        injections_connected_to_bus_b_string=[],
    )

    get_net = create_complex_grid_battery_hvdc_svc_3w_trafo
    bus_to_split = 13
    branches_connected_to_bus_b_string = ["L12", "L6", "L7"]
    open_switches = (
        "VL_MV_BREAKER",
        "SHUNT_MV_DISCONNECTOR_21_0",
    )
    close_switches = ("SHUNT_MV_DISCONNECTOR_21_1",)
    shunt_injection_test_case = BsdfTestCase(
        get_net=get_net,
        bus_to_split=bus_to_split,
        branches_connected_to_bus_b_string=branches_connected_to_bus_b_string,
        open_switches=open_switches,
        close_switches=close_switches,
        shunt_connected_to_bus_b_string=["SHUNT_MV"],
        injections_connected_to_bus_b_string=["BAT_MV"],
    )

    return [simple_bsdf_test_case, shunt_injection_test_case]


def _reassign_selected_bus_indices(
    asset_bus_indices: np.ndarray,
    selected_asset_indices: np.ndarray,
    bus_to_split: int,
    new_bus_index: int,
) -> np.ndarray:
    """Reassign the selected assets from the split bus to the new bus."""
    reassigned_bus_indices = np.asarray(asset_bus_indices).copy()
    for asset_idx in np.asarray(selected_asset_indices, dtype=np.int32).reshape(-1):
        reassigned_bus_indices[asset_idx] = np.where(
            reassigned_bus_indices[asset_idx] == bus_to_split,
            new_bus_index,
            reassigned_bus_indices[asset_idx],
        )
    return reassigned_bus_indices


def _build_split_bus_ids(original_bus_ids: np.ndarray) -> np.ndarray:
    """Append a placeholder bus ID for the newly created split bus."""
    bus_ids = np.asarray(original_bus_ids, dtype=object)
    return np.concatenate((bus_ids, np.asarray([_NEW_BUS_PLACEHOLDER_ID], dtype=object)))


def derive_bus_order(candidate_bus_ids: np.ndarray, reference_bus_ids: np.ndarray) -> np.ndarray:
    """Map candidate bus ordering to the powsybl reference bus ordering.

    The candidate ordering contains all original bus IDs plus one placeholder for the
    newly created split bus. The reference ordering contains the actual new bus ID.
    """
    candidate_ids = np.asarray(candidate_bus_ids, dtype=str)
    reference_ids = np.asarray(reference_bus_ids, dtype=str)

    if candidate_ids.size != reference_ids.size:
        raise ValueError(
            "Candidate and reference bus ID arrays must have the same length to derive bus ordering. "
            f"Got candidate={candidate_ids.size}, reference={reference_ids.size}."
        )

    candidate_lookup = {
        bus_id: idx for idx, bus_id in enumerate(candidate_ids.tolist()) if bus_id != _NEW_BUS_PLACEHOLDER_ID
    }
    bus_order = np.full(reference_ids.size, -1, dtype=np.int32)
    used_candidate_indices: set[int] = set()
    unresolved_reference_positions: list[int] = []

    for ref_pos, ref_bus_id in enumerate(reference_ids.tolist()):
        candidate_idx = candidate_lookup.get(ref_bus_id)
        if candidate_idx is None:
            unresolved_reference_positions.append(ref_pos)
            continue

        bus_order[ref_pos] = candidate_idx
        used_candidate_indices.add(candidate_idx)

    remaining_candidate_indices = [
        idx
        for idx in range(candidate_ids.size)
        if idx not in used_candidate_indices and candidate_ids[idx] == _NEW_BUS_PLACEHOLDER_ID
    ]

    if len(unresolved_reference_positions) != len(remaining_candidate_indices):
        raise ValueError(
            "Could not derive a unique bus order from bus IDs. "
            f"Unresolved reference positions={unresolved_reference_positions}, "
            f"remaining candidate indices={remaining_candidate_indices}, "
            f"reference bus ids={reference_ids.tolist()}, candidate bus ids={candidate_ids.tolist()}."
        )

    for ref_pos, candidate_idx in zip(unresolved_reference_positions, remaining_candidate_indices):
        bus_order[ref_pos] = candidate_idx

    return bus_order


def _aggregate_bus_injections(
    injection_to_bus: np.ndarray,
    injection_active_power: np.ndarray,
    injection_reactive_power: np.ndarray,
    injection_connected: np.ndarray,
    n_buses: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-injection powers back to bus totals."""
    injection_to_bus_arr = np.asarray(injection_to_bus, dtype=int)
    injection_active_power_arr = np.asarray(injection_active_power, dtype=float)
    injection_reactive_power_arr = np.asarray(injection_reactive_power, dtype=float)
    injection_connected_arr = np.asarray(injection_connected, dtype=bool)

    if injection_to_bus_arr.ndim == 1:
        bus_active_power = np.zeros(n_buses, dtype=injection_active_power_arr.dtype)
        bus_reactive_power = np.zeros(n_buses, dtype=injection_reactive_power_arr.dtype)
        connected_mask = injection_connected_arr.astype(bool)
        np.add.at(bus_active_power, injection_to_bus_arr[connected_mask], injection_active_power_arr[connected_mask])
        np.add.at(
            bus_reactive_power,
            injection_to_bus_arr[connected_mask],
            injection_reactive_power_arr[connected_mask],
        )
        return bus_active_power, bus_reactive_power

    n_timesteps = injection_to_bus_arr.shape[1]
    bus_active_power = np.zeros((n_buses, n_timesteps), dtype=injection_active_power_arr.dtype)
    bus_reactive_power = np.zeros((n_buses, n_timesteps), dtype=injection_reactive_power_arr.dtype)

    for timestep in range(n_timesteps):
        connected_mask = injection_connected_arr[:, timestep].astype(bool)
        np.add.at(
            bus_active_power[:, timestep],
            injection_to_bus_arr[connected_mask, timestep],
            injection_active_power_arr[connected_mask, timestep],
        )
        np.add.at(
            bus_reactive_power[:, timestep],
            injection_to_bus_arr[connected_mask, timestep],
            injection_reactive_power_arr[connected_mask, timestep],
        )

    return bus_active_power, bus_reactive_power


def _resolve_asset_indices_by_id(
    available_ids: np.ndarray,
    selected_ids: list[str],
    asset_kind: str,
) -> np.ndarray:
    """Resolve configured asset IDs to their DC+ internal indices."""
    available_ids_arr = np.asarray(available_ids, dtype=str)
    selected_ids_list = list(selected_ids)

    id_to_index = {asset_id: idx for idx, asset_id in enumerate(available_ids_arr.tolist())}
    missing_ids = [asset_id for asset_id in selected_ids_list if asset_id not in id_to_index]
    if missing_ids:
        raise ValueError(
            f"Unknown {asset_kind} IDs in BSDF test case: {missing_ids}. Available IDs are: {available_ids_arr.tolist()}"
        )

    return np.asarray([id_to_index[asset_id] for asset_id in selected_ids_list], dtype=np.int32)


def _validate_reassignment_targets(
    bus_to_split: int,
    branches_connected_to_bus_b: np.ndarray,
    shunt_connected_to_bus_b: np.ndarray,
    injections_connected_to_bus_b: np.ndarray,
    branch_from_bus: np.ndarray,
    branch_to_bus: np.ndarray,
    shunt_bus_indices: np.ndarray,
    injection_to_bus: np.ndarray,
) -> None:
    """Validate that all selected assets are currently attached to the split bus."""
    invalid_branch_indices = [
        int(branch_idx)
        for branch_idx in np.asarray(branches_connected_to_bus_b, dtype=np.int32).reshape(-1)
        if not np.any(np.asarray(branch_from_bus[branch_idx]) == bus_to_split)
        and not np.any(np.asarray(branch_to_bus[branch_idx]) == bus_to_split)
    ]
    invalid_shunt_indices = [
        int(shunt_idx)
        for shunt_idx in np.asarray(shunt_connected_to_bus_b, dtype=np.int32).reshape(-1)
        if not np.any(np.asarray(shunt_bus_indices[shunt_idx]) == bus_to_split)
    ]
    invalid_injection_indices = [
        int(injection_idx)
        for injection_idx in np.asarray(injections_connected_to_bus_b, dtype=np.int32).reshape(-1)
        if not np.any(np.asarray(injection_to_bus[injection_idx]) == bus_to_split)
    ]

    if invalid_branch_indices or invalid_shunt_indices or invalid_injection_indices:
        raise ValueError(
            "Invalid BSDF test case selection for bus split "
            f"{bus_to_split}: branches={invalid_branch_indices}, "
            f"shunts={invalid_shunt_indices}, injections={invalid_injection_indices}"
        )


def prepare_bsdf_test_context(bsdf_test_case: BsdfTestCase) -> BsdfTestContext:
    """Build split-bus context shared by BSDF tests to avoid duplication."""
    net, _static_info, dynamic_info, string_info, jacobian_data = _load_test_grid(bsdf_test_case.get_net)
    branches_connected_to_bus_b = _resolve_asset_indices_by_id(
        available_ids=string_info.branch_ids,
        selected_ids=bsdf_test_case.branches_connected_to_bus_b_string,
        asset_kind="branch",
    )
    shunt_connected_to_bus_b = _resolve_asset_indices_by_id(
        available_ids=string_info.shunt_ids,
        selected_ids=bsdf_test_case.shunt_connected_to_bus_b_string,
        asset_kind="shunt",
    )
    injections_connected_to_bus_b = _resolve_asset_indices_by_id(
        available_ids=string_info.injection_ids,
        selected_ids=bsdf_test_case.injections_connected_to_bus_b_string,
        asset_kind="injection",
    )
    jacobian_data_with_extra_buses, dynamic_info_with_placeholders = preprocess_jacobian_bsdf(
        jacobian_data=jacobian_data,
        max_bus_splits=1,
        dynamic_network_data=dynamic_info,
    )
    _validate_reassignment_targets(
        bus_to_split=bsdf_test_case.bus_to_split,
        branches_connected_to_bus_b=branches_connected_to_bus_b,
        shunt_connected_to_bus_b=shunt_connected_to_bus_b,
        injections_connected_to_bus_b=injections_connected_to_bus_b,
        branch_from_bus=dynamic_info.branch_from_bus,
        branch_to_bus=dynamic_info.branch_to_bus,
        shunt_bus_indices=dynamic_info.shunt_bus_indices,
        injection_to_bus=dynamic_info.injection_to_bus,
    )
    new_bus_index = dynamic_info_with_placeholders.n_buses - 1
    split_bus_ids = _build_split_bus_ids(string_info.bus_ids)
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
    for branch_idx in branches_connected_to_bus_b:
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

    shunt_bus_split = _reassign_selected_bus_indices(
        asset_bus_indices=dynamic_info_with_placeholders.shunt_bus_indices,
        selected_asset_indices=shunt_connected_to_bus_b,
        bus_to_split=bsdf_test_case.bus_to_split,
        new_bus_index=new_bus_index,
    )
    injection_to_bus_split = _reassign_selected_bus_indices(
        asset_bus_indices=dynamic_info_with_placeholders.injection_to_bus,
        selected_asset_indices=injections_connected_to_bus_b,
        bus_to_split=bsdf_test_case.bus_to_split,
        new_bus_index=new_bus_index,
    )
    bus_active_power_split, bus_reactive_power_split = _aggregate_bus_injections(
        injection_to_bus=injection_to_bus_split,
        injection_active_power=dynamic_info_with_placeholders.injection_active_power,
        injection_reactive_power=dynamic_info_with_placeholders.injection_reactive_power,
        injection_connected=dynamic_info_with_placeholders.injection_connected,
        n_buses=dynamic_info_with_placeholders.n_buses,
    )

    dynamic_info_split_manual = replace(
        dynamic_info_with_placeholders,
        branch_from_bus=branch_from_split,
        branch_to_bus=branch_to_split,
        shunt_bus_indices=shunt_bus_split,
        injection_to_bus=injection_to_bus_split,
        bus_active_power=bus_active_power_split,
        bus_reactive_power=bus_reactive_power_split,
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
    mismatch_bsdf_reference = calculate_nodal_mismatch_network_data(
        dynamic_network_data=dynamic_info_split_manual, y_matrix=y_matrix_n1
    )
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
        split_bus_ids=split_bus_ids,
        bus_to_split=bsdf_test_case.bus_to_split,
        branches_connected_to_bus_b=branches_connected_to_bus_b,
        shunt_connected_to_bus_b=shunt_connected_to_bus_b,
        injections_connected_to_bus_b=injections_connected_to_bus_b,
        y_ff=y_ff,
        y_ft=y_ft,
        y_tf=y_tf,
        y_tt=y_tt,
        branch_from_original=branch_from_original,
        branch_to_original=branch_to_original,
        v_mag_hat=v_mag_hat,
        theta_hat=theta_hat,
        mismatch_bsdf_reference=mismatch_bsdf_reference,
        theta_base=theta_base,
        vm_base=vm_base,
        pvpq_indices=pvpq_indices,
        pq_indices=pq_indices,
    )


def run_reference_one_step(
    net: Any,
    bsdf_test_case: BsdfTestCase,
) -> tuple[DynamicNetworkInformation, StringNetworkInformation]:
    """Apply the reference switch pattern and run one-step AC load-flow for the given case."""
    for switch_id in bsdf_test_case.open_switches:
        net.open_switch(switch_id)
    for switch_id in bsdf_test_case.close_switches:
        net.close_switch(switch_id)

    loadflow_parameter = get_powsybl_loadflow_parameter("one_step")
    lf_res = pypowsybl.loadflow.run_ac(net, parameters=loadflow_parameter)[0]
    assert lf_res.iteration_count == 1, (
        f"Expected the reference load-flow to converge in one step, but got {lf_res.iteration_count} iterations."
    )
    _static_info, dynamic_info_one_step, string_info_one_step = create_network_data_pypowsybl(net)
    return dynamic_info_one_step, string_info_one_step
