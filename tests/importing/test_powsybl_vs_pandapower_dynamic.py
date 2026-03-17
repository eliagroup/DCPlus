# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Cross-check Powsybl and Pandapower importers on shared benchmark grids."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import pandapower as pp
import pypowsybl
import pytest

from dc_plus.importing.pandapower.pandapower_import import (
    _get_limits_parameter_pandapower as _get_limits_parameter_pp,
)
from dc_plus.importing.powsybl.powsybl_import import (
    _get_limits_parameter_powsybl as _get_limits_parameter_psb,
)
from dc_plus.importing.powsybl.powsybl_network_helpers import (
    get_powsybl_loadflow_parameter,
    load_pandapower_net_for_powsybl,
)
from dc_plus.interfaces.network_information import (
    DynamicNetworkInformation,
    StringNetworkInformation,
)
from dc_plus.preprocess.create_network_data import (
    create_network_data,
    create_network_data_pandapower,
)

from .test_compare_imports import NetworkImportComparator


def _normalize_limit_label(limit_type: str) -> str:
    text = str(limit_type).strip().lower()
    if not text:
        return "unknown"
    if "current" in text:
        return "current"
    if "power" in text:
        return "apparent_power"
    return text


def _canonical_limit_names_from_df(limit_df, branch_id_map: dict[str, str], allowed_types: set[str] | None) -> np.ndarray:
    if limit_df is None or getattr(limit_df, "empty", True):
        return np.array([], dtype=str)
    if allowed_types is not None and allowed_types:
        mask = [str(limit_type).upper() in allowed_types for limit_type in limit_df["limit_type"]]
        limit_df = limit_df.loc[mask]
        if getattr(limit_df, "empty", True):
            return np.array([], dtype=str)
    canonical: list[str] = []
    seen: set[tuple[str, str]] = set()
    for element_id, limit_type in zip(limit_df["element_id_str"], limit_df["limit_type"], strict=False):
        canonical_branch = branch_id_map.get(str(element_id), str(element_id))
        label = _normalize_limit_label(limit_type)
        key = (canonical_branch, label)
        if key in seen:
            continue
        seen.add(key)
        canonical.append(f"{canonical_branch}::{label}")
    if not canonical:
        return np.array([], dtype=str)
    return np.sort(np.asarray(canonical, dtype=str))


def _branch_type_priority(value: str) -> int:
    branch = value.upper()
    if branch == "LINE":
        return 0
    if branch in {"IMPEDANCE", "TIE_LINE", "HVDC_LINE"}:
        return 1
    if branch in {"TWO_WINDINGS_TRANSFORMER", "TRAFO"}:
        return 2
    if branch in {"TRAFO3W_LV", "TRAFO3W_MV", "TRAFO3W_HV"}:
        return 3
    return 4


def _injection_type_priority(value: str) -> int:
    injection = value.upper()
    if injection in {"GENERATOR", "EXT_GRID", "SGEN", "BATTERY"}:
        return 0
    if injection in {"STATIC_VAR_COMPENSATOR", "SHUNT_COMPENSATOR"}:
        return 1
    if injection == "HVDC_CONVERTER_STATION":
        return 2
    if injection == "DANGLING_LINE":
        return 3
    if injection == "LOAD":
        return 4
    return 5


def _numeric_suffix(identifier: str) -> int:
    parts = str(identifier).split("_")
    for token in reversed(parts):
        if token.lstrip("-").isdigit():
            return int(token)
    digits = "".join(ch for ch in str(identifier) if ch.isdigit())
    return int(digits) if digits else 0


def _bus_indices_from_powsybl(strings: StringNetworkInformation) -> np.ndarray:
    if strings.bus_ids.size == 0:
        return np.array([], dtype=int)
    return np.array([int(bus.split("_")[1]) for bus in strings.bus_ids], dtype=int)


def _bus_indices_from_pandapower(strings: StringNetworkInformation) -> np.ndarray:
    if strings.bus_ids.size == 0:
        return np.array([], dtype=int)
    return np.array([int(bus) for bus in strings.bus_ids], dtype=int)


def _branch_indices_from_powsybl(strings: StringNetworkInformation) -> np.ndarray:
    if strings.branch_ids.size == 0:
        return np.array([], dtype=int)
    return np.array([int(branch.split("_")[2]) for branch in strings.branch_ids], dtype=int)


def _branch_indices_from_pandapower(strings: StringNetworkInformation) -> np.ndarray:
    branch_types = np.array([bt.upper() for bt in strings.branch_types], dtype=str)
    if branch_types.size == 0:
        return np.array([], dtype=int)
    offsets: dict[str, int] = {}
    total = 0
    for branch_type in sorted({bt for bt in branch_types}, key=_branch_type_priority):
        offsets[branch_type] = total
        total += int(np.count_nonzero(branch_types == branch_type))
    return np.array(
        [
            offsets[bt] + _numeric_suffix(identifier)
            for bt, identifier in zip(branch_types, strings.branch_ids, strict=False)
        ],
        dtype=int,
    )


def _canonical_injection_order(injection_types: Sequence[str], injection_bus_indices: np.ndarray) -> np.ndarray:
    if injection_bus_indices.size == 0:
        return np.array([], dtype=int)
    priorities = np.array([_injection_type_priority(t) for t in injection_types], dtype=int)
    tie_breaker = np.arange(injection_bus_indices.size)
    return np.lexsort((tie_breaker, injection_bus_indices, priorities))


def _canonical_shunt_order(shunt_bus_indices: np.ndarray, shunt_ids: Sequence[str]) -> np.ndarray:
    if shunt_bus_indices.size == 0:
        return np.array([], dtype=int)
    tie_breaker = np.arange(shunt_bus_indices.size)
    return np.lexsort((tie_breaker, shunt_bus_indices))


def _canonical_branch_ids(branch_from_bus: np.ndarray, branch_to_bus: np.ndarray) -> np.ndarray:
    return np.array(
        [f"{int(f)}-{int(t)}-{idx}" for idx, (f, t) in enumerate(zip(branch_from_bus, branch_to_bus, strict=False))],
        dtype=str,
    )


def _canonical_shunt_ids(shunt_bus_indices: np.ndarray) -> np.ndarray:
    if shunt_bus_indices.size == 0:
        return np.array([], dtype=str)
    return np.array([f"{int(bus)}-{idx}" for idx, bus in enumerate(shunt_bus_indices)], dtype=str)


def _prepare_powsybl_network(
    dynamic_psb: DynamicNetworkInformation,
    string_psb: StringNetworkInformation,
    pp_net: pp.pandapowerNet,
    psb_net: pypowsybl.network.Network,
    limit_df_psb,
    allowed_limit_types: set[str] | None,
) -> tuple[DynamicNetworkInformation, StringNetworkInformation]:
    original_branch_ids = np.asarray(string_psb.branch_ids, dtype=str)

    bus_index_map = _bus_indices_from_powsybl(string_psb)
    bus_perm = np.argsort(bus_index_map)

    branch_indices = _branch_indices_from_powsybl(string_psb)
    branch_perm = np.argsort(branch_indices)

    injection_bus_indices = (
        bus_index_map[dynamic_psb.injection_to_bus.astype(int)]
        if dynamic_psb.injection_to_bus.size
        else np.array([], dtype=int)
    )
    injection_perm = _canonical_injection_order(string_psb.injection_types, injection_bus_indices)

    shunt_bus_indices = (
        bus_index_map[dynamic_psb.shunt_bus_indices.astype(int)]
        if dynamic_psb.shunt_bus_indices.size
        else np.array([], dtype=int)
    )
    shunt_perm = _canonical_shunt_order(shunt_bus_indices, string_psb.shunt_ids)

    bus_voltage_magnitudes = dynamic_psb.bus_voltage_magnitudes[bus_perm].copy()
    bus_voltage_angles_rad = dynamic_psb.bus_voltage_angles_rad[bus_perm].copy()
    bus_active_power = (-dynamic_psb.bus_active_power[bus_perm]).copy()
    bus_reactive_power = (-dynamic_psb.bus_reactive_power[bus_perm]).copy()
    bus_type = dynamic_psb.bus_type[bus_perm].copy()

    mapped_branch_from = (
        bus_index_map[dynamic_psb.branch_from_bus.astype(int)]
        if dynamic_psb.branch_from_bus.size
        else np.array([], dtype=int)
    )
    mapped_branch_to = (
        bus_index_map[dynamic_psb.branch_to_bus.astype(int)] if dynamic_psb.branch_to_bus.size else np.array([], dtype=int)
    )
    branch_from_bus = mapped_branch_from[branch_perm].copy() if mapped_branch_from.size else mapped_branch_from
    branch_to_bus = mapped_branch_to[branch_perm].copy() if mapped_branch_to.size else mapped_branch_to

    branch_active_power_from = (
        dynamic_psb.branch_active_power_from[branch_perm].copy()
        if dynamic_psb.branch_active_power_from.size
        else dynamic_psb.branch_active_power_from
    )
    branch_active_power_to = (
        dynamic_psb.branch_active_power_to[branch_perm].copy()
        if dynamic_psb.branch_active_power_to.size
        else dynamic_psb.branch_active_power_to
    )
    branch_reactive_power_from = (
        dynamic_psb.branch_reactive_power_from[branch_perm].copy()
        if dynamic_psb.branch_reactive_power_from.size
        else dynamic_psb.branch_reactive_power_from
    )
    branch_reactive_power_to = (
        dynamic_psb.branch_reactive_power_to[branch_perm].copy()
        if dynamic_psb.branch_reactive_power_to.size
        else dynamic_psb.branch_reactive_power_to
    )

    branch_current_magnitude_from = dynamic_psb.branch_current_magnitude_from[branch_perm].copy()
    branch_current_magnitude_to = dynamic_psb.branch_current_magnitude_to[branch_perm].copy()
    branch_ratio_tap_positions = dynamic_psb.branch_ratio_tap_positions[branch_perm].copy()
    branch_phase_tap_positions = dynamic_psb.branch_phase_tap_positions[branch_perm].copy()
    branch_effective_admittance_from_to = dynamic_psb.branch_effective_admittance_from_to[branch_perm].copy()
    branch_effective_admittance_from_from = dynamic_psb.branch_effective_admittance_from_from[branch_perm].copy()
    branch_effective_admittance_to_to = dynamic_psb.branch_effective_admittance_to_to[branch_perm].copy()
    branch_effective_admittance_to_from = dynamic_psb.branch_effective_admittance_to_from[branch_perm].copy()
    branch_effective_admittance_series = dynamic_psb.branch_effective_admittance_series[branch_perm].copy()
    branch_effective_admittance_charging_symmetric = dynamic_psb.branch_effective_admittance_charging_symmetric[
        branch_perm
    ].copy()
    branch_connected = dynamic_psb.branch_connected[branch_perm].copy()
    is_branch_symmetric = dynamic_psb.is_branch_symmetric[branch_perm].copy()
    is_connected_to_slack = dynamic_psb.is_connected_to_slack[branch_perm].copy()

    injection_to_bus = injection_bus_indices[injection_perm].copy() if injection_bus_indices.size else injection_bus_indices
    injection_active_power = (-dynamic_psb.injection_active_power[injection_perm]).copy()
    injection_reactive_power = (-dynamic_psb.injection_reactive_power[injection_perm]).copy()
    injection_connected = dynamic_psb.injection_connected[injection_perm].copy()

    shunt_bus_aligned = shunt_bus_indices[shunt_perm].copy() if shunt_bus_indices.size else shunt_bus_indices
    shunt_active_power = (-dynamic_psb.shunt_active_power[shunt_perm]).copy()
    shunt_reactive_power = (-dynamic_psb.shunt_reactive_power[shunt_perm]).copy()
    shunt_section_count = dynamic_psb.shunt_section_count[shunt_perm].copy()
    shunt_effective_bus_admittance = dynamic_psb.shunt_effective_bus_admittance[shunt_perm].copy()
    shunt_connected = dynamic_psb.shunt_connected[shunt_perm].copy()

    dynamic_psb_aligned = replace(
        dynamic_psb,
        branch_from_bus=branch_from_bus,
        branch_to_bus=branch_to_bus,
        branch_active_power_from=branch_active_power_from,
        branch_reactive_power_from=branch_reactive_power_from,
        branch_active_power_to=branch_active_power_to,
        branch_reactive_power_to=branch_reactive_power_to,
        branch_current_magnitude_from=branch_current_magnitude_from,
        branch_current_magnitude_to=branch_current_magnitude_to,
        branch_ratio_tap_positions=branch_ratio_tap_positions,
        branch_phase_tap_positions=branch_phase_tap_positions,
        branch_effective_admittance_from_to=branch_effective_admittance_from_to,
        branch_effective_admittance_from_from=branch_effective_admittance_from_from,
        branch_effective_admittance_to_to=branch_effective_admittance_to_to,
        branch_effective_admittance_to_from=branch_effective_admittance_to_from,
        branch_effective_admittance_series=branch_effective_admittance_series,
        branch_effective_admittance_charging_symmetric=branch_effective_admittance_charging_symmetric,
        branch_connected=branch_connected,
        is_branch_symmetric=is_branch_symmetric,
        is_connected_to_slack=is_connected_to_slack,
        bus_voltage_magnitudes=bus_voltage_magnitudes,
        bus_voltage_angles_rad=bus_voltage_angles_rad,
        bus_active_power=bus_active_power,
        bus_reactive_power=bus_reactive_power,
        bus_type=bus_type,
        injection_to_bus=injection_to_bus,
        injection_active_power=injection_active_power,
        injection_reactive_power=injection_reactive_power,
        injection_connected=injection_connected,
        shunt_bus_indices=shunt_bus_aligned,
        shunt_active_power=shunt_active_power,
        shunt_reactive_power=shunt_reactive_power,
        shunt_section_count=shunt_section_count,
        shunt_effective_bus_admittance=shunt_effective_bus_admittance,
        shunt_connected=shunt_connected,
    )

    canonical_bus_indices_psb = bus_index_map[bus_perm]
    if canonical_bus_indices_psb.size:
        canonical_bus_indices_psb = canonical_bus_indices_psb - canonical_bus_indices_psb.min()
    canonical_bus_ids = np.array([str(int(idx)) for idx in canonical_bus_indices_psb], dtype=str)
    branch_types_canonical = np.array([bt.upper() for bt in string_psb.branch_types], dtype=str)[branch_perm]
    canonical_branch_ids = _canonical_branch_ids(branch_from_bus, branch_to_bus)
    branch_id_reference = original_branch_ids[branch_perm] if original_branch_ids.size else original_branch_ids
    branch_id_map = {
        original: canonical for original, canonical in zip(branch_id_reference, canonical_branch_ids, strict=False)
    }
    canonical_shunt_ids = _canonical_shunt_ids(shunt_bus_aligned)
    canonical_injection_types = np.array([t.upper() for t in string_psb.injection_types], dtype=str)[injection_perm]
    canonical_limit_names = _canonical_limit_names_from_df(limit_df_psb, branch_id_map, allowed_limit_types)

    string_psb_aligned = replace(
        string_psb,
        bus_ids=canonical_bus_ids,
        branch_types=branch_types_canonical,
        branch_ids=canonical_branch_ids,
        shunt_ids=canonical_shunt_ids,
        injection_types=canonical_injection_types,
        limit_names=canonical_limit_names,
    )

    return dynamic_psb_aligned, string_psb_aligned


def _prepare_pandapower_network(
    dynamic_pp: DynamicNetworkInformation,
    string_pp: StringNetworkInformation,
    pp_net: pp.pandapowerNet,
    limit_df_pp,
    allowed_limit_types: set[str] | None,
) -> tuple[DynamicNetworkInformation, StringNetworkInformation]:
    original_branch_ids = np.asarray(string_pp.branch_ids, dtype=str)

    bus_indices = _bus_indices_from_pandapower(string_pp)
    bus_perm = np.argsort(bus_indices)

    branch_indices = _branch_indices_from_pandapower(string_pp)
    branch_perm = np.argsort(branch_indices)

    injection_bus_indices = dynamic_pp.injection_to_bus.astype(int)
    injection_perm = _canonical_injection_order(string_pp.injection_types, injection_bus_indices)

    shunt_bus_indices = dynamic_pp.shunt_bus_indices.astype(int)
    shunt_perm = _canonical_shunt_order(shunt_bus_indices, string_pp.shunt_ids)

    dynamic_pp_aligned = replace(
        dynamic_pp,
        branch_from_bus=dynamic_pp.branch_from_bus[branch_perm].copy(),
        branch_to_bus=dynamic_pp.branch_to_bus[branch_perm].copy(),
        branch_active_power_from=dynamic_pp.branch_active_power_from[branch_perm].copy(),
        branch_reactive_power_from=dynamic_pp.branch_reactive_power_from[branch_perm].copy(),
        branch_active_power_to=dynamic_pp.branch_active_power_to[branch_perm].copy(),
        branch_reactive_power_to=dynamic_pp.branch_reactive_power_to[branch_perm].copy(),
        branch_current_magnitude_from=dynamic_pp.branch_current_magnitude_from[branch_perm].copy(),
        branch_current_magnitude_to=dynamic_pp.branch_current_magnitude_to[branch_perm].copy(),
        branch_ratio_tap_positions=dynamic_pp.branch_ratio_tap_positions[branch_perm].copy(),
        branch_phase_tap_positions=dynamic_pp.branch_phase_tap_positions[branch_perm].copy(),
        branch_effective_admittance_from_to=dynamic_pp.branch_effective_admittance_from_to[branch_perm].copy(),
        branch_effective_admittance_from_from=dynamic_pp.branch_effective_admittance_from_from[branch_perm].copy(),
        branch_effective_admittance_to_to=dynamic_pp.branch_effective_admittance_to_to[branch_perm].copy(),
        branch_effective_admittance_to_from=dynamic_pp.branch_effective_admittance_to_from[branch_perm].copy(),
        branch_effective_admittance_series=dynamic_pp.branch_effective_admittance_series[branch_perm].copy(),
        branch_effective_admittance_charging_symmetric=dynamic_pp.branch_effective_admittance_charging_symmetric[
            branch_perm
        ].copy(),
        branch_connected=dynamic_pp.branch_connected[branch_perm].copy(),
        is_branch_symmetric=dynamic_pp.is_branch_symmetric[branch_perm].copy(),
        is_connected_to_slack=dynamic_pp.is_connected_to_slack[branch_perm].copy(),
        bus_voltage_magnitudes=dynamic_pp.bus_voltage_magnitudes[bus_perm].copy(),
        bus_voltage_angles_rad=dynamic_pp.bus_voltage_angles_rad[bus_perm].copy(),
        bus_active_power=dynamic_pp.bus_active_power[bus_perm].copy(),
        bus_reactive_power=dynamic_pp.bus_reactive_power[bus_perm].copy(),
        bus_type=dynamic_pp.bus_type[bus_perm].copy(),
        injection_to_bus=dynamic_pp.injection_to_bus[injection_perm].copy(),
        injection_active_power=dynamic_pp.injection_active_power[injection_perm].copy(),
        injection_reactive_power=dynamic_pp.injection_reactive_power[injection_perm].copy(),
        injection_connected=dynamic_pp.injection_connected[injection_perm].copy(),
        shunt_bus_indices=dynamic_pp.shunt_bus_indices[shunt_perm].copy(),
        shunt_active_power=dynamic_pp.shunt_active_power[shunt_perm].copy(),
        shunt_reactive_power=dynamic_pp.shunt_reactive_power[shunt_perm].copy(),
        shunt_section_count=dynamic_pp.shunt_section_count[shunt_perm].copy(),
        shunt_effective_bus_admittance=dynamic_pp.shunt_effective_bus_admittance[shunt_perm].copy(),
        shunt_connected=dynamic_pp.shunt_connected[shunt_perm].copy(),
    )

    canonical_bus_indices_pp = bus_indices[bus_perm]
    if canonical_bus_indices_pp.size:
        canonical_bus_indices_pp = canonical_bus_indices_pp - canonical_bus_indices_pp.min()
    canonical_bus_ids = np.array([str(int(idx)) for idx in canonical_bus_indices_pp], dtype=str)
    branch_types_canonical = np.array([bt.upper() for bt in string_pp.branch_types], dtype=str)[branch_perm]
    canonical_branch_ids = _canonical_branch_ids(
        dynamic_pp_aligned.branch_from_bus.astype(int), dynamic_pp_aligned.branch_to_bus.astype(int)
    )
    branch_id_reference = original_branch_ids[branch_perm] if original_branch_ids.size else original_branch_ids
    branch_id_map = {
        original: canonical for original, canonical in zip(branch_id_reference, canonical_branch_ids, strict=False)
    }
    canonical_shunt_ids = _canonical_shunt_ids(dynamic_pp_aligned.shunt_bus_indices.astype(int))
    canonical_injection_types = np.array([t.upper() for t in string_pp.injection_types], dtype=str)[injection_perm]
    canonical_limit_names = _canonical_limit_names_from_df(limit_df_pp, branch_id_map, allowed_limit_types)

    string_pp_aligned = replace(
        string_pp,
        bus_ids=canonical_bus_ids,
        branch_types=branch_types_canonical,
        branch_ids=canonical_branch_ids,
        shunt_ids=canonical_shunt_ids,
        injection_types=canonical_injection_types,
        limit_names=canonical_limit_names,
    )

    return dynamic_pp_aligned, string_pp_aligned


@pytest.mark.parametrize(
    ("network_func", "network_name"),
    [
        (pp.networks.case9, "case9"),
        (pp.networks.case14, "case14"),
        (pp.networks.case30, "case30"),
    ],
)
def test_powsybl_vs_pandapower_imports(network_func, network_name):
    pp_net = network_func()
    pp.runpp(pp_net, calculate_voltage_angles=True)

    try:
        psb_net = load_pandapower_net_for_powsybl(pp_net)
    except Exception as exc:
        pytest.fail(f"{network_name}: conversion failed ({exc})")

    lf_params = get_powsybl_loadflow_parameter("academic")
    result = pypowsybl.loadflow.run_ac(psb_net, parameters=lf_params)
    if result[0].status.name != "CONVERGED":
        pytest.fail(f"{network_name}: Powsybl load flow did not converge ({result[0].status.name})")

    static_psb, dynamic_psb_raw, string_psb_raw = create_network_data(psb_net)
    static_pp, dynamic_pp_raw, string_pp_raw = create_network_data_pandapower(pp_net)

    limit_df_psb = _get_limits_parameter_psb(psb_net)
    limit_df_pp = _get_limits_parameter_pp(pp_net)
    allowed_limit_types = set()
    if limit_df_psb is not None and not getattr(limit_df_psb, "empty", True):
        allowed_limit_types = {str(value).upper() for value in limit_df_psb["limit_type"]}
    if not allowed_limit_types and limit_df_pp is not None and not getattr(limit_df_pp, "empty", True):
        allowed_limit_types = {str(value).upper() for value in limit_df_pp["limit_type"]}

    dynamic_psb, string_psb = _prepare_powsybl_network(
        dynamic_psb_raw,
        string_psb_raw,
        pp_net,
        psb_net,
        limit_df_psb,
        allowed_limit_types,
    )
    dynamic_pp, string_pp = _prepare_pandapower_network(
        dynamic_pp_raw,
        string_pp_raw,
        pp_net,
        limit_df_pp,
        allowed_limit_types,
    )

    comparator = NetworkImportComparator(tolerance=1e-9)
    comparison = comparator.compare_dynamic_network_info(dynamic_psb, dynamic_pp)

    assert comparison.buses_match, f"{network_name}: bus data mismatch (max |ΔV|={comparison.max_voltage_diff:.2e})"
    assert comparison.branches_match, f"{network_name}: branch data mismatch (max |ΔS|={comparison.max_power_diff:.2e})"
    shunt_details = comparison.details["shunts"]
    assert shunt_details.get("bus_match", True), f"{network_name}: shunt bus assignment mismatch"
    assert shunt_details.get("status_match", True), f"{network_name}: shunt status mismatch"
    assert comparison.injections_match, f"{network_name}: injection data mismatch"
    assert comparison.voltage_match, f"{network_name}: voltage mismatch (max |ΔV|={comparison.max_voltage_diff:.2e})"
    assert comparison.power_flow_match, f"{network_name}: power flow mismatch (max |ΔS|={comparison.max_power_diff:.2e})"

    assert static_psb == static_pp, f"{network_name}: static network information diverges"

    np.testing.assert_array_equal(
        string_psb.bus_ids,
        string_pp.bus_ids,
        err_msg=f"{network_name}: canonical bus identifiers mismatch",
    )
    np.testing.assert_array_equal(
        string_psb.branch_ids,
        string_pp.branch_ids,
        err_msg=f"{network_name}: canonical branch identifiers mismatch",
    )
    np.testing.assert_array_equal(
        string_psb.branch_types,
        string_pp.branch_types,
        err_msg=f"{network_name}: branch types mismatch",
    )
    np.testing.assert_array_equal(
        string_psb.injection_types,
        string_pp.injection_types,
        err_msg=f"{network_name}: injection types mismatch",
    )
    np.testing.assert_array_equal(
        string_psb.shunt_ids,
        string_pp.shunt_ids,
        err_msg=f"{network_name}: shunt identifiers mismatch",
    )
    np.testing.assert_array_equal(
        string_psb.limit_names,
        string_pp.limit_names,
        err_msg=f"{network_name}: limit names mismatch",
    )
