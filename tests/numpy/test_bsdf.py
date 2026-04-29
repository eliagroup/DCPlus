# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from dataclasses import replace

import numpy as np
import pandas as pd
import pypowsybl
import pytest

from dc_plus.importing.powsybl.powsybl_import import DANGLING_BUS_STRING_SUFFIX
from dc_plus.interfaces.jacobian_network_data import (
    _get_admittance_matrix_from_network_data,
    _get_jacobian_data_from_network_data,
    calculate_nodal_mismatch_network_data,
)
from dc_plus.interfaces.network_information import BusType
from dc_plus.numpy.bsdf_full_rank import compute_bsdf_update
from dc_plus.preprocess.create_network_data import create_network_data_pypowsybl
from dc_plus.preprocess.preprocess_jacobian_bsdf import preprocess_jacobian_bsdf
from tests.test_helper.bsdf_helper import (
    derive_bus_order,
    get_bsdf_cases,
    prepare_bsdf_test_context,
    run_reference_one_step,
)


@pytest.mark.parametrize("bsdf_test_case", get_bsdf_cases())
def test_bsdf_full_rank(bsdf_test_case):
    setup = prepare_bsdf_test_context(bsdf_test_case=bsdf_test_case)

    jacobian_data_split_direct = setup.jacobian_data_split_manual

    jacobian_inv_bsdf = compute_bsdf_update(
        jacobian_inv=setup.jacobian_data_with_extra_buses.inverse_jacobian,
        bus_to_split=setup.bus_to_split,
        new_bus_b_index=setup.new_bus_index,
        new_bus_type=2,  # force select PQ node
        branches_connected_to_bus_b=setup.branches_connected_to_bus_b,
        shunt_connected_to_bus_b=setup.shunt_connected_to_bus_b,
        branch_from=setup.branch_from_original,
        branch_to=setup.branch_to_original,
        shunt_to_bus=setup.dynamic_info.shunt_bus_indices,
        v_mag_hat=setup.v_mag_hat,
        theta_hat=setup.theta_hat,
        y_ff=setup.y_ff,
        y_ft=setup.y_ft,
        y_tf=setup.y_tf,
        y_tt=setup.y_tt,
        y_shunt=setup.dynamic_info.shunt_effective_bus_admittance,
        angle_component_indices=setup.jacobian_data_with_extra_buses.angle_component_indices,
        magnitude_component_indices=setup.jacobian_data_with_extra_buses.magnitude_component_indices,
    )

    assert setup.jacobian_data_with_extra_buses.jacobian.shape == jacobian_data_split_direct.jacobian.shape

    in_use_indices = np.flatnonzero(setup.jacobian_data_with_extra_buses.jacobian_index_in_use)
    np.testing.assert_allclose(
        jacobian_inv_bsdf[np.ix_(in_use_indices, in_use_indices)],
        jacobian_data_split_direct.inverse_jacobian[np.ix_(in_use_indices, in_use_indices)],
        rtol=1e-10,
        atol=1e-10,
    )

    # test against powsybl
    dynamic_info_one_step, string_info_one_step = run_reference_one_step(setup.net, bsdf_test_case=bsdf_test_case)
    bus_order = derive_bus_order(setup.split_bus_ids, string_info_one_step.bus_ids)
    dx = -jacobian_inv_bsdf @ setup.mismatch_bsdf_reference

    # Map Jacobian increments back to bus ordering using the Jacobian mapping

    dynamic_info_split_manual = setup.dynamic_info_split_manual
    theta_actual = dynamic_info_split_manual.bus_voltage_angles_rad
    vm_actual = dynamic_info_split_manual.bus_voltage_magnitudes
    theta_updated_J = theta_actual.copy()
    vm_updated_J = vm_actual.copy()
    theta_updated_J[setup.pvpq_indices] = (
        theta_actual[setup.pvpq_indices] + dx[setup.jacobian_data_with_extra_buses.is_angle_component]
    )
    vm_updated_J[setup.pq_indices] = (
        vm_actual[setup.pq_indices] + dx[setup.jacobian_data_with_extra_buses.is_magnitude_component]
    )

    dangling_bus_mask = np.char.endswith(np.asarray(string_info_one_step.bus_ids, dtype=str), DANGLING_BUS_STRING_SUFFIX)

    np.testing.assert_allclose(
        dynamic_info_one_step.bus_voltage_magnitudes[~dangling_bus_mask],
        vm_updated_J[bus_order][~dangling_bus_mask],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        dynamic_info_one_step.bus_voltage_angles_rad[~dangling_bus_mask],
        theta_updated_J[bus_order][~dangling_bus_mask],
        rtol=1e-10,
        atol=1e-10,
    )

    # FIXME
    # Powsybl one-step results on dangling buses are not bitwise identical to the
    # imported Jacobian linearization
    np.testing.assert_allclose(
        dynamic_info_one_step.bus_voltage_magnitudes[dangling_bus_mask],
        vm_updated_J[bus_order][dangling_bus_mask],
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        dynamic_info_one_step.bus_voltage_angles_rad[dangling_bus_mask],
        theta_updated_J[bus_order][dangling_bus_mask],
        rtol=1e-8,
        atol=1e-8,
    )


def test_bsdf_full_rank_applies_phase_tap_change_before_split(
    micro_grid_be_network_with_replaced_transformers: pypowsybl.network.Network,
):
    net = micro_grid_be_network_with_replaced_transformers
    _static_info_base, dynamic_info_base, string_info_base = create_network_data_pypowsybl(net)
    jacobian_data_base = _get_jacobian_data_from_network_data(dynamic_info_base)
    jacobian_data_with_extra_buses, dynamic_info_with_placeholders = preprocess_jacobian_bsdf(
        jacobian_data=jacobian_data_base,
        max_bus_splits=1,
        dynamic_network_data=dynamic_info_base,
    )

    phase_tap_changer = net.get_phase_tap_changers().reset_index().iloc[0]
    branch_index_by_id = {str(branch_id): idx for idx, branch_id in enumerate(string_info_base.branch_ids)}
    phase_branch_idx = branch_index_by_id[str(phase_tap_changer["id"])]

    branch_from_bus = int(dynamic_info_base.branch_from_bus[phase_branch_idx])
    branch_to_bus = int(dynamic_info_base.branch_to_bus[phase_branch_idx])
    if dynamic_info_base.bus_type[branch_from_bus] == BusType.PQ:
        bus_to_split = branch_from_bus
    else:
        bus_to_split = branch_to_bus
    assert dynamic_info_base.bus_type[bus_to_split] == BusType.PQ

    new_bus_index = dynamic_info_with_placeholders.n_buses - 1
    updated_tap = int(phase_tap_changer["high_tap"])
    if updated_tap == int(phase_tap_changer["tap"]):
        updated_tap = int(phase_tap_changer["low_tap"])

    net.update_phase_tap_changers(df=pd.DataFrame({"id": [phase_tap_changer["id"]], "tap": [updated_tap]}).set_index("id"))

    _static_info_tap, dynamic_info_tap, _string_info_tap = create_network_data_pypowsybl(net)
    jacobian_data_tap = _get_jacobian_data_from_network_data(dynamic_info_tap)
    _jacobian_data_tap_extended, dynamic_info_tap_with_placeholders = preprocess_jacobian_bsdf(
        jacobian_data=jacobian_data_tap,
        max_bus_splits=1,
        dynamic_network_data=dynamic_info_tap,
    )

    split_voltage_magnitudes = dynamic_info_tap_with_placeholders.bus_voltage_magnitudes.copy()
    split_voltage_angles = dynamic_info_tap_with_placeholders.bus_voltage_angles_rad.copy()
    split_voltage_magnitudes[new_bus_index] = dynamic_info_tap.bus_voltage_magnitudes[bus_to_split]
    split_voltage_angles[new_bus_index] = dynamic_info_tap.bus_voltage_angles_rad[bus_to_split]

    branch_from_split = np.asarray(dynamic_info_tap.branch_from_bus, dtype=np.int32).copy()
    branch_to_split = np.asarray(dynamic_info_tap.branch_to_bus, dtype=np.int32).copy()
    branch_from_split[phase_branch_idx] = np.where(
        branch_from_split[phase_branch_idx] == bus_to_split,
        new_bus_index,
        branch_from_split[phase_branch_idx],
    )
    branch_to_split[phase_branch_idx] = np.where(
        branch_to_split[phase_branch_idx] == bus_to_split,
        new_bus_index,
        branch_to_split[phase_branch_idx],
    )

    dynamic_info_split_expected = replace(
        dynamic_info_tap_with_placeholders,
        bus_voltage_magnitudes=split_voltage_magnitudes,
        bus_voltage_angles_rad=split_voltage_angles,
        branch_from_bus=branch_from_split,
        branch_to_bus=branch_to_split,
    )
    jacobian_data_split_expected = _get_jacobian_data_from_network_data(dynamic_info_split_expected)

    jacobian_inv_bsdf = compute_bsdf_update(
        jacobian_inv=jacobian_data_with_extra_buses.inverse_jacobian,
        bus_to_split=bus_to_split,
        new_bus_b_index=new_bus_index,
        new_bus_type=2,
        branches_connected_to_bus_b=np.asarray([phase_branch_idx], dtype=np.int32),
        shunt_connected_to_bus_b=np.asarray([], dtype=np.int32),
        branch_from=np.asarray(dynamic_info_base.branch_from_bus, dtype=np.int32),
        branch_to=np.asarray(dynamic_info_base.branch_to_bus, dtype=np.int32),
        shunt_to_bus=np.asarray(dynamic_info_base.shunt_bus_indices, dtype=np.int32),
        v_mag_hat=np.asarray(dynamic_info_split_expected.bus_voltage_magnitudes, dtype=float),
        theta_hat=np.asarray(dynamic_info_split_expected.bus_voltage_angles_rad, dtype=float),
        y_ff=np.asarray(dynamic_info_tap.branch_effective_admittance_from_from, dtype=np.complex128),
        y_ft=np.asarray(dynamic_info_tap.branch_effective_admittance_from_to, dtype=np.complex128),
        y_tf=np.asarray(dynamic_info_tap.branch_effective_admittance_to_from, dtype=np.complex128),
        y_tt=np.asarray(dynamic_info_tap.branch_effective_admittance_to_to, dtype=np.complex128),
        y_shunt=np.asarray(dynamic_info_tap.shunt_effective_bus_admittance, dtype=np.complex128),
        angle_component_indices=jacobian_data_with_extra_buses.angle_component_indices,
        magnitude_component_indices=jacobian_data_with_extra_buses.magnitude_component_indices,
        y_ff_base=np.asarray(dynamic_info_base.branch_effective_admittance_from_from, dtype=np.complex128),
        y_ft_base=np.asarray(dynamic_info_base.branch_effective_admittance_from_to, dtype=np.complex128),
        y_tf_base=np.asarray(dynamic_info_base.branch_effective_admittance_to_from, dtype=np.complex128),
        y_tt_base=np.asarray(dynamic_info_base.branch_effective_admittance_to_to, dtype=np.complex128),
    )

    np.testing.assert_allclose(
        jacobian_inv_bsdf,
        jacobian_data_split_expected.inverse_jacobian,
        rtol=1e-10,
        atol=1e-10,
    )


def test_bsdf_full_rank_applies_phase_tap_change_without_split(
    micro_grid_be_network_with_replaced_transformers: pypowsybl.network.Network,
):
    net = micro_grid_be_network_with_replaced_transformers
    _static_info_base, dynamic_info_base, string_info_base = create_network_data_pypowsybl(net)
    jacobian_data_base = _get_jacobian_data_from_network_data(dynamic_info_base)

    phase_tap_changer = net.get_phase_tap_changers().reset_index().iloc[0]
    branch_index_by_id = {str(branch_id): idx for idx, branch_id in enumerate(string_info_base.branch_ids)}
    phase_branch_idx = branch_index_by_id[str(phase_tap_changer["id"])]

    updated_tap = int(phase_tap_changer["high_tap"])
    if updated_tap == int(phase_tap_changer["tap"]):
        updated_tap = int(phase_tap_changer["low_tap"])

    net.update_phase_tap_changers(df=pd.DataFrame({"id": [phase_tap_changer["id"]], "tap": [updated_tap]}).set_index("id"))

    _static_info_tap, dynamic_info_tap, _string_info_tap = create_network_data_pypowsybl(net)
    jacobian_data_tap = _get_jacobian_data_from_network_data(dynamic_info_tap)

    jacobian_inv_bsdf = compute_bsdf_update(
        jacobian_inv=jacobian_data_base.inverse_jacobian,
        bus_to_split=0,
        new_bus_b_index=0,
        new_bus_type=2,
        branches_connected_to_bus_b=np.asarray([], dtype=np.int32),
        shunt_connected_to_bus_b=np.asarray([], dtype=np.int32),
        branch_from=np.asarray(dynamic_info_base.branch_from_bus, dtype=np.int32),
        branch_to=np.asarray(dynamic_info_base.branch_to_bus, dtype=np.int32),
        shunt_to_bus=np.asarray(dynamic_info_base.shunt_bus_indices, dtype=np.int32),
        v_mag_hat=np.asarray(dynamic_info_tap.bus_voltage_magnitudes, dtype=float),
        theta_hat=np.asarray(dynamic_info_tap.bus_voltage_angles_rad, dtype=float),
        y_ff=np.asarray(dynamic_info_tap.branch_effective_admittance_from_from, dtype=np.complex128),
        y_ft=np.asarray(dynamic_info_tap.branch_effective_admittance_from_to, dtype=np.complex128),
        y_tf=np.asarray(dynamic_info_tap.branch_effective_admittance_to_from, dtype=np.complex128),
        y_tt=np.asarray(dynamic_info_tap.branch_effective_admittance_to_to, dtype=np.complex128),
        y_shunt=np.asarray(dynamic_info_tap.shunt_effective_bus_admittance, dtype=np.complex128),
        angle_component_indices=jacobian_data_base.angle_component_indices,
        magnitude_component_indices=jacobian_data_base.magnitude_component_indices,
        y_ff_base=np.asarray(dynamic_info_base.branch_effective_admittance_from_from, dtype=np.complex128),
        y_ft_base=np.asarray(dynamic_info_base.branch_effective_admittance_from_to, dtype=np.complex128),
        y_tf_base=np.asarray(dynamic_info_base.branch_effective_admittance_to_from, dtype=np.complex128),
        y_tt_base=np.asarray(dynamic_info_base.branch_effective_admittance_to_to, dtype=np.complex128),
        apply_split_bus_adjustment=False,
    )

    np.testing.assert_allclose(
        jacobian_inv_bsdf,
        jacobian_data_tap.inverse_jacobian,
        rtol=1e-10,
        atol=1e-10,
    )

    mismatch_tap = calculate_nodal_mismatch_network_data(
        dynamic_network_data=dynamic_info_tap,
        y_matrix=_get_admittance_matrix_from_network_data(dynamic_info_tap),
    )
    dx_bsdf = -jacobian_inv_bsdf @ mismatch_tap
    dx_direct = -jacobian_data_tap.inverse_jacobian @ mismatch_tap

    np.testing.assert_allclose(dx_bsdf, dx_direct, rtol=1e-10, atol=1e-10)
