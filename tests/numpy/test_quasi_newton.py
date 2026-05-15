# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pypowsybl
import pytest

from dc_plus.example_grids.pypowsbl.example_grids import PANDAPOWER_NETWORKS_FOR_POWSYBL_LARGE_GRIDS, POWSYBL_NETWORKS_SHORT_LIST, PANDAPOWER_NETWORKS_FOR_POWSYBL
from dc_plus.interfaces.jacobian_network_data import (
    _apply_jacobian_dx_to_network_data,
    _get_admittance_matrix_from_network_data,
    _get_jacobian_data_from_network_data,
    calculate_nodal_mismatch_network_data,
)
from dc_plus.importing.powsybl.powsybl_import import DANGLING_BUS_STRING_SUFFIX
from dc_plus.importing.powsybl.powsybl_loadflow_parameter import get_powsybl_loadflow_parameter
from dc_plus.importing.powsybl.powsybl_network_helpers import _load_test_grid, get_bus_branch_ids_for_n1_results, powsybl_n1_analysis
from dc_plus.numpy.bsdf_full_rank import compute_bsdf_update
from dc_plus.numpy.lodf import branch_outage_update_inverse
from dc_plus.numpy.quasi_newton import run_quasi_newton_updates
from dc_plus.preprocess.create_network_data import create_network_data_pypowsybl as create_network_data_pypowsybl
from dc_plus.preprocess.helper_functions import _find_bridges
from tests.test_helper.bsdf_helper import (
    derive_bus_order,
    get_bsdf_cases,
    prepare_bsdf_test_context,
    run_reference_full_ac,
)


QN_TOL = 1e-9


def _apply_inverse_jacobian_step(dynamic_info, jacobian_inv, y_matrix):
    mismatch = calculate_nodal_mismatch_network_data(dynamic_info, y_matrix)
    return _apply_jacobian_dx_to_network_data(dynamic_info, dx=-(jacobian_inv @ mismatch))


@pytest.mark.parametrize("get_net", POWSYBL_NETWORKS_SHORT_LIST)
def test_quasi_newton_branch_outage(get_net) -> None:
    """Compare the Python quasi-Newton branch-outage solve against a Powsybl one-step N-1 result."""
    net, _static_info, dynamic_info, string_info, jacobian_data = _load_test_grid(get_net)

    is_bridge = _find_bridges(dynamic_info)
    outage_ids = string_info.branch_ids[~is_bridge]

    loadflow_parameter = get_powsybl_loadflow_parameter("hotstart_test")

    sa_res = powsybl_n1_analysis(
        net=net,
        outage_grid_ids=list(outage_ids),
        loadflow_parameter=loadflow_parameter,
    )
    sa_bus_results = get_bus_branch_ids_for_n1_results(net, sa_res)
    dangling_bus_mask = np.char.endswith(string_info.bus_ids.astype(str), DANGLING_BUS_STRING_SUFFIX)

    compared_outage = None
    for outage_idx in np.flatnonzero(~is_bridge):
        outage_id = string_info.branch_ids[outage_idx]
        if outage_id not in sa_bus_results.index:
            continue

        result = sa_res.post_contingency_results[outage_id]
        if result.status != pypowsybl._pypowsybl.PostContingencyComputationStatus.CONVERGED:
            continue

        branch_connected_n1 = np.asarray(dynamic_info.branch_connected, dtype=bool).copy()
        branch_connected_n1[outage_idx] = False
        dynamic_info_n1 = replace(dynamic_info, branch_connected=branch_connected_n1)
        y_matrix_n1 = _get_admittance_matrix_from_network_data(dynamic_info_n1)

        jacobian_inv_n1 = branch_outage_update_inverse(
            jacobian_inv=jacobian_data.inverse_jacobian,
            outage_branches_indices=np.array([outage_idx], dtype=np.int64),
            branch_from=dynamic_info.branch_from_bus,
            branch_to=dynamic_info.branch_to_bus,
            v_mag_hat=dynamic_info.bus_voltage_magnitudes,
            theta_hat=dynamic_info.bus_voltage_angles_rad,
            y_ft=dynamic_info.branch_effective_admittance_from_to,
            y_tf=dynamic_info.branch_effective_admittance_to_from,
            y_ff=dynamic_info.branch_effective_admittance_from_from,
            y_tt=dynamic_info.branch_effective_admittance_to_to,
            angle_component_indices=jacobian_data.angle_component_indices,
            magnitude_component_indices=jacobian_data.magnitude_component_indices,
        )

        dynamic_info_quasi_newton, mismatch_history, _jacobian_inv_updated = run_quasi_newton_updates(
            jacobian_inv=jacobian_inv_n1,
            dynamic_network_data=dynamic_info_n1,
            n_iterations=20,
            y_matrix=y_matrix_n1,
        )
        angle_n1 = dynamic_info_quasi_newton.bus_voltage_angles_rad
        vm_n1 = dynamic_info_quasi_newton.bus_voltage_magnitudes

        sa_bus_results_n1 = sa_bus_results.loc[outage_id]
        np.testing.assert_allclose(
            vm_n1[~dangling_bus_mask],
            sa_bus_results_n1["v_mag_pu"].values,
            rtol=QN_TOL,
            atol=QN_TOL,
        )
        np.testing.assert_allclose(
            angle_n1[~dangling_bus_mask],
            sa_bus_results_n1["v_angle_rad"].values,
            rtol=QN_TOL,
            atol=QN_TOL,
        )

        mismatch_after = calculate_nodal_mismatch_network_data(dynamic_info_quasi_newton, y_matrix_n1)
        assert len(mismatch_history) == 20
        assert np.all(np.isfinite(mismatch_history))
        assert np.max(np.abs(mismatch_after)) < QN_TOL

        compared_outage = outage_id

    if compared_outage is None:
        pytest.skip("No converged non-bridge N-1 outage was available.")


def test_quasi_newton_zero_iteration_keeps_target_state(
    micro_grid_be_network_with_replaced_transformers: pypowsybl.network.Network,
):
    net = micro_grid_be_network_with_replaced_transformers
    _static_info_base, dynamic_info_base, _string_info_base = create_network_data_pypowsybl(net)
    jacobian_data_base = _get_jacobian_data_from_network_data(dynamic_info_base)

    dynamic_info_quasi_newton, mismatch_history, jacobian_inv_updated = run_quasi_newton_updates(
        jacobian_inv=jacobian_data_base.inverse_jacobian,
        dynamic_network_data=dynamic_info_base,
        n_iterations=0,
    )

    np.testing.assert_allclose(
        dynamic_info_quasi_newton.bus_voltage_magnitudes,
        dynamic_info_base.bus_voltage_magnitudes,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        dynamic_info_quasi_newton.bus_voltage_angles_rad,
        dynamic_info_base.bus_voltage_angles_rad,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        jacobian_inv_updated,
        jacobian_data_base.inverse_jacobian,
        rtol=1e-10,
        atol=1e-10,
    )
    assert mismatch_history == []


def test_quasi_newton_reduces_phase_tap_mismatch(
    micro_grid_be_network_with_replaced_transformers: pypowsybl.network.Network,
):
    net = micro_grid_be_network_with_replaced_transformers
    _static_info_base, dynamic_info_base, _string_info_base = create_network_data_pypowsybl(net)
    jacobian_data_base = _get_jacobian_data_from_network_data(dynamic_info_base)

    phase_tap_changer = net.get_phase_tap_changers().reset_index().iloc[0]
    updated_tap = min(int(phase_tap_changer["tap"]) + 10, int(phase_tap_changer["high_tap"]))
    net.update_phase_tap_changers(df=pd.DataFrame({"id": [phase_tap_changer["id"]], "tap": [updated_tap]}).set_index("id"))

    _static_info_tap, dynamic_info_tap, _string_info_tap = create_network_data_pypowsybl(net)
    y_matrix_tap = _get_admittance_matrix_from_network_data(dynamic_info_tap)
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

    dynamic_info_quasi_newton, mismatch_history, _jacobian_inv_updated = run_quasi_newton_updates(
        jacobian_inv=jacobian_inv_bsdf,
        dynamic_network_data=dynamic_info_tap,
        n_iterations=10,
        y_matrix=y_matrix_tap,
    )
    mismatch_before = calculate_nodal_mismatch_network_data(dynamic_info_tap, y_matrix_tap)
    mismatch_after = calculate_nodal_mismatch_network_data(dynamic_info_quasi_newton, y_matrix_tap)

    assert len(mismatch_history) == 10
    assert np.max(np.abs(mismatch_after)) < np.max(np.abs(mismatch_before))
    assert np.all(np.isfinite(mismatch_history))
    assert mismatch_history[-1] < 1e-10


@pytest.mark.parametrize("bsdf_test_case", get_bsdf_cases())
def test_quasi_newton_matches_full_ac_for_bsdf_cases(bsdf_test_case):
    setup = prepare_bsdf_test_context(bsdf_test_case=bsdf_test_case)

    jacobian_inv_bsdf = compute_bsdf_update(
        jacobian_inv=setup.jacobian_data_with_extra_buses.inverse_jacobian,
        bus_to_split=setup.bus_to_split,
        new_bus_b_index=setup.new_bus_index,
        new_bus_type=2,
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

    y_matrix_split = _get_admittance_matrix_from_network_data(setup.dynamic_info_split_manual)
    dynamic_info_quasi_newton, mismatch_history, jacobian_inv_updated = run_quasi_newton_updates(
        jacobian_inv=jacobian_inv_bsdf,
        dynamic_network_data=setup.dynamic_info_split_manual,
        n_iterations=9,
        y_matrix=y_matrix_split,
    )

    dynamic_info_full_ac, string_info_full_ac = run_reference_full_ac(setup.net, bsdf_test_case=bsdf_test_case)
    bus_order = derive_bus_order(setup.split_bus_ids, string_info_full_ac.bus_ids)
    dangling_bus_mask = np.char.endswith(np.asarray(string_info_full_ac.bus_ids, dtype=str), DANGLING_BUS_STRING_SUFFIX)
    vm_original = setup.dynamic_info_split_manual.bus_voltage_magnitudes[bus_order]
    theta_original = setup.dynamic_info_split_manual.bus_voltage_angles_rad[bus_order]

    # check that the quasi-Newton result is not accidentally close to the original guess, but matches the full AC result instead
    assert not (
        np.allclose(
            dynamic_info_full_ac.bus_voltage_magnitudes[~dangling_bus_mask],
            vm_original[~dangling_bus_mask],
            rtol=1e-4,
            atol=1e-4,
        )
        and np.allclose(
            dynamic_info_full_ac.bus_voltage_angles_rad[~dangling_bus_mask],
            theta_original[~dangling_bus_mask],
            rtol=1e-4,
            atol=1e-4,
        )
    )
    assert not (
        np.allclose(
            dynamic_info_full_ac.bus_voltage_magnitudes[~dangling_bus_mask],
            vm_original[~dangling_bus_mask],
            rtol=1e-5,
            atol=1e-5,
        )
        and np.allclose(
            dynamic_info_full_ac.bus_voltage_angles_rad[~dangling_bus_mask],
            theta_original[~dangling_bus_mask],
            rtol=1e-5,
            atol=1e-5,
        )
    )

    # check that the quasi-Newton result matches the full AC result, both for regular and dangling buses
    np.testing.assert_allclose(
        dynamic_info_full_ac.bus_voltage_magnitudes[~dangling_bus_mask],
        dynamic_info_quasi_newton.bus_voltage_magnitudes[bus_order][~dangling_bus_mask],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        dynamic_info_full_ac.bus_voltage_angles_rad[~dangling_bus_mask],
        dynamic_info_quasi_newton.bus_voltage_angles_rad[bus_order][~dangling_bus_mask],
        rtol=1e-10,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        dynamic_info_full_ac.bus_voltage_magnitudes[dangling_bus_mask],
        dynamic_info_quasi_newton.bus_voltage_magnitudes[bus_order][dangling_bus_mask],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        dynamic_info_full_ac.bus_voltage_angles_rad[dangling_bus_mask],
        dynamic_info_quasi_newton.bus_voltage_angles_rad[bus_order][dangling_bus_mask],
        rtol=1e-10,
        atol=1e-10,
    )

    mismatch_after = calculate_nodal_mismatch_network_data(dynamic_info_quasi_newton, y_matrix_split)

    assert len(mismatch_history) == 9
    assert np.all(np.isfinite(mismatch_history))
    assert np.max(np.abs(mismatch_after)) < 1e-10
