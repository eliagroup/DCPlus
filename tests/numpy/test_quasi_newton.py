# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from copy import deepcopy

from dc_plus.example_grids.pypowsbl.example_grids import POWSYBL_NETWORKS_SHORT_LIST
from dc_plus.preprocess.helper_functions import _find_bridges
import numpy as np
import pandas as pd
import pypowsybl
import pytest

from dc_plus.interfaces.jacobian_network_data import (
    _apply_jacobian_dx_to_network_data,
    _get_admittance_matrix_from_network_data,
    get_jacobian_data_from_network_data,
    calculate_nodal_mismatch_network_data,
)
from dc_plus.interfaces.network_information import BusType
from dc_plus.importing.powsybl.powsybl_loadflow_parameter import get_powsybl_loadflow_parameter
from dc_plus.importing.powsybl.powsybl_import import DANGLING_BUS_STRING_SUFFIX
from dc_plus.importing.powsybl.powsybl_network_helpers import (
    _load_test_grid,
    get_bus_branch_ids_for_n1_results,
    powsybl_n1_analysis,
)
from dc_plus.numpy.bsdf_full_rank import compute_bsdf_update
from dc_plus.numpy.quasi_newton import run_quasi_newton_updates
from dc_plus.preprocess.create_network_data import create_network_data_pypowsybl as create_network_data_pypowsybl

from tests.test_helper.injection_state_helper import build_updated_dynamic_injection_state
from tests.test_helper.bsdf_helper import (
    derive_bus_order,
    get_bsdf_cases,
    prepare_bsdf_test_context,
    run_reference_full_ac,
)

powsybl_networks_short_list = POWSYBL_NETWORKS_SHORT_LIST


def test_quasi_newton_zero_iteration_keeps_target_state(
    micro_grid_be_network_with_replaced_transformers: pypowsybl.network.Network,
):
    net = micro_grid_be_network_with_replaced_transformers
    dynamic_info_base = create_network_data_pypowsybl(net).dynamic_network_data
    jacobian_data_base = get_jacobian_data_from_network_data(
        dynamic_info_base,
    )

    dynamic_info_quasi_newton, mismatch_history, jacobian_inv_updated = run_quasi_newton_updates(
        jacobian_data=jacobian_data_base,
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
    dynamic_info_base = create_network_data_pypowsybl(net).dynamic_network_data
    jacobian_data_base = get_jacobian_data_from_network_data(
        dynamic_info_base,
    )

    phase_tap_changer = net.get_phase_tap_changers().reset_index().iloc[0]
    updated_tap = min(int(phase_tap_changer["tap"]) + 10, int(phase_tap_changer["high_tap"]))
    net.update_phase_tap_changers(df=pd.DataFrame({"id": [phase_tap_changer["id"]], "tap": [updated_tap]}).set_index("id"))

    dynamic_info_tap = create_network_data_pypowsybl(net).dynamic_network_data
    jacobian_data_tap = get_jacobian_data_from_network_data(
        dynamic_info_tap,
    )
    y_matrix_tap = _get_admittance_matrix_from_network_data(dynamic_info_tap)
    jacobian_inv_bsdf = compute_bsdf_update(
        jacobian_inv=jacobian_data_base.inverse_jacobian,
        bus_to_split=0,
        new_bus_b_index=0,
        new_bus_type=2,
        branches_connected_to_bus_b=np.asarray([], dtype=np.int32),
        shunt_connected_to_bus_b=np.asarray([], dtype=np.int32),
        branch_from=dynamic_info_base.branch_from_bus,
        branch_to=dynamic_info_base.branch_to_bus,
        shunt_to_bus=dynamic_info_base.shunt_bus_indices,
        v_mag_hat=dynamic_info_tap.bus_voltage_magnitudes,
        theta_hat=dynamic_info_tap.bus_voltage_angles_rad,
        y_ff=dynamic_info_tap.branch_effective_admittance_from_from,
        y_ft=dynamic_info_tap.branch_effective_admittance_from_to,
        y_tf=dynamic_info_tap.branch_effective_admittance_to_from,
        y_tt=dynamic_info_tap.branch_effective_admittance_to_to,
        y_shunt=dynamic_info_tap.shunt_effective_bus_admittance,
        angle_component_indices=jacobian_data_base.angle_component_indices,
        magnitude_component_indices=jacobian_data_base.magnitude_component_indices,
        reactive_power_bus_mask=dynamic_info_base.bus_type == 2,
        y_ff_base=dynamic_info_base.branch_effective_admittance_from_from,
        y_ft_base=dynamic_info_base.branch_effective_admittance_from_to,
        y_tf_base=dynamic_info_base.branch_effective_admittance_to_from,
        y_tt_base=dynamic_info_base.branch_effective_admittance_to_to,
        apply_split_bus_adjustment=False,
    )

    dynamic_info_quasi_newton, mismatch_history, _jacobian_inv_updated = run_quasi_newton_updates(
        jacobian_data=jacobian_data_tap.copy_with_inverse_jacobian(jacobian_inv_bsdf),
        dynamic_network_data=dynamic_info_tap,
        n_iterations=10,
        y_matrix=y_matrix_tap,
    )
    mismatch_before = calculate_nodal_mismatch_network_data(
        dynamic_info_tap,
        y_matrix_tap,
        jacobian_data=jacobian_data_tap,
    )
    mismatch_after = calculate_nodal_mismatch_network_data(
        dynamic_info_quasi_newton,
        y_matrix_tap,
        jacobian_data=jacobian_data_tap,
    )

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
        reactive_power_bus_mask=np.asarray(setup.dynamic_info_split_manual.bus_type == 2, dtype=bool),
    )

    y_matrix_split = _get_admittance_matrix_from_network_data(setup.dynamic_info_split_manual)
    dynamic_info_quasi_newton, mismatch_history, jacobian_inv_updated = run_quasi_newton_updates(
        jacobian_data=setup.jacobian_data_with_extra_buses.copy_with_inverse_jacobian(jacobian_inv_bsdf),
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

    mismatch_after = calculate_nodal_mismatch_network_data(
        dynamic_info_quasi_newton,
        y_matrix_split,
        jacobian_data=setup.jacobian_data_with_extra_buses,
    )

    assert len(mismatch_history) == 9
    assert np.all(np.isfinite(mismatch_history))
    assert np.max(np.abs(mismatch_after)) < 1e-10


# @pytest.mark.parametrize("get_net", [pypowsybl.network.create_ieee30])
@pytest.mark.parametrize("get_net", powsybl_networks_short_list)
def test_quasi_branch_outage_numpy_full_rank_update_compare_powsybl(get_net):
    net, network_info, _jacobian_data = _load_test_grid(get_net)
    dynamic_info = network_info.dynamic_network_data
    string_info = network_info.string_network_data
    iterations = 20
    tol_powsybl = 1e-9
    tol_not_converged = 1e-4

    is_bridge = _find_bridges(dynamic_info)
    outage_candidates = np.flatnonzero(~is_bridge)

    loadflow_parameter = get_powsybl_loadflow_parameter("hotstart_test")
    outage_ids = string_info.branch_ids[~is_bridge]
    sa_res = powsybl_n1_analysis(net=net, outage_grid_ids=outage_ids, loadflow_parameter=loadflow_parameter)
    sa_bus_results = get_bus_branch_ids_for_n1_results(net, sa_res)
    loadflow_parameter_limited = get_powsybl_loadflow_parameter("hotstart_test")
    loadflow_parameter_limited.use_reactive_limits = True
    loadflow_parameter_limited.provider_parameters["useActiveLimits"] = "true"
    sa_res_limited = powsybl_n1_analysis(
        net=net,
        outage_grid_ids=outage_ids,
        loadflow_parameter=loadflow_parameter_limited,
    )

    cases_compared = 0

    for outage_branch in outage_candidates:
        outage_id = string_info.branch_ids[outage_branch]
        if outage_id not in sa_bus_results.index:
            continue

        is_n1_converged = (
            sa_res.post_contingency_results[outage_id].status
            == pypowsybl._pypowsybl.PostContingencyComputationStatus.CONVERGED
        )
        if not is_n1_converged:
            continue

        dynamic_info_n1 = deepcopy(dynamic_info)
        dynamic_info_n1.branch_connected[outage_branch] = False
        jacobian_data_n1 = get_jacobian_data_from_network_data(dynamic_info_n1)
        y_matrix_n1 = _get_admittance_matrix_from_network_data(dynamic_info_n1)

        results_numpy, mismatch_history, _updated_inverse_jacobian = run_quasi_newton_updates(
            jacobian_data=jacobian_data_n1,
            dynamic_network_data=dynamic_info_n1,
            n_iterations=iterations,
            y_matrix=y_matrix_n1,
        )

        sa_bus_results_n1 = sa_bus_results.loc[outage_id]
        if isinstance(sa_bus_results_n1, pd.Series):
            sa_bus_results_n1 = sa_bus_results_n1.to_frame().T
        sa_bus_results_n1 = sa_bus_results_n1.set_index("bus_id").reindex(string_info.bus_ids)
        dangling_mask = sa_bus_results_n1.index.str.endswith(DANGLING_BUS_STRING_SUFFIX)
        sa_bus_results_n1 = sa_bus_results_n1[~dangling_mask]
        if sa_bus_results_n1[["v_mag_pu", "v_angle_rad"]].isna().any().any():
            continue

        if (
            sa_res_limited.post_contingency_results[outage_id].status
            != pypowsybl._pypowsybl.PostContingencyComputationStatus.CONVERGED
            or outage_id == "L1-3-1"
        ):
            tol_effective = tol_not_converged
        else:
            tol_effective = tol_powsybl

        np.testing.assert_allclose(
            np.asarray(results_numpy.bus_voltage_magnitudes[~dangling_mask]),
            sa_bus_results_n1["v_mag_pu"].to_numpy(),
            rtol=tol_effective,
            atol=tol_effective,
        )
        np.testing.assert_allclose(
            np.asarray(results_numpy.bus_voltage_angles_rad[~dangling_mask]),
            sa_bus_results_n1["v_angle_rad"].to_numpy(),
            rtol=tol_effective,
            atol=tol_effective,
        )
        assert np.all(np.isfinite(np.asarray(mismatch_history))), outage_branch
        cases_compared += 1

    if cases_compared == 0:
        pytest.skip("No suitable outage cases found for comparison")
