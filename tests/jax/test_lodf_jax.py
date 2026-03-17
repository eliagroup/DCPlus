# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pypowsybl
import pytest
from jax import tree_util

from dc_plus.importing.powsybl.powsybl_import import DANGLING_BUS_STRING_SUFFIX
from dc_plus.importing.powsybl.powsybl_loadflow_parameter import get_powsybl_loadflow_parameter
from dc_plus.importing.powsybl.powsybl_network_helpers import (
    _load_test_grid,
    get_bus_branch_ids_for_n1_results,
    powsybl_n1_analysis,
)
from dc_plus.interfaces.jacobian_network_data import (
    _get_admittance_matrix_from_network_data,
    calculate_nodal_mismatch_network_data,
)
from dc_plus.preprocess.helper_functions import _find_bridges

# Enable 64-bit floats in JAX where needed.
jax.config.update("jax_enable_x64", True)
# set jax to cpu for testing
jax.config.update("jax_platform_name", "cpu")

from dc_plus.example_grids.pypowsbl.example_grids import PANDAPOWER_NETWORKS_FOR_POWSYBL, POWSYBL_NETWORKS
from dc_plus.jax.lodf_branches import line_outage_post_contingency_monitored

powsybl_networks = POWSYBL_NETWORKS
pandapower_networks = PANDAPOWER_NETWORKS_FOR_POWSYBL


@tree_util.register_pytree_node_class
@dataclass(slots=True)
class LODFInputs:
    """Inputs held on-device for the profiled kernel."""

    jacobian_inv_transposed: jnp.ndarray
    branch_idx: jnp.ndarray
    branch_from: jnp.ndarray
    branch_to: jnp.ndarray
    v_mag_hat: jnp.ndarray
    theta_hat: jnp.ndarray
    angle_component_indices: jnp.ndarray
    magnitude_component_indices: jnp.ndarray
    y_ff: jnp.ndarray
    y_ft: jnp.ndarray
    y_tf: jnp.ndarray
    y_tt: jnp.ndarray

    # Branch linearization inputs (per topology)
    branch_pq_base: jnp.ndarray
    branch_flow_jac: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.jacobian_inv_transposed,
            self.branch_idx,
            self.branch_from,
            self.branch_to,
            self.v_mag_hat,
            self.theta_hat,
            self.angle_component_indices,
            self.magnitude_component_indices,
            self.y_ff,
            self.y_ft,
            self.y_tf,
            self.y_tt,
            self.branch_pq_base,
            self.branch_flow_jac,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@pytest.mark.parametrize("get_net", powsybl_networks + pandapower_networks)
# @pytest.mark.parametrize("get_net", [pypowsybl.network.create_ieee14])
# @pytest.mark.parametrize("get_net", [create_complex_grid_battery_hvdc_svc_3w_trafo])
def test_lodf_jax_full_rank_update_compare_powsybl(get_net):
    net, static_info, dynamic_info, string_info, jacobian_data = _load_test_grid(get_net)
    theta_actual = dynamic_info.bus_voltage_angles_rad
    vm_actual = dynamic_info.bus_voltage_magnitudes
    v_pu_actual = vm_actual * np.exp(theta_actual * 1j)
    j_inverse = jacobian_data.inverse_jacobian

    ### N-1 test
    is_bridge = _find_bridges(dynamic_info)

    loadflow_parameter_one_step = get_powsybl_loadflow_parameter("one_step")
    outage_ids = string_info.branch_ids[~is_bridge]
    sa_res = powsybl_n1_analysis(net=net, outage_grid_ids=outage_ids, loadflow_parameter=loadflow_parameter_one_step)
    sa_bus_results = get_bus_branch_ids_for_n1_results(net, sa_res)

    y_matrix_n0 = _get_admittance_matrix_from_network_data(dynamic_info)
    mismatch_n0 = calculate_nodal_mismatch_network_data(dynamic_network_data=dynamic_info, y_matrix=y_matrix_n0)

    # hotstart precision
    if isinstance(get_net(), pypowsybl.network.Network):
        precition = 1e-11
    else:
        precition = 1e-9
    assert abs(mismatch_n0).max() < precition
    dx = -j_inverse @ mismatch_n0
    assert abs(dx).max() < precition

    v_mag_hat = np.asarray(dynamic_info.bus_voltage_magnitudes, dtype=float).reshape(-1)
    theta_hat = np.asarray(dynamic_info.bus_voltage_angles_rad, dtype=float).reshape(-1)

    jacobian_inv = jacobian_data.inverse_jacobian
    jacobian_inv_np = np.asarray(jacobian_inv)
    jacobian_inv_transposed_np = jacobian_inv_np.T

    branch_from = np.asarray(dynamic_info.branch_from_bus, dtype=np.int32).reshape(-1)
    branch_to = np.asarray(dynamic_info.branch_to_bus, dtype=np.int32).reshape(-1)
    y_ff = np.asarray(dynamic_info.branch_effective_admittance_from_from, dtype=np.complex128).reshape(-1)
    y_ft = np.asarray(dynamic_info.branch_effective_admittance_from_to, dtype=np.complex128).reshape(-1)
    y_tf = np.asarray(dynamic_info.branch_effective_admittance_to_from, dtype=np.complex128).reshape(-1)
    y_tt = np.asarray(dynamic_info.branch_effective_admittance_to_to, dtype=np.complex128).reshape(-1)

    is_bridge = _find_bridges(dynamic_info)
    outage_candidates = np.flatnonzero(~is_bridge)

    # Precompute per-branch endpoint complex powers at the N-0 voltage.
    # This is the per-outage mismatch seed used by the JAX kernel.
    v0 = v_mag_hat * np.exp(1j * theta_hat)
    v_from = v0[branch_from]
    v_to = v0[branch_to]

    i_from = y_ff * v_from + y_ft * v_to
    i_to = y_tf * v_from + y_tt * v_to
    s_from = v_from * np.conj(i_from)
    s_to = v_to * np.conj(i_to)

    branch_connected_base = np.asarray(dynamic_info.branch_connected, dtype=bool).reshape(-1)
    s_from = np.where(branch_connected_base, s_from, 0.0 + 0.0j)
    s_to = np.where(branch_connected_base, s_to, 0.0 + 0.0j)

    branch_pq_base = np.empty((branch_from.size, 4), dtype=jacobian_inv_np.dtype)
    branch_pq_base[:, 0] = s_from.real
    branch_pq_base[:, 1] = s_to.real
    branch_pq_base[:, 2] = s_from.imag
    branch_pq_base[:, 3] = s_to.imag

    # For profiling the branch-linearization path, we only need shapes/dtypes.
    # Replace with your production precompute when available.
    n_branches = branch_from.size
    branch_flow_jac = np.zeros((n_branches, 4, 4), dtype=jacobian_inv_np.dtype)

    angle_component_indices = np.asarray(jacobian_data.angle_component_indices, dtype=np.int32)
    magnitude_component_indices = np.asarray(jacobian_data.magnitude_component_indices, dtype=np.int32)

    inputs = LODFInputs(
        jacobian_inv_transposed=jnp.asarray(jacobian_inv_transposed_np),
        branch_idx=jnp.asarray(outage_candidates, dtype=jnp.int32),
        branch_from=jnp.asarray(branch_from, dtype=jnp.int32),
        branch_to=jnp.asarray(branch_to, dtype=jnp.int32),
        v_mag_hat=jnp.asarray(v_mag_hat),
        theta_hat=jnp.asarray(theta_hat),
        angle_component_indices=jnp.asarray(angle_component_indices),
        magnitude_component_indices=jnp.asarray(magnitude_component_indices),
        y_ff=jnp.asarray(y_ff),
        y_ft=jnp.asarray(y_ft),
        y_tf=jnp.asarray(y_tf),
        y_tt=jnp.asarray(y_tt),
        branch_pq_base=jnp.asarray(branch_pq_base),
        branch_flow_jac=jnp.asarray(branch_flow_jac),
    )
    monitor_bus_indices = jnp.arange(inputs.theta_hat.size, dtype=jnp.int32)
    monitor_branch_indices = jnp.arange(inputs.branch_from.size, dtype=jnp.int32)
    bus_to_mon_index = jnp.arange(inputs.theta_hat.size, dtype=jnp.int32)

    lf_res = line_outage_post_contingency_monitored(
        jacobian_inv_transposed=inputs.jacobian_inv_transposed,
        outage_branch_idx=inputs.branch_idx,
        branch_from=inputs.branch_from,
        branch_to=inputs.branch_to,
        v_mag_hat=inputs.v_mag_hat,
        theta_hat=inputs.theta_hat,
        angle_component_indices=inputs.angle_component_indices,
        magnitude_component_indices=inputs.magnitude_component_indices,
        y_ff=inputs.y_ff,
        y_ft=inputs.y_ft,
        y_tf=inputs.y_tf,
        y_tt=inputs.y_tt,
        monitor_bus_indices=monitor_bus_indices,
        branch_pq_base=inputs.branch_pq_base,
        monitor_branch_indices=monitor_branch_indices,
        bus_to_mon_index=bus_to_mon_index,
    )

    for pos, outage_idx in enumerate(outage_candidates):
        outage_id = string_info.branch_ids[outage_idx]
        if outage_id not in sa_bus_results.index:
            continue
        is_n1_converged = (
            sa_res.post_contingency_results[outage_id].status
            == pypowsybl._pypowsybl.PostContingencyComputationStatus.CONVERGED
        )
        if not is_n1_converged:
            continue

        sa_bus_results_n1 = sa_bus_results.loc[outage_id]
        if isinstance(sa_bus_results_n1, pd.Series):
            sa_bus_results_n1 = sa_bus_results_n1.to_frame().T
        sa_bus_results_n1 = sa_bus_results_n1.set_index("bus_id").reindex(string_info.bus_ids)
        dangling_mask = sa_bus_results_n1.index.str.endswith(DANGLING_BUS_STRING_SUFFIX)
        sa_bus_results_n1 = sa_bus_results_n1[~dangling_mask]
        if sa_bus_results_n1[["v_mag_pu", "v_angle_rad"]].isna().any().any():
            continue

        v_pu_actual_n1_sa = sa_bus_results_n1["v_mag_pu"].to_numpy() * np.exp(
            1j * sa_bus_results_n1["v_angle_rad"].to_numpy()
        )
        lf_res_voltage = lf_res.n_1_voltage[pos] * jnp.exp(1j * lf_res.n_1_theta[pos])

        assert np.allclose(
            np.asarray(lf_res_voltage[~dangling_mask]),
            v_pu_actual_n1_sa,
            atol=1e-7,
            rtol=1e-5,
        ), f"Voltage mismatch for outage {outage_id}"

        sa_branch_results_n1 = (
            sa_res.branch_results.loc[outage_id].reset_index().sort_values(by="branch_id").set_index("branch_id")
        )
        existing_branches = pd.Index(string_info.branch_ids)
        dangling_line_mask = string_info.branch_types == "DANGLING_LINE"
        sa_branch_results_n1_power = sa_branch_results_n1.reindex(existing_branches)[["p1", "p2", "q1", "q2"]].fillna(0.0)
        sa_branch_results_n1_power_pu = sa_branch_results_n1_power / net.nominal_apparent_power
        sa_branch_results_n1_power_pu = sa_branch_results_n1_power_pu.iloc[~dangling_line_mask]

        assert np.allclose(
            np.asarray(lf_res.n_1_p_from[pos][~dangling_line_mask]),
            sa_branch_results_n1_power_pu["p1"].to_numpy(),
            atol=1e-10,
        ), f"P_from mismatch for outage {outage_id}"

        assert np.allclose(
            np.asarray(lf_res.n_1_p_to[pos][~dangling_line_mask]),
            sa_branch_results_n1_power_pu["p2"].to_numpy(),
            atol=1e-10,
        ), f"P_to mismatch for outage {outage_id}"

        assert np.allclose(
            np.asarray(lf_res.n_1_q_from[pos][~dangling_line_mask]),
            sa_branch_results_n1_power_pu["q1"].to_numpy(),
            atol=1e-10,
        ), f"Q_from mismatch for outage {outage_id}"

        assert np.allclose(
            np.asarray(lf_res.n_1_q_to[pos][~dangling_line_mask]),
            sa_branch_results_n1_power_pu["q2"].to_numpy(),
            atol=1e-10,
        ), f"Q_to mismatch for outage {outage_id}"
