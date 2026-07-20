# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pypowsybl
import pandas as pd
import pytest
from dataclasses import dataclass
from pypowsybl.network import Network

from dc_plus.importing.powsybl.powsybl_loadflow_parameter import get_powsybl_loadflow_parameter
from dc_plus.interfaces.jacobian_network_data import (
    _get_admittance_matrix_from_network_data,
    get_jacobian_data_from_network_data,
)
from dc_plus.preprocess.create_network_data import create_network_data_pypowsybl

from dc_plus.importing.powsybl.powsybl_import_helpers import select_a_generator_as_slack_and_run_loadflow
from tests.test_helper.injection_state_helper import build_updated_dynamic_injection_state


@dataclass
class IEEE14QuasiNewtonQlimit10PercentSetup:
    base_loadflow_parameter: pypowsybl.loadflow.Parameters
    static_info_base: object
    dynamic_info_base: object
    string_info_base: object
    jacobian_data_base: object
    y_matrix_base: object
    load_update_df: object
    generator_target_p_df: object
    dynamic_info_updated_injections: object
    dynamic_info_reference: object


@pytest.fixture
def micro_grid_be_network_with_replaced_transformers() -> Network:
    net = pypowsybl.network.create_micro_grid_be_network()
    select_a_generator_as_slack_and_run_loadflow(net)
    pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers(net)
    net.per_unit = True
    return net


@pytest.fixture
def ieee14_quasi_newton_q_limit_10_percent_setup() -> IEEE14QuasiNewtonQlimit10PercentSetup:
    net = pypowsybl.network.create_ieee14()
    net.per_unit = True

    base_loadflow_parameter = get_powsybl_loadflow_parameter("default")
    base_loadflow_parameter.distributed_slack = False
    base_loadflow_parameter.provider_parameters["reactiveLimitsMaxPqPvSwitch"] = "0"
    base_loadflow_result = pypowsybl.loadflow.run_ac(net, parameters=base_loadflow_parameter)[0]
    assert base_loadflow_result.status == pypowsybl._pypowsybl.LoadFlowComponentStatus.CONVERGED

    network_info_base = create_network_data_pypowsybl(net)
    static_info_base = network_info_base.static_network_data
    dynamic_info_base = network_info_base.dynamic_network_data
    string_info_base = network_info_base.string_network_data
    jacobian_data_base = get_jacobian_data_from_network_data(
        dynamic_info_base,
    )
    y_matrix_base = _get_admittance_matrix_from_network_data(dynamic_info_base)

    load_update = {
        "p0": {
            "B2-L": 0.2387,
            "B3-L": 1.0362,
            "B4-L": 0.5258,
            "B9-L": 0.3245,
            "B5-L": 0.0836,
            "B6-L": 0.1232,
            "B10-L": 0.099,
            "B11-L": 0.0385,
            "B12-L": 0.0671,
            "B13-L": 0.1485,
            "B14-L": 0.1639,
        },
        "q0": {
            "B2-L": 0.1397,
            "B3-L": 0.209,
            "B4-L": -0.0429,
            "B9-L": 0.1826,
            "B5-L": 0.0176,
            "B6-L": 0.0825,
            "B10-L": 0.0638,
            "B11-L": 0.0198,
            "B12-L": 0.0176,
            "B13-L": 0.0638,
            "B14-L": 0.055,
        },
    }
    generator_target_p = {
        "target_p": {
            "B1-G": 2.5449676945668136,
            "B2-G": 0.4380323054331865,
            "B3-G": 0.0,
            "B6-G": 0.0,
            "B8-G": 0.0,
        }
    }
    load_update_df = pd.DataFrame(load_update).reset_index().rename(columns={"index": "id"}).set_index("id")
    generator_target_p_df = pd.DataFrame(generator_target_p).reset_index().rename(columns={"index": "id"}).set_index("id")

    dynamic_info_updated_injections = build_updated_dynamic_injection_state(
        dynamic_info_base=dynamic_info_base,
        string_info_base=string_info_base,
        load_update_df=load_update_df,
        generator_target_p_df=generator_target_p_df,
    )

    net.update_loads(load_update_df)
    net.update_generators(generator_target_p_df)
    fully_converged_result = pypowsybl.loadflow.run_ac(net, parameters=base_loadflow_parameter)[0]
    assert fully_converged_result.status == pypowsybl._pypowsybl.LoadFlowComponentStatus.CONVERGED

    dynamic_info_reference = create_network_data_pypowsybl(net).dynamic_network_data

    return IEEE14QuasiNewtonQlimit10PercentSetup(
        base_loadflow_parameter=base_loadflow_parameter,
        static_info_base=static_info_base,
        dynamic_info_base=dynamic_info_base,
        string_info_base=string_info_base,
        jacobian_data_base=jacobian_data_base,
        y_matrix_base=y_matrix_base,
        load_update_df=load_update_df,
        generator_target_p_df=generator_target_p_df,
        dynamic_info_updated_injections=dynamic_info_updated_injections,
        dynamic_info_reference=dynamic_info_reference,
    )
