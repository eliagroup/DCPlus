# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper function to get powsybl load flow parameters."""

import logging
from typing import Literal

from pypowsybl.loadflow import ConnectedComponentMode, VoltageInitMode
from pypowsybl.loadflow import Parameters as LoadFlowParameters

logger = logging.getLogger(__name__)


def get_powsybl_loadflow_parameter(
    mode: Literal["default", "academic", "hotstart_test", "one_step", "real"],
) -> LoadFlowParameters:
    """Get the powsybl load flow parameters.

    Parameters
    ----------
    mode : Literal["default", "academic", "hotstart_test", "one_step", "real"]
        Specifies the load flow mode:
        - "default": use limits and distributed slack
        - "academic": no limits, no distributed slack
        - "hotstart_test": same as default, but with increased accuracy
        - "one_step": one step mode
        - "real": real mode

    Returns
    -------
    LoadFlowParameters
        The powsybl load flow parameters.
    """
    provider_param = {
        "newtonRaphsonConvEpsPerEq": "1e-6",  # hotstart accuracy
        "maxNewtonRaphsonIterations": "20",
        "svcVoltageMonitoring": "false",
        "newtonRaphsonStoppingCriteriaType": "UNIFORM_CRITERIA",  # hotstart accuracy
        "generatorReactivePowerRemoteControl": "false",
        "useActiveLimits": "true",
        "phaseShifterRegulationOn": "false",
    }

    powsybl_loadflow_parameter = LoadFlowParameters(
        voltage_init_mode=VoltageInitMode.PREVIOUS_VALUES,
        transformer_voltage_control_on=False,
        use_reactive_limits=True,
        phase_shifter_regulation_on=False,
        twt_split_shunt_admittance=True,  # See *1 Note
        shunt_compensator_voltage_control_on=False,
        read_slack_bus=True,
        write_slack_bus=None,
        distributed_slack=True,
        # balance_type=BalanceType.PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN,  # BalanceType
        dc_use_transformer_ratio=True,
        countries_to_balance=None,  # Sequence[str]
        connected_component_mode=ConnectedComponentMode.MAIN,  # ConnectedComponentMode
        dc_power_factor=None,
        provider_parameters=provider_param,  # Dict[str, str]
    )
    if mode == "academic":
        # DC+ cannot compute limits / control -> deactivate for fair comparison
        powsybl_loadflow_parameter.voltage_init_mode = VoltageInitMode.DC_VALUES
        powsybl_loadflow_parameter.provider_parameters["useActiveLimits"] = "false"
        powsybl_loadflow_parameter.provider_parameters["generatorReactivePowerRemoteControl"] = "false"
        powsybl_loadflow_parameter.provider_parameters["svcVoltageMonitoring"] = "false"
        powsybl_loadflow_parameter.use_reactive_limits = False
        powsybl_loadflow_parameter.transformer_voltage_control_on = False
        powsybl_loadflow_parameter.shunt_compensator_voltage_control_on = False
        powsybl_loadflow_parameter.distributed_slack = False
    elif mode == "hotstart_test":
        powsybl_loadflow_parameter.voltage_init_mode = VoltageInitMode.DC_VALUES
        powsybl_loadflow_parameter.distributed_slack = False
        powsybl_loadflow_parameter.use_reactive_limits = False
        powsybl_loadflow_parameter.provider_parameters["useActiveLimits"] = "false"
        powsybl_loadflow_parameter.provider_parameters["newtonRaphsonConvEpsPerEq"] = "1e-10"
        powsybl_loadflow_parameter.provider_parameters["maxNewtonRaphsonIterations"] = "50"
    elif mode == "one_step":
        # "referenceBusSelectionMode": "GENERATOR_REFERENCE_PRIORITY",
        powsybl_loadflow_parameter.voltage_init_mode = VoltageInitMode.PREVIOUS_VALUES
        powsybl_loadflow_parameter.provider_parameters["useActiveLimits"] = "false"
        powsybl_loadflow_parameter.provider_parameters["alwaysUpdateNetwork"] = (
            "true"  # Update network even if Newton-Raphson algorithm has diverged
        )
        powsybl_loadflow_parameter.provider_parameters["newtonRaphsonConvEpsPerEq"] = (
            "1e30"  # Convergence criterion for the Newton-Raphson method
        )
        powsybl_loadflow_parameter.use_reactive_limits = False
        powsybl_loadflow_parameter.distributed_slack = False
    elif mode == "default":
        # default mode
        pass

    return powsybl_loadflow_parameter
