# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions for importing Powsybl networks."""

import pypowsybl
import pypowsybl.loadflow
from pypowsybl.loadflow import ConnectedComponentMode, VoltageInitMode
from pypowsybl.loadflow import Parameters as LoadFlowParameters
from pypowsybl.network import Network


def select_a_generator_as_slack_and_run_loadflow(network: Network) -> None:
    """Select a generator as slack and run loadflow.

    Powsybl tends to set the reference bus and slack as two different buses.
    Additionally, in some cases the slack is not a generator bus.
    This function selects a generator as slack bus and runs the loadflow again.

    Parameters
    ----------
    network : Network
        The Powsybl network to modify and run loadflow on.

    Raises
    ------
    ValueError
        If the loadflow does not converge after setting the slack.
        If the slack bus is not a generator.
    """
    try:
        # try to get slack from CGMES data
        b = network.get_buses()
        ref_bus = b[(b["v_angle"] == 0) & (b["connected_component"] == 0)]

        slack_voltage_id = ref_bus["voltage_level_id"].values[0]
        slack_bus_id = ref_bus.index.values[0]
    except Exception:
        # if not found, set first largest generator as slack
        generators = network.get_generators(attributes=["bus_id", "voltage_level_id", "max_p"])
        generators = generators.sort_values(by="max_p", ascending=False)
        first = 1
        slack_bus_id = generators["bus_id"].values[first]
        slack_voltage_id = generators["voltage_level_id"].values[first]

    dict_slack = {"voltage_level_id": slack_voltage_id, "bus_id": slack_bus_id}
    pypowsybl.network.Network.create_extensions(network, extension_name="slackTerminal", **dict_slack)
    network.get_extensions("slackTerminal")

    powsybl_loadflow_parameter = LoadFlowParameters(
        voltage_init_mode=VoltageInitMode.DC_VALUES,
        read_slack_bus=True,
        distributed_slack=True,
        use_reactive_limits=True,
        connected_component_mode=ConnectedComponentMode.MAIN,  # ConnectedComponentMode
    )

    loadflow_res = pypowsybl.loadflow.run_ac(network, parameters=powsybl_loadflow_parameter)[0]
    if loadflow_res.status != pypowsybl._pypowsybl.LoadFlowComponentStatus.CONVERGED:
        raise ValueError(
            f"Load flow did not converge. Status: {loadflow_res.status}, "
            f"Status text: {loadflow_res.status_text}, "
            f"Reference bus ID: {loadflow_res.reference_bus_id}"
        )

    slack_bus_id = loadflow_res.slack_bus_results[0].id
    generators = network.get_generators(attributes=["bus_id"])
    if slack_bus_id not in generators["bus_id"].values:
        raise ValueError("The slack bus must be a generator.")
