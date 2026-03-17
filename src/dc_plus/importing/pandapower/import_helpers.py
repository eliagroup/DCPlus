# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions specific to pandapower import."""

from pandapower.auxiliary import pandapowerNet


def _get_slack_bus_id(net: pandapowerNet) -> int:
    """Get the slack bus ID from a pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.

    Returns
    -------
    int
        The slack bus ID (bus index).
    """
    # Slack bus is typically connected to ext_grid
    if len(net.ext_grid) > 0:
        return net.ext_grid["bus"].iloc[0]
    # Fallback: find bus with type 3 (slack)
    slack_buses = net.bus[net.bus["type"] == "b"].index
    if len(slack_buses) > 0:
        return slack_buses[0]
    raise ValueError("No slack bus found in pandapower network")
