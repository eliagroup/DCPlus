# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from dataclasses import dataclass
from jaxtyping import Bool, Float, Int

import numpy as np

from dc_plus.preprocess.helper_functions import _find_bridges


@dataclass
class MockDynamicNetworkInformation:
    branch_from_bus: Int[np.ndarray, " n_branch n_timestep"]
    """The from bus of the branches"""

    branch_to_bus: Int[np.ndarray, " n_branch n_timestep"]
    """The to bus of the branches"""

    branch_connected: Bool[np.ndarray, " n_branch n_timestep"]
    """Indicates whether a branch is connected for all branches and timesteps.
    Becomes imporant for reconnecting branches. E.g. Grid planning or canceling outage plans.
    """

    bus_voltage_magnitudes: Float[np.ndarray, " n_buses n_timestep"]
    """Voltage magnitudes in per unit for all buses and timesteps."""


def test_parallel_branches_not_marked_as_bridge():
    mock_dynamic_network_data = MockDynamicNetworkInformation(
        branch_from_bus=np.array([0, 0], dtype=int),
        branch_to_bus=np.array([1, 1], dtype=int),
        branch_connected=np.ones(2, dtype=bool),
        bus_voltage_magnitudes=np.array([1, 1], dtype=float),
    )

    is_bridge = _find_bridges(mock_dynamic_network_data)
    assert not is_bridge.any()
