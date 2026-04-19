# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pytest
from pypowsybl.network import Network
import pypowsybl

from dc_plus.importing.powsybl.powsybl_import_helpers import select_a_generator_as_slack_and_run_loadflow


@pytest.fixture
def micro_grid_be_network_with_replaced_transformers() -> Network:
    net = pypowsybl.network.create_micro_grid_be_network()
    select_a_generator_as_slack_and_run_loadflow(net)
    pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers(net)

    return net
