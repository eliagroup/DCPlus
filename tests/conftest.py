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
