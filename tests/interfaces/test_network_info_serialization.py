# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


from dc_plus.preprocess.create_network_data import create_network_data_pypowsybl
import pypowsybl
from dc_plus.interfaces.network_information import (
    BusType,
    DynamicNetworkInformation,
    StaticNetworkInformation,
    StringNetworkInformation,
    TransformerTapInformation,
    NetworkInformation,
)


def test_network_info_serialization():
    net = pypowsybl.network.create_ieee30()
    network_info = create_network_data_pypowsybl(net)
    json_str = network_info.static_network_data.model_dump_json(round_trip=True)
    StaticNetworkInformation.model_validate_json(json_str)
    json_str = network_info.dynamic_network_data.model_dump_json(round_trip=True)
    DynamicNetworkInformation.model_validate_json(json_str)
    json_str = network_info.string_network_data.model_dump_json(round_trip=True)
    StringNetworkInformation.model_validate_json(json_str)
    json_str = network_info.model_dump_json(round_trip=True)
    NetworkInformation.model_validate_json(json_str)
