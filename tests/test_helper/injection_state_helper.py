# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from dc_plus.interfaces.network_information import replace_network_data


import numpy as np
import pandas as pd


def build_updated_dynamic_injection_state(
    dynamic_info_base,
    string_info_base,
    load_update_df: pd.DataFrame,
    generator_target_p_df: pd.DataFrame,
):
    """Return a new dynamic state with only injection powers updated.

    This keeps the original N-0 voltage state and network model intact while
    applying new load ``p0/q0`` values and generator ``target_p`` dispatches.
    """
    updated_injection_active_power = dynamic_info_base.injection_active_power.copy()
    updated_injection_reactive_power = dynamic_info_base.injection_reactive_power.copy()

    injection_ids = string_info_base.injection_ids
    injection_types = string_info_base.injection_types

    for load_id, load_row in load_update_df.iterrows():
        load_mask = (injection_ids == load_id) & (injection_types == "LOAD")
        if not np.any(load_mask):
            continue
        updated_injection_active_power[load_mask] = float(load_row["p0"])
        updated_injection_reactive_power[load_mask] = float(load_row["q0"])

    for generator_id, generator_row in generator_target_p_df.iterrows():
        generator_mask = (injection_ids == generator_id) & (injection_types == "GENERATOR")
        if not np.any(generator_mask):
            continue
        updated_injection_active_power[generator_mask] = -float(generator_row["target_p"])

    active_power_delta = updated_injection_active_power - dynamic_info_base.injection_active_power
    reactive_power_delta = updated_injection_reactive_power - dynamic_info_base.injection_reactive_power

    updated_bus_active_power = dynamic_info_base.bus_active_power.copy()
    updated_bus_reactive_power = dynamic_info_base.bus_reactive_power.copy()
    np.add.at(updated_bus_active_power, dynamic_info_base.injection_to_bus, active_power_delta)
    np.add.at(updated_bus_reactive_power, dynamic_info_base.injection_to_bus, reactive_power_delta)

    return replace_network_data(
        dynamic_info_base,
        injection_active_power=updated_injection_active_power,
        injection_reactive_power=updated_injection_reactive_power,
        bus_active_power=updated_bus_active_power,
        bus_reactive_power=updated_bus_reactive_power,
    )
