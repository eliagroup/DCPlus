# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Pandapower import functions for network data extraction.

This module provides functions to extract network data from pandapower networks
and convert them to the standardized DCplus format.
"""

import numpy as np
import pandas as pd

import pandapower as pp
from dc_plus.importing.import_schema import (
    BranchParamSchema,
    BusParamSchema,
    InjectionParamSchema,
    LimitParamSchema,
    ShuntParamSchema,
)
from dc_plus.interfaces.network_information import BusType
from pandapower.auxiliary import pandapowerNet


def _get_line_parameters_pandapower(net: pandapowerNet) -> pd.DataFrame:
    """Extract line parameters from pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.

    Returns
    -------
    pd.DataFrame
        Line parameters in standardized format.
    """
    lines = []
    line_id = 0

    # Process lines
    for idx, line in net.line.iterrows():
        from_bus = int(line["from_bus"])
        to_bus = int(line["to_bus"])

        # Get results
        res_line = net.res_line.loc[idx]

        # Calculate admittances from pandapower parameters
        # Pandapower stores r_ohm_per_km, x_ohm_per_km, c_nf_per_km, g_us_per_km
        length_km = line["length_km"]
        vn_kv = net.bus.loc[from_bus, "vn_kv"]
        z_base = vn_kv**2 / net.sn_mva

        r_pu = line["r_ohm_per_km"] * length_km / z_base
        x_pu = line["x_ohm_per_km"] * length_km / z_base
        c_nf_total = line["c_nf_per_km"] * length_km
        g_us_total = line["g_us_per_km"] * length_km

        # Convert to admittance (per unit)
        y_base = 1.0 / z_base
        b_shunt = 2 * np.pi * 50 * c_nf_total * 1e-9 / y_base  # Capacitive susceptance
        g_shunt = g_us_total * 1e-6 / y_base  # Conductance

        # For lines, shunt is split equally between both ends
        g1 = g_shunt / 2
        b1 = b_shunt / 2
        g2 = g_shunt / 2
        b2 = b_shunt / 2

        lines.append(
            {
                "id_int": line_id,
                "id_str": line["name"] if pd.notna(line["name"]) else f"line_{idx}",
                "name": line["name"] if pd.notna(line["name"]) else f"line_{idx}",
                "connected": line["in_service"],
                "r": r_pu,
                "x": x_pu,
                "g1": g1,
                "b1": b1,
                "g2": g2,
                "b2": b2,
                "p1": res_line["p_from_mw"] / net.sn_mva if pd.notna(res_line["p_from_mw"]) else np.nan,
                "q1": res_line["q_from_mvar"] / net.sn_mva if pd.notna(res_line["q_from_mvar"]) else np.nan,
                "i1": res_line["i_from_ka"] * vn_kv / net.sn_mva if pd.notna(res_line["i_from_ka"]) else np.nan,
                "p2": res_line["p_to_mw"] / net.sn_mva if pd.notna(res_line["p_to_mw"]) else np.nan,
                "q2": res_line["q_to_mvar"] / net.sn_mva if pd.notna(res_line["q_to_mvar"]) else np.nan,
                "i2": res_line["i_to_ka"] * vn_kv / net.sn_mva if pd.notna(res_line["i_to_ka"]) else np.nan,
                "rho": 1.0,  # No tap for lines
                "alpha": 0.0,  # No phase shift for lines
                "from_bus_index": from_bus,
                "to_bus_index": to_bus,
                "branch_type": "LINE",
            }
        )
        line_id += 1

    df = pd.DataFrame(lines)
    return BranchParamSchema.validate(df)


def _get_transformer_parameters_pandapower(net: pandapowerNet, split_trafo_charging: bool) -> pd.DataFrame:
    """Extract transformer parameters from pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.
    split_trafo_charging : bool
        If True, split transformer charging admittance symmetrically.

    Returns
    -------
    pd.DataFrame
        Transformer parameters in standardized format.
    """
    transformers = []
    transformer_id = 0
    # Process transformers
    for idx, trafo in net.trafo.iterrows():
        # Skip 3-winding transformers (they should be in net.trafo3w, but check as safeguard)
        if "mv_bus" in trafo and pd.notna(trafo.get("mv_bus")):
            continue

        hv_bus = int(trafo["hv_bus"])
        lv_bus = int(trafo["lv_bus"])

        # Get results
        res_trafo = net.res_trafo.loc[idx]

        # Get transformer parameters
        vn_hv_kv = trafo["vn_hv_kv"]
        vn_lv_kv = trafo["vn_lv_kv"]
        sn_trafo_mva = trafo["sn_mva"]

        # Calculate impedance in per unit
        vk_percent = trafo["vk_percent"]
        vkr_percent = trafo["vkr_percent"]

        z_sc_pu = vk_percent / 100.0 * (net.sn_mva / sn_trafo_mva)
        r_pu = vkr_percent / 100.0 * (net.sn_mva / sn_trafo_mva)
        x_pu = np.sqrt(z_sc_pu**2 - r_pu**2)

        # Tap position and ratio
        tap_pos = trafo.get("tap_pos", trafo.get("tap_neutral", 0))
        tap_neutral = trafo.get("tap_neutral", 0)
        tap_step_percent = trafo.get("tap_step_percent", 0)
        tap_side = trafo.get("tap_side", "hv")

        # Calculate tap ratio
        tap_diff = tap_pos - tap_neutral if pd.notna(tap_pos) and pd.notna(tap_neutral) else 0
        tap_factor = 1.0 + (tap_diff * tap_step_percent / 100.0) if pd.notna(tap_step_percent) else 1.0

        if tap_side == "lv":
            rho = (vn_hv_kv / vn_lv_kv) / tap_factor
        else:
            rho = (vn_hv_kv / vn_lv_kv) * tap_factor

        # Phase shift
        shift_degree = trafo.get("shift_degree", 0.0)
        alpha = np.deg2rad(shift_degree) if pd.notna(shift_degree) else 0.0

        # Shunt admittance (magnetizing branch)
        pfe_kw = trafo.get("pfe_kw", 0.0)
        i0_percent = trafo.get("i0_percent", 0.0)

        if split_trafo_charging and pd.notna(i0_percent) and i0_percent > 0:
            # Calculate shunt parameters
            g_shunt = (pfe_kw / 1000.0) / net.sn_mva if pd.notna(pfe_kw) else 0.0
            y_m = (i0_percent / 100.0) * (net.sn_mva / sn_trafo_mva)
            b_shunt = np.sqrt(y_m**2 - g_shunt**2) if y_m**2 > g_shunt**2 else 0.0

            # Split shunt equally if requested
            g1 = g_shunt / 2
            b1 = -b_shunt / 2  # Negative for inductive
            g2 = g_shunt / 2
            b2 = -b_shunt / 2
        else:
            g1 = 0.0
            b1 = 0.0
            g2 = 0.0
            b2 = 0.0

        transformers.append(
            {
                "id_int": transformer_id,
                "id_str": trafo["name"] if pd.notna(trafo["name"]) else f"trafo_{idx}",
                "name": trafo["name"] if pd.notna(trafo["name"]) else f"trafo_{idx}",
                "connected": trafo["in_service"],
                "r": r_pu,
                "x": x_pu,
                "g1": g1,
                "b1": b1,
                "g2": g2,
                "b2": b2,
                "p1": res_trafo["p_hv_mw"] / net.sn_mva if pd.notna(res_trafo["p_hv_mw"]) else np.nan,
                "q1": res_trafo["q_hv_mvar"] / net.sn_mva if pd.notna(res_trafo["q_hv_mvar"]) else np.nan,
                "i1": res_trafo["i_hv_ka"] * vn_hv_kv / net.sn_mva if pd.notna(res_trafo["i_hv_ka"]) else np.nan,
                "p2": res_trafo["p_lv_mw"] / net.sn_mva if pd.notna(res_trafo["p_lv_mw"]) else np.nan,
                "q2": res_trafo["q_lv_mvar"] / net.sn_mva if pd.notna(res_trafo["q_lv_mvar"]) else np.nan,
                "i2": res_trafo["i_lv_ka"] * vn_lv_kv / net.sn_mva if pd.notna(res_trafo["i_lv_ka"]) else np.nan,
                "rho": rho,
                "alpha": alpha,
                "from_bus_index": hv_bus,
                "to_bus_index": lv_bus,
                "branch_type": "TWO_WINDINGS_TRANSFORMER",
            }
        )
        transformer_id += 1

    if len(transformers) == 0:
        return pd.DataFrame(columns=BranchParamSchema.__fields__.keys())  # Return empty DataFrame with correct columns
    df = pd.DataFrame(transformers)
    return BranchParamSchema.validate(df)


def _get_branches_parameter_pandapower(net: pandapowerNet, split_trafo_charging: bool = True) -> pd.DataFrame:
    """Extract branch parameters from pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.
    split_trafo_charging : bool, optional
        If True, split transformer charging admittance symmetrically, by default True.

    Returns
    -------
    pd.DataFrame
        Branch parameters in standardized format.
    """
    # Make sure power flow has been run
    if "res_line" not in net or len(net.res_line) == 0:
        pp.runpp(net, calculate_voltage_angles=True)

    line_df = _get_line_parameters_pandapower(net)
    trafo_df = _get_transformer_parameters_pandapower(net, split_trafo_charging)
    branches = pd.concat([line_df, trafo_df], ignore_index=True)
    branches.reset_index(drop=True, inplace=True)
    branches["id_int"] = branches.index  # Reassign sequential IDs after concatenation

    return BranchParamSchema.validate(branches)


def _get_buses_pandapower(net: pandapowerNet, slack_id: NotImplementedError) -> pd.DataFrame:
    """Extract bus parameters from pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.
    slack_id : int
        The slack bus ID.

    Returns
    -------
    pd.DataFrame
        Bus parameters in standardized format.
    """
    # Make sure power flow has been run
    if "res_bus" not in net or len(net.res_bus) == 0:
        pp.runpp(net, calculate_voltage_angles=True)

    buses = []

    for idx, bus in net.bus.iterrows():
        if not bus["in_service"]:
            continue

        res_bus = net.res_bus.loc[idx]

        # Determine bus type
        if idx == slack_id:
            bus_type = BusType.SLACK
        elif idx in net.gen["bus"].values or idx in net.sgen["bus"].values:
            # Check if voltage control is enabled
            is_voltage_controlled = False
            if idx in net.gen["bus"].values:
                gen_at_bus = net.gen[net.gen["bus"] == idx]
                is_voltage_controlled = gen_at_bus["vm_pu"].notna().any()
            if not is_voltage_controlled and idx in net.sgen["bus"].values:
                sgen_at_bus = net.sgen[net.sgen["bus"] == idx]
                is_voltage_controlled = sgen_at_bus.get("controllable", pd.Series([False])).any()

            bus_type = BusType.PV if is_voltage_controlled else BusType.PQ
        else:
            bus_type = BusType.PQ

        # Determine grid island (0 = main grid, others = isolated)
        # Pandapower doesn't have a direct grid_island_id, so we assume all in-service buses are in main grid
        grid_island_id = 0

        buses.append(
            {
                "id_int": int(idx),
                "id_str": bus["name"] if pd.notna(bus["name"]) else f"bus_{idx}",
                "name": bus["name"] if pd.notna(bus["name"]) else f"bus_{idx}",
                "voltage_magnitude": res_bus["vm_pu"] if pd.notna(res_bus["vm_pu"]) else np.nan,
                "voltage_angle": (np.deg2rad(res_bus["va_degree"]) if pd.notna(res_bus["va_degree"]) else np.nan),
                "bus_type": int(bus_type),
                "grid_island_id": grid_island_id,
            }
        )

    df = pd.DataFrame(buses)
    return BusParamSchema.validate(df)


def _process_generators_pandapower(net: pandapowerNet) -> list[dict]:
    """Process generator injections from pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.

    Returns
    -------
    list[dict]
        List of generator injection dictionaries.
    """
    generator_injections = []

    for idx, gen in net.gen.iterrows():
        if not gen["in_service"]:
            continue

        bus_idx = int(gen["bus"])
        res_gen = net.res_gen.loc[idx]

        generator_injections.append(
            {
                "id_str": gen["name"] if pd.notna(gen["name"]) else f"gen_{idx}",
                "injection_type": "GENERATOR",
                "p": res_gen["p_mw"] / net.sn_mva if pd.notna(res_gen["p_mw"]) else np.nan,
                "q": res_gen["q_mvar"] / net.sn_mva if pd.notna(res_gen["q_mvar"]) else np.nan,
                "i": res_gen["va_degree"]
                if pd.notna(res_gen.get("va_degree"))
                else np.nan,  # Current not directly available
                "setpoint_p": gen["p_mw"] / net.sn_mva if pd.notna(gen["p_mw"]) else np.nan,
                "setpoint_q": gen.get("vm_pu") if pd.notna(gen.get("vm_pu")) else np.nan,
                "min_q": gen["min_q_mvar"] / net.sn_mva if pd.notna(gen.get("min_q_mvar")) else np.nan,
                "max_q": gen["max_q_mvar"] / net.sn_mva if pd.notna(gen.get("max_q_mvar")) else np.nan,
                "min_p": gen["min_p_mw"] / net.sn_mva if pd.notna(gen.get("min_p_mw")) else np.nan,
                "max_p": gen["max_p_mw"] / net.sn_mva if pd.notna(gen.get("max_p_mw")) else np.nan,
                "bus_index": bus_idx,
                "connected": gen["in_service"],
                "voltage_regulation": pd.notna(gen.get("vm_pu")),
                "regulated_bus_id_str": (
                    net.bus.loc[bus_idx, "name"] if pd.notna(net.bus.loc[bus_idx, "name"]) else f"bus_{bus_idx}"
                ),
                "regulated_bus_id_int": bus_idx if pd.notna(gen.get("vm_pu")) else -1,
            }
        )

    return generator_injections


def _process_static_generators_pandapower(net: pandapowerNet) -> list[dict]:
    """Process static generator injections from pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.

    Returns
    -------
    list[dict]
        List of static generator injection dictionaries.
    """
    static_generator_injections = []

    for idx, sgen in net.sgen.iterrows():
        if not sgen["in_service"]:
            continue

        bus_idx = int(sgen["bus"])
        res_sgen = net.res_sgen.loc[idx]

        static_generator_injections.append(
            {
                "id_str": sgen["name"] if pd.notna(sgen["name"]) else f"sgen_{idx}",
                "injection_type": "GENERATOR",
                "p": res_sgen["p_mw"] / net.sn_mva if pd.notna(res_sgen["p_mw"]) else np.nan,
                "q": res_sgen["q_mvar"] / net.sn_mva if pd.notna(res_sgen["q_mvar"]) else np.nan,
                "i": np.nan,
                "setpoint_p": sgen["p_mw"] / net.sn_mva if pd.notna(sgen["p_mw"]) else np.nan,
                "setpoint_q": sgen["q_mvar"] / net.sn_mva if pd.notna(sgen.get("q_mvar")) else np.nan,
                "min_q": sgen.get("min_q_mvar", np.nan) / net.sn_mva if pd.notna(sgen.get("min_q_mvar")) else np.nan,
                "max_q": sgen.get("max_q_mvar", np.nan) / net.sn_mva if pd.notna(sgen.get("max_q_mvar")) else np.nan,
                "min_p": sgen.get("min_p_mw", np.nan) / net.sn_mva if pd.notna(sgen.get("min_p_mw")) else np.nan,
                "max_p": sgen.get("max_p_mw", np.nan) / net.sn_mva if pd.notna(sgen.get("max_p_mw")) else np.nan,
                "bus_index": bus_idx,
                "connected": sgen["in_service"],
                "voltage_regulation": sgen.get("controllable", False) if "controllable" in sgen else False,
                "regulated_bus_id_str": (
                    net.bus.loc[bus_idx, "name"] if pd.notna(net.bus.loc[bus_idx, "name"]) else f"bus_{bus_idx}"
                ),
                "regulated_bus_id_int": (bus_idx if sgen.get("controllable", False) and "controllable" in sgen else -1),
            }
        )

    return static_generator_injections


def _process_external_grids_pandapower(net: pandapowerNet) -> list[dict]:
    """Process external grid injections from pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.

    Returns
    -------
    list[dict]
        List of external grid injection dictionaries (converted to generators).
    """
    external_grid_injections = []

    for idx, ext_grid in net.ext_grid.iterrows():
        if not ext_grid["in_service"]:
            continue

        bus_idx = int(ext_grid["bus"])
        res_ext_grid = net.res_ext_grid.loc[idx]

        external_grid_injections.append(
            {
                "id_str": ext_grid["name"] if pd.notna(ext_grid["name"]) else f"ext_grid_{idx}",
                "injection_type": "GENERATOR",  # Treat as generator (slack bus)
                "p": res_ext_grid["p_mw"] / net.sn_mva if pd.notna(res_ext_grid["p_mw"]) else np.nan,
                "q": res_ext_grid["q_mvar"] / net.sn_mva if pd.notna(res_ext_grid["q_mvar"]) else np.nan,
                "i": np.nan,
                "setpoint_p": np.nan,  # External grid adjusts P to balance system
                "setpoint_q": ext_grid.get("vm_pu") if pd.notna(ext_grid.get("vm_pu")) else np.nan,
                "min_q": ext_grid.get("min_q_mvar", np.nan) / net.sn_mva if pd.notna(ext_grid.get("min_q_mvar")) else np.nan,
                "max_q": ext_grid.get("max_q_mvar", np.nan) / net.sn_mva if pd.notna(ext_grid.get("max_q_mvar")) else np.nan,
                "min_p": ext_grid.get("min_p_mw", np.nan) / net.sn_mva if pd.notna(ext_grid.get("min_p_mw")) else np.nan,
                "max_p": ext_grid.get("max_p_mw", np.nan) / net.sn_mva if pd.notna(ext_grid.get("max_p_mw")) else np.nan,
                "bus_index": bus_idx,
                "connected": ext_grid["in_service"],
                "voltage_regulation": True,  # External grid always regulates voltage
                "regulated_bus_id_str": (
                    net.bus.loc[bus_idx, "name"] if pd.notna(net.bus.loc[bus_idx, "name"]) else f"bus_{bus_idx}"
                ),
                "regulated_bus_id_int": bus_idx,
            }
        )

    return external_grid_injections


def _process_loads_pandapower(net: pandapowerNet) -> list[dict]:
    """Process load injections from pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.

    Returns
    -------
    list[dict]
        List of load injection dictionaries.
    """
    load_injections = []

    for idx, load in net.load.iterrows():
        if not load["in_service"]:
            continue

        bus_idx = int(load["bus"])
        res_load = net.res_load.loc[idx]

        load_injections.append(
            {
                "id_str": load["name"] if pd.notna(load["name"]) else f"load_{idx}",
                "injection_type": "LOAD",
                "p": -res_load["p_mw"] / net.sn_mva if pd.notna(res_load["p_mw"]) else np.nan,  # Negative for loads
                "q": -res_load["q_mvar"] / net.sn_mva if pd.notna(res_load["q_mvar"]) else np.nan,
                "i": np.nan,
                "setpoint_p": -load["p_mw"] / net.sn_mva if pd.notna(load["p_mw"]) else np.nan,
                "setpoint_q": -load["q_mvar"] / net.sn_mva if pd.notna(load["q_mvar"]) else np.nan,
                "min_q": np.nan,
                "max_q": np.nan,
                "min_p": np.nan,
                "max_p": np.nan,
                "bus_index": bus_idx,
                "connected": load["in_service"],
                "voltage_regulation": False,
                "regulated_bus_id_str": "",
                "regulated_bus_id_int": -1,
            }
        )

    return load_injections


def _process_storage_pandapower(net: pandapowerNet) -> list[dict]:
    """Process storage (battery) injections from pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.

    Returns
    -------
    list[dict]
        List of storage injection dictionaries.
    """
    storage_injections = []

    if "storage" in net and len(net.storage) > 0:
        for idx, storage in net.storage.iterrows():
            if not storage["in_service"]:
                continue

            bus_idx = int(storage["bus"])
            res_storage = net.res_storage.loc[idx]

            storage_injections.append(
                {
                    "id_str": storage["name"] if pd.notna(storage["name"]) else f"storage_{idx}",
                    "injection_type": "BATTERY",
                    "p": res_storage["p_mw"] / net.sn_mva if pd.notna(res_storage["p_mw"]) else np.nan,
                    "q": res_storage["q_mvar"] / net.sn_mva if pd.notna(res_storage["q_mvar"]) else np.nan,
                    "i": np.nan,
                    "setpoint_p": storage["p_mw"] / net.sn_mva if pd.notna(storage["p_mw"]) else np.nan,
                    "setpoint_q": np.nan,
                    "min_q": np.nan,
                    "max_q": np.nan,
                    "min_p": storage.get("min_p_mw", np.nan) / net.sn_mva if pd.notna(storage.get("min_p_mw")) else np.nan,
                    "max_p": storage.get("max_p_mw", np.nan) / net.sn_mva if pd.notna(storage.get("max_p_mw")) else np.nan,
                    "bus_index": bus_idx,
                    "connected": storage["in_service"],
                    "voltage_regulation": False,
                    "regulated_bus_id_str": "",
                    "regulated_bus_id_int": -1,
                }
            )

    return storage_injections


def _get_injections_pandapower(net: pandapowerNet) -> pd.DataFrame:
    """Extract injection parameters from pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.

    Returns
    -------
    pd.DataFrame
        Injection parameters in standardized format.
    """
    # Make sure power flow has been run
    if "res_gen" not in net or "res_load" not in net:
        pp.runpp(net, calculate_voltage_angles=True)

    # Collect all injections from different sources
    all_injections = []
    all_injections.extend(_process_generators_pandapower(net))
    all_injections.extend(_process_static_generators_pandapower(net))
    all_injections.extend(_process_external_grids_pandapower(net))
    all_injections.extend(_process_loads_pandapower(net))
    all_injections.extend(_process_storage_pandapower(net))

    # Assign sequential IDs
    for injection_id, injection in enumerate(all_injections):
        injection["id_int"] = injection_id

    df = pd.DataFrame(all_injections)
    return InjectionParamSchema.validate(df)


def _get_shunts_pandapower(net: pandapowerNet) -> pd.DataFrame:
    """Extract shunt parameters from pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.

    Returns
    -------
    pd.DataFrame
        Shunt parameters in standardized format.
    """
    # Make sure power flow has been run
    if "res_shunt" not in net or len(net.res_shunt) == 0:
        if len(net.shunt) > 0:
            pp.runpp(net, calculate_voltage_angles=True)

    shunts = []

    for idx, shunt in net.shunt.iterrows():
        if not shunt["in_service"]:
            continue

        bus_idx = int(shunt["bus"])
        res_shunt = net.res_shunt.loc[idx]

        # Pandapower shunt parameters
        q_mvar = shunt["q_mvar"]
        p_mw = shunt.get("p_mw", 0.0)
        step = shunt.get("step", 1)
        max_step = shunt.get("max_step", 1)

        # Convert to per unit admittance
        g = (p_mw / net.sn_mva) / 1.0**2 if pd.notna(p_mw) else 0.0  # Conductance in pu
        b = (q_mvar / net.sn_mva) / 1.0**2 if pd.notna(q_mvar) else 0.0  # Susceptance in pu

        shunts.append(
            {
                "id_int": int(idx),
                "id_str": shunt["name"] if pd.notna(shunt["name"]) else f"shunt_{idx}",
                "name": shunt["name"] if pd.notna(shunt["name"]) else f"shunt_{idx}",
                "connected": shunt["in_service"],
                "g": g,
                "b": b,
                "p": res_shunt["p_mw"] / net.sn_mva if pd.notna(res_shunt["p_mw"]) else np.nan,
                "q": res_shunt["q_mvar"] / net.sn_mva if pd.notna(res_shunt["q_mvar"]) else np.nan,
                "i": np.nan,
                "bus_index": bus_idx,
                "section_count": int(step) if pd.notna(step) else 1,
                "max_section_count": int(max_step) if pd.notna(max_step) else 1,
                "voltage_regulation": shunt.get("vn_kv", False) is not False,
                "regulated_bus_id_str": (
                    net.bus.loc[bus_idx, "name"] if pd.notna(net.bus.loc[bus_idx, "name"]) else f"bus_{bus_idx}"
                ),
                "regulated_bus_id_int": bus_idx if shunt.get("vn_kv", False) is not False else -1,
            }
        )

    df = pd.DataFrame(shunts)
    if len(df) == 0:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=ShuntParamSchema.to_schema().columns.keys())
    return ShuntParamSchema.validate(df)


def _get_limits_parameter_pandapower(net: pandapowerNet) -> pd.DataFrame:
    """Extract limit parameters from pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network.

    Returns
    -------
    pd.DataFrame
        Limit parameters in standardized format.
    """
    limits = []
    limit_id = 0

    # Process line current limits
    for idx, line in net.line.iterrows():
        if not line["in_service"]:
            continue

        max_i_ka = line.get("max_i_ka")
        if pd.notna(max_i_ka) and max_i_ka > 0:
            vn_kv = net.bus.loc[int(line["from_bus"]), "vn_kv"]

            line_name = line["name"] if pd.notna(line["name"]) else f"line_{idx}"

            limits.append(
                {
                    "id_int": limit_id,
                    "element_id_str": line_name,
                    "limit_type": "CURRENT",
                    "element_type": "LINE",
                    "acceptable_duration": np.inf,
                    "side": "ONE",
                    "name": f"{line_name}_current_limit_1",
                    "value": max_i_ka * vn_kv / net.sn_mva,  # per unit
                }
            )
            limit_id += 1

            limits.append(
                {
                    "id_int": limit_id,
                    "element_id_str": line_name,
                    "limit_type": "CURRENT",
                    "element_type": "LINE",
                    "acceptable_duration": np.inf,
                    "side": "TWO",
                    "name": f"{line_name}_current_limit_2",
                    "value": max_i_ka * vn_kv / net.sn_mva,  # per unit
                }
            )
            limit_id += 1

    # Process transformer limits
    for idx, trafo in net.trafo.iterrows():
        if not trafo["in_service"]:
            continue

        sn_mva = trafo.get("sn_mva")
        if pd.notna(sn_mva) and sn_mva > 0:
            trafo_name = trafo["name"] if pd.notna(trafo["name"]) else f"trafo_{idx}"

            limits.append(
                {
                    "id_int": limit_id,
                    "element_id_str": trafo_name,
                    "limit_type": "APPARENT_POWER",
                    "element_type": "TWO_WINDINGS_TRANSFORMER",
                    "acceptable_duration": np.inf,
                    "side": "ONE",
                    "name": f"{trafo_name}_power_limit",
                    "value": sn_mva / net.sn_mva,  # per unit
                }
            )
            limit_id += 1

    df = pd.DataFrame(limits)
    if len(df) == 0:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=LimitParamSchema.to_schema().columns.keys())
    return LimitParamSchema.validate(df)
