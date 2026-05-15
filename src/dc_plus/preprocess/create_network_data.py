# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Functions to create the network data from different input formats."""

import numpy as np
import pypowsybl
from pandapower.auxiliary import pandapowerNet

from dc_plus.importing.import_helpers import (
    _empty_schema_dataframe,
    _filter_branch_tap_data,
    _filter_main_grid_network_data,
    _get_admittance_branches,
    _get_branch_admittance_terms,
    _get_branch_current_limits,
    _get_bus_active_power_injections,
    _get_bus_admittance_shunts,
    _get_bus_reactive_power_injections,
    _get_shunt_section_information,
)
from dc_plus.importing.import_schema import (
    BranchParamSchema,
    BusParamSchema,
    InjectionParamSchema,
    LimitParamSchema,
    ShuntParamSchema,
    TapChangerParamSchema,
    TapPositionParamSchema,
)
from dc_plus.importing.pandapower.import_helpers import _get_slack_bus_id
from dc_plus.importing.pandapower.pandapower_import import (
    _get_branches_parameter_pandapower,
    _get_buses_pandapower,
    _get_injections_pandapower,
    _get_limits_parameter_pandapower,
    _get_shunts_pandapower,
)
from dc_plus.importing.powsybl.powsybl_import import (
    _get_branches_parameter_powsybl,
    _get_buses_powsybl,
    _get_injections_powsybl,
    _get_limits_parameter_powsybl,
    _get_shunts_powsybl,
    _get_tap_changer_parameter_powsybl,
    _get_tap_steps_parameter_powsybl,
)
from dc_plus.interfaces.network_information import (
    BusType,
    DynamicNetworkInformation,
    StaticNetworkInformation,
    StringNetworkInformation,
    TransformerTapInformation,
    _check_network_data_consistency,
)
from dc_plus.preprocess.helper_functions import _is_branch_symmetric, _is_connected_to_slack


def _filter_main_grid_create_network_data_inputs(
    buses: BusParamSchema,
    branches: BranchParamSchema,
    injections: InjectionParamSchema,
    shunts: ShuntParamSchema,
    ratio_changer: TapChangerParamSchema,
    phase_changer: TapChangerParamSchema,
    ratio_positions: TapPositionParamSchema,
    phase_positions: TapPositionParamSchema,
) -> tuple[
    BusParamSchema,
    BranchParamSchema,
    InjectionParamSchema,
    ShuntParamSchema,
    TapChangerParamSchema,
    TapChangerParamSchema,
    TapPositionParamSchema,
    TapPositionParamSchema,
]:
    """Filter dynamic and tap import tables down to the main grid branch set."""
    buses, branches, injections, shunts = _filter_main_grid_network_data(
        buses,
        branches,
        injections,
        shunts,
    )
    ratio_changer, ratio_positions = _filter_branch_tap_data(branches, ratio_changer, ratio_positions)
    phase_changer, phase_positions = _filter_branch_tap_data(branches, phase_changer, phase_positions)
    return (
        buses,
        branches,
        injections,
        shunts,
        ratio_changer,
        phase_changer,
        ratio_positions,
        phase_positions,
    )


def _get_branch_index_by_id(branches: BranchParamSchema) -> dict[str, int]:
    """Build a branch-index lookup keyed by branch id."""
    return {str(branch_id): idx for idx, branch_id in enumerate(branches["id_str"].astype(str).values)}


def _create_transformer_tap_information(
    tap_changer: TapChangerParamSchema,
    tap_positions: TapPositionParamSchema,
) -> TransformerTapInformation:
    """Build static tap information for a single transformer branch."""
    tap_positions = TapPositionParamSchema.validate(tap_positions.sort_values("position").reset_index(drop=True))

    resistance = tap_positions["offset_r"].to_numpy(dtype=float)
    reactance = tap_positions["offset_x"].to_numpy(dtype=float)
    conductance_from = tap_positions["offset_g1"].to_numpy(dtype=float)
    susceptance_from = tap_positions["offset_b1"].to_numpy(dtype=float)
    conductance_to = tap_positions["offset_g2"].to_numpy(dtype=float)
    susceptance_to = tap_positions["offset_b2"].to_numpy(dtype=float)
    shift_ratio_rho = 1 / tap_positions["offset_rho"].to_numpy(dtype=float)
    shift_angle = -tap_positions["offset_alpha"].to_numpy(dtype=float)

    y_series, y_charging_from, y_charging_to, _ = _get_branch_admittance_terms(
        r=resistance,
        x=reactance,
        g1=conductance_from,
        b1=susceptance_from,
        g2=conductance_to,
        b2=susceptance_to,
        rho=shift_ratio_rho,
        alpha=shift_angle,
    )

    neutral_shift_angle = 0.0
    neutral_shift_ratio_rho = 1.0

    y_series_neutral, y_charging_from_neutral, y_charging_to_neutral, _ = _get_branch_admittance_terms(
        r=np.asarray([float(tap_changer["neutral_r"])], dtype=float),
        x=np.asarray([float(tap_changer["neutral_x"])], dtype=float),
        g1=np.asarray([float(tap_changer["neutral_g1"])], dtype=float),
        b1=np.asarray([float(tap_changer["neutral_b1"])], dtype=float),
        g2=np.asarray([float(tap_changer["neutral_g2"])], dtype=float),
        b2=np.asarray([float(tap_changer["neutral_b2"])], dtype=float),
        rho=np.asarray([neutral_shift_ratio_rho], dtype=float),
        alpha=np.asarray([neutral_shift_angle], dtype=float),
    )
    # check that y_series is not 0 for any branch
    if np.any(np.isclose(y_series_neutral, 0.0)):
        raise ValueError("Zero impedance branch detected. Check network Data!")
    neutral_y_series = y_series_neutral[0]
    neutral_y_charging_from = y_charging_from_neutral[0]
    neutral_y_charging_to = y_charging_to_neutral[0]

    return TransformerTapInformation(
        n_max_tap_positions=len(tap_positions),
        neutral_conductance_series=float(np.real(neutral_y_series)),
        neutral_susceptance_series=float(np.imag(neutral_y_series)),
        neutral_conductance_charging_from=float(np.real(neutral_y_charging_from)),
        neutral_susceptance_charging_from=float(np.imag(neutral_y_charging_from)),
        neutral_conductance_charging_to=float(np.real(neutral_y_charging_to)),
        neutral_susceptance_charging_to=float(np.imag(neutral_y_charging_to)),
        neutral_shift_angle=neutral_shift_angle,
        neutral_shift_ratio_rho=neutral_shift_ratio_rho,
        tap_offset_conductance_series=np.real(y_series - neutral_y_series),
        tap_offset_susceptance_series=np.imag(y_series - neutral_y_series),
        tap_offset_conductance_charging_from=np.real(y_charging_from - neutral_y_charging_from),
        tap_offset_susceptance_charging_from=np.imag(y_charging_from - neutral_y_charging_from),
        tap_offset_conductance_charging_to=np.real(y_charging_to - neutral_y_charging_to),
        tap_offset_susceptance_charging_to=np.imag(y_charging_to - neutral_y_charging_to),
        tap_offset_shift_angle=shift_angle,
        tap_offset_shift_ratio_rho=shift_ratio_rho,
    )


def _build_transformer_tap_metadata(
    branches: BranchParamSchema,
    tap_changers: TapChangerParamSchema,
    tap_positions: TapPositionParamSchema,
) -> tuple[np.ndarray, np.ndarray, dict[int, TransformerTapInformation]]:
    """Build branch-aligned transformer tap flags, positions, and metadata."""
    n_branches = len(branches)
    has_transformer = np.zeros(n_branches, dtype=bool)
    current_tap_positions = np.zeros(n_branches, dtype=int)
    tap_info: dict[int, TransformerTapInformation] = {}
    if tap_changers.empty:
        return has_transformer, current_tap_positions, tap_info

    branch_index_by_id = _get_branch_index_by_id(branches)
    tap_positions_by_id = {
        str(branch_id): branch_positions.copy()
        for branch_id, branch_positions in tap_positions.groupby("id_str", sort=False)
    }

    for _, tap_changer in tap_changers.iterrows():
        branch_idx = branch_index_by_id.get(str(tap_changer["id_str"]))
        if branch_idx is None:
            continue

        current_tap_positions[branch_idx] = int(tap_changer["current_tap_pos"])
        branch_tap_positions = tap_positions_by_id.get(str(tap_changer["id_str"]))
        if branch_tap_positions is None or branch_tap_positions.empty:
            continue

        has_transformer[branch_idx] = True
        tap_info[branch_idx] = _create_transformer_tap_information(tap_changer, branch_tap_positions)

    return has_transformer, current_tap_positions, tap_info


def _get_powsybl_tap_imports(
    network: pypowsybl.network.Network, split_trafo_charging: bool = True
) -> tuple[
    TapChangerParamSchema,
    TapChangerParamSchema,
    TapPositionParamSchema,
    TapPositionParamSchema,
]:
    """Import ratio and phase tap changer tables from a Powsybl network."""
    transformers = network.get_2_windings_transformers(attributes=["r", "x", "g", "b"])
    ratio_changer = _get_tap_changer_parameter_powsybl(
        network.get_ratio_tap_changers(), transformers.copy(), split_trafo_charging=split_trafo_charging
    )
    phase_changer = _get_tap_changer_parameter_powsybl(
        network.get_phase_tap_changers(), transformers.copy(), split_trafo_charging=split_trafo_charging
    )
    ratio_positions = _get_tap_steps_parameter_powsybl(network.get_ratio_tap_changer_steps())
    phase_positions = _get_tap_steps_parameter_powsybl(network.get_phase_tap_changer_steps())

    return ratio_changer, phase_changer, ratio_positions, phase_positions


def _create_network_data(
    buses: BusParamSchema,
    branches: BranchParamSchema,
    injections: InjectionParamSchema,
    limits: LimitParamSchema,
    shunts: ShuntParamSchema,
    ratio_changer: TapChangerParamSchema,
    phase_changer: TapChangerParamSchema,
    ratio_positions: TapPositionParamSchema,
    phase_positions: TapPositionParamSchema,
) -> tuple[StaticNetworkInformation, DynamicNetworkInformation, StringNetworkInformation]:
    """Create the network data from a Powsybl network.

    Creates the central network data structures used in DCplus from a Powsybl network.

    Parameters
    ----------
    buses : BusParamSchema
        The bus parameters of the network.
    branches : BranchParamSchema
        The branch parameters of the network.
    injections : InjectionParamSchema
        The injection parameters of the network.
    limits : LimitParamSchema
        The limit parameters of the network.
    shunts : ShuntParamSchema
        The shunt parameters of the network.
    ratio_changer : TapChangerParamSchema
        The ratio changer parameters of the network.
    phase_changer : TapChangerParamSchema
        The phase changer parameters of the network.
    ratio_positions : TapPositionParamSchema
        The ratio changer tap positions of the network.
    phase_positions : TapPositionParamSchema
        The phase changer tap positions of the network.

    Returns
    -------
    tuple[StaticNetworkInformation, DynamicNetworkInformation, StringNetworkInformation]
        The static, dynamic and string network information.
    """
    buses, branches, injections, shunts, ratio_changer, phase_changer, ratio_positions, phase_positions = (
        _filter_main_grid_create_network_data_inputs(
            buses=buses,
            branches=branches,
            injections=injections,
            shunts=shunts,
            ratio_changer=ratio_changer,
            phase_changer=phase_changer,
            ratio_positions=ratio_positions,
            phase_positions=phase_positions,
        )
    )

    y_ff, y_ft, y_tf, y_tt, y_series, y_charging_symmetric = _get_admittance_branches(branches=branches)
    y_shunts = _get_bus_admittance_shunts(shunts=shunts)
    limit_names, branch_current_limits = _get_branch_current_limits(branches, limits)
    has_ratio_changing_transformer, branch_ratio_tap_positions, ratio_shift_info = _build_transformer_tap_metadata(
        branches=branches,
        tap_changers=ratio_changer,
        tap_positions=ratio_positions,
    )
    has_phase_shifting_transformer, branch_phase_tap_positions, phase_shift_info = _build_transformer_tap_metadata(
        branches=branches,
        tap_changers=phase_changer,
        tap_positions=phase_positions,
    )

    bus_active_power = _get_bus_active_power_injections(injections=injections, n_buses=len(buses))
    bus_reactive_power = _get_bus_reactive_power_injections(injections=injections, n_buses=len(buses))
    is_branch_symmetric = _is_branch_symmetric(
        y_ff=y_ff,
        y_ft=y_ft,
        y_tf=y_tf,
        y_tt=y_tt,
    )

    slack_bus_ids = buses[buses["bus_type"] == BusType.SLACK]["id_int"].values
    is_connected_to_slack = _is_connected_to_slack(
        branch_from_nodes=branches["from_bus_index"].to_numpy(dtype=int),
        branch_to_nodes=branches["to_bus_index"].to_numpy(dtype=int),
        slack_bus_indices=slack_bus_ids,
    )

    static_info = StaticNetworkInformation(
        injection_limits=np.full(len(injections), np.nan, dtype=float),
        shunt_section_info=_get_shunt_section_information(shunts),
        n_limits=len(limit_names),
        branch_current_limits=branch_current_limits,
        has_phase_shifting_transformer=has_phase_shifting_transformer,
        has_ratio_changing_transformer=has_ratio_changing_transformer,
        phase_shift_info=phase_shift_info,
        ratio_shift_info=ratio_shift_info,
    )
    dynamic_info = DynamicNetworkInformation(
        branch_from_bus=branches["from_bus_index"].to_numpy(dtype=int),
        branch_to_bus=branches["to_bus_index"].to_numpy(dtype=int),
        branch_active_power_from=branches["p1"].to_numpy(),
        branch_active_power_to=branches["p2"].to_numpy(),
        branch_reactive_power_from=branches["q1"].to_numpy(),
        branch_reactive_power_to=branches["q2"].to_numpy(),
        branch_current_magnitude_from=branches["i1"].to_numpy(),
        branch_current_magnitude_to=branches["i2"].to_numpy(),
        branch_ratio_tap_positions=branch_ratio_tap_positions,
        branch_phase_tap_positions=branch_phase_tap_positions,
        branch_effective_admittance_from_to=y_ft,
        branch_effective_admittance_from_from=y_ff,
        branch_effective_admittance_to_to=y_tt,
        branch_effective_admittance_to_from=y_tf,
        branch_effective_admittance_series=y_series,
        branch_r=branches["r"].to_numpy(dtype=float),
        branch_x=branches["x"].to_numpy(dtype=float),
        branch_g_from=branches["g1"].to_numpy(dtype=float),
        branch_b_from=branches["b1"].to_numpy(dtype=float),
        branch_g_to=branches["g2"].to_numpy(dtype=float),
        branch_b_to=branches["b2"].to_numpy(dtype=float),
        branch_rho=branches["rho"].to_numpy(dtype=float),
        branch_shift_angle_rad=branches["alpha"].to_numpy(dtype=float),
        branch_effective_admittance_charging_symmetric=y_charging_symmetric,
        branch_connected=branches["connected"].to_numpy(),
        is_branch_symmetric=is_branch_symmetric,
        is_connected_to_slack=is_connected_to_slack,
        bus_voltage_magnitudes=buses["voltage_magnitude"].to_numpy(),
        bus_voltage_angles_rad=buses["voltage_angle"].to_numpy(),
        bus_active_power=bus_active_power,
        bus_reactive_power=bus_reactive_power,
        bus_type=buses["bus_type"].to_numpy(dtype=int),
        bus_is_angle_reference=buses["is_angle_reference"].to_numpy(dtype=bool),
        injection_to_bus=injections["bus_index"].to_numpy(dtype=int),
        injection_active_power=injections["p"].to_numpy(),
        injection_reactive_power=injections["q"].to_numpy(),
        injection_connected=injections["connected"].to_numpy(),
        shunt_bus_indices=shunts["bus_index"].to_numpy(dtype=int),
        shunt_active_power=shunts["p"].to_numpy(),
        shunt_reactive_power=shunts["q"].to_numpy(),
        shunt_section_count=shunts["section_count"].to_numpy(dtype=int),
        shunt_effective_bus_admittance=y_shunts,
        shunt_connected=shunts["connected"].to_numpy(),
    )
    string_info = StringNetworkInformation(
        bus_ids=buses["id_str"].to_numpy(),
        shunt_ids=shunts["id_str"].to_numpy(),
        branch_types=branches["branch_type"].to_numpy(),
        branch_ids=branches["id_str"].to_numpy(),
        limit_names=limit_names,
        injection_types=injections["injection_type"].to_numpy(),
        injection_ids=injections["id_str"].values,
    )

    _check_network_data_consistency(dynamic_network_data=dynamic_info, string_network_data=string_info)
    return static_info, dynamic_info, string_info


def create_network_data_pypowsybl(
    network: pypowsybl.network.Network, split_trafo_charging: bool = True
) -> tuple[StaticNetworkInformation, DynamicNetworkInformation, StringNetworkInformation]:
    """Create the network data from a Powsybl network.

    Creates the central network data structures used in DCplus from a Powsybl network.

    Parameters
    ----------
    network : pypowsybl.network.Network
        The Powsybl network.
    split_trafo_charging : bool, optional
        Whether to split transformer charging admittances into from and to components, by default True.

    Returns
    -------
    tuple[StaticNetworkInformation, DynamicNetworkInformation, StringNetworkInformation]
        The static, dynamic and string network information.
    """
    network.per_unit = True

    branches = _get_branches_parameter_powsybl(network, split_trafo_charging=split_trafo_charging)
    injections = _get_injections_powsybl(network)
    shunts = _get_shunts_powsybl(network)
    slack_id = network.get_extensions("slackTerminal")["bus_id"].values[0]
    references = network.get_extensions("referencePriorities")
    if len(references) > 0:
        reference_id = network.get_extensions("referencePriorities").index[0]
    else:
        reference_id = slack_id
    buses = _get_buses_powsybl(net=network, slack_id=slack_id, injections=injections, reference_id=reference_id)
    limits = _get_limits_parameter_powsybl(network)
    ratio_changer, phase_changer, ratio_positions, phase_positions = _get_powsybl_tap_imports(network)

    return _create_network_data(
        buses=buses,
        branches=branches,
        injections=injections,
        limits=limits,
        shunts=shunts,
        ratio_changer=ratio_changer,
        phase_changer=phase_changer,
        ratio_positions=ratio_positions,
        phase_positions=phase_positions,
    )


def _get_empty_tap_imports() -> tuple[
    TapChangerParamSchema,
    TapChangerParamSchema,
    TapPositionParamSchema,
    TapPositionParamSchema,
]:
    """Create empty validated tap changer and tap position tables."""
    return (
        _empty_schema_dataframe(TapChangerParamSchema),
        _empty_schema_dataframe(TapChangerParamSchema),
        _empty_schema_dataframe(TapPositionParamSchema),
        _empty_schema_dataframe(TapPositionParamSchema),
    )


def create_network_data_pandapower(
    network: pandapowerNet,
) -> tuple[StaticNetworkInformation, DynamicNetworkInformation, StringNetworkInformation]:
    """Create the network data from a Pandapower network.

    Creates the central network data structures used in DCplus from a Pandapower network.

    Parameters
    ----------
    network : pandapowerNet
        The Pandapower network.

    Returns
    -------
    tuple[StaticNetworkInformation, DynamicNetworkInformation, StringNetworkInformation]
        The static, dynamic and string network information.
    """
    branches = _get_branches_parameter_pandapower(network, split_trafo_charging=True)
    injections = _get_injections_pandapower(network)
    shunts = _get_shunts_pandapower(network)
    slack_id = _get_slack_bus_id(network)
    buses = _get_buses_pandapower(net=network, slack_id=slack_id)
    limits = _get_limits_parameter_pandapower(network)
    ratio_changer, phase_changer, ratio_positions, phase_positions = _get_empty_tap_imports()

    return _create_network_data(
        buses=buses,
        branches=branches,
        injections=injections,
        limits=limits,
        shunts=shunts,
        ratio_changer=ratio_changer,
        phase_changer=phase_changer,
        ratio_positions=ratio_positions,
        phase_positions=phase_positions,
    )
