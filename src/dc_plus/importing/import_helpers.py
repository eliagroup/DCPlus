# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions for importing network data.

Helpers independent of the specific source (e.g., Powsybl, Pandapower).
"""

from typing import TypeVar

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pandera.typing as pat
from jaxtyping import Complex128

from dc_plus.importing.import_schema import (
    BranchParamSchema,
    BusParamSchema,
    InjectionParamSchema,
    LimitParamSchema,
    ShuntParamSchema,
    TapChangerParamSchema,
    TapPositionParamSchema,
)
from dc_plus.interfaces.network_information import ShuntSectionInformation

DataFrameModelT = TypeVar("DataFrameModelT", bound=pa.DataFrameModel)


def _empty_schema_dataframe(schema_model: type[DataFrameModelT]) -> pat.DataFrame[DataFrameModelT]:
    """Create an empty validated dataframe for a Pandera schema model.

    Parameters
    ----------
    schema_model : type[pa.DataFrameModel]
        The Pandera dataframe model that defines the expected columns and dtypes.

    Returns
    -------
    pd.DataFrame
        An empty dataframe validated against the provided schema model.
    """
    return schema_model.validate(pd.DataFrame().reindex(columns=list(schema_model.to_schema().columns.keys())))


def _remove_isolated_buses_injections(
    buses: BusParamSchema,
    injections: InjectionParamSchema | ShuntParamSchema,
) -> InjectionParamSchema | ShuntParamSchema:
    """Remove isolated buses and corresponding bus-connected elements.

    Keeps only main grid buses.

    Parameters
    ----------
    buses : BusParamSchema
        The bus parameters of the network.
    injections : InjectionParamSchema | ShuntParamSchema
        The bus-connected elements of the network.

    Returns
    -------
    InjectionParamSchema | ShuntParamSchema
        The filtered bus-connected elements without isolated buses.
    """
    main_grid = buses[buses["grid_island_id"] == 0]
    injections = injections[(injections["bus_index"] >= 0) & injections["bus_index"].isin(main_grid["id_int"])]
    return injections


def _filter_main_grid_network_data(
    buses: BusParamSchema,
    branches: BranchParamSchema,
    injections: InjectionParamSchema,
    shunts: ShuntParamSchema,
) -> tuple[
    BusParamSchema,
    BranchParamSchema,
    InjectionParamSchema,
    ShuntParamSchema,
]:
    """Keep only the main grid across all imported parameter tables."""
    buses = _remove_isolated_buses(buses)
    injections = _remove_isolated_buses_injections(buses, injections)
    shunts = _remove_isolated_buses_injections(buses, shunts)
    branches = _remove_isolated_branches(buses, branches)
    buses, branches, injections, shunts = _reindex_main_grid_network_data(buses, branches, injections, shunts)
    return buses, branches, injections, shunts


def _remap_bus_reference_column(
    dataframe: pd.DataFrame,
    column: str,
    bus_index_by_old_id: dict[int, int],
) -> pd.DataFrame:
    """Remap a bus reference column to compact main-grid-local indices."""
    if column not in dataframe.columns:
        return dataframe

    dataframe[column] = dataframe[column].map(bus_index_by_old_id).fillna(-1).astype(int)
    return dataframe


def _reindex_main_grid_network_data(
    buses: BusParamSchema,
    branches: BranchParamSchema,
    injections: InjectionParamSchema,
    shunts: ShuntParamSchema,
) -> tuple[
    BusParamSchema,
    BranchParamSchema,
    InjectionParamSchema,
    ShuntParamSchema,
]:
    """Compact main-grid bus numbering and remap all surviving bus references."""
    buses = buses.sort_values("id_int").reset_index(drop=True).copy()
    bus_index_by_old_id = {int(old_id): new_idx for new_idx, old_id in enumerate(buses["id_int"].to_numpy(dtype=int))}
    buses["id_int"] = np.arange(len(buses), dtype=int)

    branches = branches.copy()
    branches = _remap_bus_reference_column(branches, "from_bus_index", bus_index_by_old_id)
    branches = _remap_bus_reference_column(branches, "to_bus_index", bus_index_by_old_id)

    injections = injections.copy()
    injections = _remap_bus_reference_column(injections, "bus_index", bus_index_by_old_id)
    injections = _remap_bus_reference_column(injections, "regulated_bus_id_int", bus_index_by_old_id)

    shunts = shunts.copy()
    shunts = _remap_bus_reference_column(shunts, "bus_index", bus_index_by_old_id)
    shunts = _remap_bus_reference_column(shunts, "regulated_bus_id_int", bus_index_by_old_id)

    return buses, branches, injections, shunts


def _filter_branch_tap_data(
    branches: BranchParamSchema,
    tap_changers: TapChangerParamSchema,
    tap_positions: TapPositionParamSchema,
) -> tuple[TapChangerParamSchema, TapPositionParamSchema]:
    """Keep only transformer tap data belonging to the provided branch table."""
    branch_ids = branches["id_str"].astype(str)
    tap_changers = tap_changers[tap_changers["id_str"].astype(str).isin(branch_ids)]
    tap_positions = tap_positions[tap_positions["id_str"].astype(str).isin(branch_ids)]
    return tap_changers, tap_positions


def _get_unique_limit_names(limits: LimitParamSchema) -> np.ndarray:
    """Get unique limit names while preserving importer order."""
    if limits.empty:
        return np.array([], dtype=str)
    return limits["name"].astype(str).drop_duplicates().to_numpy(dtype=str)


def _get_branch_current_limits(
    branches: BranchParamSchema,
    limits: LimitParamSchema,
) -> tuple[np.ndarray, np.ndarray]:
    """Map imported branch limits to a branch-aligned 2D array."""
    limit_names = _get_unique_limit_names(limits)
    branch_current_limits = np.full((len(branches), len(limit_names)), np.nan, dtype=float)
    if limits.empty or branches.empty:
        return limit_names, branch_current_limits

    branch_index_by_id = {str(branch_id): idx for idx, branch_id in enumerate(branches["id_str"].astype(str).values)}
    limit_index_by_name = {limit_name: idx for idx, limit_name in enumerate(limit_names)}

    for limit in limits.to_dict("records"):
        branch_idx = branch_index_by_id.get(str(limit["element_id_str"]))
        limit_idx = limit_index_by_name.get(str(limit["name"]))
        if branch_idx is None or limit_idx is None:
            continue
        branch_current_limits[branch_idx, limit_idx] = float(limit["value"])

    return limit_names, branch_current_limits


def _get_shunt_section_information(shunts: ShuntParamSchema) -> ShuntSectionInformation:
    """Build branch-independent shunt section metadata from imported shunt tables."""
    n_shunts = len(shunts)
    if n_shunts == 0:
        return ShuntSectionInformation.empty(n_shunts=0)

    max_section_count = shunts["max_section_count"].fillna(0).astype(int).to_numpy()
    section_count = shunts["section_count"].fillna(0).astype(int).to_numpy()
    n_max_shunt_sections = int(max_section_count.max()) if max_section_count.size else 0
    shunt_conductance_at_section = np.zeros((n_shunts, n_max_shunt_sections), dtype=float)
    shunt_susceptance_at_section = np.zeros((n_shunts, n_max_shunt_sections), dtype=float)

    for shunt_idx in range(n_shunts):
        n_sections = int(max_section_count[shunt_idx])
        if n_sections <= 0:
            continue

        scaling_sections = int(section_count[shunt_idx]) if int(section_count[shunt_idx]) > 0 else n_sections
        if scaling_sections <= 0:
            continue

        conductance_step = float(shunts.iloc[shunt_idx]["g"]) / scaling_sections
        susceptance_step = float(shunts.iloc[shunt_idx]["b"]) / scaling_sections
        active_sections = np.arange(1, n_sections + 1, dtype=float)
        shunt_conductance_at_section[shunt_idx, :n_sections] = conductance_step * active_sections
        shunt_susceptance_at_section[shunt_idx, :n_sections] = susceptance_step * active_sections

    return ShuntSectionInformation(
        n_max_shunt_sections=n_max_shunt_sections,
        min_shunt_section=np.zeros(n_shunts, dtype=int),
        max_shunt_section=max_section_count,
        shunt_conductance_at_section=shunt_conductance_at_section,
        shunt_susceptance_at_section=shunt_susceptance_at_section,
    )


def _remove_isolated_branches(
    buses: BusParamSchema,
    branches: BranchParamSchema,
) -> BranchParamSchema:
    """Remove isolated branches.

    Keeps only branches that are connected to main grid buses.

    Parameters
    ----------
    buses : BusParamSchema
        The bus parameters of the network.
    branches : BranchParamSchema
        The branch parameters of the network.

    Returns
    -------
    BranchParamSchema
        The branch parameters of the network without isolated branches.
    """
    main_grid = buses[buses["grid_island_id"] == 0]
    branches = branches[
        (branches["from_bus_index"] >= 0)
        & (branches["to_bus_index"] >= 0)
        & (branches["from_bus_index"].isin(main_grid["id_int"]))
        & (branches["to_bus_index"].isin(main_grid["id_int"]))
    ]
    return branches


def _remove_isolated_buses(buses: BusParamSchema) -> BusParamSchema:
    """Remove isolated buses.

    Keeps only main grid buses.

    Parameters
    ----------
    buses : BusParamSchema
        The bus parameters of the network.

    Returns
    -------
    BusParamSchema
        The bus parameters of the network without isolated buses.
    """
    main_grid = buses[buses["grid_island_id"] == 0]
    return main_grid


def _get_branch_admittance_terms(
    r: np.ndarray,
    x: np.ndarray,
    g1: np.ndarray,
    b1: np.ndarray,
    g2: np.ndarray,
    b2: np.ndarray,
    rho: np.ndarray,
    alpha: np.ndarray,
) -> tuple[
    Complex128[np.ndarray, " n_branches"],
    Complex128[np.ndarray, " n_branches"],
    Complex128[np.ndarray, " n_branches"],
    Complex128[np.ndarray, " n_branches"],
]:
    """Build reusable branch admittance primitives from electrical parameters."""
    if np.any(np.isclose(r + x, 0.0, rtol=1e-8)):
        y_series = np.zeros_like(r, dtype=complex)
    else:
        y_series = 1 / (r + 1j * x)
    y_charging_from = g1 + 1j * b1
    y_charging_to = g2 + 1j * b2
    rho_alpha = rho * np.exp(1j * alpha)
    return y_series, y_charging_from, y_charging_to, rho_alpha


def _get_admittance_branches(
    branches: BranchParamSchema,
) -> tuple[
    Complex128[np.ndarray, " n_branches"],
    Complex128[np.ndarray, " n_branches"],
    Complex128[np.ndarray, " n_branches"],
    Complex128[np.ndarray, " n_branches"],
    Complex128[np.ndarray, " n_branches"],
    Complex128[np.ndarray, " n_branches"],
]:
    """Get the admittance matrix of the branches.

    Returns
    -------
    Float[np.ndarray, "n_branches, n_branches, n_branches, n_branches"]
        The admittance matrix of the branches.
        [branch_effective_admittance_from_to, branch_effective_admittance_from_from,
         branch_effective_admittance_to_to, branch_effective_admittance_to_from, branch_effective_admittance_series]
    """
    y_series, y_charging_from, y_charging_to, rho_alpha = _get_branch_admittance_terms(
        r=branches["r"].to_numpy(dtype=float),
        x=branches["x"].to_numpy(dtype=float),
        g1=branches["g1"].to_numpy(dtype=float),
        b1=branches["b1"].to_numpy(dtype=float),
        g2=branches["g2"].to_numpy(dtype=float),
        b2=branches["b2"].to_numpy(dtype=float),
        rho=branches["rho"].to_numpy(dtype=float),
        alpha=branches["alpha"].to_numpy(dtype=float),
    )
    # check that y_series is not 0 for any branch
    if np.any(np.isclose(y_series, 0.0)):
        raise ValueError("Zero impedance branch detected. Check network Data!")
    y_charging_symmetric = (y_charging_from + y_charging_to) / 2

    y_ff = (y_series + y_charging_from) / (rho_alpha * np.conj(rho_alpha))
    y_ft = -y_series / np.conj(rho_alpha)
    y_tf = -y_series / rho_alpha
    y_tt = y_series + y_charging_to

    return y_ff, y_ft, y_tf, y_tt, y_series, y_charging_symmetric


def _get_bus_admittance_shunts(
    shunts: ShuntParamSchema,
) -> np.ndarray:
    """Get the admittance matrix of the shunts.

    Parameters
    ----------
    shunts : ShuntParamSchema
        The shunt parameters of the network.

    Returns
    -------
    Float[np.ndarray, "n_buses"]
        The node admittance of the shunts.
    """
    y_shunt = shunts["g"].to_numpy(dtype=float) + 1j * shunts["b"].to_numpy(dtype=float)
    return y_shunt


def _get_bus_active_power_injections(
    injections: InjectionParamSchema,
    n_buses: int,
) -> np.ndarray:
    """Get the nodal active power injections.

    Parameters
    ----------
    injections : InjectionParamSchema
        The injection parameters of the network.
    n_buses : int
        The number of buses in the network.

    Returns
    -------
    Float[np.ndarray, "n_buses"]
        The nodal active power injections.
    """
    p_injections = np.zeros(n_buses)
    np.add.at(
        p_injections,
        injections["bus_index"].to_numpy(dtype=int),
        injections["p"].to_numpy(dtype=float),
    )
    return p_injections


def _get_bus_reactive_power_injections(
    injections: InjectionParamSchema,
    n_buses: int,
) -> np.ndarray:
    """Get the nodal reactive power injections.

    Parameters
    ----------
    injections : InjectionParamSchema
        The injection parameters of the network.
    n_buses : int
        The number of buses in the network.

    Returns
    -------
    Float[np.ndarray, "n_buses"]
        The nodal reactive power injections.
    """
    q_injections = np.zeros(n_buses)
    np.add.at(
        q_injections,
        injections["bus_index"].to_numpy(dtype=int),
        injections["q"].to_numpy(dtype=float),
    )
    return q_injections
