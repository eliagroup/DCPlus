# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions for importing network data.

Helpers independent of the specific source (e.g., Powsybl, Pandapower).
"""

import numpy as np

from dc_plus.importing.import_schema import BranchParamSchema, BusParamSchema, InjectionParamSchema, ShuntParamSchema


def _remove_isolated_buses_injections(
    buses: BusParamSchema, injections: InjectionParamSchema
) -> tuple[BusParamSchema, InjectionParamSchema]:
    """Remove isolated buses and corresponding injections.

    Keeps only main grid buses.

    Parameters
    ----------
    buses : BusParamSchema
        The bus parameters of the network.
    injections : InjectionParamSchema
        The injection parameters of the network.

    Returns
    -------
    tuple[BusParamSchema, InjectionParamSchema]
        The bus and injection parameters of the network without isolated buses.
    """
    main_grid = buses[buses["grid_island_id"] == 0]
    injections = injections[injections["bus_index"].isin(main_grid["id_int"])]
    return injections


def _remove_isolated_branches(buses: BusParamSchema, branches: BranchParamSchema) -> BranchParamSchema:
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
        (branches["from_bus_index"].isin(main_grid["id_int"])) & (branches["to_bus_index"].isin(main_grid["id_int"]))
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


def _get_admittance_branches(
    branches: BranchParamSchema,
) -> tuple[np.complex128, np.complex128, np.complex128, np.complex128, np.complex128]:
    """Get the admittance matrix of the branches.

    Returns
    -------
    Float[np.ndarray, "n_branches, n_branches, n_branches, n_branches"]
        The admittance matrix of the branches.
        [branch_effective_admittance_from_to, branch_effective_admittance_from_from,
         branch_effective_admittance_to_to, branch_effective_admittance_to_from, branch_effective_admittance_series]
    """
    y_series = 1 / (branches["r"] + 1j * branches["x"])
    y_charging_from = branches["g1"] + 1j * branches["b1"]
    y_charging_to = branches["g2"] + 1j * branches["b2"]
    rho_alpha = branches["rho"] * np.exp(1j * branches["alpha"])
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
    y_shunt = shunts["g"] + 1j * shunts["b"]
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
    np.add.at(p_injections, injections.bus_index.values, injections.p.values)
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
    np.add.at(q_injections, injections.bus_index.values, injections.q.values)
    return q_injections
