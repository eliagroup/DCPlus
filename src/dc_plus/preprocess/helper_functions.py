# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions for DC+ preprocessing."""

import networkx as nx
import numpy as np
from jaxtyping import Bool, Complex128, Int
from scipy import sparse

from dc_plus.interfaces.network_information import DynamicNetworkInformation


def _flatten_time_dimension(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[:, 0]
    raise ValueError("Expected array with at most two dimensions for bridge detection.")


def _find_bridges(
    dynamic_network_data: DynamicNetworkInformation,
) -> np.ndarray:
    """Find the bridges in the network.

    Parameters
    ----------
    dynamic_network_data : DynamicNetworkInformation
        The dynamic network data.

    Returns
    -------
    np.ndarray
        Boolean array indicating which branches are bridges.
    """
    branch_from_nodes = _flatten_time_dimension(dynamic_network_data.branch_from_bus).astype(int)
    branch_to_nodes = _flatten_time_dimension(dynamic_network_data.branch_to_bus).astype(int)

    if dynamic_network_data.branch_connected is None:
        active_mask = np.ones_like(branch_from_nodes, dtype=bool)
    else:
        active_mask = _flatten_time_dimension(np.asarray(dynamic_network_data.branch_connected)).astype(bool)

    bridges = np.zeros(branch_from_nodes.shape[0], dtype=bool)

    active_indices = np.flatnonzero(active_mask)
    if active_indices.size == 0:
        return bridges

    number_edges = active_indices.size
    number_nodes = dynamic_network_data.bus_voltage_magnitudes.shape[0]

    from_nodes_active = branch_from_nodes[active_indices]
    to_nodes_active = branch_to_nodes[active_indices]

    data = np.concatenate([np.ones(number_edges), -np.ones(number_edges)])
    row_indices = np.concatenate([np.arange(number_edges), np.arange(number_edges)])
    column_indices = np.concatenate([from_nodes_active, to_nodes_active])
    connectivity_matrix = sparse.csc_array(
        (data, (row_indices, column_indices)),
        shape=(number_edges, number_nodes),
        dtype=int,
    )

    graph = connectivity_matrix.T @ connectivity_matrix
    graph_nx = nx.from_scipy_sparse_array(graph)

    bridge_pairs = {frozenset(edge) for edge in nx.bridges(graph_nx)}

    for branch_index in active_indices:
        if frozenset((branch_from_nodes[branch_index], branch_to_nodes[branch_index])) in bridge_pairs:
            bridges[branch_index] = True

    return bridges


def _is_branch_symmetric(
    y_ff: Complex128[np.ndarray, " n_branches"],
    y_ft: Complex128[np.ndarray, " n_branches"],
    y_tf: Complex128[np.ndarray, " n_branches"],
    y_tt: Complex128[np.ndarray, " n_branches"],
    tol: float = 1e-12,
) -> Bool[np.ndarray, " n_branches"]:
    """Check if branches are symmetric in admittance representation.

    Parameters
    ----------
    y_ff : Complex128[np.ndarray, " n_branches"]
        Self-admittance at the "from" bus for all branches.
    y_ft : Complex128[np.ndarray, " n_branches"]
        Mutual admittance from "from" to "to" bus for all branches.
    y_tf : Complex128[np.ndarray, " n_branches"]
        Mutual admittance from "to" to "from" bus for all branches.
    y_tt : Complex128[np.ndarray, " n_branches"]
        Self-admittance at the "to" bus for all branches.
    tol : float, optional
        Tolerance for symmetry check.

    Returns
    -------
    is_symmetric : Bool[np.ndarray, " n_branches"]
        Boolean array indicating if each branch is symmetric.
    """
    cond1 = np.abs(y_ff - y_tt) < tol
    cond2 = np.abs(y_ft - y_tf) < tol
    is_symmetric = cond1 & cond2
    return is_symmetric


def _is_connected_to_slack(
    branch_from_nodes: Int[np.ndarray, " n_branches"],
    branch_to_nodes: Int[np.ndarray, " n_branches"],
    slack_bus_indices: np.ndarray,
) -> Bool[np.ndarray, " n_branches"]:
    """Determine if branches are connected to a slack bus.

    Parameters
    ----------
    branch_from_nodes : Int[np.ndarray, " n_branches"]
        Array of "from" bus indices for all branches.
    branch_to_nodes : Int[np.ndarray, " n_branches"]
        Array of "to" bus indices for all branches.
    slack_bus_indices : np.ndarray
        Array of slack bus indices.

    Returns
    -------
    Bool[np.ndarray, " n_branches"]
        Boolean array indicating if each branch is connected to a slack bus.
    """
    connected_to_slack_bus = np.zeros(branch_from_nodes.shape[0], dtype=bool)
    slack_set = set(slack_bus_indices.tolist())

    for i in range(branch_from_nodes.shape[0]):
        if (branch_from_nodes[i] in slack_set) or (branch_to_nodes[i] in slack_set):
            connected_to_slack_bus[i] = True

    return connected_to_slack_bus
