# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Contains classes to define network information for the DC+ solver.

NetworkInformation: is seperated into static and dynamic parts.
    StaticNetworkInformation: contains all network information that does not change during
        the solving process.
        TODO: add description of static network information
    DynamicNetworkInformation: contains all network information that can change during
        the solving process.
        TODO: add description of dynamic network information

    TODO: add documation about regulating generators -> here the PV bus is simply set to the regulated bus
    -> generator bus will not have the same result as a true PV bus with voltage regulation.

Note:
    - classes need to be gpu friendly -> separate gpu and human friendly parts like strings.
    - in the past there has been a difference between cold and hot start
      only the hot start will be considered here.
    - everything is in per unit.

"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Bool, Float, Int


class BusType(IntEnum):
    """Defines the type of the bus."""

    SLACK = 0
    PV = 1
    PQ = 2


BranchTypePandapower: TypeAlias = Literal[
    "line",
    "trafo",
    "trafo3w_lv",
    "trafo3w_mv",
    "trafo3w_hv",
    "impedance",
]
BranchTypePowsybl: TypeAlias = Literal[
    "LINE",
    "TWO_WINDINGS_TRANSFORMER",
    "TIE_LINE",
]
BranchType: TypeAlias = Literal[BranchTypePandapower, BranchTypePowsybl]

InjectionTypePandapower: TypeAlias = Literal[
    "ext_grid",
    "gen",
    "load",
    "shunt",
    "sgen",
    "ward",
    "ward_load",
    "ward_shunt",
    "xward",
    "xward_load",
    "xward_shunt",
    "dcline_from",
    "dcline_to",
]
InjectionTypePowsybl: TypeAlias = Literal[
    "LOAD",
    "GENERATOR",
    "DANGLING_LINE",
    "HVDC_CONVERTER_STATION",
    "STATIC_VAR_COMPENSATOR",
    "SHUNT_COMPENSATOR",
]
InjectionType: TypeAlias = Literal[InjectionTypePandapower, InjectionTypePowsybl]
AssetType: TypeAlias = Literal[BranchType, InjectionType]


class InjectionTypeBusBranch(IntEnum):
    """Defines the type of the injection"""

    LOAD = 0
    GENERATOR = 1


StringArray: TypeAlias = np.ndarray[np.str_, ...]


@dataclass
class TransformerTapInformation:
    """Contains transformer tap information for a single transformer.

    Either for ratio-changing or phase-shifting transformers.

    A implementation of the cim:PhaseTapChangerTablePoint module.
    Or the pypowsybl.network.Network.get_phase_tap_changer_steps method.
    """

    n_max_tap_positions: int
    """Number of maximum tap positions.

    If a transformer has fewer positions, the remaining positions are padded as zeros.
    """

    min_tap_pos: Int[np.ndarray, " n_branch"]
    """Minimum tap position of the tap-changing transformers."""

    max_tap_pos: Int[np.ndarray, " n_branch"]
    """Maximum tap position of the tap-changing transformers."""

    neutral_tap_pos: Int[np.ndarray, " n_branch"]
    """Neutral tap position of the tap-changing transformers."""

    tap_conductance_series: Float[np.ndarray, " n_branch n_max_tap_positions"]
    """Conductance of series for different tap positions."""

    tap_susceptance_series: Float[np.ndarray, " n_branch n_max_tap_positions"]
    """Susceptance of series for different tap positions."""

    tap_conductance_charging_from: Float[np.ndarray, " n_branch n_max_tap_positions"]
    """Conductance of charging for different tap positions."""

    tap_susceptance_charging_from: Float[np.ndarray, " n_branch n_max_tap_positions"]
    """Susceptance of charging for different tap positions."""

    tap_conductance_charging_to: Float[np.ndarray, " n_branch n_max_tap_positions"]
    """Conductance of charging for different tap positions."""

    tap_susceptance_charging_to: Float[np.ndarray, " n_branch n_max_tap_positions"]
    """Susceptance of charging for different tap positions."""

    tap_shift_angle: Float[np.ndarray, " n_branch n_max_tap_positions"]
    """Shift angle for different tap positions."""

    tap_shift_ratio_rho: Float[np.ndarray, " n_branch n_max_tap_positions"]
    """Tap shift ratio for different tap positions."""


@dataclass
class ShuntSectionInformation:
    """Contains shunt section information for shunt elements."""

    n_max_shunt_sections: int
    """Number of maximum shunt sections.

    If a shunt has fewer sections, the remaining sections are padded as zeros.
    """

    min_shunt_section: Int[np.ndarray, " n_shunts"]
    """Minimum number of shunt sections."""

    max_shunt_section: Int[np.ndarray, " n_shunts"]
    """Maximum number of shunt sections."""

    shunt_conductance_at_section: Float[np.ndarray, " n_shunts n_max_shunt_sections"]
    """Conductance for different shunt sections.

    Note: this is the absolute conductance, don't add the base conductance.
    Note: this is in the ideal case lossless and therefore 0.0
    """

    shunt_susceptance_at_section: Float[np.ndarray, " n_shunts n_max_shunt_sections"]
    """Susceptance for different shunt sections.

    Note: this is the absolute susceptance, don't add the base susceptance.
    """


@dataclass
class StaticNetworkInformation:
    """Contains all static network information required for the DC+ solver.

    This class contains all network information and is paired with the JacobianData class.

    This data will be transfered to the GPU for the solving process.
    Do not add gpu unfriendly data here (e.g. strings).

    Note:
    - everything is in per unit.
    - at this point e.g. HCVDC and Battery elements expected to be converted to equivalent injections.

    """

    # Injection parameters

    injection_limits: Float[np.ndarray, " n_injections"]
    """The limits of the injections in the network.
    Note:
        - This is currently not used in the DC+ solver.
        - Future versions may use a fixed jacobian approach to iterate and respect injection limits.
    """

    # injection_to_bus can change due to BDSF

    # Shunt parameters
    shunt_section_info: ShuntSectionInformation
    """Contains shunt section information for the shunt elements.
    Note: this is not optional, even if no shunts are present.
    """

    # Branch parameters
    n_limits: Int
    """The number of branch limits

    Branches may have multiple limits (e.g., permanent, short-term-15-min).
    """

    branch_current_limits: Float[np.ndarray, " n_limits n_branch"]
    """The thermal limits of the branches"""

    # Tap Information

    has_ratio_changing_transformer: Bool[np.ndarray, " n_branch"]
    """Indicates whether a branch has a ratio-changing transformer."""

    has_phase_shifting_transformer: Bool[np.ndarray, " n_branch"]
    """Indicates whether a branch has a phase-shifting transformer."""

    phase_shift_info: TransformerTapInformation
    """Contains transformer tap information for the branches.

    Note: this is not optional, even if no transformers are present.
    """

    ratio_shift_info: TransformerTapInformation
    """Contains transformer tap information for the branches.

    Note: this is not optional, even if no transformers are present.
    """


@dataclass
class DynamicNetworkInformation:
    """Contains all dynamic network information required for the DC+ solver.

    To initialize, the values are expected to be in per unit.
    Fill with the hotstart /converged basecase values.
    The Dynamic Network Information has a time dimension n_timestep to
    allow for time series simulations.

    This data will be transfered to the GPU for the solving process.
    Do not add gpu unfriendly data here (e.g. strings).
    """

    # branch data

    branch_from_bus: Int[np.ndarray, " n_branch n_timestep"]
    """The from bus of the branches"""

    branch_to_bus: Int[np.ndarray, " n_branch n_timestep"]
    """The to bus of the branches"""

    branch_active_power_from: Float[np.ndarray, " n_branch n_timestep"]
    """Active power flows from side for all branches and timesteps."""

    branch_reactive_power_from: Float[np.ndarray, " n_branch n_timestep"]
    """Reactive power flows from side for all branches and timesteps."""

    branch_active_power_to: Float[np.ndarray, " n_branch n_timestep"]
    """Active power flows to side for all branches and timesteps."""

    branch_reactive_power_to: Float[np.ndarray, " n_branch n_timestep"]
    """Reactive power flows to side for all branches and timesteps."""

    branch_current_magnitude_from: Float[np.ndarray, " n_branch n_timestep"]
    """Current magnitudes from side for all branches and timesteps."""

    branch_current_magnitude_to: Float[np.ndarray, " n_branch n_timestep"]
    """Current magnitudes to side for all branches and timesteps."""

    branch_ratio_tap_positions: Int[np.ndarray, " n_branch n_timestep"]
    """Ratio tap positions for all branches and timesteps."""

    branch_phase_tap_positions: Int[np.ndarray, " n_branch n_timestep"]
    """Phase tap positions for all branches and timesteps."""

    branch_effective_admittance_from_to: Float[np.complex128, " n_branch n_timestep"]
    """Admittance from-to for the branches.
    Gets updated when tap positions change.
    """

    branch_effective_admittance_from_from: Float[np.complex128, " n_branch n_timestep"]
    """Admittance from-from for the branches.
    Gets updated when tap positions change.
    """

    branch_effective_admittance_to_to: Float[np.complex128, " n_branch n_timestep"]
    """Admittance to-to for the branches.
    Gets updated when tap positions change.
    """

    branch_effective_admittance_to_from: Float[np.complex128, " n_branch n_timestep"]
    """Admittance to-from for the branches.
    Gets updated when tap positions change.
    """

    branch_effective_admittance_series: Float[np.complex128, " n_branch n_timestep"]
    """Series admittance for the branches.
    Gets updated when tap positions change.
    """

    branch_effective_admittance_charging_symmetric: Float[np.complex128, " n_branch n_timestep"]
    """Symmetric charging admittance for the branches.
    Gets updated when tap positions change.
    """

    branch_connected: Bool[np.ndarray, " n_branch n_timestep"]
    """Indicates whether a branch is connected for all branches and timesteps.
    Becomes imporant for reconnecting branches. E.g. Grid planning or canceling outage plans.
    """

    is_branch_symmetric: Bool[np.ndarray, " n_branch n_timestep"]
    """Indicates whether a branch is symmetric in admittance representation.
    Note: this might change do to tap changes. Make sure this is updated when tap positions change.
    """

    is_connected_to_slack: Bool[np.ndarray, " n_branch n_timestep"]
    """Indicates whether a branch is connected to the slack bus for all branches and timesteps.
    is a different lodf calculation is used for those branches.
    Note: needs update when topology changes occur.
    """

    # bus data

    bus_voltage_magnitudes: Float[np.ndarray, " n_buses n_timestep"]
    """Voltage magnitudes in per unit for all buses and timesteps."""

    bus_voltage_angles_rad: Float[np.ndarray, " n_buses n_timestep"]
    """Voltage angles in radians for all buses and timesteps."""

    bus_active_power: Float[np.ndarray, " n_buses n_timestep"]
    """Active power injections for all buses and timesteps."""

    bus_reactive_power: Float[np.ndarray, " n_buses n_timestep"]
    """Reactive power injections for all buses and timesteps."""

    bus_type: Int[np.ndarray, " n_buses"]
    """The type of each bus in the network.
    0: slack
    1: pv
    2: pq
    """

    # injection data

    injection_to_bus: Int[np.ndarray, " n_injections n_timestep"]
    """The bus index for each injection."""

    injection_active_power: Float[np.ndarray, " n_injections n_timestep"]
    """Active power injections for all injections and timesteps."""

    injection_reactive_power: Float[np.ndarray, " n_injections n_timestep"]
    """Reactive power injections for all injections and timesteps."""

    injection_connected: Bool[np.ndarray, " n_injections n_timestep"]
    """Indicates whether an injection is connected for all injections and timesteps."""

    # shunt data

    shunt_bus_indices: Int[np.ndarray, " n_shunts n_timestep"]
    """The bus index for each shunt."""

    shunt_active_power: Float[np.ndarray, " n_shunts n_timestep"]
    """Active power injections for all shunts and timesteps."""

    shunt_reactive_power: Float[np.ndarray, " n_shunts n_timestep"]
    """Reactive power injections for all shunts and timesteps."""

    shunt_section_count: Int[np.ndarray, " n_shunts n_timestep"]
    """Number of active shunt sections for all shunts and timesteps."""

    shunt_effective_bus_admittance: Float[np.ndarray, " n_shunts"]
    """Conductance for different shunt sections."""

    shunt_connected: Bool[np.ndarray, " n_shunts n_timestep"]
    """Indicates whether a shunt is connected for all shunts and timesteps.
    Becomes imporant for reconnecting shunts. E.g. Grid planning or canceling outage plans."""

    # properties for easy access

    # branches
    @property
    def n_branches(self) -> int:
        """Return the number of branches in the network.

        Returns
        -------
        n_branches : int
            The number of branches in the network.
        """
        return self.branch_from_bus.shape[0]

    # buses

    @property
    def slack_indices(self) -> np.ndarray:
        """Return the indices of the slack bus.

        Returns
        -------
        slack_index : np.ndarray
            The indices of the slack bus.
        """
        slack_index = np.flatnonzero(self.bus_type == BusType.SLACK)
        return slack_index

    def is_pv_bus(self, bus_index: int) -> bool:
        """Check if a bus is a PV bus.

        Parameters
        ----------
        bus_index : int
            The index of the bus to check.

        Returns
        -------
        is_pv_bus : bool
            Indicates if the bus is a PV bus.
        """
        is_pv_bus = self.bus_type[bus_index] == BusType.PV
        return is_pv_bus

    def is_pq_bus(self, bus_index: int) -> bool:
        """Check if a bus is a PQ bus.

        Parameters
        ----------
        bus_index : int
            The index of the bus to check.

        Returns
        -------
        is_pq_bus : bool
            Indicates if the bus is a PQ bus.
        """
        is_pq_bus = self.bus_type[bus_index] == BusType.PQ
        return is_pq_bus

    @property
    def n_buses(self) -> int:
        """Return the number of buses in the network.

        Returns
        -------
        n_buses : int
            The number of buses in the network.
        """
        return self.bus_voltage_magnitudes.shape[0]

    @property
    def n_pq_buses(self) -> int:
        """Return the number of PQ buses in the network.

        Returns
        -------
        n_pq_buses : int
            The number of PQ buses in the network.
        """
        n_pq_buses = np.sum(self.bus_type == BusType.PQ)
        return n_pq_buses

    @property
    def n_pv_buses(self) -> int:
        """Return the number of PV buses in the network.

        Returns
        -------
        n_pv_buses : int
            The number of PV buses in the network.
        """
        n_pv_buses = np.sum(self.bus_type == BusType.PV)
        return n_pv_buses

    @property
    def pv_buses_mask(self) -> np.ndarray:
        """Return a boolean mask indicating which buses are PV buses.

        Returns
        -------
        pv_buses_mask : np.ndarray
            A boolean mask indicating which buses are PV buses.
        """
        pv_buses_mask = self.bus_type == BusType.PV
        return pv_buses_mask

    @property
    def pq_buses_mask(self) -> np.ndarray:
        """Return a boolean mask indicating which buses are PQ buses.

        Returns
        -------
        pq_buses_mask : np.ndarray
            A boolean mask indicating which buses are PQ buses.
        """
        pq_buses_mask = self.bus_type == BusType.PQ
        return pq_buses_mask

    @property
    def pvpq_buses_mask(self) -> np.ndarray:
        """Return a boolean mask indicating which buses are PV or PQ buses.

        Returns
        -------
        pvpq_buses_mask : np.ndarray
            A boolean mask indicating which buses are PV or PQ buses.
        """
        pvpq_buses_mask = (self.bus_type == BusType.PV) | (self.bus_type == BusType.PQ)
        return pvpq_buses_mask

    @property
    def pv_buses_indices(self) -> np.ndarray:
        """Return the indices of the PV buses.

        Returns
        -------
        pv_buses_indices : np.ndarray
            The indices of the PV buses.
        """
        pv_buses_indices = np.flatnonzero(self.bus_type == BusType.PV)
        return pv_buses_indices

    @property
    def pq_buses_indices(self) -> np.ndarray:
        """Return the indices of the PQ buses.

        Returns
        -------
        pq_buses_indices : np.ndarray
            The indices of the PQ buses.
        """
        pq_buses_indices = np.flatnonzero(self.bus_type == BusType.PQ)
        return pq_buses_indices

    @property
    def pvpq_buses_indices(self) -> np.ndarray:
        """Return the indices of the PV and PQ buses.

        Returns
        -------
        pvpq_buses_indices : np.ndarray
            The indices of the PV and PQ buses.
        """
        pvpq_buses_indices = np.flatnonzero((self.bus_type == BusType.PV) | (self.bus_type == BusType.PQ))
        return pvpq_buses_indices

    @property
    def pvpq_buses_indices_pvpq_order(self) -> np.ndarray:
        """Return the indices of the PV and PQ buses.

        Returns
        -------
        pvpq_buses_indices : np.ndarray
            The indices of the PV and PQ buses.
        """
        pvpq_buses_indices = np.concatenate((self.pv_buses_indices, self.pq_buses_indices))
        return pvpq_buses_indices

    # injections
    @property
    def n_injections(self) -> int:
        """Return the number of injections in the network.

        Returns
        -------
        n_injections : int
            The number of injections in the network.
        """
        return self.injection_to_bus.shape[0]

    # shunts
    @property
    def n_shunts(self) -> int:
        """Return the number of shunts in the network.

        Returns
        -------
        n_shunts : int
            The number of shunts in the network.
        """
        return self.shunt_bus_indices.shape[0]


@dataclass
class StringNetworkInformation:
    """Contains all human-friendly network information required for the DC+ solver.

    This data will not be transferred to the GPU and for this reason
    seperated from the Static and Dynamic Network Information.
    """

    bus_ids: StringArray
    """ids of the buses, shape (n_buses,)"""

    shunt_ids: StringArray
    """ids of the shunts, shape (n_shunts,)"""

    branch_types: BranchType
    """Types of the branches, shape (n_branches,).
    E.g., line, transformer, etc.
    """

    branch_ids: StringArray
    """ids of the branches, shape (n_branches,)"""

    limit_names: StringArray
    """Names of the branch limits, shape (n_limits,)"""

    injection_types: InjectionType
    """Types of the injections, shape (n_injections,).
    E.g., load, generator, etc.
    """


def _check_network_data_consistency(
    dynamic_network_data: DynamicNetworkInformation,
    string_network_data: StringNetworkInformation,
) -> None:
    """Check the consistency of the network data.

    Assert that the dimensions of the different arrays are consistent with each other.

    Parameters
    ----------
    dynamic_network_data : DynamicNetworkInformation
        The dynamic network data.
    string_network_data : StringNetworkInformation
        The string network data.
    """
    # check branch data
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_to_bus.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_to_nodes."
    )
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_active_power_from.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_active_power_from."
    )
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_active_power_to.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_active_power_to."
    )
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_reactive_power_from.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_reactive_power_from."
    )
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_reactive_power_to.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_reactive_power_to."
    )
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_current_magnitude_from.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_current_magnitude_from."
    )
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_current_magnitude_to.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_current_magnitude_to."
    )
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_ratio_tap_positions.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_ratio_tap_positions."
    )
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_phase_tap_positions.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_phase_tap_positions."
    )
    assert (
        dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_effective_admittance_from_to.shape[0]
    ), "Inconsistent number of branches between branch_from_nodes and branch_effective_admittance_from_to."
    assert (
        dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_effective_admittance_from_from.shape[0]
    ), "Inconsistent number of branches between branch_from_nodes and branch_effective_admittance_from_from."
    assert (
        dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_effective_admittance_to_to.shape[0]
    ), "Inconsistent number of branches between branch_from_nodes and branch_effective_admittance_to_to."
    assert (
        dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_effective_admittance_to_from.shape[0]
    ), "Inconsistent number of branches between branch_from_nodes and branch_effective_admittance_to_from."
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_connected.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_connected."
    )

    # check bus data
    assert dynamic_network_data.bus_voltage_magnitudes.shape[0] == dynamic_network_data.bus_voltage_angles_rad.shape[0], (
        "Inconsistent number of buses between bus_voltage_magnitudes and bus_voltage_angles_rad."
    )
    assert dynamic_network_data.bus_voltage_magnitudes.shape[0] == dynamic_network_data.bus_active_power.shape[0], (
        "Inconsistent number of buses between bus_voltage_magnitudes and bus_active_power."
    )
    assert dynamic_network_data.bus_voltage_magnitudes.shape[0] == dynamic_network_data.bus_reactive_power.shape[0], (
        "Inconsistent number of buses between bus_voltage_magnitudes and bus_reactive_power."
    )
    assert dynamic_network_data.bus_voltage_magnitudes.shape[0] == dynamic_network_data.bus_type.shape[0], (
        "Inconsistent number of buses between bus_voltage_magnitudes and bus_type."
    )

    # check injection data
    assert dynamic_network_data.injection_to_bus.shape[0] == dynamic_network_data.injection_active_power.shape[0], (
        "Inconsistent number of injections between injection_to_bus and injection_active_power."
    )
    assert dynamic_network_data.injection_to_bus.shape[0] == dynamic_network_data.injection_reactive_power.shape[0], (
        "Inconsistent number of injections between injection_to_bus and injection_reactive_power."
    )
    assert dynamic_network_data.injection_to_bus.shape[0] == dynamic_network_data.injection_connected.shape[0], (
        "Inconsistent number of injections between injection_to_bus and injection_connected."
    )

    # check shunt data
    assert dynamic_network_data.shunt_bus_indices.shape[0] == dynamic_network_data.shunt_active_power.shape[0], (
        "Inconsistent number of shunts between shunt_bus_indices and shunt_active_power."
    )
    assert dynamic_network_data.shunt_bus_indices.shape[0] == dynamic_network_data.shunt_reactive_power.shape[0], (
        "Inconsistent number of shunts between shunt_bus_indices and shunt_reactive_power."
    )
    assert dynamic_network_data.shunt_bus_indices.shape[0] == dynamic_network_data.shunt_section_count.shape[0], (
        "Inconsistent number of shunts between shunt_bus_indices and shunt_section_count."
    )
    assert dynamic_network_data.shunt_bus_indices.shape[0] == dynamic_network_data.shunt_effective_bus_admittance.shape[0], (
        "Inconsistent number of shunts between shunt_bus_indices and shunt_effective_bus_admittance."
    )
    assert dynamic_network_data.shunt_bus_indices.shape[0] == dynamic_network_data.shunt_connected.shape[0], (
        "Inconsistent number of shunts between shunt_bus_indices and shunt_connected."
    )

    # check string data
    assert string_network_data.bus_ids.shape[0] == dynamic_network_data.bus_voltage_magnitudes.shape[0], (
        "Inconsistent number of buses between bus_ids and bus_voltage_magnitudes."
    )
    assert string_network_data.shunt_ids.shape[0] == dynamic_network_data.shunt_bus_indices.shape[0], (
        "Inconsistent number of shunts between shunt_ids and shunt_bus_indices."
    )
    assert string_network_data.branch_ids.shape[0] == dynamic_network_data.branch_from_bus.shape[0], (
        "Inconsistent number of branches between branch_ids and branch_from_nodes."
    )
    assert string_network_data.branch_types.shape[0] == dynamic_network_data.branch_from_bus.shape[0], (
        "Inconsistent number of branches between branch_types and branch_from_nodes."
    )
    assert string_network_data.injection_types.shape[0] == dynamic_network_data.injection_to_bus.shape[0], (
        "Inconsistent number of injections between injection_types and injection_to_bus."
    )
