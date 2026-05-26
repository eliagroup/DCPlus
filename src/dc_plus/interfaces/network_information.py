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

from enum import IntEnum
from typing import Dict, Literal, TypeAlias, TypeVar

import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


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


class _NetworkBaseModel(BaseModel):
    """Shared Pydantic configuration for network information containers."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


T = TypeVar("T", bound=_NetworkBaseModel)


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)


def replace_network_data(obj: T, **kwargs: dict) -> T:
    """Replace properties of pydantic model instances."""
    return type(obj).model_validate(obj.model_dump() | kwargs)


class TransformerTapInformation(_NetworkBaseModel):
    """Contains branch-aligned transformer tap information.

    Either for ratio-changing or phase-shifting transformers.
    Branches without the corresponding tap changer type are represented with zeros.

    A implementation of the cim:PhaseTapChangerTablePoint module.
    Or the pypowsybl.network.Network.get_phase_tap_changer_steps method.
    """

    n_max_tap_positions: int
    """Number of maximum tap positions."""

    neutral_conductance_series: float
    """Conductance of series at the neutral tap position."""

    neutral_susceptance_series: float
    """Susceptance of series at the neutral tap position."""

    neutral_conductance_charging_from: float
    """Conductance of charging from at the neutral tap position."""

    neutral_susceptance_charging_from: float
    """Susceptance of charging from at the neutral tap position."""

    neutral_conductance_charging_to: float
    """Conductance of charging to at the neutral tap position."""

    neutral_susceptance_charging_to: float
    """Susceptance of charging to at the neutral tap position."""

    neutral_shift_angle: float
    """Shift angle at the neutral tap position."""

    neutral_shift_ratio_rho: float
    """Tap shift ratio at the neutral tap position."""

    tap_offset_conductance_series: NDArray[Shape[" * n_max_tap_positions"], float]
    """Conductance offset of series for different tap positions."""

    tap_offset_susceptance_series: NDArray[Shape[" * n_max_tap_positions"], float]
    """Susceptance  offset of series for different tap positions."""

    tap_offset_conductance_charging_from: NDArray[Shape[" * n_max_tap_positions"], float]
    """Conductance offset of charging for different tap positions."""

    tap_offset_susceptance_charging_from: NDArray[Shape[" * n_max_tap_positions"], float]
    """Susceptance offset of charging for different tap positions."""

    tap_offset_conductance_charging_to: NDArray[Shape[" * n_max_tap_positions"], float]
    """Conductance offset of charging for different tap positions."""

    tap_offset_susceptance_charging_to: NDArray[Shape[" * n_max_tap_positions"], float]
    """Susceptance offset of charging for different tap positions."""

    tap_offset_shift_angle: NDArray[Shape[" * n_max_tap_positions"], float]
    """Tap angle offset for different tap positions."""

    tap_offset_shift_ratio_rho: NDArray[Shape[" * n_max_tap_positions"], float]
    """Tap ratio offset for different tap positions."""

    @model_validator(mode="after")
    @classmethod
    def validate_array_lengths(cls, values: "TransformerTapInformation") -> dict:
        """Validate that all tap offset arrays have the same length as n_max_tap_positions."""
        n_max_tap_positions = values.n_max_tap_positions
        for key, value in values.model_dump().items():
            if key.startswith("tap_offset_"):
                if len(value) != n_max_tap_positions:
                    raise ValueError(
                        f"Length of {key} must be equal to n_max_tap_positions ({n_max_tap_positions}), "
                        f"but got {len(value)}."
                    )
        return values


class ShuntSectionInformation(_NetworkBaseModel):
    """Contains shunt section information for shunt elements."""

    n_max_shunt_sections: int
    """Number of maximum shunt sections.

    If a shunt has fewer sections, the remaining sections are padded as zeros.
    """

    min_shunt_section: NDArray[Shape[" * n_shunts"], int]
    """Minimum number of shunt sections."""

    max_shunt_section: NDArray[Shape[" * n_shunts"], int]
    """Maximum number of shunt sections."""

    shunt_conductance_at_section: NDArray[Shape[" * n_shunts, * n_max_shunt_sections"], float]
    """Conductance for different shunt sections.

    Note: this is the absolute conductance, don't add the base conductance.
    Note: this is in the ideal case lossless and therefore 0.0
    """

    shunt_susceptance_at_section: NDArray[Shape[" * n_shunts, * n_max_shunt_sections"], float]
    """Susceptance for different shunt sections.

    Note: this is the absolute susceptance, don't add the base susceptance.
    """

    @model_validator(mode="after")
    @classmethod
    def validate_array_lengths(cls, values: "ShuntSectionInformation") -> dict:
        """Validate that all shunt section arrays have the same length as n_max_shunt_sections."""
        n_max_shunt_sections = values.n_max_shunt_sections
        n_shunts = values.min_shunt_section.shape[0]
        for key, value in values.model_dump().items():
            # arrays with shape (n_shunts, n_max_shunt_sections)
            if key.endswith("_at_section"):
                if value.shape[1] != n_max_shunt_sections:
                    raise ValueError(
                        f"Length of {key} must be equal to n_max_shunt_sections ({n_max_shunt_sections}), "
                        f"but got {value.shape[1]}."
                    )
            # arrays with shape (n_shunts,)
            if key != "n_max_shunt_sections":
                if value.shape[0] != n_shunts:
                    raise ValueError(
                        f"Length of {key} must be equal to number of shunts ({n_shunts}), but got {value.shape[0]}."
                    )
        return values

    @classmethod
    def empty(cls, n_shunts: int) -> "ShuntSectionInformation":
        """Create an empty shunt section container."""
        return cls(
            n_max_shunt_sections=0,
            min_shunt_section=np.zeros((n_shunts), dtype=int),
            max_shunt_section=np.zeros((n_shunts), dtype=int),
            shunt_conductance_at_section=np.zeros((n_shunts, 0), dtype=float),
            shunt_susceptance_at_section=np.zeros((n_shunts, 0), dtype=float),
        )

    def __eq__(self, other: object) -> bool:
        """Check the equality of two ShuntSectionInformation objects."""
        if not isinstance(other, ShuntSectionInformation):
            return NotImplemented
        return (
            self.n_max_shunt_sections == other.n_max_shunt_sections
            and _arrays_equal(self.min_shunt_section, other.min_shunt_section)
            and _arrays_equal(self.max_shunt_section, other.max_shunt_section)
            and _arrays_equal(self.shunt_conductance_at_section, other.shunt_conductance_at_section)
            and _arrays_equal(self.shunt_susceptance_at_section, other.shunt_susceptance_at_section)
        )


class StaticNetworkInformation(_NetworkBaseModel):
    """Contains all static network information required for the DC+ solver.

    This class contains all network information and is paired with the JacobianData class.

    This data will be transfered to the GPU for the solving process.
    Do not add gpu unfriendly data here (e.g. strings).

    Note:
    - everything is in per unit.
    - at this point e.g. HCVDC and Battery elements expected to be converted to equivalent injections.

    """

    # Injection parameters

    injection_limit_reactive_power_min: NDArray[Shape[" * n_injections"], float]
    """The minimum reactive power limits of the injections in the network."""

    injection_limit_reactive_power_max: NDArray[Shape[" * n_injections"], float]
    """The maximum reactive power limits of the injections in the network."""

    injection_limit_active_power_min: NDArray[Shape[" * n_injections"], float]
    """The minimum active power limits of the injections in the network."""

    injection_limit_active_power_max: NDArray[Shape[" * n_injections"], float]
    """The maximum active power limits of the injections in the network."""

    # injection_to_bus can change due to BDSF

    # Shunt parameters
    shunt_section_info: ShuntSectionInformation
    """Contains shunt section information for the shunt elements.
    Note: this is not optional, even if no shunts are present.
    """

    # Branch parameters
    n_limits: int
    """The number of branch limits

    Branches may have multiple limits (e.g., permanent, short-term-15-min).
    """

    branch_current_limits: NDArray[Shape[" * n_branches, * n_limits"], float]
    """The thermal limits of the branches"""

    # Tap Information

    has_ratio_changing_transformer: NDArray[Shape[" * n_branches"], bool]
    """Indicates whether a branch has a ratio-changing transformer."""

    has_phase_shifting_transformer: NDArray[Shape[" * n_branches"], bool]
    """Indicates whether a branch has a phase-shifting transformer."""

    phase_shift_info: Dict[int, TransformerTapInformation]
    """Contains branch-aligned phase-shifting transformer tap information.

    Note: this is not optional, even if no transformers are present.
    key: branch index
    value: TransformerTapInformation for the phase-shifting transformer of the branch.
    """

    ratio_shift_info: Dict[int, TransformerTapInformation]
    """Contains branch-aligned ratio-changing transformer tap information.

    Note: this is not optional, even if no transformers are present.
    key: branch index
    value: TransformerTapInformation for the ratio-changing transformer of the branch.
    """

    @model_validator(mode="after")
    @classmethod
    def validate_array_shapes(cls, values: "StaticNetworkInformation") -> dict:
        """Validate that all branch-aligned arrays have the same length."""
        n_branches = values.has_ratio_changing_transformer.shape[0]
        n_injections = values.injection_limit_active_power_min.shape[0]
        n_limits = values.n_limits
        for key, value in values.model_dump().items():
            if key in ["has_ratio_changing_transformer", "has_phase_shifting_transformer"]:
                if value.shape[0] != n_branches:
                    raise ValueError(
                        f"Length of {key} must be equal to number of branches ({n_branches}), but got {value.shape[0]}."
                    )
            if key in [
                "injection_limit_active_power_min",
                "injection_limit_active_power_max",
                "injection_limit_reactive_power_min",
                "injection_limit_reactive_power_max",
            ]:
                if value.shape[0] != n_injections:
                    raise ValueError(
                        f"Length of {key} must be equal to number of injections ({n_injections}), but got {value.shape[0]}."
                    )
            if key == "branch_current_limits":
                if value.shape != (n_branches, n_limits):
                    raise ValueError(
                        f"Shape of {key} must be equal to (number of branches, number of limits) "
                        f"({n_branches}, {n_limits}), but got {value.shape}."
                    )
        return values

    @model_validator(mode="after")
    @classmethod
    def validate_phase_ratio_len(cls, values: "StaticNetworkInformation") -> dict:
        """Validate that all branch-aligned arrays have the same length."""
        n_ratio_changing_transformers = np.sum(values.has_ratio_changing_transformer)
        n_phase_shifting_transformers = np.sum(values.has_phase_shifting_transformer)
        if len(values.phase_shift_info) != n_phase_shifting_transformers:
            raise ValueError(
                f"Length of phase_shift_info must be equal to number of phase-shifting transformers "
                f"({n_phase_shifting_transformers}), but got {len(values.phase_shift_info)}."
            )
        if len(values.ratio_shift_info) != n_ratio_changing_transformers:
            raise ValueError(
                f"Length of ratio_shift_info must be equal to number of ratio-changing transformers "
                f"({n_ratio_changing_transformers}), but got {len(values.ratio_shift_info)}."
            )
        return values

    def __eq__(self, other: object) -> bool:
        """Check the equality of two StaticNetworkInformation objects."""
        if not isinstance(other, StaticNetworkInformation):
            return NotImplemented
        return (
            _arrays_equal(self.injection_limit_reactive_power_min, other.injection_limit_reactive_power_min)
            and _arrays_equal(self.injection_limit_reactive_power_max, other.injection_limit_reactive_power_max)
            and _arrays_equal(self.injection_limit_active_power_min, other.injection_limit_active_power_min)
            and _arrays_equal(self.injection_limit_active_power_max, other.injection_limit_active_power_max)
            and self.shunt_section_info == other.shunt_section_info
            and self.n_limits == other.n_limits
            and _arrays_equal(self.branch_current_limits, other.branch_current_limits)
            and _arrays_equal(self.has_ratio_changing_transformer, other.has_ratio_changing_transformer)
            and _arrays_equal(self.has_phase_shifting_transformer, other.has_phase_shifting_transformer)
            and self.phase_shift_info == other.phase_shift_info
            and self.ratio_shift_info == other.ratio_shift_info
        )


class DynamicNetworkInformation(_NetworkBaseModel):
    """Contains all dynamic network information required for the DC+ solver.

    To initialize, the values are expected to be in per unit.
    Fill with the hotstart /converged basecase values.
    The Dynamic Network Information has a time dimension n_timestep to
    allow for time series simulations.

    This data will be transfered to the GPU for the solving process.
    Do not add gpu unfriendly data here (e.g. strings).
    """

    # branch data

    branch_from_bus: NDArray[Shape[" * n_branches"], int]
    """The from bus of the branches"""

    branch_to_bus: NDArray[Shape[" * n_branches"], int]
    """The to bus of the branches"""

    branch_active_power_from: NDArray[Shape[" * n_branches"], float]
    """Active power flows from side for all branches."""

    branch_reactive_power_from: NDArray[Shape[" * n_branches"], float]
    """Reactive power flows from side for all branches."""

    branch_active_power_to: NDArray[Shape[" * n_branches"], float]
    """Active power flows to side for all branches."""

    branch_reactive_power_to: NDArray[Shape[" * n_branches"], float]
    """Reactive power flows to side for all branches."""

    branch_current_magnitude_from: NDArray[Shape[" * n_branches"], float]
    """Current magnitudes from side for all branches."""

    branch_current_magnitude_to: NDArray[Shape[" * n_branches"], float]
    """Current magnitudes to side for all branches."""

    branch_ratio_tap_positions: NDArray[Shape[" * n_branches"], int]
    """Ratio tap positions for all branches."""

    branch_phase_tap_positions: NDArray[Shape[" * n_branches"], int]
    """Phase tap positions for all branches."""

    branch_effective_admittance_from_to: NDArray[Shape[" * n_branches"], complex]
    """Admittance from-to for the branches.
    Gets updated when tap positions change.
    """

    branch_effective_admittance_from_from: NDArray[Shape[" * n_branches"], complex]
    """Admittance from-from for the branches.
    Gets updated when tap positions change.
    """

    branch_effective_admittance_to_to: NDArray[Shape[" * n_branches"], complex]
    """Admittance to-to for the branches.
    Gets updated when tap positions change.
    """

    branch_effective_admittance_to_from: NDArray[Shape[" * n_branches"], complex]
    """Admittance to-from for the branches.
    Gets updated when tap positions change.
    """

    branch_effective_admittance_series: NDArray[Shape[" * n_branches"], complex]
    """Series admittance for the branches.
    Gets updated when tap positions change.
    """

    branch_r: NDArray[Shape[" * n_branches"], float]
    """Resistance for the branches."""

    branch_x: NDArray[Shape[" * n_branches"], float]
    """Reactance for the branches."""

    branch_g_from: NDArray[Shape[" * n_branches"], float]
    """Conductance from side for the branches."""

    branch_b_from: NDArray[Shape[" * n_branches"], float]
    """Susceptance from side for the branches."""

    branch_g_to: NDArray[Shape[" * n_branches"], float]
    """Conductance to side for the branches."""

    branch_b_to: NDArray[Shape[" * n_branches"], float]
    """Susceptance to side for the branches."""

    branch_rho: NDArray[Shape[" * n_branches"], float]
    """Tap ratio for the branches."""

    branch_shift_angle_rad: NDArray[Shape[" * n_branches"], float]
    """Shift angle in radians for the branches."""

    branch_effective_admittance_charging_symmetric: NDArray[Shape[" * n_branches"], complex]
    """Symmetric charging admittance for the branches.
    Gets updated when tap positions change.
    """

    branch_connected: NDArray[Shape[" * n_branches"], bool]
    """Indicates whether a branch is connected for all branches.
    Becomes imporant for reconnecting branches. E.g. Grid planning or canceling outage plans.
    """

    branch_is_symmetric: NDArray[Shape[" * n_branches"], bool]
    """Indicates whether a branch is symmetric in admittance representation.
    Note: this might change do to tap changes. Make sure this is updated when tap positions change.
    """

    branch_connected_to_slack: NDArray[Shape[" * n_branches"], bool]
    """Indicates whether a branch is connected to the slack bus for all branches.
    is a different lodf calculation is used for those branches.
    Note: needs update when topology changes occur.
    """

    # bus data

    bus_voltage_magnitudes: NDArray[Shape[" * n_buses"], float]
    """Voltage magnitudes in per unit for all buses."""

    bus_voltage_angles_rad: NDArray[Shape[" * n_buses"], float]
    """Voltage angles in radians for all buses."""

    bus_active_power: NDArray[Shape[" * n_buses"], float]
    """Active power injections for all buses."""

    bus_reactive_power: NDArray[Shape[" * n_buses"], float]
    """Reactive power injections for all buses."""

    bus_type: NDArray[Shape[" * n_buses"], int]
    """The type of each bus in the network.
    0: slack
    1: pv
    2: pq
    """

    bus_is_angle_reference: NDArray[Shape[" * n_buses"], bool]
    """Indicates which bus is used as the angle reference.

    This is distinct from the bus voltage-control mode. A reference bus can lose
    voltage control after an outage and become a PQ bus while still fixing the
    global angle reference.
    """

    bus_voltage_magnitude_setpoint: NDArray[Shape[" * n_buses"], float]
    """Voltage magnitude target for each bus.

    For PV buses this is the controlled voltage setpoint used by the superset
    residual. For PQ buses the value is carried only for shape consistency.
    """

    # injection data

    injection_to_bus: NDArray[Shape[" * n_injections"], int]
    """The bus index for each injection."""

    injection_active_power: NDArray[Shape[" * n_injections"], float]
    """Active power injections for all injections."""

    injection_reactive_power: NDArray[Shape[" * n_injections"], float]
    """Reactive power injections for all injections."""

    injection_connected: NDArray[Shape[" * n_injections"], bool]
    """Indicates whether an injection is connected for all injections."""

    injection_voltage_regulation: NDArray[Shape[" * n_injections"], bool]
    """Indicates whether an injection participates in voltage regulation."""

    injection_regulated_bus: NDArray[Shape[" * n_injections"], int]
    """The regulated bus index for each injection, or -1 when not applicable."""

    # shunt data

    shunt_bus_indices: NDArray[Shape[" * n_shunts"], int]
    """The bus index for each shunt."""

    shunt_active_power: NDArray[Shape[" * n_shunts"], float]
    """Active power injections for all shunts."""

    shunt_reactive_power: NDArray[Shape[" * n_shunts"], float]
    """Reactive power injections for all shunts."""

    shunt_section_count: NDArray[Shape[" * n_shunts"], int]
    """Number of active shunt sections for all shunts."""

    shunt_effective_bus_admittance: NDArray[Shape[" * n_shunts"], complex]
    """Conductance for different shunt sections."""

    shunt_connected: NDArray[Shape[" * n_shunts"], bool]
    """Indicates whether a shunt is connected for all shunts.
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
        """Return the indices of the angle-reference bus.

        Returns
        -------
        slack_index : np.ndarray
            The indices of the angle-reference bus.
        """
        slack_index = np.flatnonzero(self.bus_is_angle_reference)
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
        pv_buses_mask = (self.bus_type == BusType.PV) & ~self.bus_is_angle_reference
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
        """Return a boolean mask indicating which buses participate in angle equations.

        Returns
        -------
        pvpq_buses_mask : np.ndarray
            A boolean mask indicating which buses are PV or PQ buses.
        """
        pvpq_buses_mask = ~self.bus_is_angle_reference
        return pvpq_buses_mask

    @property
    def pv_buses_indices(self) -> np.ndarray:
        """Return the indices of the PV buses.

        Returns
        -------
        pv_buses_indices : np.ndarray
            The indices of the PV buses.
        """
        pv_buses_indices = np.flatnonzero(self.pv_buses_mask)
        return pv_buses_indices

    @property
    def pq_buses_indices(self) -> np.ndarray:
        """Return the indices of the PQ buses.

        Returns
        -------
        pq_buses_indices : np.ndarray
            The indices of the PQ buses.
        """
        pq_buses_indices = np.flatnonzero(self.pq_buses_mask)
        return pq_buses_indices

    @property
    def pvpq_buses_indices(self) -> np.ndarray:
        """Return the indices of buses participating in angle equations.

        Returns
        -------
        pvpq_buses_indices : np.ndarray
            The indices of the PV and PQ buses.
        """
        pvpq_buses_indices = np.flatnonzero(self.pvpq_buses_mask)
        return pvpq_buses_indices

    @property
    def pvpq_buses_indices_pvpq_order(self) -> np.ndarray:
        """Return the indices of buses participating in angle equations.

        Returns
        -------
        pvpq_buses_indices : np.ndarray
            The indices of the PV and PQ buses.
        """
        pq_angle_indices = self.pq_buses_indices[~self.bus_is_angle_reference[self.pq_buses_indices]]
        pvpq_buses_indices = np.concatenate((self.pv_buses_indices, pq_angle_indices))
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

    @model_validator(mode="after")
    @classmethod
    def validate_array_shapes(cls, values: "DynamicNetworkInformation") -> dict:
        """Validate that all branch-aligned arrays have the same length."""
        n_branches = values.branch_from_bus.shape[0]
        n_buses = values.bus_voltage_magnitudes.shape[0]
        n_injections = values.injection_to_bus.shape[0]
        n_shunts = values.shunt_bus_indices.shape[0]
        for key, value in values.model_dump().items():
            if key.startswith("branch_"):
                if value.shape[0] != n_branches:
                    raise ValueError(
                        f"Length of {key} must be equal to number of branches ({n_branches}), but got {value.shape[0]}."
                    )
            if key.startswith("bus_"):
                if value.shape[0] != n_buses:
                    raise ValueError(
                        f"Length of {key} must be equal to number of buses ({n_buses}), but got {value.shape[0]}."
                    )
            if key.startswith("injection_"):
                if value.shape[0] != n_injections:
                    raise ValueError(
                        f"Length of {key} must be equal to number of injections ({n_injections}), but got {value.shape[0]}."
                    )
            if key.startswith("shunt_"):
                if value.shape[0] != n_shunts:
                    raise ValueError(
                        f"Length of {key} must be equal to number of shunts ({n_shunts}), but got {value.shape[0]}."
                    )
        return values


class StringNetworkInformation(_NetworkBaseModel):
    """Contains all human-friendly network information required for the DC+ solver.

    This data will not be transferred to the GPU and for this reason
    seperated from the Static and Dynamic Network Information.
    """

    bus_ids: NDArray[Shape[" * n_buses"], str]
    """ids of the buses, shape (n_buses,)"""

    shunt_ids: NDArray[Shape[" * n_shunts"], str]
    """ids of the shunts, shape (n_shunts,)"""

    branch_types: NDArray[Shape[" * n_branches"], str]
    """Types of the branches, shape (n_branches,).
    E.g., line, transformer, etc.
    """

    branch_ids: NDArray[Shape[" * n_branches"], str]
    """ids of the branches, shape (n_branches,)"""

    limit_names: NDArray[Shape[" * n_limits"], str]
    """Names of the branch limits, shape (n_limits,)"""

    injection_types: NDArray[Shape[" * n_injections"], str]
    """Types of the injections, shape (n_injections,).
    E.g., load, generator, etc.
    """

    injection_ids: NDArray[Shape[" * n_injections"], str]
    """ids of the injections, shape (n_injections,)"""

    @field_validator("branch_types", mode="after")
    @classmethod
    def validate_type(cls, value: NDArray[Shape[" * n_branches"], str]) -> NDArray[Shape[" * n_branches"], str]:
        """Validate that the branch and injection types are valid."""
        valid_branch_types = AssetType.__args__  # type: list
        for branch_type in value:
            if branch_type not in valid_branch_types:
                raise ValueError(f"Invalid branch type: {branch_type}. Valid types are: {valid_branch_types}.")
        return value

    @field_validator("injection_types", mode="after")
    @classmethod
    def validate_injection_type(
        cls, value: NDArray[Shape[" * n_injections"], str]
    ) -> NDArray[Shape[" * n_injections"], str]:
        """Validate that the injection types are valid."""
        valid_injection_types = AssetType.__args__  # type: list
        for injection_type in value:
            if injection_type not in valid_injection_types:
                raise ValueError(f"Invalid injection type: {injection_type}. Valid types are: {valid_injection_types}.")
        return value


class NetworkInformation(_NetworkBaseModel):
    """Contains all network information required for the DC+ solver.

    This class contains both static and dynamic network information and is paired with the JacobianData class.

    This data will be transfered to the GPU for the solving process.
    Do not add gpu unfriendly data here (e.g. strings).
    """

    static_network_data: StaticNetworkInformation
    """Contains all static network information required for the DC+ solver."""

    dynamic_network_data: DynamicNetworkInformation
    """Contains all dynamic network information required for the DC+ solver."""

    string_network_data: StringNetworkInformation
    """Contains all human-friendly network information required for the DC+ solver."""

    @model_validator(mode="after")
    @classmethod
    def validate_consistency(cls, values: "NetworkInformation") -> dict:
        """Validate the consistency of the network data."""
        _check_network_data_consistency(values.dynamic_network_data, values.string_network_data)
        return values


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
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_g_from.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_g_from."
    )
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_b_from.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_b_from."
    )
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_g_to.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_g_to."
    )
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_b_to.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_b_to."
    )
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_rho.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_rho."
    )
    assert dynamic_network_data.branch_from_bus.shape[0] == dynamic_network_data.branch_shift_angle_rad.shape[0], (
        "Inconsistent number of branches between branch_from_nodes and branch_shift_angle_rad."
    )
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
    assert dynamic_network_data.bus_voltage_magnitudes.shape[0] == dynamic_network_data.bus_is_angle_reference.shape[0], (
        "Inconsistent number of buses between bus_voltage_magnitudes and bus_is_angle_reference."
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
    assert string_network_data.injection_ids.shape[0] == dynamic_network_data.injection_to_bus.shape[0], (
        "Inconsistent number of injections between injection_ids and injection_to_bus."
    )
