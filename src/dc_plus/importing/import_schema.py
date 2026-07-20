# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Schemas for imported network data before preprocessing into solver structures."""

import logging

import pandera.pandas as pa
import pandera.typing as pat

logger = logging.getLogger(__name__)


# Note *1:
# The shunt admittance of the two-winding transformers is split into the series and shunt admittance.
# The vanilla powsybl implementation:
# https://powsybl.readthedocs.io/projects/powsybl-core/en/stable/grid_model/network_subnetwork.html#two-winding-transformer
# DCplus uses the split Pi model


class BranchParamSchema(pa.DataFrameModel):
    """Branch parameter needed for the DC+ network model."""

    id_int: pat.Series[int] = pa.Field(coerce=True)
    id_str: pat.Series[str] = pa.Field(coerce=True)
    name: pat.Series[str] = pa.Field(coerce=True)
    connected: pat.Series[bool] = pa.Field(coerce=True)
    r: pat.Series[float] = pa.Field(coerce=True)
    x: pat.Series[float] = pa.Field(coerce=True)
    g1: pat.Series[float] = pa.Field(coerce=True)
    b1: pat.Series[float] = pa.Field(coerce=True)
    g2: pat.Series[float] = pa.Field(coerce=True)
    b2: pat.Series[float] = pa.Field(coerce=True)
    p1: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    q1: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    i1: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    p2: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    q2: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    i2: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    rho: pat.Series[float] = pa.Field(coerce=True)
    alpha: pat.Series[float] = pa.Field(coerce=True)
    from_bus_index: pat.Series[int] = pa.Field(coerce=True)
    to_bus_index: pat.Series[int] = pa.Field(coerce=True)
    branch_type: pat.Series[str] = pa.Field(coerce=True)

    class Config:
        """Define Pandera class config."""

        strict = True


class InjectionParamSchema(pa.DataFrameModel):
    """Injection parameter needed for the DC+ network model."""

    id_int: pat.Series[int] = pa.Field(coerce=True)
    id_str: pat.Series[str] = pa.Field(coerce=True)
    injection_type: pat.Series[str] = pa.Field(coerce=True)
    p: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    q: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    i: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    setpoint_p: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    setpoint_q: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    voltage_setpoint: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    min_q: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    max_q: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    min_p: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    max_p: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    bus_index: pat.Series[int] = pa.Field(coerce=True)
    connected: pat.Series[bool] = pa.Field(coerce=True)
    voltage_regulation: pat.Series[bool] = pa.Field(coerce=True)
    regulated_bus_id_str: pat.Series[str] = pa.Field(coerce=True)
    regulated_bus_id_int: pat.Series[int] = pa.Field(coerce=True, description="Set to -1 if not regulated")

    class Config:
        """Define Pandera class config."""

        strict = True


class ShuntParamSchema(pa.DataFrameModel):
    """Shunt parameter needed for the DC+ network model."""

    id_int: pat.Series[int] = pa.Field(coerce=True)
    id_str: pat.Series[str] = pa.Field(coerce=True)
    name: pat.Series[str] = pa.Field(coerce=True)
    connected: pat.Series[bool] = pa.Field(coerce=True)
    g: pat.Series[float] = pa.Field(coerce=True)
    b: pat.Series[float] = pa.Field(coerce=True)
    p: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    q: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    i: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    bus_index: pat.Series[int] = pa.Field(coerce=True)
    section_count: pat.Series[int] = pa.Field(coerce=True)
    max_section_count: pat.Series[int] = pa.Field(coerce=True)
    voltage_regulation: pat.Series[bool] = pa.Field(coerce=True)
    regulated_bus_id_str: pat.Series[str] = pa.Field(coerce=True)
    regulated_bus_id_int: pat.Series[int] = pa.Field(coerce=True, description="Set to -1 if not regulated")

    class Config:
        """Define Pandera class config."""

        strict = True


class BusParamSchema(pa.DataFrameModel):
    """Bus parameter needed for the DC+ network model."""

    id_int: pat.Series[int] = pa.Field(coerce=True, description="Integer ID of the bus.")
    id_str: pat.Series[str] = pa.Field(coerce=True, description="String ID of the bus, e.g. the UCTE or CGMES id.")
    name: pat.Series[str] = pa.Field(coerce=True)
    voltage_magnitude: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    regulating_generator_reached_limit: pat.Series[bool] = pa.Field(coerce=True)
    voltage_angle: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    bus_type: pat.Series[int] = pa.Field(coerce=True, description="0:Slack, 1:PV, 2:PQ")
    is_angle_reference: pat.Series[bool] = pa.Field(
        coerce=True,
        description="True for the bus used as the angle reference, independent of its voltage-control mode.",
    )
    grid_island_id: pat.Series[int] = pa.Field(
        coerce=True, description="ID of the grid island the bus belongs to. 0 indicates the main grid."
    )

    class Config:
        """Define Pandera class config."""

        strict = True


class LimitParamSchema(pa.DataFrameModel):
    """Limit parameter needed for the DC+ network model."""

    id_int: pat.Series[int] = pa.Field(coerce=True, description="Corresponds to own unique limit ID.")
    element_id_str: pat.Series[str] = pa.Field(coerce=True, description="Corresponding element ID, e.g. a branch_id_str.")
    limit_type: pat.Series[str] = pa.Field(coerce=True)
    element_type: pat.Series[str] = pa.Field(coerce=True)
    acceptable_duration: pat.Series[float] = pa.Field(coerce=True)
    side: pat.Series[str] = pa.Field(coerce=True)
    name: pat.Series[str] = pa.Field(coerce=True)
    value: pat.Series[float] = pa.Field(coerce=True)

    class Config:
        """Define Pandera class config."""

        strict = True


class TapChangerParamSchema(pa.DataFrameModel):
    """Tap changer parameter needed for the DC+ network model."""

    id_str: pat.Series[str] = pa.Field(coerce=True, description="Corresponding element ID, e.g. a branch_id_str.")
    min_tap_pos: pat.Series[int] = pa.Field(
        coerce=True, description="The minimum tap position of the tap changer, must be found in to TapPositionParamSchema."
    )
    max_tap_pos: pat.Series[int] = pa.Field(
        coerce=True, description="The maximum tap position of the tap changer, must be found in to TapPositionParamSchema."
    )
    current_tap_pos: pat.Series[int] = pa.Field(
        coerce=True, description="The current tap position of the tap changer, must be found in to TapPositionParamSchema."
    )
    step_count: pat.Series[int] = pa.Field(coerce=True, description="The number of steps of the tap changer.")
    side: pat.Series[str] = pa.Field(
        coerce=True, description="The side of the tap changer, must be same as in TapPositionParamSchema."
    )
    neutral_r: pat.Series[float] = pa.Field(coerce=True, description="The resistance at the neutral tap position.")
    neutral_x: pat.Series[float] = pa.Field(coerce=True, description="The reactance at the neutral tap position.")
    neutral_g1: pat.Series[float] = pa.Field(
        coerce=True, description="The conductance at the from side of the neutral tap position."
    )
    neutral_b1: pat.Series[float] = pa.Field(
        coerce=True, description="The susceptance at the from side of the neutral tap position."
    )
    neutral_g2: pat.Series[float] = pa.Field(
        coerce=True, description="The conductance at the to side of the neutral tap position."
    )
    neutral_b2: pat.Series[float] = pa.Field(
        coerce=True, description="The susceptance at the to side of the neutral tap position."
    )

    class Config:
        """Define Pandera class config."""

        strict = True


class TapPositionParamSchema(pa.DataFrameModel):
    """Generic Tap position parameter needed for transformer ratio and phase tap-step imports.

    Note: The tap position parameter are the offset values at the tap position, based on the neutral position.
    There are different ways to model tap positions e.g. relative to the neutral position.
    """

    id_str: pat.Series[str] = pa.Field(coerce=True, description="Corresponding element ID, e.g. a branch_id_str.")
    position: pat.Series[int] = pa.Field(coerce=True, description="The tap position of the tap changer.")
    offset_r: pat.Series[float] = pa.Field(coerce=True, description="The offset resistance at the tap position.")
    offset_x: pat.Series[float] = pa.Field(coerce=True, description="The offset reactance at the tap position.")
    offset_g1: pat.Series[float] = pa.Field(
        coerce=True, description="The offset conductance at the from side of the tap position."
    )
    offset_g2: pat.Series[float] = pa.Field(
        coerce=True, description="The offset conductance at the to side of the tap position."
    )
    offset_b1: pat.Series[float] = pa.Field(
        coerce=True, description="The offset susceptance at the from side of the tap position."
    )
    offset_b2: pat.Series[float] = pa.Field(
        coerce=True, description="The offset susceptance at the to side of the tap position."
    )
    offset_alpha: pat.Series[float] = pa.Field(coerce=True, description="The offset branch phase angle at the tap position.")
    offset_rho: pat.Series[float] = pa.Field(coerce=True, description="The offset branch tap ratio at the tap position.")

    class Config:
        """Define Pandera class config."""

        strict = True
