# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Import data from powsybl network."""

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
    voltage_angle: pat.Series[float] = pa.Field(coerce=True, nullable=True)
    bus_type: pat.Series[int] = pa.Field(coerce=True, description="0:Slack, 1:PV, 2:PQ")
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
