# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandapower as pp
import pypowsybl
from pandapower.auxiliary import pandapowerNet

from dc_plus.interfaces.network_information import (
    DynamicNetworkInformation,
    StaticNetworkInformation,
    StringNetworkInformation,
)
from dc_plus.preprocess.create_network_data import (
    create_network_data,
    create_network_data_pandapower,
)


@dataclass
class NetworkDataComparison:
    """Comparison results between Powsybl and Pandapower network data.

    Attributes
    ----------
    buses_match : bool
        Whether bus data matches between implementations.
    branches_match : bool
        Whether branch data matches between implementations.
    injections_match : bool
        Whether injection data matches between implementations.
    shunts_match : bool
        Whether shunt data matches between implementations.
    admittance_match : bool
        Whether admittance matrices match between implementations.
    voltage_match : bool
        Whether voltage results match between implementations.
    power_flow_match : bool
        Whether power flow results match between implementations.
    max_voltage_diff : float
        Maximum voltage magnitude difference.
    max_angle_diff : float
        Maximum voltage angle difference in radians.
    max_power_diff : float
        Maximum power flow difference.
    max_admittance_diff : float
        Maximum admittance difference.
    details : dict
        Detailed comparison information.
    """

    buses_match: bool
    branches_match: bool
    injections_match: bool
    shunts_match: bool
    admittance_match: bool
    voltage_match: bool
    power_flow_match: bool
    max_voltage_diff: float
    max_angle_diff: float
    max_power_diff: float
    max_admittance_diff: float
    details: dict


class NetworkImportComparator:
    """Compare Powsybl and Pandapower network data imports.

    This class provides methods to compare network data extracted from Powsybl
    and Pandapower networks, validating that both implementations produce
    consistent results.
    """

    def __init__(self, tolerance: float = 1e-6):
        """Initialize the comparator.

        Parameters
        ----------
        tolerance : float, optional
            Numerical tolerance for comparisons, by default 1e-6.
        """
        self.tolerance = tolerance

    def compare_networks(
        self,
        powsybl_network: pypowsybl.network.Network,
        pandapower_network: pandapowerNet,
    ) -> NetworkDataComparison:
        """Compare network data from Powsybl and Pandapower.

        Parameters
        ----------
        powsybl_network : pypowsybl.network.Network
            The Powsybl network.
        pandapower_network : pandapowerNet
            The Pandapower network.

        Returns
        -------
        NetworkDataComparison
            Comparison results.
        """
        # Extract network data from both implementations
        static_psb, dynamic_psb, string_psb = create_network_data(powsybl_network)
        static_pp, dynamic_pp, string_pp = create_network_data_pandapower(pandapower_network)

        return self.compare_dynamic_network_info(dynamic_psb, dynamic_pp)

    def compare_dynamic_network_info(
        self,
        dynamic_powsybl: DynamicNetworkInformation,
        dynamic_pandapower: DynamicNetworkInformation,
    ) -> NetworkDataComparison:
        """Compare DynamicNetworkInformation from Powsybl and Pandapower.

        This method directly compares two DynamicNetworkInformation objects,
        which is useful when you already have the extracted data structures.

        Parameters
        ----------
        dynamic_powsybl : DynamicNetworkInformation
            Dynamic network information from Powsybl import.
        dynamic_pandapower : DynamicNetworkInformation
            Dynamic network information from Pandapower import.

        Returns
        -------
        NetworkDataComparison
            Comparison results with detailed component-by-component analysis.

        Examples
        --------
        >>> import pypowsybl
        >>> import pandapower as pp
        >>> from dcplus.preprocess.create_network_data import (
        ...     create_network_data, create_network_data_pandapower
        ... )
        >>>
        >>> # Extract data from both sources
        >>> psb_net = pypowsybl.network.create_eurostag_tutorial_example1_network()
        >>> pp_net = pp.networks.case9()
        >>>
        >>> _, dynamic_psb, _ = create_network_data(psb_net)
        >>> _, dynamic_pp, _ = create_network_data_pandapower(pp_net)
        >>>
        >>> # Compare
        >>> comparator = NetworkImportComparator(tolerance=1e-6)
        >>> comparison = comparator.compare_dynamic_network_info(dynamic_psb, dynamic_pp)
        >>>
        >>> # Check results
        >>> print(f"Buses match: {comparison.buses_match}")
        >>> print(f"Max voltage diff: {comparison.max_voltage_diff:.2e}")
        """
        # Compare components
        buses_match, bus_details = self._compare_buses(dynamic_powsybl, dynamic_pandapower)
        branches_match, branch_details = self._compare_branches(dynamic_powsybl, dynamic_pandapower)
        injections_match, injection_details = self._compare_injections(dynamic_powsybl, dynamic_pandapower)
        shunts_match, shunt_details = self._compare_shunts(dynamic_powsybl, dynamic_pandapower)
        admittance_match, admittance_details = self._compare_admittances(dynamic_powsybl, dynamic_pandapower)
        voltage_match, voltage_details = self._compare_voltages(dynamic_powsybl, dynamic_pandapower)
        power_flow_match, power_details = self._compare_power_flows(dynamic_powsybl, dynamic_pandapower)

        return NetworkDataComparison(
            buses_match=buses_match,
            branches_match=branches_match,
            injections_match=injections_match,
            shunts_match=shunts_match,
            admittance_match=admittance_match,
            voltage_match=voltage_match,
            power_flow_match=power_flow_match,
            max_voltage_diff=voltage_details.get("max_voltage_diff", 0.0),
            max_angle_diff=voltage_details.get("max_angle_diff", 0.0),
            max_power_diff=power_details.get("max_power_diff", 0.0),
            max_admittance_diff=admittance_details.get("max_admittance_diff", 0.0),
            details={
                "buses": bus_details,
                "branches": branch_details,
                "injections": injection_details,
                "shunts": shunt_details,
                "admittance": admittance_details,
                "voltage": voltage_details,
                "power_flow": power_details,
            },
        )

    def _compare_buses(
        self, dynamic_psb: DynamicNetworkInformation, dynamic_pp: DynamicNetworkInformation
    ) -> tuple[bool, dict]:
        """Compare bus data.

        Parameters
        ----------
        dynamic_psb : DynamicNetworkInformation
            Powsybl network data.
        dynamic_pp : DynamicNetworkInformation
            Pandapower network data.

        Returns
        -------
        tuple[bool, dict]
            Match status and details.
        """
        n_buses_psb = len(dynamic_psb.bus_voltage_magnitudes)
        n_buses_pp = len(dynamic_pp.bus_voltage_magnitudes)

        if n_buses_psb != n_buses_pp:
            return False, {"error": f"Different number of buses: {n_buses_psb} vs {n_buses_pp}"}

        # Compare bus types
        bus_type_match = np.allclose(dynamic_psb.bus_type, dynamic_pp.bus_type, rtol=0, atol=0)

        details = {
            "n_buses": n_buses_psb,
            "bus_type_match": bus_type_match,
            "slack_buses_psb": np.sum(dynamic_psb.bus_type == 0),
            "slack_buses_pp": np.sum(dynamic_pp.bus_type == 0),
            "pv_buses_psb": np.sum(dynamic_psb.bus_type == 1),
            "pv_buses_pp": np.sum(dynamic_pp.bus_type == 1),
            "pq_buses_psb": np.sum(dynamic_psb.bus_type == 2),
            "pq_buses_pp": np.sum(dynamic_pp.bus_type == 2),
        }

        return bus_type_match, details

    def _compare_branches(
        self, dynamic_psb: DynamicNetworkInformation, dynamic_pp: DynamicNetworkInformation
    ) -> tuple[bool, dict]:
        """Compare branch data.

        Parameters
        ----------
        dynamic_psb : DynamicNetworkInformation
            Powsybl network data.
        dynamic_pp : DynamicNetworkInformation
            Pandapower network data.

        Returns
        -------
        tuple[bool, dict]
            Match status and details.
        """
        n_branches_psb = len(dynamic_psb.branch_from_bus)
        n_branches_pp = len(dynamic_pp.branch_from_bus)

        if n_branches_psb != n_branches_pp:
            return False, {"error": f"Different number of branches: {n_branches_psb} vs {n_branches_pp}"}

        # Compare connectivity
        connectivity_match = np.allclose(
            dynamic_psb.branch_from_bus, dynamic_pp.branch_from_bus, rtol=0, atol=0
        ) and np.allclose(dynamic_psb.branch_to_bus, dynamic_pp.branch_to_bus, rtol=0, atol=0)

        # Compare branch status
        status_match = np.allclose(dynamic_psb.branch_connected, dynamic_pp.branch_connected, rtol=0, atol=0)

        # Compare symmetry
        symmetry_match = np.allclose(dynamic_psb.is_branch_symmetric, dynamic_pp.is_branch_symmetric, rtol=0, atol=0)

        details = {
            "n_branches": n_branches_psb,
            "connectivity_match": connectivity_match,
            "status_match": status_match,
            "symmetry_match": symmetry_match,
            "connected_branches_psb": np.sum(dynamic_psb.branch_connected),
            "connected_branches_pp": np.sum(dynamic_pp.branch_connected),
            "symmetric_branches_psb": np.sum(dynamic_psb.is_branch_symmetric),
            "symmetric_branches_pp": np.sum(dynamic_pp.is_branch_symmetric),
        }

        match = connectivity_match and status_match
        return match, details

    def _compare_injections(
        self, dynamic_psb: DynamicNetworkInformation, dynamic_pp: DynamicNetworkInformation
    ) -> tuple[bool, dict]:
        """Compare injection data.

        Parameters
        ----------
        dynamic_psb : DynamicNetworkInformation
            Powsybl network data.
        dynamic_pp : DynamicNetworkInformation
            Pandapower network data.

        Returns
        -------
        tuple[bool, dict]
            Match status and details.
        """
        n_injections_psb = len(dynamic_psb.injection_to_bus)
        n_injections_pp = len(dynamic_pp.injection_to_bus)

        if n_injections_psb != n_injections_pp:
            return False, {"error": f"Different number of injections: {n_injections_psb} vs {n_injections_pp}"}

        # Compare bus connections
        bus_match = np.allclose(dynamic_psb.injection_to_bus, dynamic_pp.injection_to_bus, rtol=0, atol=0)

        # Compare status
        status_match = np.allclose(dynamic_psb.injection_connected, dynamic_pp.injection_connected, rtol=0, atol=0)

        psb_bus_indices = dynamic_psb.injection_to_bus.astype(int)
        pp_bus_indices = dynamic_pp.injection_to_bus.astype(int)
        slack_mask_psb = np.isin(psb_bus_indices, dynamic_psb.slack_indices)
        slack_mask_pp = np.isin(pp_bus_indices, dynamic_pp.slack_indices)
        slack_mask_match = np.array_equal(slack_mask_psb, slack_mask_pp)

        compare_mask = ~slack_mask_psb if slack_mask_match else np.ones_like(slack_mask_psb, dtype=bool)

        psb_active = dynamic_psb.injection_active_power
        pp_active = dynamic_pp.injection_active_power
        psb_reactive = dynamic_psb.injection_reactive_power
        pp_reactive = dynamic_pp.injection_reactive_power

        if compare_mask.any():
            p_match = np.allclose(
                psb_active[compare_mask],
                pp_active[compare_mask],
                rtol=self.tolerance,
                atol=self.tolerance,
                equal_nan=True,
            )
            q_match = np.allclose(
                psb_reactive[compare_mask],
                pp_reactive[compare_mask],
                rtol=self.tolerance,
                atol=self.tolerance,
                equal_nan=True,
            )
            max_p_diff = float(np.max(np.abs(psb_active[compare_mask] - pp_active[compare_mask])))
            max_q_diff = float(np.max(np.abs(psb_reactive[compare_mask] - pp_reactive[compare_mask])))
        else:
            p_match = True
            q_match = True
            max_p_diff = 0.0
            max_q_diff = 0.0

        if slack_mask_psb.any() and slack_mask_pp.any():
            slack_p_diff = (
                float(np.max(np.abs(psb_active[slack_mask_psb] - pp_active[slack_mask_pp])))
                if slack_mask_match
                else float("nan")
            )
            slack_q_diff = (
                float(np.max(np.abs(psb_reactive[slack_mask_psb] - pp_reactive[slack_mask_pp])))
                if slack_mask_match
                else float("nan")
            )
        else:
            slack_p_diff = 0.0
            slack_q_diff = 0.0

        details = {
            "n_injections": n_injections_psb,
            "bus_match": bus_match,
            "status_match": status_match,
            "p_match": p_match,
            "q_match": q_match,
            "max_p_diff": max_p_diff,
            "max_q_diff": max_q_diff,
            "slack_mask_match": bool(slack_mask_match),
            "slack_p_diff": slack_p_diff,
            "slack_q_diff": slack_q_diff,
            "connected_injections_psb": np.sum(dynamic_psb.injection_connected),
            "connected_injections_pp": np.sum(dynamic_pp.injection_connected),
        }

        match = bus_match and status_match and p_match and q_match and slack_mask_match
        return match, details

    def _compare_shunts(
        self, dynamic_psb: DynamicNetworkInformation, dynamic_pp: DynamicNetworkInformation
    ) -> tuple[bool, dict]:
        """Compare shunt data.

        Parameters
        ----------
        dynamic_psb : DynamicNetworkInformation
            Powsybl network data.
        dynamic_pp : DynamicNetworkInformation
            Pandapower network data.

        Returns
        -------
        tuple[bool, dict]
            Match status and details.
        """
        n_shunts_psb = len(dynamic_psb.shunt_bus_indices)
        n_shunts_pp = len(dynamic_pp.shunt_bus_indices)

        if n_shunts_psb == 0 and n_shunts_pp == 0:
            return True, {"n_shunts": 0, "no_shunts": True}

        if n_shunts_psb != n_shunts_pp:
            return False, {"error": f"Different number of shunts: {n_shunts_psb} vs {n_shunts_pp}"}

        # Compare bus connections
        bus_match = np.allclose(dynamic_psb.shunt_bus_indices, dynamic_pp.shunt_bus_indices, rtol=0, atol=0)

        # Compare status
        status_match = np.allclose(dynamic_psb.shunt_connected, dynamic_pp.shunt_connected, rtol=0, atol=0)

        # Compare admittances
        admittance_match = np.allclose(
            dynamic_psb.shunt_effective_bus_admittance,
            dynamic_pp.shunt_effective_bus_admittance,
            rtol=self.tolerance,
            atol=self.tolerance,
            equal_nan=True,
        )

        details = {
            "n_shunts": n_shunts_psb,
            "bus_match": bus_match,
            "status_match": status_match,
            "admittance_match": admittance_match,
            "connected_shunts_psb": np.sum(dynamic_psb.shunt_connected),
            "connected_shunts_pp": np.sum(dynamic_pp.shunt_connected),
        }

        match = bus_match and status_match and admittance_match
        return match, details

    def _compare_admittances(
        self, dynamic_psb: DynamicNetworkInformation, dynamic_pp: DynamicNetworkInformation
    ) -> tuple[bool, dict]:
        """Compare branch admittances.

        Parameters
        ----------
        dynamic_psb : DynamicNetworkInformation
            Powsybl network data.
        dynamic_pp : DynamicNetworkInformation
            Pandapower network data.

        Returns
        -------
        tuple[bool, dict]
            Match status and details.
        """
        # Compare series admittances
        y_series_match = np.allclose(
            dynamic_psb.branch_effective_admittance_series,
            dynamic_pp.branch_effective_admittance_series,
            rtol=self.tolerance,
            atol=self.tolerance,
        )

        # Compare charging admittances
        y_charging_match = np.allclose(
            dynamic_psb.branch_effective_admittance_charging_symmetric,
            dynamic_pp.branch_effective_admittance_charging_symmetric,
            rtol=self.tolerance,
            atol=self.tolerance,
        )

        # Compare from-to admittances
        y_ft_match = np.allclose(
            dynamic_psb.branch_effective_admittance_from_to,
            dynamic_pp.branch_effective_admittance_from_to,
            rtol=self.tolerance,
            atol=self.tolerance,
        )

        # Compare from-from admittances
        y_ff_match = np.allclose(
            dynamic_psb.branch_effective_admittance_from_from,
            dynamic_pp.branch_effective_admittance_from_from,
            rtol=self.tolerance,
            atol=self.tolerance,
        )

        # Compare to-to admittances
        y_tt_match = np.allclose(
            dynamic_psb.branch_effective_admittance_to_to,
            dynamic_pp.branch_effective_admittance_to_to,
            rtol=self.tolerance,
            atol=self.tolerance,
        )

        # Compare to-from admittances
        y_tf_match = np.allclose(
            dynamic_psb.branch_effective_admittance_to_from,
            dynamic_pp.branch_effective_admittance_to_from,
            rtol=self.tolerance,
            atol=self.tolerance,
        )

        max_y_series_diff = np.max(
            np.abs(dynamic_psb.branch_effective_admittance_series - dynamic_pp.branch_effective_admittance_series)
        )
        max_y_charging_diff = np.max(
            np.abs(
                dynamic_psb.branch_effective_admittance_charging_symmetric
                - dynamic_pp.branch_effective_admittance_charging_symmetric
            )
        )

        details = {
            "y_series_match": y_series_match,
            "y_charging_match": y_charging_match,
            "y_ft_match": y_ft_match,
            "y_ff_match": y_ff_match,
            "y_tt_match": y_tt_match,
            "y_tf_match": y_tf_match,
            "max_y_series_diff": float(max_y_series_diff),
            "max_y_charging_diff": float(max_y_charging_diff),
        }

        match = y_series_match and y_charging_match and y_ft_match and y_ff_match and y_tt_match and y_tf_match
        return match, details

    def _compare_voltages(
        self, dynamic_psb: DynamicNetworkInformation, dynamic_pp: DynamicNetworkInformation
    ) -> tuple[bool, dict]:
        """Compare voltage results.

        Parameters
        ----------
        dynamic_psb : DynamicNetworkInformation
            Powsybl network data.
        dynamic_pp : DynamicNetworkInformation
            Pandapower network data.

        Returns
        -------
        tuple[bool, dict]
            Match status and details.
        """
        # Compare voltage magnitudes
        vm_match = np.allclose(
            dynamic_psb.bus_voltage_magnitudes,
            dynamic_pp.bus_voltage_magnitudes,
            rtol=self.tolerance,
            atol=self.tolerance,
            equal_nan=True,
        )

        # Compare voltage angles
        va_match = np.allclose(
            dynamic_psb.bus_voltage_angles_rad,
            dynamic_pp.bus_voltage_angles_rad,
            rtol=self.tolerance,
            atol=self.tolerance,
            equal_nan=True,
        )

        max_vm_diff = np.max(np.abs(dynamic_psb.bus_voltage_magnitudes - dynamic_pp.bus_voltage_magnitudes))
        max_va_diff = np.max(np.abs(dynamic_psb.bus_voltage_angles_rad - dynamic_pp.bus_voltage_angles_rad))

        details = {
            "vm_match": vm_match,
            "va_match": va_match,
            "max_voltage_diff": float(max_vm_diff),
            "max_angle_diff": float(max_va_diff),
            "mean_voltage_psb": float(np.mean(dynamic_psb.bus_voltage_magnitudes)),
            "mean_voltage_pp": float(np.mean(dynamic_pp.bus_voltage_magnitudes)),
        }

        match = vm_match and va_match
        return match, details

    def _compare_power_flows(
        self, dynamic_psb: DynamicNetworkInformation, dynamic_pp: DynamicNetworkInformation
    ) -> tuple[bool, dict]:
        """Compare power flow results.

        Parameters
        ----------
        dynamic_psb : DynamicNetworkInformation
            Powsybl network data.
        dynamic_pp : DynamicNetworkInformation
            Pandapower network data.

        Returns
        -------
        tuple[bool, dict]
            Match status and details.
        """
        # Compare branch active power flows
        p_from_match = np.allclose(
            dynamic_psb.branch_active_power_from,
            dynamic_pp.branch_active_power_from,
            rtol=self.tolerance,
            atol=self.tolerance,
            equal_nan=True,
        )

        p_to_match = np.allclose(
            dynamic_psb.branch_active_power_to,
            dynamic_pp.branch_active_power_to,
            rtol=self.tolerance,
            atol=self.tolerance,
            equal_nan=True,
        )

        # Compare branch reactive power flows
        q_from_match = np.allclose(
            dynamic_psb.branch_reactive_power_from,
            dynamic_pp.branch_reactive_power_from,
            rtol=self.tolerance,
            atol=self.tolerance,
            equal_nan=True,
        )

        q_to_match = np.allclose(
            dynamic_psb.branch_reactive_power_to,
            dynamic_pp.branch_reactive_power_to,
            rtol=self.tolerance,
            atol=self.tolerance,
            equal_nan=True,
        )

        max_p_diff = max(
            np.max(np.abs(dynamic_psb.branch_active_power_from - dynamic_pp.branch_active_power_from)),
            np.max(np.abs(dynamic_psb.branch_active_power_to - dynamic_pp.branch_active_power_to)),
        )

        max_q_diff = max(
            np.max(np.abs(dynamic_psb.branch_reactive_power_from - dynamic_pp.branch_reactive_power_from)),
            np.max(np.abs(dynamic_psb.branch_reactive_power_to - dynamic_pp.branch_reactive_power_to)),
        )

        details = {
            "p_from_match": p_from_match,
            "p_to_match": p_to_match,
            "q_from_match": q_from_match,
            "q_to_match": q_to_match,
            "max_power_diff": float(max_p_diff),
            "max_reactive_diff": float(max_q_diff),
        }

        match = p_from_match and p_to_match and q_from_match and q_to_match
        return match, details

    def print_comparison_report(self, comparison: NetworkDataComparison) -> None:
        """Print a formatted comparison report.

        Parameters
        ----------
        comparison : NetworkDataComparison
            The comparison results to print.
        """
        print("=" * 80)
        print("NETWORK DATA COMPARISON REPORT")
        print("=" * 80)
        print(f"\n{'Component':<20} {'Status':<10} {'Details'}")
        print("-" * 80)
        print(
            f"{'Buses':<20} {'✓' if comparison.buses_match else '✗':<10} "
            f"{comparison.details['buses'].get('n_buses', 'N/A')} buses"
        )
        print(
            f"{'Branches':<20} {'✓' if comparison.branches_match else '✗':<10} "
            f"{comparison.details['branches'].get('n_branches', 'N/A')} branches"
        )
        print(
            f"{'Injections':<20} {'✓' if comparison.injections_match else '✗':<10} "
            f"{comparison.details['injections'].get('n_injections', 'N/A')} injections"
        )
        print(
            f"{'Shunts':<20} {'✓' if comparison.shunts_match else '✗':<10} "
            f"{comparison.details['shunts'].get('n_shunts', 'N/A')} shunts"
        )
        print(f"{'Admittances':<20} {'✓' if comparison.admittance_match else '✗':<10}")
        print(
            f"{'Voltages':<20} {'✓' if comparison.voltage_match else '✗':<10} Max diff: {comparison.max_voltage_diff:.2e} pu"
        )
        print(
            f"{'Power Flows':<20} {'✓' if comparison.power_flow_match else '✗':<10} "
            f"Max diff: {comparison.max_power_diff:.2e} pu"
        )
        print("-" * 80)
        print(f"\n{'Overall Match:':<20} {'✓ ALL PASSED' if self._all_match(comparison) else '✗ FAILURES DETECTED'}")
        print("=" * 80)

    def _all_match(self, comparison: NetworkDataComparison) -> bool:
        """Check if all components match.

        Parameters
        ----------
        comparison : NetworkDataComparison
            The comparison results.

        Returns
        -------
        bool
            True if all components match.
        """
        return (
            comparison.buses_match
            and comparison.branches_match
            and comparison.injections_match
            and comparison.shunts_match
            and comparison.admittance_match
            and comparison.voltage_match
            and comparison.power_flow_match
        )
