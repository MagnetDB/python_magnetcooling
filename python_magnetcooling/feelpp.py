"""
Adapter to connect thermohydraulics.py with the FeelPP workflow.
"""

from typing import Dict, Optional
import pandas as pd

from .channel import CoolingLevel
from .thermohydraulics import (
    ThermalHydraulicCalculator,
    ThermalHydraulicInput,
    ThermalHydraulicOutput,
    ChannelInput,
    ChannelGeometry,
    AxialDiscretization,
)


class FeelppThermalHydraulicAdapter:
    """
    Adapter to use standalone ThermalHydraulicCalculator with FeelPP data
    structures.

    Handles conversion between:
    - FeelPP parameter dictionaries → ThermalHydraulicInput
    - ThermalHydraulicOutput → FeelPP parameter updates and dict_df
    """

    def __init__(self, calculator: ThermalHydraulicCalculator):
        self.calculator = calculator

    def compute_from_feelpp_data(
        self,
        target: str,
        dict_df: dict,
        p_params: dict,
        parameters: dict,
        targets: dict,
        args,
        basedir: str,
    ) -> tuple:
        """
        Compute thermal-hydraulics from FeelPP data structures.

        Returns:
            (ThermalHydraulicOutput, parameters_update, dict_df_update)
        """
        waterflow = targets[target]["waterflow"]
        objectif = abs(targets[target]["objectif"])

        th_input = self._build_input_from_feelpp(
            target, dict_df, p_params, parameters, targets, args, basedir, objectif
        )

        th_output = self.calculator.compute(th_input)

        parameters_update = self._extract_parameter_updates(th_output, p_params, args)
        dict_df_update = self._update_dict_df(dict_df, target, th_output, p_params)

        return th_output, parameters_update, dict_df_update

    def _build_input_from_feelpp(
        self, target, dict_df, p_params, parameters, targets, args, basedir, objectif
    ) -> ThermalHydraulicInput:
        """Build ThermalHydraulicInput from FeelPP data."""

        # Parse cooling level from CLI argument (e.g. "gradHZ").
        try:
            cooling_level = CoolingLevel(args.cooling)
        except ValueError:
            raise ValueError(
                f"Unknown cooling level '{args.cooling}'. "
                f"Valid values: {[l.value for l in CoolingLevel]}"
            )

        waterflow = targets[target]["waterflow"]
        pressure = waterflow.pressure(objectif)
        dpressure = waterflow.pressure_drop(objectif)

        Dh = [parameters[p] for p in p_params["Dh"]]
        Sh = [parameters[p] for p in p_params["Sh"]]

        # Total flow rate is needed for mean / meanH (from pump curve).
        total_flow_rate: Optional[float] = None
        if cooling_level.is_mean:
            total_flow_rate = waterflow.flow_rate(objectif)

        channels = []

        if cooling_level.is_per_channel:
            # ---- Per-channel mode (any H variant) --------------------------
            TwH = [parameters[p] for p in p_params["TwH"]]
            dTwH = [parameters[p] for p in p_params["dTwH"]]
            hwH = [parameters[p] for p in p_params["hwH"]]
            Lh = [
                abs(parameters[p] - parameters[p.replace("max", "min")])
                for p in p_params["ZmaxH"]
            ]

            for i, (d, s, L) in enumerate(zip(Dh, Sh, Lh)):
                cname = p_params["Dh"][i].replace("Dh_", "")
                power = dict_df[target]["Flux"][cname].iloc[-1]

                axial_disc = None
                if cooling_level.is_axial and isinstance(TwH[i], dict):
                    axial_disc = self._extract_axial_discretization(
                        i, cname, TwH[i], dict_df[target].get("FluxZ"), basedir
                    )

                if isinstance(TwH[i], dict):
                    _csvfile = TwH[i]["filename"].replace("$cfgdir", basedir)
                    _tw_data = pd.read_csv(_csvfile, sep=",")
                    Tw_inlet = float(_tw_data["Tw"].iloc[0])
                else:
                    Tw_inlet = TwH[i]
                T_out_guess = (
                    Tw_inlet + dTwH[i] if not isinstance(dTwH[i], dict) else None
                )
                h_guess = hwH[i] if not isinstance(hwH[i], dict) else None

                channels.append(
                    ChannelInput(
                        geometry=ChannelGeometry(
                            hydraulic_diameter=d,
                            cross_section=s,
                            length=L,
                            name=cname,
                        ),
                        power=power,
                        temp_inlet=Tw_inlet,
                        temp_outlet_guess=T_out_guess,
                        heat_coeff_guess=h_guess,
                        axial_discretization=axial_disc,
                    )
                )

        else:
            # ---- Global (single-magnet) mode --------------------------------
            Tw = [parameters[p] for p in p_params["Tw"]]
            dTw = [parameters[p] for p in p_params["dTw"]]
            hw = [parameters[p] for p in p_params["hw"]]
            L = [
                abs(parameters[p] - parameters[p.replace("max", "min")])
                for p in p_params["Zmax"]
            ]
            total_power = dict_df[target]["PowerM"].iloc[-1, 0]

            channels.append(
                ChannelInput(
                    geometry=ChannelGeometry(
                        hydraulic_diameter=Dh[0],
                        cross_section=sum(Sh),
                        length=L[0],
                        name="global",
                    ),
                    power=total_power,
                    temp_inlet=Tw[0],
                    temp_outlet_guess=Tw[0] + dTw[0],
                    heat_coeff_guess=hw[0],
                )
            )

        return ThermalHydraulicInput(
            channels=channels,
            pressure_inlet=pressure,
            pressure_drop=dpressure,
            cooling_level=cooling_level,
            total_flow_rate=total_flow_rate,
            heat_correlation=args.heatcorrelation,
            friction_model=args.friction,
            fuzzy_factor=targets[target]["fuzzy"],
            extra_pressure_loss=targets[target]["pextra"],
            relaxation_factor=targets[target]["relax"],
        )

    def _extract_axial_discretization(
        self, channel_idx, channel_name, Tw_dict, FluxZ_df, basedir
    ) -> Optional[AxialDiscretization]:
        """Extract axial discretization from FeelPP data."""

        if FluxZ_df is None:
            return None

        csvfile = Tw_dict["filename"].replace("$cfgdir", basedir)
        Tw_data = pd.read_csv(csvfile, sep=",")
        z_positions = Tw_data["Z"].to_list()

        key_dz = [fkey for fkey in FluxZ_df.columns if fkey.endswith(channel_name)]
        power_distribution = [
            FluxZ_df.at[FluxZ_df.index[-1], f"FluxZ{i}_{channel_name}"]
            for i in range(len(key_dz))
        ]

        return AxialDiscretization(
            z_positions=z_positions, power_distribution=power_distribution
        )

    def _extract_parameter_updates(
        self,
        th_output: ThermalHydraulicOutput,
        p_params: dict,
        args,
    ) -> Dict[str, float]:
        """Extract parameter updates for FeelPP from solver output."""

        updates = {}
        cooling_level = CoolingLevel(args.cooling)

        if cooling_level.is_per_channel:
            for channel_out, param_dT, param_h in zip(
                th_output.channels,
                p_params.get("dTwH", []),
                p_params.get("hwH", []),
            ):
                # Mean heat coefficient (all per-channel levels).
                if param_h:
                    updates[param_h] = channel_out.heat_coeff

                if cooling_level.is_axial:
                    # gradHZ / gradHZH: per-section temperature rises.
                    # p_params["dTwHZ"] is expected to be a list of lists:
                    # p_params["dTwHZ"][channel_idx] = [param_name_k0, param_name_k1, ...]
                    # If not provided, fall back to total rise only.
                    dTwHZ_params = p_params.get("dTwHZ", [])
                    if dTwHZ_params and channel_out.temp_rise_distribution is not None:
                        ch_idx = list(th_output.channels).index(channel_out)
                        if ch_idx < len(dTwHZ_params):
                            for param_name, dTw_k in zip(
                                dTwHZ_params[ch_idx],
                                channel_out.temp_rise_distribution,
                            ):
                                updates[param_name] = dTw_k
                    elif param_dT:
                        updates[param_dT] = channel_out.temp_rise

                    # gradHZH: per-section h coefficients.
                    hwHZ_params = p_params.get("hwHZ", [])
                    if hwHZ_params and channel_out.heat_coeff_distribution is not None:
                        ch_idx = list(th_output.channels).index(channel_out)
                        if ch_idx < len(hwHZ_params):
                            for param_name, h_k in zip(
                                hwHZ_params[ch_idx],
                                channel_out.heat_coeff_distribution,
                            ):
                                updates[param_name] = h_k
                else:
                    # mean / meanH / gradH: single dTw per channel.
                    if param_dT:
                        updates[param_dT] = channel_out.temp_rise
        else:
            if th_output.channels:
                ch = th_output.channels[0]
                if p_params.get("dTw"):
                    updates[p_params["dTw"][0]] = ch.temp_rise
                if p_params.get("hw"):
                    updates[p_params["hw"][0]] = ch.heat_coeff

        return updates

    def _update_dict_df(
        self,
        dict_df: dict,
        target: str,
        th_output: ThermalHydraulicOutput,
        p_params: dict,
    ) -> dict:
        """Update dict_df with results."""

        for i, channel_out in enumerate(th_output.channels):
            cname = channel_out.geometry.name if hasattr(channel_out, "geometry") else f"ch_{i}"

            dict_df[target]["HeatCoeff"][f"hw_{cname}"] = [round(channel_out.heat_coeff, 3)]
            dict_df[target]["DT"][f"dTw_{cname}"] = [round(channel_out.temp_rise, 3)]
            dict_df[target]["Uw"][f"Uw_{cname}"] = [round(channel_out.velocity, 3)]

            if "cf" in dict_df[target]:
                dict_df[target]["cf"][f"cf_{cname}"] = [channel_out.friction_factor]

            # gradHZ / gradHZH: store per-section Tw data for feelpp BCs.
            # T_in is already in the parameters; dTw per section is new.
            if channel_out.temp_rise_distribution is not None:
                dict_df[target].setdefault("DTZ", {})[f"dTwZ_{cname}"] = (
                    channel_out.temp_rise_distribution
                )

            # gradHZH: store per-section h distribution for feelpp BCs.
            if channel_out.heat_coeff_distribution is not None:
                dict_df[target].setdefault("HeatCoeffZ", {})[f"hwZ_{cname}"] = (
                    channel_out.heat_coeff_distribution
                )

        dict_df[target]["Tout"] = th_output.outlet_temp_mixed
        dict_df[target]["flow"] = th_output.total_flow_rate

        return dict_df
