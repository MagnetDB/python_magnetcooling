#! /usr/bin/python3

from __future__ import unicode_literals

import os
import sys
from python_magnetrun.MagnetRun import MagnetRun

import matplotlib

# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available

import pandas as pd

from python_magnetrun.processing import smoothers as smoothtools
from python_magnetrun.processing import filters as filtertools

# Import from python_magnetcooling module
from python_magnetcooling.heatexchanger_primary import (
    calculate_extended_temperature_fields,
    calculate_heat_transfer_coefficients,
    plot_temperature_comparison,
    plot_mixed_temperatures,
    plot_heat_transfer_coefficient,
    display_Q,
    display_T,
)
from python_magnetcooling.heat_exchanger_config import HeatExchangerConfig


if __name__ == "__main__":
    command_line = None

    import argparse

    parser = argparse.ArgumentParser("Cooling loop Heat Exchanger")
    parser.add_argument("input_file", help="input txt file (ex. M10_2020.10.04_20-2009_43_31.txt)")
    parser.add_argument("--nhelices", help="specify number of helices", type=int, default=14)
    parser.add_argument(
        "--ohtc",
        help="specify heat exchange coefficient (ex. 4000 W/K/m^2 or None)",
        type=str,
        default="None",
    )
    parser.add_argument(
        "--dT",
        help="specify dT for Tout (aka accounting for alim cooling, ex. 0)",
        type=float,
        default=0,
    )
    parser.add_argument("--site", help="specify a site (ex. M8, M9,...)", type=str)
    parser.add_argument(
        "--debit_alim",
        help="specify flowrate for power cooling - one half only (default: 60 m3/h)",
        type=float,
        default=60,
    )
    parser.add_argument(
        "--show",
        help="display graphs (requires X11 server active)",
        action="store_true",
    )
    parser.add_argument("--debug", help="activate debug mode", action="store_true")
    # parser.add_argument("--save", help="save graphs to png", action='store_true')

    # raw|filter|smooth post-traitement of data
    parser.add_argument(
        "--pre",
        help="select a pre-traitment for data",
        type=str,
        choices=["raw", "filtered", "smoothed"],
        default="smoothed",
    )
    # define params for post traitment of data
    parser.add_argument(
        "--pre_params",
        help="pass param for pre-traitment method",
        type=str,
        default="400",
    )

    parser.add_argument(
        "--Q",
        help="specify Q factor for Flow (aka cooling magnets, ex. 1)",
        type=float,
        default=1,
    )
    args = parser.parse_args(command_line)

    tau = 400
    if args.pre == "smoothed":
        print("smoothed options")
        tau = float(args.pre_params)

    threshold = 0.5
    twindows = 10
    if args.pre == "filtered":
        print("filtered options")
        params = args.pre_params.split(";")
        threshold = float(params[0])
        twindows = int(params[1])

    print("args: ", args)

    # check extension
    f_extension = os.path.splitext(args.input_file)[-1]
    if f_extension != ".txt":
        print("so far only txt file support is implemented")
        sys.exit(0)

    housing = args.site
    filename = os.path.basename(args.input_file)
    result = filename.startswith("M")
    if result:
        try:
            index = filename.index("_")
            args.site = filename[0:index]
            housing = args.site
            print(f"site detected: {args.site}")
        except Exception:
            print("no site detected - use args.site argument instead")
            pass

    mrun = MagnetRun.fromtxt(housing, args.site, args.input_file)
    if not args.site:
        args.site = mrun.getSite()

    experiment = mrun.getInsert().replace(r"_", r"\_")

    # Adapt filtering and smoothing params to run duration
    duration = mrun.getMData().getDuration()
    if duration <= 10 * tau:
        tau = min(duration // 10, 10)
        print(f"Modified smoothing param: {tau} over {duration} s run")
        # args.markevery = 8 * tau

    # print("type(mrun):", type(mrun))
    start_timestamp = mrun.getMData().getStartDate()

    if "Flow" not in mrun.getKeys():
        mrun.getMData().addData("Flow", "Flow = FlowH + FlowB")
    if "Tin" not in mrun.getKeys():
        mrun.getMData().addData("Tin", "Tin = (TinH + TinB)/2.")
    if "HP" not in mrun.getKeys():
        mrun.getMData().addData("HP", "HP = (HPH + HPB)/2.")
    if "TAlimout" not in mrun.getKeys():
        # Talim not defined, try to estimate it
        print("TAlimout key not present - set TAlimout=0")
        mrun.getMData().addData("Talim", "TAlimout = 0")

    pretreatment_keys = ["flow_secondary", "Flow", "temp_secondary_in", "Tout", "Pmagnet", "Ptot"]
    if "TAlimout" in mrun.getKeys():
        pretreatment_keys.append("TAlimout")
    else:
        mrun.getMData().addData("TAlimout", "TAlimout = TinH")

    # filter spikes
    # see: https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/
    if args.pre == "filtered":
        for key in pretreatment_keys:
            mrun = filtertools.filterpikes(
                mrun,
                key,
                inplace=True,
                threshold=threshold,
                twindows=twindows,
                debug=args.debug,
                show=args.show,
                input_file=args.input_file,
            )
        print("Filtered pikes done")

    # smooth data Locally Weighted Linear Regression (Loess)
    # see: https://xavierbourretsicotte.github.io/loess.html(
    if args.pre == "smoothed":
        for key in pretreatment_keys:
            mrun = smoothtools.smooth(
                mrun,
                key,
                inplace=True,
                tau=tau,
                debug=args.debug,
                show=args.show,
                input_file=args.input_file,
            )
        print("smooth data done")
    print(mrun.getKeys())

    # extract data
    keys = [
        "t",
        "temp_secondary_in",
        "temp_secondary_out",
        "flow_secondary",
        "Tout",
        "Tin",
        "Flow",
        "BP",
        "HP",
        "Pmagnet",
    ]
    units = ["s", "C", "C", "m\u00b3/h", "C", "C", "l/s", "bar", "MW"]
    # df = mrun.getMData().extractData(keys)

    if args.debug:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)

    if "PH" not in mrun.getKeys():
        mrun.getMData().addData("PH", "PH = UH * IH")
    if "PB" not in mrun.getKeys():
        mrun.getMData().addData("PB", "PB = UB * IB")
    if "Pt" not in mrun.getKeys():
        mrun.getMData().addData("Pt", "Pt = (PH + PB)/1.e+6")
    df = mrun.getMData().getPandasData(key=None)

    # Calculate extended temperature fields using module function
    df = calculate_extended_temperature_fields(df, args.debit_alim)

    # Plot temperature comparison using module function
    plot_temperature_comparison(
        df,
        experiment,
        show=args.show,
        save_path=None if args.show else args.input_file.replace(".txt", "-Tout.png"),
    )

    # Create heat exchanger configuration with student-fitted parameters
    # Use HeatExchangerConfig(use_nominal_params=True) for nominal values instead
    hx_config = HeatExchangerConfig()

    # Calculate heat transfer coefficients using module function
    df = calculate_heat_transfer_coefficients(df, args.debit_alim, hx_config)

    # Plot mixed temperatures using module function
    plot_mixed_temperatures(df, show=args.show)

    # Plot heat transfer coefficient using module function
    plot_heat_transfer_coefficient(
        df,
        experiment,
        show=args.show,
        save_path=None if args.show else args.input_file.replace(".txt", "-ohtc.png"),
    )

    # Convert ohtc string to appropriate type for module functions
    ohtc_value = None if args.ohtc == "None" else float(args.ohtc)

    # Use module functions for display_T and display_Q
    display_T(
        args.input_file,
        f_extension,
        df,
        experiment,
        "itsb",
        "iTin",
        ohtc=ohtc_value,
        dT=args.dT,
        show=args.show,
        extension="-coolingloop.png",
        debug=args.debug,
    )
    display_Q(
        args.input_file,
        f_extension,
        df,
        experiment,
        args.debit_alim,
        ohtc=ohtc_value,
        dT=args.dT,
        show=args.show,
        extension="-Q.png",
    )
