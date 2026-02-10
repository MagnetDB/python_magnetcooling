"""
Extract flow params from records using a fit
"""


import tempfile
import os
import re

import numpy as np
from scipy import optimize
from math import floor

import datetime

import json
import pandas as pd
from rich.progress import track
from . import utils

# from txt2csv import load_files
from python_magnetrun.utils.files import concat_files
from python_magnetrun.utils.plots import plot_files
from python_magnetrun.magnetdata import MagnetData
from python_magnetrun.processing.stats import nplateaus


def stats(
    Ikey: str,
    Okey: str,
    Ostring: str,
    threshold: float,
    files: list,
    wd: str,
    filename: str,
    debug: bool = False,
):
    df = concat_files(files, keys=[Ikey, Okey], debug=debug)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # # drop values for Icoil1 > Imax
    result = df.query(f"{Ikey} <= {threshold}")  # , inplace=True)
    if result is not None and debug:
        print(f"df: nrows={df.shape[0]}, results: nrows={result.shape[0]}")
        print(f"result max: {result[Ikey].max()}")

    plot_files(
        filename,
        files,
        key1=Ikey,
        key2=Okey,
        fit=None,
        show=debug,
        debug=debug,
        wd=wd,
    )

    if debug:
        stats = result[Okey].describe(include="all")
        print(f"{Okey}: stats")
    return (result[Okey].mean(), result[Okey].std())


def fit(
    Ikey: str,
    Okey: str,
    Ostring: str,
    threshold: float,
    fit_function,
    files: list,
    wd: str,
    filename: str,
    debug: bool = False,
):
    """
    perform fit
    """

    df = concat_files(files, keys=[Ikey, Okey], debug=debug)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # # drop values for Icoil1 > Imax
    result = df.query(f"{Ikey} <= {threshold}")  # , inplace=True)
    if result is not None and debug:
        print(f"df: nrows={df.shape[0]}, results: nrows={result.shape[0]}")
        print(f"result max: {result[Ikey].max()}")

    x_data = result[f"{Ikey}"].to_numpy()
    y_data = result[Okey].to_numpy()
    params, params_covariance = optimize.curve_fit(fit_function, x_data, y_data)

    print(f"{Ostring} Fit:")
    print(f"\tparams: {params}")
    # print(f"\tcovariance: {params_covariance}")
    print(f"\tstderr: {np.sqrt(np.diag(params_covariance))}")

    # TODO update interface with name=f'{sname}_{mname}'
    plot_files(
        filename,
        files,
        key1=Ikey,
        key2=Okey,
        fit=(x_data, [fit_function(x, params[0], params[1]) for x in x_data]),
        show=debug,
        debug=debug,
        wd=wd,
    )

    return params


def compute(session, api_server: str, headers: dict, oid: int, samples: int=20, debug: bool = False):
    """
    compute flow_params for a given magnet
    """
    print(f"flow_params.compute: api_server={api_server}, id={oid}")
    cwd = os.getcwd()
    print(f"cwd={cwd}")

    # default value
    # set Imax to 40 kA to enable real Imax detection
    flow_params = {
        "Vp0": {"value": 1000, "unit": "rpm"},
        "Vpmax": {"value": 2840, "unit": "rpm"},
        "F0": {"value": 0, "unit": "l/s"},
        "Fmax": {"value": 61.71612272405876, "unit": "l/s"},
        "Pmax": {"value": 22, "unit": "bar"},
        "Pmin": {"value": 4, "unit": "bar"},
        "Pout": {"value": 4, "unit": "bar"},
        "Imax": {"value": 28000, "unit": "A"},
    }

    Imax = flow_params["Imax"]["value"]  # 28000

    # get magnet type: aka bitter|helix|supra ??
    odata = utils.get_object(
        session,
        api_server,
        headers=headers,
        mtype="magnet",
        id=oid,
        debug=debug,
    )
    if debug:
        print(f"magnet data: {json.dumps(odata, indent=2, default=str)}")
    mname = odata["name"]
    mpart = odata["magnet_parts"][0]
    otype = mpart["part"]["type"]
    # print(f"magnet type: {otype}")

    # TODO: change according to magnet type
    # or better store data with RpmH and RpmB
    # similarely keep only Ih, Ib instead of Icoil
    # and Ih_ref, Ib_ref instead of of Iddcct1
    # Iddct are values of measured current
    # Icoil  are actually referenced values required by the user
    # on M9: FlowH = Flow1, FlowB = Flow2
    # on M8,M10: FlowH = Flow2, FlowB = Flow1
    fit_data = {
        "M9": {"Rpm": "Rpm1", "Flow": "Flow1", "Pin": "HP1", "Pout": "BP", "rlist": []},
        "M10": {
            "Rpm": "Rpm2",
            "Flow": "Flow2",
            "Pin": "HP2",
            "Pout": "BP",
            "rlist": [],
        },
    }
    if otype == "bitter":
        fit_data = {
            "M9": {
                "Rpm": "Rpm2",
                "Flow": "Flow2",
                "Pin": "HP2",
                "Pout": "BP",
                "rlist": [],
            },
            "M10": {
                "Rpm": "Rpm1",
                "Flow": "Flow1",
                "Pin": "HP1",
                "Pout": "BP",
                "rlist": [],
            },
        }

    sites = utils.get_history(
        session, api_server, headers, oid, mtype="magnet", otype="site", debug=debug
    )
    if debug:
        print(f"sites: {json.dumps(sites, indent=2, default=str)}")
        for i, site in enumerate(sites):
            print(f"site[{i}/{len(sites)}]: {json.dumps(site, indent=2, default=str)}")

    with tempfile.TemporaryDirectory() as tempdir:
        os.chdir(tempdir)
        if debug:
            print(f"moving to {tempdir}")

        for site in sites:
            sname = site["site"]["name"]
            records = utils.get_history(
                session,
                api_server,
                headers,
                site["site_id"],
                mtype="site",
                otype="record",
                verbose=debug,
                debug=debug,
            )

            # download files
            files = []
            total = 0
            nrecords = len(records)
            ithreshold = nrecords
            if nrecords > samples:
                ithreshold = min(samples, floor(nrecords * 0.80))
            if debug:
                print(f"site[{site['site']['name']}]: nrecords={nrecords}")

            housing = None
            import random

            num_records = range(ithreshold)
            if nrecords > 20:
                random.seed()
                num_records = random.sample(range(0, nrecords), ithreshold)
            print(f"randomly selected records ({ithreshold}): {num_records}")
            for i in track(
                range(ithreshold),
                description=f"Processing records for site {site['site']['name']} (pick {ithreshold}/ {nrecords})",
            ):
                f = records[num_records[i]]
                # print(f'f={f}')
                attach = f["attachment_id"]
                filename = utils.download(
                    session, api_server, headers, attach, verbose=debug, debug=debug
                )
                housing = filename.split("_")[0]
                files.append(filename)
                """
                if i >= ithreshold:
                    break
                """

            if files:
                # get keys to be extracted
                df = pd.read_csv(files[0], sep=r"\s+", engine="python", skiprows=1)

                df_emptycolumns = df.mask(df != 0).dropna(axis=1)
                keys_emptycolumns = [
                    _key
                    for _key in df_emptycolumns.columns.values.tolist()
                    if re.match(r"Icoil\d+", _key)
                ]
                for _key in ["Icoil15", "Icoil16"]:
                    try:
                        keys_emptycolumns.remove(_key)
                    except ValueError:
                        pass
                # print(f"keys_emptycolumns={keys_emptycolumns}")

                # get first Icoil column (not necessary Icoil1)
                keys = df.columns.values.tolist()
                if debug:
                    print(f"{files[0]}: keys={keys}")

                # key first or latest header that match Icoil\d+ depending on mtype
                Ikeys = []
                for _key in keys:
                    _found = re.match(r"(Icoil\d+)", _key)
                    if _found and _key not in keys_emptycolumns:
                        Ikeys.append(_found.group())
                Ikey = Ikeys[0]
                if otype == "bitter":
                    Ikey = Ikeys[-1]
                print(f"Ikey={Ikey}")
                df = pd.DataFrame()

                dropped_files = []

                # Imax detection
                new_Imax = []
                for file in files:
                    _df = pd.read_csv(file, sep=r"\s+", engine="python", skiprows=1)
                    if Ikey not in _df.columns.values.tolist():
                        print(f"{Ikey}: no such key in {file} - ignore {file}")
                        dropped_files.append(file)
                    else:
                        # drop if duration is less than threshold ??
                        if (
                            "Date" in _df.columns.values.tolist()
                            and "Time" in _df.columns.values.tolist()
                        ):
                            tformat = "%Y.%m.%d %H:%M:%S"
                            t0 = datetime.datetime.strptime(
                                _df["Date"].iloc[0] + " " + _df["Time"].iloc[0], tformat
                            )
                            _df["t"] = _df.apply(
                                lambda row: (
                                    datetime.datetime.strptime(
                                        row.Date + " " + row.Time, tformat
                                    )
                                    - t0
                                ).total_seconds(),
                                axis=1,
                            )
                        duration = _df["t"].iloc[-1] - _df["t"][0]
                        if duration <= 15 * 60:
                            dropped_files.append(file)

                        else:
                            _Rpmmax = _df[fit_data[housing]["Rpm"]].max()
                            threshold = _Rpmmax * (1 - 0.1 / 100.0)
                            result = _df.query(
                                f'{fit_data[housing]["Rpm"]} >= {threshold}'
                            )
                            if not result.empty:
                                if (
                                    result[Ikey].std() >= 10
                                    and result[Ikey].count() >= 100
                                    and result["Field"].max() >= 0.5
                                ):
                                    if debug:
                                        _Istats = result[Ikey].describe(include="all")
                                        print(
                                            f'Rpmmax={_Rpmmax}, thresold={threshold} {Ikey}: {_Istats}, Field: {result["Field"].max()}'
                                        )
                                        """ """
                                        import matplotlib.pyplot as plt

                                        result.plot.scatter(
                                            x=Ikey,
                                            y=fit_data[housing]["Rpm"],
                                            grid=True,
                                        )
                                        lname = file.replace("_", "-")
                                        lname = lname.replace(".txt", "")
                                        lname = lname.split("/")
                                        plt.title(lname[-1])
                                        plt.show()
                                        plt.close()
                                        """ """

                                    new_Imax.append(result[Ikey].min())

                    """
                    # xField = (Ikey, "A")
                    # yField = (fit_data[housing]["Rpm"], "rpm")
                    # threshold = 2.0e-2
                    # num_points_threshold = 600
                    Data = MagnetData.fromtxt(file)
                    plateaus = nplateaus(
                        Data, xField, yField, threshold, num_points_threshold, show=True
                    )
                    if plateaus:
                        new_Imax = {min(plateaus[0]["start"], Imax)}
                        print(f"new_Imax = {new_Imax}")
                    """

                if new_Imax:
                    new_Imax_mean = sum(new_Imax) / len(new_Imax)
                    if Imax != new_Imax_mean:
                        print(f"new_Imax = {new_Imax_mean} raw={new_Imax}")
                        flow_params["Imax"]["value"] = new_Imax_mean
                        Imax = new_Imax_mean

                for file in dropped_files:
                    files.remove(file)

                def vpump_func(x, a: float, b: float):
                    return a * (x / Imax) ** 2 + b

                params = fit(
                    Ikey,
                    fit_data[housing]["Rpm"],
                    "Rpm",
                    Imax,
                    vpump_func,
                    files,
                    cwd,
                    f"{sname}-{mname}",
                    debug,
                )
                flow_params["Vp0"]["value"] = params[1]
                flow_params["Vpmax"]["value"] = params[0]
                vp0 = flow_params["Vp0"]["value"]
                vpmax = flow_params["Vpmax"]["value"]
                params = []

                # Fit for Flow
                def flow_func(x, a: float, b: float):
                    return a + b * vpump_func(x, vpmax, vp0) / (vpmax + vp0)

                params = fit(
                    Ikey,
                    fit_data[housing]["Flow"],
                    "Flow",
                    Imax,
                    flow_func,
                    files,
                    cwd,
                    f"{sname}-{mname}",
                    debug,
                )
                flow_params["F0"]["value"] = params[0]
                flow_params["Fmax"]["value"] = params[1]
                params = []

                # Fit for Pressure
                def pressure_func(x, a: float, b: float):
                    return a + b * (vpump_func(x, vpmax, vp0) / (vpmax + vp0)) ** 2

                params = fit(
                    Ikey,
                    fit_data[housing]["Pin"],
                    "Pin",
                    Imax,
                    pressure_func,
                    files,
                    cwd,
                    f"{sname}-{mname}",
                    debug,
                )
                flow_params["Pmin"]["value"] = params[0]
                flow_params["Pmax"]["value"] = params[1]
                P0 = flow_params["Pmin"]["value"]
                Pmax = flow_params["Pmax"]["value"]
                params = []

                # correlation Pout
                params = stats(
                    Ikey,
                    fit_data[housing]["Pout"],
                    "Pout",
                    Imax,
                    files,
                    cwd,
                    f"{sname}-{mname}",
                    debug,
                )
                print(f"Pout(mean, std): {params}")
                Pout = params[0]
                flow_params["Pout"]["value"] = Pout

                # save flow_params
                print(f"flow_params: {json.dumps(flow_params, indent=4)}")
                filename = f"{cwd}/{sname}_{mname}-flow_params.json"
                with open(filename, "w") as f:
                    f.write(json.dumps(flow_params, indent=4))

        os.chdir(cwd)
