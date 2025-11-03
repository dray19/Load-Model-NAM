import json
import glob
import configparser
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# suppress only pandas future warnings (optional)
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from load_math import Calc  # keep as-is
from utils import *  # keep as-is (consider importing explicit names)


class ModelClean:
    """
    Refactored, more efficient version of Model_Clean.
    """

    def __init__(self, dts: str, config_path: str = "/mnt/trade05_data/load_v5_new_code/Production/config.ini"):
        self.dts = dts
        cfg = configparser.ConfigParser()
        cfg.read(config_path)
        paths = cfg["paths"]
        self.fcst_data = Path(paths["fcst_data"])
        self.mtlf_data = Path(paths["mtlf_data"])
        self.var_path = Path(paths["var_path"])

    @staticmethod
    def _read_lines(path: Path) -> List[str]:
        with path.open("r") as f:
            return f.readlines()

    def load_data(self, file_path: str) -> Tuple[List[List[str]], List[str], str, str]:
        """
        Read custom-formatted load file and return parsed rows, column names, load zone and init date.
        """
        lines = self._read_lines(Path(file_path))
        # Defensive checks
        if len(lines) < 4:
            raise ValueError(f"Unexpected file format (too few lines): {file_path}")

        # Parse header line 1 (example: line contains tokens; we want token[1] and tokens[2]+tokens[3])
        header_tokens = lines[1].split()
        lz = header_tokens[1]
        init = header_tokens[2] + header_tokens[3]

        # Column names are on line 2; drop the two middle tokens if necessary (replicates original logic)
        tokens_line2 = lines[2].split()
        colnames = tokens_line2[:2] + tokens_line2[4:]

        # For each subsequent data line build values consistent with colnames
        data_values = []
        for ln in lines[3:]:
            toks = ln.split()
            # mirror original selection: first two tokens and then tokens from index 3 onward
            vals = toks[:2] + toks[3:]
            data_values.append(vals)

        return data_values, colnames, lz, init

    def load_var(self, path: Optional[str] = None) -> Dict[str, str]:
        """
        Load variable mapping file into dict: {num: name}
        Original logic removed parentheses and trims whitespace.
        """
        p = Path(path) if path else self.var_path
        lines = self._read_lines(p)
        out: Dict[str, str] = {}
        for line in lines:
            # normalize whitespace and remove parentheses/newlines
            s = line.replace("(", "").replace(")", "").strip()
            parts = " ".join(s.split()).split()
            if len(parts) < 3:
                continue
            num = parts[-1]
            # assume name is everything between first token and last two tokens (like original)
            name = " ".join(parts[1:-2]) if len(parts) > 3 else parts[1]
            out[num] = name
        return out

    def create_df(
        self,
        data_list: List[List[str]],
        col_names: List[str],
        new_col: Dict[str, str],
        load_zone: Optional[str],
        init_date: str,
    ) -> pd.DataFrame:
        """
        Build DataFrame from parsed rows and standardize columns.
        new_col is a mapping used to rename columns (original code passed dict).
        """
        df = pd.DataFrame(data_list, columns=col_names)

        # Rename columns using provided mapping; safe fallback to identity mapping
        if isinstance(new_col, dict) and new_col:
            df = df.rename(columns=new_col)

        if load_zone:
            df = df.assign(Load_Zone=load_zone)

        # Add init_date and compose fcst_date from DATE+TIME then parse both
        df = df.assign(init_date=init_date)
        # Construct fcst_date column and parse
        df["fcst_date"] = pd.to_datetime(df["DATE"].astype(str) + df["TIME"].astype(str), format="%y%m%d%H%M", utc=True)
        df["init_date"] = pd.to_datetime(df["init_date"].astype(str), format="%y%m%d%H%M", utc=True)

        # Drop the raw DATE and TIME
        df = df.drop(columns=["DATE", "TIME"], errors="ignore")

        # Sort and reorder to put init_date and fcst_date first
        df = df.sort_values(["init_date", "fcst_date"], ignore_index=True)
        cols = df.columns.tolist()
        # Ensure fcst_date and init_date are first two columns
        remaining = [c for c in cols if c not in ("init_date", "fcst_date")]
        df = df[["init_date", "fcst_date"] + remaining]

        # Look_Ahead: original code set to index — keep same behavior but explicit
        df = df.reset_index(drop=True)
        df["Look_Ahead"] = df.index

        # Call external Calc to add load values (keeps original behavior)
        df = Calc(df).load_values()

        return df

    def download_mtlf(self) -> pd.DataFrame:
        """
        Load MTLF CSV for a given date (self.dts without last two chars as original).
        Normalize Load_Zone values and keep only required columns.
        """
        path = self.mtlf_data / f"{self.dts[:-2]}.csv"
        mtlf = pd.read_csv(path)
        # Filter out global MISO aggregate
        mtlf = mtlf[mtlf["Load_Zone"] != "MISO"].copy()

        replacements = {
            "LRZ1": "L01",
            "LRZ2_7": "L27",
            "LRZ3_5": "L35",
            "LRZ4": "L04",
            "LRZ6": "L06",
            "LRZ8_9_10": "L89",
        }
        mtlf["Load_Zone"] = mtlf["Load_Zone"].replace(replacements)
        mtlf = mtlf.rename(columns={"MTLF": "MISO_FORECAST", "Load_Zone": "LZ", "out_date_HB": "Date"})
        mtlf = mtlf[["Date", "LZ", "MISO_FORECAST"]].copy()
        mtlf["Date"] = pd.to_datetime(mtlf["Date"])
        return mtlf

    @staticmethod
    def past_actuals(data: pd.DataFrame) -> pd.DataFrame:
        """
        Build past actuals D-1 and D-2 aligned to fcst_date (DateTime + offsets).
        """
        # Ensure a copy to avoid SettingWithCopyWarning
        tmp = data[["DateTime", "LZ", "OBS DATA"]].copy()
        tmp["D-1_date"] = tmp["DateTime"] + pd.Timedelta(days=2)
        tmp["D-2_date"] = tmp["DateTime"] + pd.Timedelta(days=3)

        d1 = tmp[["D-1_date", "OBS DATA", "LZ"]].rename(columns={"D-1_date": "fcst_date", "OBS DATA": "D-1", "LZ": "Load_Zone"})
        d2 = tmp[["D-2_date", "OBS DATA", "LZ"]].rename(columns={"D-2_date": "fcst_date", "OBS DATA": "D-2", "LZ": "Load_Zone"})

        # Merge on fcst_date and Load_Zone (inner preserves only matching pairs)
        past_day = pd.merge(d1, d2, on=["fcst_date", "Load_Zone"], how="inner")
        return past_day

    @staticmethod
    def miso_error(df_act: pd.DataFrame, df_da: pd.DataFrame) -> pd.DataFrame:
        """
        Compute MISO errors for D1 and D2 relative to MTLF and align to fcst_date.
        """
        merged = pd.merge(df_act, df_da, on=["DateTime", "LZ"], how="inner", suffixes=("_act", "_da"))

        # Compute errors and shift DateTime to fcst_date for D1 and D2
        m1 = merged[["DateTime", "LZ", "OBS DATA", "MTLF"]].copy()
        m1["Miso_error_D1"] = m1["OBS DATA"] - m1["MTLF"]
        m1["fcst_date"] = m1["DateTime"] + pd.Timedelta(days=2)
        m1 = m1[["fcst_date", "Miso_error_D1", "LZ"]].rename(columns={"LZ": "Load_Zone"})

        m2 = merged[["DateTime", "LZ", "OBS DATA", "MTLF"]].copy()
        m2["Miso_error_D2"] = m2["OBS DATA"] - m2["MTLF"]
        m2["fcst_date"] = m2["DateTime"] + pd.Timedelta(days=3)
        m2 = m2[["fcst_date", "Miso_error_D2", "LZ"]].rename(columns={"LZ": "Load_Zone"})

        df_me = pd.merge(m1, m2, on=["fcst_date", "Load_Zone"], how="inner")
        return df_me

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean columns by applying numeric conversions and clipping rules.
        Operates on a copy and returns it.
        """
        cols_to_float = [
            "80 m Temperature",
            "925 mb Model Relative U Component",
            "HDD/CDD-3",
            "Surface PBL Height",
            "Surface CAPE",
            "Surface Freezing Rain",
            "500 mb Vert. Comp. Abs Vorticity",
        ]
        # Convert to numeric (coerce errors to NaN)
        total = df.copy()
        total[cols_to_float] = total[cols_to_float].apply(pd.to_numeric, errors="coerce")

        # Apply vectorized conditions
        total.loc[total["80 m Temperature"] < 0, "80 m Temperature"] = np.nan
        total.loc[total["925 mb Model Relative U Component"] < -50, "925 mb Model Relative U Component"] = np.nan
        total.loc[total["HDD/CDD-3"] < -51.2914, "HDD/CDD-3"] = np.nan
        total.loc[total["Surface PBL Height"] < 0, "Surface PBL Height"] = np.nan
        total.loc[total["Surface CAPE"] < 0, "Surface CAPE"] = np.nan
        total.loc[total["Surface Freezing Rain"] < 0, "Surface Freezing Rain"] = np.nan
        total.loc[total["500 mb Vert. Comp. Abs Vorticity"] > 0.2, "500 mb Vert. Comp. Abs Vorticity"] = np.nan

        return total

    def run(self) -> pd.DataFrame:
        """
        Main pipeline to load forecast files, clean, merge with MTLF, DA and actuals and return final dataset.
        """
        # Find matching forecast files
        pattern = str(self.fcst_data / f"*{self.dts}*")
        load_paths = glob.glob(pattern)

        frames = []
        var_map = self.load_var(self.var_path)

        for p in load_paths:
            data_rows, colnames, lz, init = self.load_data(p)
            df = self.create_df(
                data_list=data_rows,
                col_names=colnames,
                new_col=var_map,
                load_zone=lz,
                init_date=init,
            )
            frames.append(df)

        if not frames:
            raise FileNotFoundError(f"No forecast files found for pattern: {pattern}")

        load_mod = pd.concat(frames, axis=0, ignore_index=True)

        # Keep only desired look ahead range and drop 'Date' if it exists
        load_total = load_mod[(load_mod["Look_Ahead"] >= 5) & (load_mod["Look_Ahead"] <= 46)].copy()
        if "Date" in load_total.columns:
            load_total = load_total.drop(columns=["Date"])

        cleaned = self.clean(load_total)

        mtlf = self.download_mtlf()

        # merge with MTLF on fcst_date and Load_Zone
        tr_load = pd.merge(
            cleaned,
            mtlf,
            left_on=["fcst_date", "Load_Zone"],
            right_on=["Date", "LZ"],
            how="inner",
        ).drop(columns=["Date", "LZ"], errors="ignore")

        # Read day-ahead and actuals; convert datetime
        da = pd.read_csv("Live_da.csv", parse_dates=["DateTime"])
        act = pd.read_csv("Live_act.csv", parse_dates=["DateTime"])

        pa = self.past_actuals(act)
        df_error = self.miso_error(act, da)

        df_day = pd.merge(df_error, pa, on=["fcst_date", "Load_Zone"], how="inner")

        final = pd.merge(tr_load, df_day, on=["fcst_date", "Load_Zone"], how="inner")
        final = final.rename(columns={"Load_Zone": "LZ"})

        return final