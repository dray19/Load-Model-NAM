"""
Optimized script to prepare training data and run XGB_model.

Notes:
- If 'bias' appears in cols_json, bias is computed after loading the data and retained in cols_list.
- If cols_json is absent, a fallback feature list is built (excluding bias by default).
"""

from datetime import timedelta
from dateutil.relativedelta import relativedelta
import argparse
import json
import warnings
import sys

import numpy as np
import pandas as pd

from ML_model import train_xgb_model
from utils import create_folder_ml, get_cols

warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data and run XGB_model")
    parser.add_argument("model_num", help="Model number (string)")
    parser.add_argument("test_dts", help="Test datetime in YYYYMMDDHH (int or str)")
    parser.add_argument("months_back", type=int, help="Months back to consider for training (int)")
    parser.add_argument(
        "--input-csv",
        default="../New_ML_data/NAM_data_2017_2025_v2.csv",
        help="Path to input CSV",
    )
    parser.add_argument(
        "--cols-json",
        default="selected_cols/cols_all.json",
        help="JSON file containing selected columns (expects key 'cols')",
    )
    return parser.parse_args()


def read_and_clean_csv(path, usecols=None, parse_dates=None, na_sentinels=None, dtypes=None):
    """
    Read CSV with optional usecols, parse_dates, and dtypes. Then do initial sentinel filtering.
    na_sentinels: list of sentinel values (e.g. [-999, -999.99]) to treat as NA for numeric columns
    """
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes, parse_dates=parse_dates)

    # Replace sentinel values with NaN for numeric columns
    if na_sentinels:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for s in na_sentinels:
            df[num_cols] = df[num_cols].replace(s, np.nan)

    return df


def main():
    args = parse_args()
    model_num = str(args.model_num)
    try:
        test_dts_int = int(args.test_dts)
    except ValueError:
        raise SystemExit("test_dts must be an integer in format YYYYMMDDHH")

    # convert to timezone-aware UTC datetime
    test_dts_val = pd.to_datetime(test_dts_int, format="%Y%m%d%H", utc=True)
    train_start_val = test_dts_val - relativedelta(months=args.months_back)

    out_folder = create_folder_ml(model_num, test_dts_int, "all")

    # load columns list (if file exists)
    try:
        cols_list = get_cols(args.cols_json).get("cols", None)
    except Exception:
        cols_list = None  # fall back to loading entire CSV if something fails

    # Required columns for filtering / time logic
    required_cols = [
        "OBS DATA",
        "MISO FORECAST",
        "init_date",
        "fcst_date",
        "Look_Ahead",
    ]

    # If cols_list exists, include it but do NOT force 'bias' into usecols since it's computed
    usecols = None
    if cols_list:
        # remove 'bias' from usecols if present because we will compute it after loading
        cols_for_read = [c for c in cols_list if c != "bias"]
        # ensure required columns are present for filtering and bias computation
        usecols = list(dict.fromkeys(cols_for_read + required_cols))
    else:
        usecols = None

    dtypes = {
        "OBS DATA": "float32",
        "MISO FORECAST": "float32",
        "Look_Ahead": "int16",
    }

    parse_dates = ["init_date", "fcst_date"]

    # Read once with parsing and initial NA handling
    train = read_and_clean_csv(
        args.input_csv,
        usecols=usecols,
        parse_dates=parse_dates,
        na_sentinels=[-999, -999.0, -999.99],
        dtypes=dtypes,
    )

    # compute bias immediately after loading and cleaning the data
    # (important: bias is not expected to be in the CSV; compute from MISO FORECAST - OBS DATA)
    if {"MISO FORECAST", "OBS DATA"}.issubset(train.columns):
        train = train.copy()
        train["bias"] = train["MISO FORECAST"] - train["OBS DATA"]

    # Basic initial info
    print("Rows read:", len(train))

    # Ensure datetime columns are timezone-aware (they were parsed with pandas; ensure utc)
    for dt_col in ["init_date", "fcst_date"]:
        if dt_col in train.columns:
            if train[dt_col].dt.tz is None:
                train[dt_col] = train[dt_col].dt.tz_localize("UTC")
            else:
                train[dt_col] = train[dt_col].dt.tz_convert("UTC")

    # Vectorized filtering
    mask = pd.Series(True, index=train.index)
    if "OBS DATA" in train.columns:
        mask &= train["OBS DATA"].notna() & (train["OBS DATA"] != 0)
    if "MISO FORECAST" in train.columns:
        mask &= train["MISO FORECAST"].notna() & (train["MISO FORECAST"] != 0)
    if "Look_Ahead" in train.columns:
        mask &= train["Look_Ahead"].between(23, 46)

    train = train[mask].reset_index(drop=True)
    print("After filtering for sentinels / zeros / lookahead:", len(train))

    # Additional constraint: we only keep forecast dates that have exactly 6 records
    if "fcst_date" in train.columns:
        date_counts = train["fcst_date"].value_counts()
        valid_dates = date_counts[date_counts == 6].index
        train = train[train["fcst_date"].isin(valid_dates)].reset_index(drop=True)
        print("After requiring fcst_date count == 6:", len(train))

    # Time window for training set
    train2 = train[
        (train["init_date"] >= train_start_val) & (train["init_date"] < test_dts_val)
    ].reset_index(drop=True)
    print("Train window rows:", len(train2))

    # Drop MISO FORECAST as original script did (bias already present)
    if "MISO FORECAST" in train2.columns:
        train2 = train2.drop(columns=["MISO FORECAST"])

    # Save training data
    training_csv_path = f"{out_folder}/Training_data_{test_dts_int}.csv"
    train2.to_csv(training_csv_path, index=False)
    print("Saved training data to", training_csv_path)

    # Write model metadata
    train_start = min(train2["init_date"]).strftime("%Y%m%d%H") if not train2.empty else ""
    train_end = max(train2["init_date"]).strftime("%Y%m%d%H") if not train2.empty else ""
    meta = {
        "Model Num": model_num,
        "Test Year": str(test_dts_int),
        "Train Start": train_start,
        "Train End": train_end,
    }
    info_path = f"{out_folder}/Model{model_num}_info_{test_dts_int}.json"
    with open(info_path, "w") as f:
        json.dump(meta, f)
    print("Wrote model info to", info_path)

    # If cols_list is missing, build a fallback list of columns to train on.
    # Note: by default exclude bias from fallback features; if cols_json included 'bias', it will be in cols_list already.
    if not cols_list:
        exclude = {"bias", "OBS DATA", "init_date", "fcst_date", "MISO FORECAST"}
        cols_list = [c for c in train2.columns if c not in exclude]
    else:
        # Ensure requested cols_list contains only columns present in train2 (bias was computed)
        cols_list = [c for c in cols_list if c in train2.columns]

    # Call the XGB_model (assumed to return forecast and error metrics)
    fcst_data, error_met = train_xgb_model(train2, cols_list, "XGB", out_folder, model_num, test_dts_int)

    print("XGB_model finished. Output folder:", out_folder)


if __name__ == "__main__":
    main()
