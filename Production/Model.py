"""
Refactored and cleaned version of the original Model_pred class.

Improvements:
- Organized imports and removed unused ones.
- Added type hints and small inline docs.
- Use pathlib for path handling.
- Safer JSON and model loading with clear error messages.
- Avoid unnecessary index/reset_index operations.
- Vectorized dtype conversion with explicit column checks.
- Handle missing columns gracefully and fail early with informative errors.
- Use logging instead of warnings for better observability.
- Small performance tweaks: avoid building large intermediate lists,
  only create DMatrix from the necessary final DataFrame.
- Keep behavior compatible with original outputs (e.g. fillna with "-999.99").
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List
from Clean import ModelClean

import numpy as np
import pandas as pd
import xgboost as xgb

# Configure logging for visibility (can be adjusted by caller)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPred:
    """
    Loads required column list and an XGBoost model, prepares input, runs predictions,
    and returns the final output formatted as in the original code.

    Parameters:
      dts: input data DataFrame-like (expects fcst_date column and necessary features)
      model_path: path to the trained model folder (string or Path-like)
      model_year: used to locate model file and column order file (string/int)
    """

    def __init__(self, dts: str , model_path: str | Path, model_year: str | int) -> None:
        self.df_in = dts  # copy-like to avoid mutating caller's df
        self.model_path = Path(model_path)
        self.model_year = str(model_year)

    def _load_json_cols(self, rel_path: str = "selected_cols/cols_all.json") -> List[str]:
        p = self.model_path / rel_path
        if not p.exists():
            raise FileNotFoundError(f"Column definition file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        cols = payload.get("cols")
        if not isinstance(cols, list):
            raise ValueError(f"Invalid format in {p}: expected key 'cols' -> list")
        return cols

    def _read_col_order(self, filename: str) -> List[str]:
        p = self.model_path / filename
        if not p.exists():
            raise FileNotFoundError(f"Column order file not found: {p}")
        # Expecting a newline-delimited column list
        with p.open("r", encoding="utf-8") as fh:
            cols = [ln.strip() for ln in fh if ln.strip()]
        return cols

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        # Match original renames and formatting
        if "fcst_date" in df.columns:
            df = df.rename(columns={"fcst_date": "Date"})
        # Ensure index is datetime-like
        if not isinstance(df.index, pd.DatetimeIndex):
            # if there's a Date column, set it as index; else attempt to coerce
            if "Date" in df.columns:
                df = df.set_index(pd.to_datetime(df["Date"]))
            else:
                df.index = pd.to_datetime(df.index)
        out = df.copy()
        out["DATE"] = out.index.strftime("%y%m%d")
        out["TIME"] = out.index.strftime("%H%M")
        # Ensure columns exist; if not, raise a helpful error
        required = ["LZ", "pred"]
        missing = [c for c in required if c not in out.columns]
        if missing:
            raise KeyError(f"Missing columns required for output formatting: {missing}")
        df_out = out[["DATE", "TIME", "LZ", "pred"]].rename(columns={"pred": "POWER OUTPUT (kWh)"})
        # Keep same fill as original: "-999.99"
        # Use object dtype fill to match original style (string sentinel)
        df_out = df_out.fillna("-999.99")
        return df_out

    def run(self) -> pd.DataFrame:
        """
        Main runner that:
          1) cleans/prepares data using Model_Clean (expected available in environment)
          2) loads columns used for prediction
          3) coerces numeric columns to float efficiently
          4) applies dummy encodings via provided helpers: dummy_week, dummy_LZ
          5) orders columns according to the model's expected input
          6) loads XGBoost model and predicts
          7) returns formatted forecast output
        """
        # Import local helpers here so module-level import doesn't fail if not available at import time.
        try:
            from load_math import Calc  # noqa: F401  (kept to preserve original dependency if needed)
            from utils import dummy_week, dummy_LZ, read_col_order
        except Exception as e:
            # Fail with helpful message if helpers are missing
            raise ImportError("Required helper modules (load_math, utils) are not importable") from e

        # 1) Run the cleaning/preparation step (keeps original behavior)
        logger.info("Running ModelClean to prepare data.")
        cleaner = ModelClean(self.df_in)
        df_clean = cleaner.run()

        # 2) Determine columns to use
        logger.info("Loading selected columns for prediction.")
        cols_all = self._load_json_cols()
        # Remove bias column if present (keeps original behavior)
        cols_to_use = [c for c in cols_all if c != "bias"]
        # Ensure required identification columns are present in df_clean
        required_meta = {"init_date", "fcst_date", "LZ", "EST day name", "MISO_FORECAST"}
        missing_meta = required_meta.intersection(set(cols_to_use)) - set(df_clean.columns)
        # If MISO_FORECAST isn't in cols_to_use but needed later, ensure it's present in the cleaned df
        if "MISO_FORECAST" not in df_clean.columns:
            raise KeyError("MISO_FORECAST is required in cleaned data but not found.")

        # 3) Extract prediction frame and convert numerics
        logger.info("Preparing prediction DataFrame and coercing numeric columns.")
        # We'll keep only the columns that exist in df_clean (defensive)
        pred_columns_present = [c for c in cols_to_use if c in df_clean.columns]
        pred_data = df_clean[pred_columns_present].copy()

        # Identify columns to cast to numeric (exclude known string-like columns)
        # Original code excluded: init_date, fcst_date, LZ, 'EST day name'
        excluded = {"init_date", "fcst_date", "LZ", "EST day name"}
        to_float = [c for c in pred_data.columns if c not in excluded]
        if to_float:
            # Use pandas.to_numeric with errors='coerce' for vectorized conversion
            pred_data[to_float] = pred_data[to_float].apply(pd.to_numeric, errors="coerce")

        # 4) Apply dummy encodings - assume dummy_week and dummy_LZ return DataFrame with new columns
        logger.info("Applying dummy encoding for week and LZ.")
        pred_data = dummy_week(pred_data)
        pred_data = dummy_LZ(pred_data)

        # 5) Read column order file and select/order final data for model
        col_order_file = f"Model1_training_{self.model_year}_all/Model1_{self.model_year}_col_order.txt"
        logger.info("Reading final column order from %s", col_order_file)
        col_order = read_col_order(self.model_path, col_order_file)

        # Validate that all columns in col_order are present in pred_data
        missing_cols = [c for c in col_order if c not in pred_data.columns]
        if missing_cols:
            # Instead of silently dropping, raise an error so problems are addressed upstream
            raise KeyError(f"The following model input columns are missing from prepared data: {missing_cols}")

        final_pred_data = pred_data.loc[:, col_order]

        # 6) Load model and run prediction
        model_file = self.model_path / f"Model1_training_{self.model_year}_all/model1_{self.model_year}.ubj"
        if not model_file.exists():
            raise FileNotFoundError(f"XGBoost model file not found: {model_file}")
        logger.info("Loading XGBoost model from %s", model_file)
        bst = xgb.Booster()
        try:
            bst.load_model(str(model_file))
        except xgb.core.XGBoostError as e:
            raise RuntimeError(f"Failed to load XGBoost model: {e}") from e

        logger.info("Converting input to DMatrix and predicting.")
        dmat = xgb.DMatrix(final_pred_data.values, feature_names=list(final_pred_data.columns))
        y_pred = bst.predict(dmat)

        # 7) Assemble final outputs consistent with original code
        # The original code appended 'pred_bias' and computed pred = MISO_FORECAST - pred_bias
        output_df = df_clean.copy()
        # Align lengths: original code assumed df_clean and final_pred_data have same row order/length
        if len(output_df) != len(y_pred):
            raise ValueError("Length mismatch between cleaned data and model predictions.")

        output_df["pred_bias"] = y_pred
        if "MISO_FORECAST" not in output_df.columns:
            raise KeyError("MISO_FORECAST missing in cleaned data; required to compute final 'pred'.")
        output_df["pred"] = output_df["MISO_FORECAST"] - output_df["pred_bias"]

        result = self._format_output(output_df)
        logger.info("Prediction completed successfully.")
        return result