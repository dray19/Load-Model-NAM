import json
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

import Model
from Model import ModelPred


def _write_json_cols(model_path: Path, cols: list):
    sel_dir = model_path / "selected_cols"
    sel_dir.mkdir(parents=True, exist_ok=True)
    with (sel_dir / "cols_all.json").open("w", encoding="utf-8") as fh:
        json.dump({"cols": cols}, fh)


def _write_col_order(model_path: Path, year: str, cols: list):
    dirp = model_path / f"Model1_training_{year}_all"
    dirp.mkdir(parents=True, exist_ok=True)
    with (dirp / f"Model1_{year}_col_order.txt").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(cols))


def _write_model_stub(model_path: Path, year: str):
    # touch model file so ModelPred sees it exists; actual content not used in tests because we monkeypatch xgb
    model_file = model_path / f"Model1_training_{year}_all" / f"model1_{year}.ubj"
    model_file.write_text("stub")
    return model_file


def test_run_success(monkeypatch, tmp_path):
    # Prepare model files
    model_path = tmp_path / "model_dir"
    model_path.mkdir()
    year = "2025"
    # selected columns (include bias but ModelPred drops it)
    cols = ["bias", "init_date", "fcst_date", "LZ", "EST day name", "MISO_FORECAST", "feat1", "feat2"]
    _write_json_cols(model_path, cols)
    _write_col_order(model_path, year, ["feat1", "feat2"])
    _write_model_stub(model_path, year)

    # Prepare input DataFrame (two rows)
    df = pd.DataFrame({
        "init_date": ["2025-11-07", "2025-11-07"],
        "fcst_date": ["2025-11-07 00:00:00", "2025-11-07 01:00:00"],
        "LZ": ["LZ1", "LZ1"],
        "EST day name": ["Friday", "Friday"],
        "MISO_FORECAST": [100.0, 150.0],
        "feat1": [1.0, 2.0],
        "feat2": [3.0, 4.0],
    })

    # Monkeypatch ModelClean to return our df (model_pred imported ModelClean at module import; replace name in module)
    class FakeClean:
        def __init__(self, df_in):
            self.df_in = df_in
        def run(self):
            return self.df_in.copy()

    monkeypatch.setattr(Model, "ModelClean", FakeClean)

    # Monkeypatch xgboost Booster and DMatrix in model_pred module
    class FakeBooster:
        def load_model(self, path):
            # accept any path
            self._loaded = True
        def predict(self, dmat):
            # Return simple biases matching number of rows in df
            return np.array([10.0, 20.0])

    monkeypatch.setattr(Model.xgb, "Booster", FakeBooster)
    monkeypatch.setattr(Model.xgb, "DMatrix", lambda values, feature_names=None: ("DMAT", values, feature_names))

    # Run ModelPred
    mp = ModelPred(df, model_path, year)
    result = mp.run()

    # Validate output DataFrame
    assert list(result.columns) == ["DATE", "TIME", "LZ", "POWER OUTPUT (kWh)"]
    assert len(result) == 2
    # Compare predicted values: MISO_FORECAST - pred_bias = [90, 130]
    expected = [90.0, 130.0]
    # Because fillna writes strings only for NAs, numeric results remain numeric; compare numerically
    assert pytest.approx(result["POWER OUTPUT (kWh)"].astype(float).tolist()) == expected


def test_missing_json_raises(monkeypatch, tmp_path):
    # model_path without selected_cols
    model_path = tmp_path / "model_dir"
    model_path.mkdir()
    year = "2025"
    _write_col_order(model_path, year, ["feat1", "feat2"])
    _write_model_stub(model_path, year)

    df = pd.DataFrame({
        "init_date": ["2025-11-07"],
        "fcst_date": ["2025-11-07 00:00:00"],
        "LZ": ["LZ1"],
        "EST day name": ["Friday"],
        "MISO_FORECAST": [100.0],
        "feat1": [1.0],
        "feat2": [3.0],
    })

    # Keep default ModelClean from module (it will return df as-is because Clean.py provides one)
    mp = ModelPred(df, model_path, year)
    with pytest.raises(FileNotFoundError):
        mp.run()



def test_format_output_missing_columns_raises():
    # Build a DataFrame missing required columns (LZ and/or pred)
    df = pd.DataFrame({
        "fcst_date": ["2025-11-07 00:00:00"],
        # intentionally missing LZ and pred
    })
    mp = ModelPred(df, Path("."), "2025")
    with pytest.raises(KeyError):
        mp._format_output(df)