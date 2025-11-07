import configparser
from pathlib import Path

import pandas as pd
import pytest

import Get_mean

from Get_mean import MeanModel, _read_out_path_from_config


def write_config(path: Path, out_path: Path):
    cfg = configparser.ConfigParser()
    cfg['paths'] = {'out_path': str(out_path)}
    path.write_text("")
    with open(path, "w") as f:
        cfg.write(f)


def create_model_csv(out_path: Path, model_name: str, dts: str, rows):
    """
    rows: iterable of dicts with keys 'DATE','TIME','LZ','POWER OUTPUT (kWh)'
    """
    target_dir = out_path / model_name / dts
    target_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = target_dir / f"{dts}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_read_out_path_from_config_success(tmp_path):
    cfg_path = tmp_path / "config.ini"
    out_dir = tmp_path / "out_dir"
    write_config(cfg_path, out_dir)

    result = _read_out_path_from_config(str(cfg_path))
    assert isinstance(result, Path)
    assert result == out_dir


def test_read_out_path_from_config_missing_raises(tmp_path):
    # create an invalid config (no [paths] or out_path)
    cfg_path = tmp_path / "bad_config.ini"
    cfg_path.write_text("[other]\nkey = value\n")

    with pytest.raises(RuntimeError):
        _read_out_path_from_config(str(cfg_path))


def test__read_model_series_success(tmp_path):
    out_dir = tmp_path / "out"
    model = "modelA"
    dts = "20250101"
    rows = [
        {"DATE": "2025-01-01", "TIME": " 5", "LZ": "LZ1", "POWER OUTPUT (kWh)": 10.0},
        {"DATE": "2025-01-01", "TIME": "0900", "LZ": "LZ2", "POWER OUTPUT (kWh)": 20.5},
    ]
    create_model_csv(out_dir, model, dts, rows)

    mm = MeanModel(dts=dts, model_list=[model], out_path=out_dir)
    series = mm._read_model_series(model)

    # check Series name and index
    assert series.name == model
    # index should be MultiIndex of (DATE, TIME, LZ)
    assert list(series.index.names) == ["DATE", "TIME", "LZ"]
    # TIME should be stripped (leading space removed)
    assert ("2025-01-01", "5", "LZ1") in series.index
    assert series.loc[("2025-01-01", "5", "LZ1")] == 10.0
    assert series.loc[("2025-01-01", "0900", "LZ2")] == 20.5



def test_compute_mean_df_success(tmp_path):
    out_dir = tmp_path / "out"
    dts = "20250103"
    model_a = "A"
    model_b = "B"

    rows_a = [
        {"DATE": "2025-01-03", "TIME": "0100", "LZ": "X", "POWER OUTPUT (kWh)": 10.0},
        {"DATE": "2025-01-03", "TIME": "0200", "LZ": "X", "POWER OUTPUT (kWh)": 20.0},
    ]
    rows_b = [
        {"DATE": "2025-01-03", "TIME": "0100", "LZ": "X", "POWER OUTPUT (kWh)": 30.0},
        {"DATE": "2025-01-03", "TIME": "0200", "LZ": "X", "POWER OUTPUT (kWh)": 40.0},
    ]
    create_model_csv(out_dir, model_a, dts, rows_a)
    create_model_csv(out_dir, model_b, dts, rows_b)

    mm = MeanModel(dts=dts, model_list=[model_a, model_b], out_path=out_dir)
    df_mean = mm.compute_mean_df()

    # expect two rows and 'pred' column
    assert list(df_mean.columns) == ["DATE", "TIME", "LZ", "pred"]
    # check means
    row1 = df_mean.set_index(["DATE", "TIME", "LZ"]).loc[("2025-01-03", "0100", "X")]
    assert pytest.approx(row1["pred"]) == 20.0
    row2 = df_mean.set_index(["DATE", "TIME", "LZ"]).loc[("2025-01-03", "0200", "X")]
    assert pytest.approx(row2["pred"]) == 30.0




def test_format_output_zfill_and_fill():
    # craft input dataframe similar to compute_mean_df output
    df = pd.DataFrame({
        "DATE": ["2025-01-05", "2025-01-05"],
        "TIME": ["5", "090"],
        "LZ": ["L1", "L2"],
        "pred": [1.2345, float("nan")],
    })
    out = MeanModel.format_output(df)

    # TIME should be zero-padded to 4 digits
    assert out.loc[0, "TIME"] == "0005"
    assert out.loc[1, "TIME"] == "0090"
    # NaN pred should become -999.99 and column renamed
    assert out.loc[1, "POWER OUTPUT (kWh)"] == -999.99
    assert list(out.columns) == ["DATE", "TIME", "LZ", "POWER OUTPUT (kWh)"]