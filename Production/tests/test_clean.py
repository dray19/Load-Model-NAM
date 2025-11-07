import io
import textwrap
from pathlib import Path
import configparser
import pandas as pd
import numpy as np
import pytest

import Clean
from Clean import ModelClean


class DummyCalc:
    """
    A simple stand-in for load_math.Calc used in tests.
    It should accept a DataFrame and return a DataFrame when load_values() is called.
    """
    def __init__(self, df):
        self._df = df.copy()

    def load_values(self):
        # Mark that calc was run; do not modify index or fcst_date/init_date behavior
        self._df["__calc_marker__"] = True
        return self._df


@pytest.fixture(autouse=True)
def patch_calc(monkeypatch):
    """
    Replace the Calc used in the module with DummyCalc for all tests to avoid external dependency.
    """
    monkeypatch.setattr(Clean, "Calc", DummyCalc)
    yield


def write_file(path: Path, content: str):
    path.write_text(content)


def test_load_data_parses_expected_structure(tmp_path):
    # Build a fake forecast file with the expected structure
    file = tmp_path / "fcst.txt"
    content = textwrap.dedent(
        """\
        IGNORE THIS LINE
        HDR LRZ1 210101 0000
        DATE TIME X Y Z
        210102 0000 a b c d e
        210103 0100 p q r s t u
        """
    )
    write_file(file, content)

    assert callable(ModelClean._read_lines)

    # call instance method load_data
    model = ModelClean.__new__(ModelClean)
    # monkeypatch the private _read_lines to use our file
    model._read_lines = lambda p: content.splitlines(keepends=True)
    # call load_data using the temp file path as string
    data, cols, lz_out, init_out = ModelClean.load_data(model, str(file))

    # Expectations:
    # - header token[1] is LRZ1, init is '2101010000'
    assert lz_out == "LRZ1"
    assert init_out == "2101010000"
    # - colnames: tokens_line2[:2] + tokens_line2[4:] -> DATE, TIME, Z
    assert cols == ["DATE", "TIME", "Z"]
    # - data rows: for each data line, tokens[:2] + tokens[3:]
    assert data[0][:2] == ["210102", "0000"]
    assert data[1][:2] == ["210103", "0100"]
    # ensure length matches lines after header
    assert len(data) == 2


def test_load_var_produces_expected_mapping(tmp_path):
    var_file = tmp_path / "vars.txt"
    # Format: first token, some name, ..., last token numeric key
    content = textwrap.dedent(
        """\
        1 VAR NAME abc 42
        2 SINGLE 99
        (comment) 3 BADLINE
        """
    )
    write_file(var_file, content)

    # instantiate and call load_var
    model = ModelClean.__new__(ModelClean)
    model.var_path = var_file
    # reuse private _read_lines so it reads from our file
    model._read_lines = lambda p: content.splitlines(keepends=True)

    mapping = ModelClean.load_var(model)
    # mapping should contain key '42' -> 'VAR NAME' and '99' -> 'SINGLE'
    assert mapping["42"] == "VAR NAME"
    assert mapping["99"] == "SINGLE"


def test_create_df_parses_dates_and_assigns_columns(tmp_path):
    # Prepare data_list and col_names consistent with expectations
    data_list = [
        # DATE, TIME, Z (colnames planned DATE, TIME, Z) plus extra token that will be included due to selection
        ["210102", "0000", "foo"],
        ["210103", "0100", "bar"],
    ]
    col_names = ["DATE", "TIME", "Z"]

    model = ModelClean.__new__(ModelClean)
    # Provide new_col mapping to rename Z -> NewZ
    new_col = {"Z": "Renamed"}
    load_zone = "L01"
    init_date = "2101010000"

    # Call create_df; Calc is patched to DummyCalc by fixture
    df = ModelClean.create_df(model, data_list, col_names, new_col, load_zone, init_date)

    # Check that init_date and fcst_date exist and are datetime
    assert "init_date" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["init_date"])
    assert pd.api.types.is_datetime64_any_dtype(df["fcst_date"])

    # Check that renaming occurred
    assert "Renamed" in df.columns

    # Check Load_Zone assigned
    assert (df["Load_Zone"] == load_zone).all()

    # Look_Ahead is created and monotonic starting at 0
    assert df["Look_Ahead"].iloc[0] == 0
    assert (df["Look_Ahead"].diff().fillna(1) >= 0).all()

    # DummyCalc added marker column
    assert "__calc_marker__" in df.columns
    assert df["__calc_marker__"].all()


def test_past_actuals_merges_d1_d2_correctly():
    # Create two rows where one row's D-2 aligns with other's D-1 after offsets to produce a merge
    t0 = pd.Timestamp("2021-01-01 00:00:00", tz="UTC")
    t1 = t0 + pd.Timedelta(days=1)

    act = pd.DataFrame({
        "DateTime": [t0, t1],
        "LZ": ["L01", "L01"],
        "OBS DATA": [100, 110],
    })

    # call past_actuals
    past = ModelClean.past_actuals(act)

    # Should contain columns fcst_date, D-1, D-2, Load_Zone
    assert set(["fcst_date", "D-1", "D-2", "Load_Zone"]).issubset(set(past.columns))
    # compute expected fcst_date: for t1's d2_date equals t1+3d; must match somewhere; verify at least one row exists
    assert len(past) >= 1
    # check that D-1 and D-2 values correspond to OBS DATA values from appropriate rows
    assert past["D-1"].notna().all()
    assert past["D-2"].notna().all()


def test_miso_error_computes_and_shifts_dates():
    # Create act and da DataFrames with matching DateTime and LZ and known MTLF so errors are deterministic
    t0 = pd.Timestamp("2021-01-01 00:00:00", tz="UTC")
    t1 = pd.Timestamp("2021-01-02 00:00:00", tz="UTC")
    t2 = pd.Timestamp("2021-01-03 00:00:00", tz="UTC")
    act = pd.DataFrame({
        "DateTime": [t0,t1,t2],
        "LZ": ["L01","L01","L01"],
        "OBS DATA": [150, 160, 170] # present in df_act for merging; df_da also has MTLF column expected by code
    })
    da = pd.DataFrame({
        "DateTime": [t0,t1,t2],
        "LZ": ["L01","L01","L01"],
        "MTLF": [140, 150, 160]
    })

    df_me = ModelClean.miso_error(act, da)

    # Should have fcst_date, Miso_error_D1, Miso_error_D2, Load_Zone
    assert set(["fcst_date", "Miso_error_D1", "Miso_error_D2", "Load_Zone"]).issubset(df_me.columns)
    # Errors should be OBS - MTLF = 10
    assert (df_me["Miso_error_D1"] == 10).all()
    assert (df_me["Miso_error_D2"] == 10).all()
    # fcst_date should be t0 + 2 days (for D1) and present
    assert (df_me["fcst_date"] == (t2 + pd.Timedelta(days=2))).any()


def test_clean_applies_thresholds_and_clips():
    # Build DataFrame containing the columns used by clean()
    df = pd.DataFrame({
        "80 m Temperature": [10, -5, 0],
        "925 mb Model Relative U Component": [-60, -40, 0],
        "HDD/CDD-3": [-52.0, -51.0, -30.0],
        "Surface PBL Height": [10, -1, 5],
        "Surface CAPE": [100, -10, 0],
        "Surface Freezing Rain": [0.1, -0.2, 0.0],
        "500 mb Vert. Comp. Abs Vorticity": [0.1, 0.3, 0.0],
    })

    cleaned = ModelClean.clean(df)

    # negative 80 m Temperature should become NaN
    assert np.isnan(cleaned.loc[1, "80 m Temperature"])
    # 925 mb less than -50 should become NaN (row 0)
    assert np.isnan(cleaned.loc[0, "925 mb Model Relative U Component"])
    # HDD/CDD-3 less than -51.2914 should be NaN (row 0)
    assert np.isnan(cleaned.loc[0, "HDD/CDD-3"])
    # Surface PBL < 0 -> NaN (row 1)
    assert np.isnan(cleaned.loc[1, "Surface PBL Height"])
    # Surface CAPE < 0 -> NaN (row 1)
    assert np.isnan(cleaned.loc[1, "Surface CAPE"])
    # Surface Freezing Rain < 0 -> NaN (row 1)
    assert np.isnan(cleaned.loc[1, "Surface Freezing Rain"])
    # Vorticity > 0.2 -> NaN (row 1, value 0.3)
    assert np.isnan(cleaned.loc[1, "500 mb Vert. Comp. Abs Vorticity"])