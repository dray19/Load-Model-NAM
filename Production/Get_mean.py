"""
Improved and more efficient implementation of the original Mean_model.
"""
from pathlib import Path
from typing import Iterable, List, Union
import configparser
import logging

import pandas as pd

_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _read_out_path_from_config(config_path: str) -> Path:
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    try:
        return Path(cfg['paths']['out_path'])
    except Exception as exc:
        raise RuntimeError(f"Could not load out_path from config '{config_path}': {exc}") from exc


class MeanModel:
    """
    Compute the mean prediction across multiple model CSV outputs.
    """

    REQUIRED_COLS = ['DATE', 'TIME', 'LZ', 'POWER OUTPUT (kWh)']

    def __init__(self, dts: str, model_list: Iterable[str], out_path: Union[str, Path] = None, config_path: str = None):
        """
        Provide either out_path or config_path (path to a config.ini with [paths].out_path).
        """
        self.dts = dts
        self.model_list: List[str] = list(model_list)

        if out_path is not None:
            self.out_path = Path(out_path)
        elif config_path is not None:
            self.out_path = _read_out_path_from_config(config_path)
        else:
            # default location (keeps backward compatibility with original)
            self.out_path = _read_out_path_from_config('/mnt/trade05_data/load_v5_new_code/Production/config.ini')

    def _read_model_series(self, model_name: str) -> pd.Series:
        csv_path = self.out_path / model_name / self.dts / f"{self.dts}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected CSV not found: {csv_path}")

        try:
            df = pd.read_csv(csv_path, usecols=self.REQUIRED_COLS, dtype={'TIME': str})
        except ValueError:
            df = pd.read_csv(csv_path)
            missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
            if missing:
                raise ValueError(f"CSV {csv_path} is missing columns: {missing}")
            df = df[self.REQUIRED_COLS]

        df['TIME'] = df['TIME'].astype(str).str.strip()
        series = df.set_index(['DATE', 'TIME', 'LZ'])['POWER OUTPUT (kWh)']
        series = series.rename(model_name)
        return series

    def compute_mean_df(self) -> pd.DataFrame:
        if not self.model_list:
            raise ValueError("model_list is empty")

        series_list = []
        for name in self.model_list:
            _LOG.debug("Reading model %s", name)
            s = self._read_model_series(name)
            series_list.append(s)

        df_concat = pd.concat(series_list, axis=1, join='inner')
        if df_concat.empty:
            raise ValueError("No overlapping rows between model outputs; result is empty after inner join")

        df_concat['pred'] = df_concat.mean(axis=1)
        return df_concat[['pred']].reset_index()

    @staticmethod
    def format_output(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out['TIME'] = out['TIME'].astype(str).str.zfill(4)
        out['pred'] = out['pred'].fillna(-999.99)
        out = out.rename(columns={'pred': 'POWER OUTPUT (kWh)'})
        out = out[['DATE', 'TIME', 'LZ', 'POWER OUTPUT (kWh)']]
        return out

    def run(self) -> pd.DataFrame:
        df_mean = self.compute_mean_df()
        result = self.format_output(df_mean)
        return result