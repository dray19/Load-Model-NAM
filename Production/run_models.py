"""

Simple, efficient, serial version of the script you provided.
- No parallelism: everything runs in the main process in sequence.
- Uses argparse for clearer CLI.
- Caches config lookups to avoid repeated config reads.
- Basic logging and error handling.
- Preserves original behavior: runs each ModelPred, writes Output for each,
  then computes mean and writes mean output.

Usage:
    python run_models_serial.py 2025-10-31T00:00:00Z mean_type_key
"""

import argparse
import configparser
import logging
import sys
from typing import List, Tuple

from Model import ModelPred
from Fcst_out import Output
from Get_mean import MeanModel


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    read_files = cfg.read(path)
    if not read_files:
        raise FileNotFoundError(f"Could not read config file at '{path}'")
    return cfg


def parse_model_list(cfg: configparser.ConfigParser, mean_type: str) -> List[str]:
    try:
        raw = cfg['Mean_models'][mean_type]
    except KeyError:
        raise KeyError(f"Mean_models section or key '{mean_type}' not found in config")
    if not raw:
        return []
    return [m.strip() for m in raw.split(',') if m.strip()]


def get_model_info(cfg: configparser.ConfigParser, model_key: str) -> Tuple[str, str]:
    try:
        section = cfg[model_key]
    except KeyError:
        raise KeyError(f"Model section '{model_key}' not found in config")
    model_path = section.get('model_path', fallback='').strip()
    model_name = section.get('model_name', fallback='').strip()
    if not model_path or not model_name:
        raise ValueError(f"Missing model_path or model_name for section '{model_key}'")
    return model_path, model_name


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Run model predictions and outputs (serial)")
    parser.add_argument("dts", help="Date/time string to pass to model code")
    parser.add_argument("mean_type", help="Mean type key as present in config under [Mean_models]")
    parser.add_argument("--config", default="/mnt/trade05_data/load_v5_new_code/Production/config.ini",
                        help="Path to config.ini")
    args = parser.parse_args()

    logging.info("Loading config from %s", args.config)
    cfg = load_config(args.config)

    model_keys = parse_model_list(cfg, args.mean_type)
    if not model_keys:
        logging.warning("No models configured for mean_type '%s'", args.mean_type)
        sys.exit(0)

    # Pre-read model info to fail fast on config issues
    models_info = []
    for mk in model_keys:
        try:
            mp, mn = get_model_info(cfg, mk)
            models_info.append((mk, mp, mn))
        except Exception:
            logging.exception("Skipping model '%s' due to config error", mk)

    # Run each model serially and create outputs immediately after each run
    results = {}
    for mk, mp, mn in models_info:
        try:
            logging.info("Running ModelPred for %s (%s)", mk, mn)
            df_pred = ModelPred(args.dts, mp, mn).run()
            results[mk] = (mn, df_pred)
        except Exception:
            logging.exception("ModelPred failed for %s; skipping output", mk)
            continue

        try:
            logging.info("Creating output for %s (%s)", mk, mn)
            Output(args.dts, mk, mn, df_pred).Create()
        except Exception:
            logging.exception("Failed creating output for %s", mk)

    # Compute and write mean model output
    try:
        logging.info("Computing mean model for mean_type '%s'", args.mean_type)
        df_mean = MeanModel(args.dts, model_keys).run()
        logging.info("Creating output for mean_type %s", args.mean_type)
        Output(args.dts, args.mean_type, args.mean_type, df_mean).Create()
    except Exception:
        logging.exception("Failed computing or creating mean output")

    logging.info("Done.")


if __name__ == "__main__":
    main()