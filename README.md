# **NAM Load Forecasting Models**

This repository contains a suite of machine-learning models and production utilities for forecasting electricity load using NAM (North American Mesoscale) weather model data and historical load data.  
It includes multiple training pipelines, feature configurations, and production forecasting scripts designed for MLOps workflows.

---

## **📁 Repository Structure**

```
├── fcstout/                 # Forecast output storage
├── model_1y_4m/             # Train on ~1 year, forecast 4 months
│   ├── Create_ML.py         # Training script
│   ├── ML_model.py          # Model architecture
│   ├── run.sh               # Run training job
│   ├── col_order/           # Column ordering for training
│   ├── selected_cols/       # Feature lists (cols_all.json)
│   └── utils.py
├── model_4y_1y/             # Train ~4 years, forecast ~1 year
├── model_4y_lin_bias/       # Linear bias-adjusted model variant
├── model_5y_1y/             # Train ~5 years, forecast 1 year
├── model_5y_lin_bias/       # Linear bias-adjusted model
├── model_6m_2w_dc_bias/     # 6-month model, bias-corrected
├── model_6m_dc_bias/        # 6-month bias-corrected model
│   └── (all follow same pattern as above)
│
├── MTLF/                    # Mid-Term Load Forecast data tools
│   ├── get_data.py          # Primary MTLF data ingestion
│   ├── GetMTLF.py
│   ├── send_mtlf.py
│   └── utils.py
│
├── New_ML_data/             # Intermediate datasets
├── pro_data/                # Processed datasets
│
├── Production/              # Forecast production pipeline
│   ├── config.ini           # Model & path configuration
│   ├── run_fcst.sh          # Run production forecast job
│   ├── run_models.py        # Batch model inference
│   ├── Fcst_out.py          # Post-processing
│   └── utils.py
│
├── run_loop.sh              # Rolling model training script
├── run_train.sh             # Manual training launcher
└── README.md
```