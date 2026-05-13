# Tea Demand Forecasting using Machine Learning & Deep Learning

## Project Overview
This project focuses on forecasting tea demand using Machine Learning, Statistical Forecasting, and Deep Learning models. The system predicts future tea demand across multiple tea grades using historical weekly tea market data.

The objective is to improve production planning, inventory management, and business decision-making in the tea industry by providing accurate demand forecasts.

---

## Features
- Exploratory Data Analysis (EDA) of tea demand data
- Data preprocessing and feature engineering
- Baseline forecasting models
- Statistical forecasting using SARIMA and Holt-Winters
- Machine Learning forecasting using:
  - XGBoost
  - LightGBM
- Deep Learning forecasting using:
  - LSTM (Long Short-Term Memory)
- Model performance evaluation
- Visualization of forecasts and comparisons
- Feature importance analysis
- Probabilistic forecasting with confidence intervals

---

## Project Structure

```bash
tea_forecast/
│
├── 01_eda.ipynb                  # Exploratory data analysis
├── 02_preprocessing.ipynb        # Data cleaning & preprocessing
├── 03_baseline_models.ipynb      # Naive forecasting baselines
├── 04_ml_models.ipynb            # XGBoost & LightGBM models
├── 05_lstm_model.ipynb           # LSTM deep learning forecasting
├── 06_evaluation.ipynb           # Final evaluation & comparison
│
├── data/
│   ├── tea_demand_dataset.csv
│   └── tea_demand_processed.csv
│
├── models/
│   ├── xgb_model.pkl
│   ├── lgbm_model.pkl
│   ├── lstm_BOP.keras
│   ├── lstm_BOPF.keras
│   ├── lstm_Dust.keras
│   ├── lstm_OP.keras
│   ├── lstm_Pekoe.keras
│   └── scaler files
│
├── plots/
│   ├── demand_timeseries.png
│   ├── model_comparison_bar.png
│   ├── seasonal_decomposition_all.png
│   ├── xgb_feature_importance.png
│   └── forecast visualizations
│
├── results/
│   ├── baseline_results.csv
│   ├── ml_results.csv
│   ├── lstm_results.csv
│   ├── final_comparison_table.csv
│   └── findings_summary.txt
│
└── utils.py                      # Helper functions
