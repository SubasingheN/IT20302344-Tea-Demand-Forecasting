import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def evaluate_model(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smap = smape(y_true, y_pred)
    
    print(f"\n{'='*40}")
    print(f"  Model: {name}")
    print(f"{'='*40}")
    print(f"  MAE   : {mae:.2f} kg")
    print(f"  RMSE  : {rmse:.2f} kg")
    print(f"  MAPE  : {mape:.2f}%")
    print(f"  sMAPE : {smap:.2f}%")
    
    return {'model': name, 'MAE': round(mae,2), 'RMSE': round(rmse,2),
            'MAPE': round(mape,2), 'sMAPE': round(smap,2)}

def temporal_split(df_grade, test_year=2024):
    train = df_grade[df_grade['year'] < test_year].copy()
    test  = df_grade[df_grade['year'] >= test_year].copy()
    return train, test