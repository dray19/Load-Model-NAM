from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from datetime import datetime
import os
import pandas as pd
import numpy as np
import glob
import json

def create_folder_ml(mv, ty,loadzone):
    if not os.path.exists(f'Model{mv}_training_{ty}_{loadzone}'):
           os.makedirs(f'Model{mv}_training_{ty}_{loadzone}')
    return f'Model{mv}_training_{ty}_{loadzone}'

def get_cols(path):
    with open(path) as f:
        data = json.load(f)
    return data

def get_hyp(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_columns(path,name, cols_list):
    with open(f'{path}/{name}.txt', "w") as f:
        for val in cols_list:
            f.write(str(val) + "\n")

def read_col_order(path,name):
    with open(f'{path}/{name}', "r") as f:
        data_cols = f.readlines()
    data_cols = [i.strip() for i in data_cols]
    return data_cols


def Error(data,name):
    mae = mean_absolute_error(data['bias'], data[name])
    rmse = np.sqrt(mean_squared_error(data['bias'], data[name]))
    pct_mae = (mae/np.mean(data['bias'])) * 100
    pct_rmse = (rmse/np.mean(data['bias']))* 100
    mean_error = np.mean(data['bias'] - data[name])
        
    return mae, rmse, pct_mae, pct_rmse,mean_error



def feat_selecter_all(anal, fst, feat_list, feat_name):
    
    train_data = anal.drop(['Categorical_freezing_rain', 'Categorical_rain', 'Categorical_snow','day_of_year',
                            'hour', 'Year','Date','LZ','OBS_DATA'] + feat_list ,axis = 1)
    fcst_data = fst.drop(['Categorical_freezing_rain', 'Categorical_rain', 'Categorical_snow','day_of_year',
                            'hour', 'Init_date','Date', 'hour_count','Year','LZ','OBS_DATA'] + feat_list ,axis = 1)


    train_data = pd.get_dummies(train_data,prefix='day_of_week',columns=['day_of_week'], drop_first = True)
    fcst_data = pd.get_dummies(fcst_data,prefix='day_of_week',columns=['day_of_week'], drop_first = True)
    print(fcst_data.shape, train_data.shape)

    missing_cols = set(train_data.columns ) - set(fcst_data.columns )
    print(missing_cols)
    for c in missing_cols:
        print(c)
        fcst_data[c] = 0
    fcst_data = fcst_data[train_data.columns]

    x_train = train_data.drop('bias',axis = 1)
    y_train = np.array(train_data['bias'])

    x_fcst = fcst_data.drop('bias',axis = 1)
    y_fcst = np.array(fcst_data['bias'])

    best_params = {"subsample": 0.8, "reg_lambda": 0.3, "reg_alpha": 0.0, "n_estimators": 520,
 "max_depth":6, "learning_rate": 0.1, "colsample_bytree": 0.9} 
    xgb_reg = xgb.XGBRegressor(**best_params,objective =  'reg:squarederror',random_state = 44,  n_jobs= 2)
    xgb_reg.fit(x_train, y_train)


    y_pred_train = xgb_reg.predict(x_train)
    y_pred_fcst = xgb_reg.predict(x_fcst)

    res_train = pd.concat([anal, pd.DataFrame(y_pred_train, columns = ['XGB'])], axis = 1)
    res_fcst = pd.concat([fst, pd.DataFrame(y_pred_fcst, columns = ['XGB'])], axis = 1)


    mae_test2, rmse_test2, pct_mae_test2, pct_rmse_test2,mean_error_test2 = Error(res_fcst, 'XGB')
    out33 = []

    out33.append({'Col_remove': feat_name, "MAE":mae_test2, 'RMSE':rmse_test2, "MAE %":pct_mae_test2, 
                                 'RMSE %':pct_rmse_test2,'Bias (Actual - Predicted)': mean_error_test2, 'Year':"23-24"})
    return pd.DataFrame(out33)
#############################################################################
def feat_selecter_all_fr(anal, fst, feat_list, feat_name):
    
    train_data = anal.drop(['Categorical_freezing_rain', 'Categorical_rain', 'Categorical_snow','day_of_year',
                            'hour', 'Year','Date','LZ','OBS_DATA'] ,axis = 1)
    fcst_data = fst.drop(['Categorical_freezing_rain', 'Categorical_rain', 'Categorical_snow','day_of_year',
                            'hour', 'Init_date','Date', 'hour_count','Year','LZ','OBS_DATA'] ,axis = 1)

    cols = feat_list + ['Downward_short_wave_radiation_flux_pop',
       'Downward_short_wave_radiation_flux_solar', 'cooling_6_housing',
       'cooling_wet_bulb_6', 'heating_6_housing', 'heating_6_electricity','cosHour_Angle','Holiday','cosDOY_Angle','day_of_week','bias' ]
    train_data = train_data[cols]
    fcst_data = fcst_data[cols]

    train_data = pd.get_dummies(train_data,prefix='day_of_week',columns=['day_of_week'], drop_first = True)
    fcst_data = pd.get_dummies(fcst_data,prefix='day_of_week',columns=['day_of_week'], drop_first = True)
    print(fcst_data.shape, train_data.shape)

    missing_cols = set(train_data.columns ) - set(fcst_data.columns )
    print(missing_cols)
    for c in missing_cols:
        print(c)
        fcst_data[c] = 0
    fcst_data = fcst_data[train_data.columns]

    x_train = train_data.drop('bias',axis = 1)
    y_train = np.array(train_data['bias'])

    x_fcst = fcst_data.drop('bias',axis = 1)
    y_fcst = np.array(fcst_data['bias'])

    best_params = {"subsample": 0.8, "reg_lambda": 0.3, "reg_alpha": 0.0, "n_estimators": 520,
 "max_depth":6, "learning_rate": 0.1, "colsample_bytree": 0.9} 
    xgb_reg = xgb.XGBRegressor(**best_params,objective =  'reg:squarederror',random_state = 44,  n_jobs= 2)
    xgb_reg.fit(x_train, y_train)


    y_pred_train = xgb_reg.predict(x_train)
    y_pred_fcst = xgb_reg.predict(x_fcst)

    res_train = pd.concat([anal, pd.DataFrame(y_pred_train, columns = ['XGB'])], axis = 1)
    res_fcst = pd.concat([fst, pd.DataFrame(y_pred_fcst, columns = ['XGB'])], axis = 1)

    
    mae_test2, rmse_test2, pct_mae_test2, pct_rmse_test2,mean_error_test2 = Error(res_fcst, 'XGB')
    out33 = []

    out33.append({'Col_remove': feat_name, "MAE":mae_test2, 'RMSE':rmse_test2, "MAE %":pct_mae_test2, 
                                 'RMSE %':pct_rmse_test2,'Bias (Actual - Predicted)': mean_error_test2, 'Year':"23-24"})
    return pd.DataFrame(out33)

###########################################################################################
def XGB_model_check(anal, fst, feat_list, feat_name):
    
    train_data = anal.drop(['Categorical_freezing_rain', 'Categorical_rain', 'Categorical_snow','day_of_year',
                            'hour', 'Year','Date','LZ','OBS_DATA'] ,axis = 1)
    fcst_data = fst.drop(['Categorical_freezing_rain', 'Categorical_rain', 'Categorical_snow','day_of_year',
                            'hour', 'Init_date','Date', 'hour_count','Year','LZ','OBS_DATA'] ,axis = 1)

    # cols = feat_list + ['cosHour_Angle','Holiday','cosDOY_Angle','day_of_week','OBS_DATA' ]
    train_data = train_data[feat_list]
    fcst_data = fcst_data[feat_list]

    train_data = pd.get_dummies(train_data,prefix='day_of_week',columns=['day_of_week'], drop_first = True)
    fcst_data = pd.get_dummies(fcst_data,prefix='day_of_week',columns=['day_of_week'], drop_first = True)
    print(fcst_data.shape, train_data.shape)

    missing_cols = set(train_data.columns ) - set(fcst_data.columns )
    print(missing_cols)
    for c in missing_cols:
        print(c)
        fcst_data[c] = 0
    fcst_data = fcst_data[train_data.columns]

    x_train = train_data.drop('bias',axis = 1)
    y_train = np.array(train_data['bias'])

    col_order = list(x_train.columns)

    x_fcst = fcst_data.drop('bias',axis = 1)
    y_fcst = np.array(fcst_data['bias'])

    best_params = {"subsample": 0.8, "reg_lambda": 0.3, "reg_alpha": 0.0, "n_estimators": 520,
 "max_depth":6, "learning_rate": 0.1, "colsample_bytree": 0.9} 
    xgb_reg = xgb.XGBRegressor(**best_params,objective =  'reg:squarederror',random_state = 44,  n_jobs= -1)
    xgb_reg.fit(x_train, y_train)


    y_pred_train = xgb_reg.predict(x_train)
    y_pred_fcst = xgb_reg.predict(x_fcst)

    res_train = pd.concat([anal, pd.DataFrame(y_pred_train, columns = ['XGB'])], axis = 1)
    res_fcst = pd.concat([fst, pd.DataFrame(y_pred_fcst, columns = ['XGB'])], axis = 1)

    mae_test2, rmse_test2, pct_mae_test2, pct_rmse_test2,mean_error_test2 = Error(res_fcst, 'XGB')
    out33 = []

    out33.append({'Type': feat_name, "MAE":mae_test2, 'RMSE':rmse_test2, "MAE %":pct_mae_test2, 
                                 'RMSE %':pct_rmse_test2,'Bias (Actual - Predicted)': mean_error_test2, 'Year':"23-24"})
    return pd.DataFrame(out33), col_order