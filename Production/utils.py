import pandas as pd
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# def dummy_LZ(data):
#     """ 
#     Dummy load zone column
#     param data: dataframe
#     return dataframe
#     """
#     lz = ['L89', 'L35', 'L27', 'L06', 'L04', 'L01']
#     lz_df = pd.concat((pd.get_dummies(data['LZ'], columns=lz), pd.DataFrame(columns=lz))).fillna(0)
#     lz_df.columns = ["LZ_" + str(i) for i in list(lz_df.columns)]
#     df = pd.concat([data, lz_df], axis = 1)
#     return df

def dummy_LZ(data):
    """ 
    Dummy load zone column
    param data: dataframe
    return dataframe
    """
    lz = ['L89', 'L35', 'L27', 'L06', 'L04', 'L01']
    lz_df = pd.concat((pd.get_dummies(data['LZ'], columns=lz), pd.DataFrame(columns=lz))).fillna(0)
    lz_df.columns = ["LZ_" + str(i) for i in list(lz_df.columns)]
    df = pd.concat([data, lz_df], axis = 1)
    return df

# def dummy_dow(data):
#     """ 
#     Dummy day of week column
#     param data: dataframe
#     return dataframe
#     """
#     dw = ['Monday','Tuesday', 'Wednesday', 'Thursday','Friday','Saturday', 'Sunday']
#     dw_df = pd.concat((pd.get_dummies(data['day_of_week'], columns=dw), pd.DataFrame(columns=dw))).fillna(0)
#     dw_df.columns = ["day_of_week_" + str(i) for i in list(dw_df.columns)]
#     df = pd.concat([data, dw_df], axis = 1)
#     return df

def dummy_week(data):
    """ 
    Dummy name day of week columns
    param data: dataframe
    return dataframe
    """
    days = ['Monday','Saturday', 'Sunday', 'Thursday','Tuesday', 'Wednesday','Friday']
    days_df = pd.concat((pd.get_dummies(data['EST day name'], columns=days), pd.DataFrame(columns=days))).fillna(0)
    days_df.columns = ["EST_day_name_" + i for i in list(days_df.columns)]
    df = pd.concat([data, days_df], axis = 1)
    return df

def read_col_order(path,name):
    with open(f'{path}/{name}', "r") as f:
        data_cols = f.readlines()
    data_cols = [i.strip() for i in data_cols]
    return data_cols

    