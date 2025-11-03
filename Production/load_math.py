import pandas as pd
import numpy as np

import glob

from datetime import datetime

import math
import json

import holidays
import pytz

class Calc:
    def __init__(self, data):
        self.data = data
        
    def HDD_CDD(self, data):
        """ 
        Create HDD CDD values
        param data: dataframe
        return: dataframe
        """
        df = data.copy()
        df['2 m Temperature'] = df['2 m Temperature'].astype(float)
        df['HDD/CDD'] = df['2 m Temperature'] - 291.48
        df['2 m Temperature_1'] = df['2 m Temperature'].shift(1)
        df['2 m Temperature_3'] = df['2 m Temperature'].shift(3)
        df['HDD/CDD-1'] = df['2 m Temperature_1'] - 291.48
        df['HDD/CDD-3'] = df['2 m Temperature_3'] - 291.48
        df = df.drop(['2 m Temperature','2 m Temperature_1', '2 m Temperature_3'], axis = 1)
        return df
    
    def doy_angle(self,val):
        """
        Calculate Day of Year Angle
        param val: int (day number)
        return: float
        """
        pi = 3.14159
        res = math.cos((pi * val)/183)
        return res

    def cos_zen(self, doy, input_hour):
        """
        Calculate cos(Zenith)
        param doy: int
        param input_hour: int
        return: float
        """
        lon = -89.5890
        lat = 40.6936
        pi = 3.14159
        DEGRAD = pi/180
        RADDEG = 180/pi
        w = 360.0/365.24

        a = w * (doy+ 9)

        B = a + (360/pi) * 0.0167 * math.sin(DEGRAD * w * (doy-3))

        c = (a - RADDEG * math.atan(math.tan(DEGRAD * B)/math.cos(DEGRAD * 23.44)))/ 180

        edt = 720 * (c - round(c, 0))

                ##################
        DEC = - math.asin(math.sin(DEGRAD * 23.44) * math.cos(DEGRAD * B))
                ###################
        LSTM = 15 * round(int(lon/15),0)
                ###################
        TC = 4 * (lon - LSTM) + edt

        utc = input_hour
        SOLTIM = utc + LSTM/15 + TC/60
        if SOLTIM < 0:
            SOLTIM = SOLTIM + 24
        elif SOLTIM > 24:
            SOLTIM = SOLTIM - 24
        else:
            pass

        solhour = DEGRAD * 15 * (SOLTIM - 12)

        coszen = math.sin(DEGRAD * lat) * math.sin(DEC) + math.cos(DEGRAD * lat) * math.cos(DEC) * math.cos(solhour)
        return coszen
    
    def hour_angle(self, input_hour):
        """
        Calculate hour angle
        param input_hour: int
        return: float
        """
        ha = math.cos((3.14159 * ((input_hour) + (0/60)))/12)
        return ha
    
    def other_vars(self, data):
        """
        Calculate other values needed
        param data: dataframe
        return dataframe
        """
        data['80 m Model Relative U Component'] = data['80 m Model Relative U Component'].astype(float)
        data['80 m Model Relative V Component'] = data['80 m Model Relative V Component'].astype(float)
        data['850 mb Model Relative U Component'] = data['850 mb Model Relative U Component'].astype(float)
        data['975 mb Model Relative U Component'] = data['975 mb Model Relative U Component'].astype(float)


        data['850 mb Model Relative V Component'] = data['850 mb Model Relative V Component'].astype(float)
        data['975 mb Model Relative V Component'] = data['975 mb Model Relative V Component'].astype(float)
        
        
        data["80 m Model Speed Cubed"] = np.sqrt((data['80 m Model Relative U Component']**2) + (data['80 m Model Relative V Component']**2))**3
        data["975-850 mb Model U Wind Shear"] = data['850 mb Model Relative U Component'] - data['975 mb Model Relative U Component'] 
        data["975-850 mb Model V Wind Shear"] = data['850 mb Model Relative V Component'] - data['975 mb Model Relative V Component']
        return data
    
    def name_day(self, data):
        """
        Create name day column for load dataset
        param data: dataframe
        return: dataframe
        """
        data['Eastern Time'] = data['fcst_date'].dt.tz_convert('US/Eastern')
        data['EST day name'] = data['Eastern Time'].dt.day_name()
        return data
  
    def holiday_vars(self, data):
        """ 
        Create Hoilday column
        param data: dataframe
        return: dataframe
        """
        data['Eastern Time'] = data['fcst_date'].dt.tz_convert('US/Eastern')
        data['Date'] = data['Eastern Time'].dt.date
        yr_list = list(data['fcst_date'].dt.year.unique())
        total_hol = []
        for vv in yr_list:
            hol = list(holidays.UnitedStates(years=int(vv)).keys())
            total_hol = total_hol + hol
        data['Holiday'] = 0
        data.loc[data['Date'].isin(total_hol), 'Holiday'] = 1
        return data
    
    def load_values(self):
        """ 
        Create values for load dataset
        return: dataframe
        """
        load_mod = self.data.copy()
        load_mod['day_of_year'] = load_mod['fcst_date'].dt.strftime('%j').astype(int)
        load_mod['hour'] = load_mod['fcst_date'].dt.hour
        load_mod['cos(DOY_Angle)'] = load_mod['day_of_year'].apply(self.doy_angle)
        load_mod['cos(Zenith)'] = load_mod.apply(lambda x: self.cos_zen(x['cos(DOY_Angle)'], x['hour']), axis=1)
        load_mod['cos(Hour_Angle)'] = load_mod['hour'].apply(self.hour_angle)
        load_mod = self.HDD_CDD(load_mod)
        load_mod = self.other_vars(load_mod)
        load_mod = self.name_day(load_mod)
        load_mod = self.holiday_vars(load_mod)
        return load_mod
    
    def wind_values(self):
        """ 
        Create values for wind dataset
        return: dataframe
        """
        wind_mod = self.data.copy()
        wind_mod['day_of_year'] = wind_mod['fcst_date'].dt.strftime('%j').astype(int)
        wind_mod['hour'] = wind_mod['fcst_date'].dt.hour
        wind_mod['cos(DOY_Angle)'] = wind_mod['day_of_year'].apply(self.doy_angle)
        wind_mod['cos(Zenith)'] = wind_mod.apply(lambda x: self.cos_zen(x['cos(DOY_Angle)'], x['hour']), axis=1)
        wind_mod['cos(Hour_Angle)'] = wind_mod['hour'].apply(self.hour_angle)
        wind_mod = self.other_vars(wind_mod)
        return wind_mod