#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:18:08 2018

@author: Gr3at
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('hourly_datasets_to_use/energy_weather_global_data.csv',parse_dates=[0],index_col=0)

# set frequency to one Hour
df=df.resample('30T').ffill().reindex(pd.date_range(df.index[0],df.index[-1],freq='H'))
df.isnull().values.sum()
df.index.freq

# decomposition of data (seasonality, trend, residual)
from pylab import rcParams
rcParams['figure.figsize'] = 11, 9

decomposition = sm.tsa.seasonal_decompose(df['energy'], model='additive')
fig = decomposition.plot()
plt.show()

# demand distribution of data
plt.hist(df['energy'].dropna(), bins=100)
plt.title('Demand distribution')
plt.show()

# temperature to energy demand scater plot
plt.plot(df['temperature'], df['energy'], 'ro', markersize=2)
plt.title('Demand vs temperature')
plt.xlabel('temperature')
plt.ylabel('demand')
plt.show()

#plot all features
df.plot(subplots=True)

#%% copy df to new DataFrame
import os
from scipy import stats

demand_features = df.copy()

def generate_lagged_features(dfa, var, max_lag):
    for t in range(1, max_lag+1):
        dfa[var+'_lag'+str(t)] = dfa[var].shift(t, freq='1H')

generate_lagged_features(demand_features, 'temperature', 6)
generate_lagged_features(demand_features, 'energy', 6)

demand_features.dropna(how='any', inplace=True)

demand_features.reset_index(inplace=True)
dt_idx = demand_features['index']
demand_features = demand_features.reindex(dt_idx)
train, test = (demand_features.loc[demand_features['timeStamp']<'2013-07-01'], demand_features.loc[demand_features['timeStamp']>='2013-07-01'])
