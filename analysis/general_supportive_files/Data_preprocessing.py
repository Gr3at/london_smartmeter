# coding: utf-8
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
'''
# first we want to find unique Acorn_grouped and Acorns
df = pd.read_csv('smart_meters_london/informations_households.csv')
df.head()
df['Acorn_grouped'].unique()
df['Acorn'].unique()
# there are gonna be 6 groups of users
user_groups = {
    'Affluent_Achievers':['ACORN-A', 'ACORN-B', 'ACORN-C'],
    'Rising_Prosperity':['ACORN-D', 'ACORN-E'],
    'Comfortable_Communities':['ACORN-F', 'ACORN-G', 'ACORN-H', 'ACORN-I', 'ACORN-J'],
    'Financially_Stretched':['ACORN-K', 'ACORN-L', 'ACORN-M', 'ACORN-N'],
    'Urban_Adversity':['ACORN-O', 'ACORN-P', 'ACORN-Q'],
    'Not_Private_Households':'ACORN-U'
              }
'''
#%% load original data file
# start by dividing data into 6 groups, as stated above
df = pd.read_csv('cleanPowerDataset.csv', skiprows=1, header=None, names=["DateTime","energy","Acorn"], usecols=[1,2,3])
#df.to_csv('original_data/cleanPower.csv')
df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
df["DateTime"]=pd.to_datetime(df["DateTime"])
'''
#df["day"] = df["DateTime"].dt.day
#df["day_of_week"] = df["DateTime"].dt.dayofweek
#df["month"] = df["DateTime"].dt.month
#df["date"] = df["DateTime"].dt.date
#df["time"] = df["DateTime"].dt.time
'''
#df['energy'].describe()
#df.isnull().values.sum()
df['energy'] = df['energy'].ffill(limit=3).bfill(limit=3)
df['energy'].fillna(value=0)
#%% group data by Acorn and DateTime
aggregations={
    "energy":"sum"
#    ,"day":"first",
#    "day_of_week":"first",
#    "month":"first",
}

df_group=df.groupby(["Acorn","DateTime"]).agg(aggregations)
df_group.head()
#df_group.isnull().values.any()
#df_group.isnull().values.sum()

df_global = pd.DataFrame()
for acorn in df['Acorn'].unique():
    df_temp = df_group.loc[acorn]
    df_global[acorn]=df_temp.energy.resample('30min').sum()
#df_global.to_csv('half_hour_energy_by_Acorn.csv')
#df_global=df_global.bfill().ffill()
df_global=df_global.resample('60min').sum()

#%% Create hourly acorn energy DataFrame to store info of all Acorn Groups
hourly_acorn_energy = pd.DataFrame()
hourly_acorn_energy['Affluent_Achievers'] = df_global['ACORN-A'] + df_global['ACORN-B'] + df_global['ACORN-C'] + df_global['ACORN-']
hourly_acorn_energy['Rising_Prosperity'] = df_global['ACORN-D'] + df_global['ACORN-E']
hourly_acorn_energy['Comfortable_Communities'] = df_global['ACORN-F'] + df_global['ACORN-G'] + df_global['ACORN-H'] + df_global['ACORN-I'] + df_global['ACORN-J']
hourly_acorn_energy['Financially_Stretched'] = df_global['ACORN-K'] + df_global['ACORN-L'] + df_global['ACORN-M'] + df_global['ACORN-N']
hourly_acorn_energy['Urban_Adversity'] = df_global['ACORN-O'] + df_global['ACORN-P'] + df_global['ACORN-Q']
hourly_acorn_energy['Not_Private_Households'] = df_global['ACORN-U']
df=hourly_acorn_energy.reset_index()
df["hour"] = df["DateTime"].dt.hour
df["day_of_week"] = df["DateTime"].dt.dayofweek
df["month"] = df["DateTime"].dt.month
hourly_acorn_energy=df
hourly_acorn_energy=hourly_acorn_energy.set_index('DateTime')
#hourly_acorn_energy.to_csv('hourly_acorn_energy.csv')

#%% get also the global energy consumption for all Acorn groups combined
hourly_global_energy = pd.DataFrame()
col_list = list(hourly_acorn_energy)
col_list.remove('hour')
col_list.remove('day_of_week')
col_list.remove('month')

hourly_global_energy["energy"]=hourly_acorn_energy[col_list].sum(axis=1)
hourly_global_energy["hour"]=hourly_acorn_energy["hour"]
hourly_global_energy["day_of_week"]=hourly_acorn_energy["day_of_week"]
hourly_global_energy["month"]=hourly_acorn_energy["month"]
hourly_global_energy["energy"].describe()
#hourly_global_energy.to_csv('hourly_global_energy.csv')

'''
check if everything is fine
there are 5567 households taken into account for this study
and according to web information each household in UK 
consumes 4600 kwH on average

df=hourly_global_energy["energy"].resample('A').sum() # calculate annual consumption
df.describe()
'''
#%% plot energy related data
hge=hourly_global_energy["energy"].resample('M').sum()
ax = hge.plot(figsize=(15, 8))
fig = ax.get_figure()
plt.xlabel('Months')
plt.ylabel('Energy Consumption[kWH]')
plt.show()
fig.savefig("energy_consumption_per_month_London.png")
plt.close(fig)
#%% process weather data
df_weather = pd.read_csv('weather_hourly_darksky.csv')
df_weather = df_weather.drop(df_weather.columns[[0,1,4,5,6,8,9,11]], axis=1)
df_weather["time"]=pd.to_datetime(df_weather["time"])

df_weather=df_weather.set_index(["time"])
df_weather=df_weather[(df_weather["windSpeed"]<150) & (df_weather["windSpeed"]>=0)]
df_weather=df_weather[(df_weather["temperature"]>-20) & (df_weather["temperature"]<40)]
df_weather=df_weather[(df_weather["humidity"]>=0) & (df_weather["humidity"]<=100)]
df_weather.isnull().values.sum()

#%% join weather and energy datasets
# first combine with Acorn dataset
df_acorn=df_weather.join(hourly_acorn_energy,how='inner')[["windSpeed","humidity","temperature","Affluent_Achievers","Rising_Prosperity","Comfortable_Communities","Financially_Stretched","Urban_Adversity","Not_Private_Households","hour","day_of_week","month"]]
df_acorn.head()
df_acorn = df_acorn.truncate(before='2012-05-01')
df_acorn.plot()
plt.show()
df_acorn.to_csv("energy_weather_acorn_data.csv")

# After join weather with Global Energy dataset
df_global=df_weather.join(hourly_global_energy,how='inner')[["windSpeed","humidity","temperature","energy","hour","day_of_week","month"]]
df_global.head()
df_global = df_global.truncate(before='2012-05-01')
df_global.plot()
plt.show()
df_global.to_csv("energy_weather_global_data.csv")
