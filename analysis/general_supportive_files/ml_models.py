#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:38:29 2018

@author: Gr3at
"""
import os
import sys
import datetime
import time
import pandas as pd
import numpy as np
from math import radians, sin, cos, acos
import json
import scipy
from statsmodels.tsa.arima_model import ARIMA
import sqlite3
#%matplotlib inline
from matplotlib import patheffects
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

df = pd.read_csv('hourly_datasets_to_use/energy_weather_global_data.csv',parse_dates=[0],index_col=0)

#df=df.iloc[:,[3,4,5,6,7,8]] # select data
df=df.resample('H').sum() # upsample
# set frequency to one Hour DOWNSAMPLE
#df=df.resample('60T').ffill().reindex(pd.date_range(df.index[0],df.index[-1],freq='H'))
df.isnull().values.sum()
df.index.freq
'''report - exporatory analysis'''
df.describe()

df_ptg=df
#%% testing

from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import explained_variance_score, r2_score,mean_absolute_error
from sklearn.model_selection import KFold

df_totest=df_ptg.copy()

# normalize data
df_split=df_totest.copy()
for param in ['temperature','humidity','windSpeed','month','day_of_week','hour']:
    df_split[param]=df_split.apply(lambda row: (row[param]-df_split[param].mean())/(df_split[param].max()-df_split[param].min()),axis=1)
print(df_split.head())

#%% Define Number of Folds
kf = KFold(n_splits=10)

#%% Polynomial Regression Model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.base import clone


def get_bestpolynomialmodel(X_train, X_test, y_train, y_test,kf,degree):
    r2score_ref=-1*np.inf
    executiontime_ref=0
    best_model={}
    for train_index, validation_index in kf.split(X_train):
        tic=time.time()
        X_training, X_validation = X_train[train_index], X_train[validation_index]
        y_training, y_validation = y_train[train_index], y_train[validation_index]
        
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X_training, y_training)
        
        toc=time.time()-tic
        
        y_pred=model.predict(X_validation)
        
        print(r2_score(y_validation, y_pred),mean_absolute_error(y_validation, y_pred))

        score=r2_score(y_validation, y_pred)
        if score>r2score_ref:
            r2score_ref=score
            executiontime_ref=toc
            best_model=model
            print("Best model:",r2score_ref,toc)



    y_pred=best_model.predict(X_test)
    r2score_test=r2_score(y_test, y_pred)
    meanabserror_test= mean_absolute_error(y_test, y_pred)
    print("Final results:",r2score_test,meanabserror_test,executiontime_ref)  
    return best_model,r2score_test,meanabserror_test,executiontime_ref

#%% test Random Forest
from sklearn.ensemble import RandomForestRegressor

def get_bestrandomforestmodel(X_train, X_test, y_train, y_test,kf,max_depth):
    r2score_ref=-1*np.inf
    executiontime_ref=0
    best_model={}
    for train_index, validation_index in kf.split(X_train):
        tic=time.time()
        X_training, X_validation = X_train[train_index], X_train[validation_index]
        y_training, y_validation = y_train[train_index], y_train[validation_index]
        
        model= RandomForestRegressor(max_depth=max_depth, random_state=0)
        model.fit(X_training, y_training)
        
        toc=time.time()-tic
        
        y_pred=model.predict(X_validation)
        
#         print(r2_score(y_validation, y_pred),mean_absolute_error(y_validation, y_pred))

        score=r2_score(y_validation, y_pred)
        if score>r2score_ref:
            r2score_ref=score
            executiontime_ref=toc
            best_model=model
#             print("Best model:",r2score_ref,toc)



    y_pred=best_model.predict(X_test)
    r2score_test=r2_score(y_test, y_pred)
    meanabserror_test= mean_absolute_error(y_test, y_pred)
    print("Final results:",r2score_test,meanabserror_test,executiontime_ref)  
    return best_model,r2score_test,meanabserror_test,executiontime_ref

#%%test Decision Trees
from sklearn.tree import DecisionTreeRegressor

def get_bestdecisiontreemodel(X_train, X_test, y_train, y_test,kf,max_depth):
    r2score_ref=-1*np.inf
    executiontime_ref=0
    best_model={}
    for train_index, validation_index in kf.split(X_train):
        tic=time.time()
        X_training, X_validation = X_train[train_index], X_train[validation_index]
        y_training, y_validation = y_train[train_index], y_train[validation_index]
        
        model= DecisionTreeRegressor(max_depth=max_depth, random_state=0)
        model.fit(X_training, y_training)
        
        toc=time.time()-tic
        
        y_pred=model.predict(X_validation)
        
#         print(r2_score(y_validation, y_pred),mean_absolute_error(y_validation, y_pred))

        score=r2_score(y_validation, y_pred)
        if score>r2score_ref:
            r2score_ref=score
            executiontime_ref=toc
            best_model=model
#             print("Best model:",r2score_ref,toc)



    y_pred=best_model.predict(X_test)
    r2score_test=r2_score(y_test, y_pred)
    meanabserror_test= mean_absolute_error(y_test, y_pred)
    print("Final results:",r2score_test,meanabserror_test,executiontime_ref)  
    return best_model,r2score_test,meanabserror_test,executiontime_ref

#%%K-Nearest-Neighbor
from sklearn.neighbors import KNeighborsRegressor

def get_bestknearestmodel(X_train, X_test, y_train, y_test,kf,param_model):
    r2score_ref=-1*np.inf
    executiontime_ref=0
    best_model={}
    for train_index, validation_index in kf.split(X_train):
        tic=time.time()
        X_training, X_validation = X_train[train_index], X_train[validation_index]
        y_training, y_validation = y_train[train_index], y_train[validation_index]
        
        model=KNeighborsRegressor(param_model[0], weights=param_model[1])
        model.fit(X_training, y_training)
        
        toc=time.time()-tic
        
        y_pred=model.predict(X_validation)
        
#         print(r2_score(y_validation, y_pred),mean_absolute_error(y_validation, y_pred))

        score=r2_score(y_validation, y_pred)
        if score>r2score_ref:
            r2score_ref=score
            executiontime_ref=toc
            best_model=model
#             print("Best model:",r2score_ref,toc)



    y_pred=best_model.predict(X_test)
    r2score_test=r2_score(y_test, y_pred)
    meanabserror_test= mean_absolute_error(y_test, y_pred)
#     print("Final results:",r2score_test,meanabserror_test,executiontime_ref)  
    return best_model,r2score_test,meanabserror_test,executiontime_ref

#%% Neural Network MLP Regression
from sklearn.neural_network import MLPRegressor
model=MLPRegressor(max_iter=1000)
def get_bestmlp(X_train, X_test, y_train, y_test,kf,param_model):
    r2score_ref=-1*np.inf
    executiontime_ref=0
    best_model={}
    for train_index, validation_index in kf.split(X_train):
        tic=time.time()
        X_training, X_validation = X_train[train_index], X_train[validation_index]
        y_training, y_validation = y_train[train_index], y_train[validation_index]
        
        model=MLPRegressor(hidden_layer_sizes=param_model[0],activation=param_model[1],solver=param_model[2],random_state=0)
        model.fit(X_training, y_training)
        
        toc=time.time()-tic
        
        y_pred=model.predict(X_validation)
        
#         print(r2_score(y_validation, y_pred),mean_absolute_error(y_validation, y_pred))
        
        score=r2_score(y_validation, y_pred)
        if score>r2score_ref:
            r2score_ref=score
            executiontime_ref=toc
            best_model=model
#             print("Best model:",r2score_ref,toc)



    y_pred=best_model.predict(X_test)
    r2score_test=r2_score(y_test, y_pred)
    meanabserror_test= mean_absolute_error(y_test, y_pred)
    print("Final results:",r2score_test,meanabserror_test,executiontime_ref)  
    return best_model,r2score_test,meanabserror_test,executiontime_ref

#%% Test new features with the models
X_train, X_test, y_train, y_test = train_test_split(np.array(df_split[["windSpeed","humidity","temperature","hour","day_of_week","month"]]),np.array(df_split["energy"]),test_size=0.2,random_state=42)
print("Number of elements for the training set: {} samples".format(len(X_train)))
print("Number of elements for the testing set: {} samples".format(len(X_test)))

#%%Polynomial Degree Test with all features
list_results=[]
r2score_ref=-1*np.inf
for degree in range(0,16):
    print(r2score_ref)
    print("DEGREE:",degree)
    results_poly=get_bestpolynomialmodel(X_train, X_test, y_train.reshape(-1, 1), y_test.reshape(-1, 1),kf,degree)
    list_results.append(["PR:{}".format(degree),results_poly[1],results_poly[2],results_poly[3]])
    if results_poly[1]>r2score_ref:
        print(degree)
        r2score_ref=results_poly[1]
        ref_model=results_poly
print('-------------FINAL_BM-------------',ref_model)
df_resultspoly=pd.DataFrame(list_results,columns=["models","r2_score","mean_abs_error","execution_time"])
print(df_resultspoly.head())

#%% Random Forest Test with all features
list_results=[]
r2score_ref=-1*np.inf
for max_depth in range(2,15):
    print("MAX_DEPTH:",max_depth)
    results_randomforest=get_bestrandomforestmodel(X_train, X_test, y_train, y_test,kf,max_depth)
    list_results.append(["RF:{}".format(max_depth),results_randomforest[1],results_randomforest[2],results_randomforest[3]])
    if results_randomforest[1]>r2score_ref:
        print(degree)
        r2score_ref=results_randomforest[1]
        ref_model=results_randomforest
print('-------------FINAL_BM-------------',ref_model)    
df_resultsrandomforest=pd.DataFrame(list_results,columns=["models","r2_score","mean_abs_error","execution_time"])
print(df_resultsrandomforest.head())

#%% Decision Trees Test with all features
list_results=[]
r2score_ref=-1*np.inf
for max_depth in range(2,15):
    print("MAX_DEPTH:",max_depth)
    results_randomforest=get_bestdecisiontreemodel(X_train, X_test, y_train, y_test,kf,max_depth)
    list_results.append(["RF:{}".format(max_depth),results_randomforest[1],results_randomforest[2],results_randomforest[3]])
    if results_randomforest[1]>r2score_ref:
        print(degree)
        r2score_ref=results_randomforest[1]
        ref_model=results_randomforest
print('-------------FINAL_BM-------------',ref_model)    
df_resultsrandomforest=pd.DataFrame(list_results,columns=["models","r2_score","mean_abs_error","execution_time"])
print(df_resultsrandomforest.head())

#%% KNN Test with all features
list_results=[]
r2score_ref=-1*np.inf
for neighbors in range(1,30):
    print("NUMBER OF NEIGHBORS:",neighbors)
    for weights in ['uniform','distance']:
        print("WEIGHTS",weights)
        results_knearest=get_bestknearestmodel(X_train, X_test, y_train, y_test,kf,[neighbors,weights])
        list_results.append(["KNN:{}/{}".format(neighbors,weights),results_knearest[1],results_knearest[2],results_knearest[3]])
        if results_knearest[1]>r2score_ref:
            r2score_ref=results_knearest[1]
            ref_model=results_knearest
print('-------------FINAL_BM-------------',ref_model) 
df_resultsknearest=pd.DataFrame(list_results,columns=["models","r2_score","mean_abs_error","execution_time"])
print(df_resultsknearest.head())

#%% NN MLP Test with all features
list_results=[]
r2score_ref=-1*np.inf
for hidden_layers in [2,4,8,16,32,64,128,256]:
    for activation in ["identity","logistic","tanh","relu"]:
        for solver in ["lbfgs","adam"]:
            print(hidden_layers,activation,solver)
            results_mlp=get_bestmlp(X_train, X_test, y_train, y_test,kf,[hidden_layers,activation,solver])
            list_results.append(["NN_MLP:{}/{}/{}".format(hidden_layers,activation,solver),results_mlp[1],results_mlp[2],results_mlp[3]])
            if results_mlp[1]>r2score_ref:
                r2score_ref=results_mlp[1]
                ref_model=results_mlp
print('-------------FINAL_BM-------------',ref_model) 
df_resultsmlp=pd.DataFrame(list_results,columns=["models","r2_score","mean_abs_error","execution_time"])
print(df_resultsmlp.head())


#%% Detecting Overfitting
X_train, X_test, y_train, y_test = train_test_split(np.array(df_split[["temperature"]]),np.array(df_split["energy"]),test_size=0.2,random_state=42)

print("Number of elements for the training set: {} samples".format(len(X_train)))
print("Number of elements for the testing set: {} samples".format(len(X_test)))


dict_efficiency={}
# polynmial regression
list_efficiency=[]
for size in [30,60,120,240,480,560,1120,2240]:
    results=get_bestpolynomialmodel(X_train[:size,:].reshape(-1, 1), X_test[:size,:].reshape(-1, 1), y_train[:size], y_test[:size],kf,15)
    list_efficiency.append(results[1]) 
dict_efficiency["polynomial_regressor"]=list_efficiency
#random forest
list_efficiency=[]
for size in [30,60,120,240,480,560,1120,2240]:
    results=get_bestrandomforestmodel(X_train[:size,:].reshape(-1, 1), X_test[:size,:].reshape(-1, 1), y_train[:size], y_test[:size],kf,4)
    list_efficiency.append(results[1]) 
dict_efficiency["random_forest_regressor"]=list_efficiency
#Decision tree
list_efficiency=[]
for size in [30,60,120,240,480,560,1120,2240]:
    results=get_bestdecisiontreemodel(X_train[:size,:].reshape(-1, 1), X_test[:size,:].reshape(-1, 1), y_train[:size], y_test[:size],kf,4)
    list_efficiency.append(results[1]) 
dict_efficiency["decision_tree_regressor"]=list_efficiency
#knearest meighbours
list_efficiency=[]
for size in [30,60,120,240,480,560,1120,2240]:
    results=get_bestknearestmodel(X_train[:size,:].reshape(-1, 1), X_test[:size,:].reshape(-1, 1), y_train[:size], y_test[:size],kf,[19,"uniform"])
    list_efficiency.append(results[1]) 
dict_efficiency["Knearest_neighbours"]=list_efficiency
#mlp regressor
list_efficiency=[]
for size in [30,60,120,240,480,560,1120,2240]:
    results=get_bestmlp(X_train[:size,:], X_test[:size,:].reshape(-1, 1), y_train[:size].reshape(-1, 1), y_test[:size],kf,[256,"relu","lbfgs"])
    list_efficiency.append(results[1]) 
dict_efficiency["neural_network_mlp_regressor"]=list_efficiency

df_impact=pd.DataFrame(dict_efficiency,index=[30,60,120,240,480,560,1120,2240])


fig, ax = plt.subplots(figsize=(12,12))
for model in df_impact.columns:
    df_impact.plot(ax=ax,y=model,kind="line",label=model)

ax.set_xlabel("Number of points", fontsize=15)
ax.set_ylabel("r_score", fontsize=15)
plt.legend(prop={'size': 15})
plt.show()

#ax.figure.savefig("../../reports/pictures/trainingset_impact.png")

#%% Hourly Forecast
# Normalisation of the data
df_split=df.copy()
for param in ['temperature','humidity','windSpeed','month','day_of_week','hour']:
    print(param)
    mean_p=df_split[param].mean()
    max_p=df_split[param].max()
    min_p=df_split[param].min()
    df_split[param]=df_split.apply(lambda row: (row[param]-mean_p)/(max_p-min_p),axis=1)
print(df_split.head())

X_train, X_test, y_train, y_test = train_test_split(np.array(df_split[["temperature",'humidity','windSpeed','month','day_of_week','hour']]),np.array(df_split["energy"]),test_size=0.2,random_state=42)

print("Number of elements for the training set: {} samples".format(len(X_train)))
print("Number of elements for the testing set: {} samples".format(len(X_test)))


#%%ARIMA MODEL
series = df["energy"]
X = series.values
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
tic=time.time()
model = ARIMA(history, order=(5,1,0))
toc=time.time()-tic
model_fit = model.fit(disp=0)
for t in range(len(test)):
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)

print(r2_score(test, predictions),toc)

#%%Polynomial Regression
ist_results=[]
r2score_ref=-1*np.inf
for degree in range(0,8):
    print(r2score_ref)
    print("DEGREE:",degree)
    results_poly=get_bestpolynomialmodel(X_train, X_test, y_train, y_test,kf,degree)
    list_results.append(["PR:{}".format(degree),results_poly[1],results_poly[2],results_poly[3]])
    if results_poly[1]>r2score_ref:
        print(degree)
        r2score_ref=results_poly[1]
        ref_model=results_poly
print('-------------FINAL_BM-------------',ref_model)
df_resultspoly=pd.DataFrame(list_results,columns=["models","r2_score","mean_abs_error","execution_time"])
print(df_resultspoly.head())

#%%Random Forest
list_results=[]
r2score_ref=-1*np.inf
for max_depth in range(2,20):
    print("MAX_DEPTH:",max_depth)
    results_randomforest=get_bestrandomforestmodel(X_train, X_test, y_train, y_test,kf,max_depth)
    list_results.append(["RF:{}".format(max_depth),results_randomforest[1],results_randomforest[2],results_randomforest[3]])
    if results_randomforest[1]>r2score_ref:
        print(degree)
        r2score_ref=results_randomforest[1]
        ref_model=results_randomforest
print('-------------FINAL_BM-------------',ref_model)    
df_resultsrandomforest=pd.DataFrame(list_results,columns=["models","r2_score","mean_abs_error","execution_time"])
print(df_resultsrandomforest.head())

#%%Decision Trees
list_results=[]
r2score_ref=-1*np.inf
for max_depth in range(2,20):
    print("MAX_DEPTH:",max_depth)
    results_tdecisiontree=get_bestdecisiontreemodel(X_train, X_test, y_train, y_test,kf,max_depth)
    list_results.append(["DT:{}".format(max_depth),results_tdecisiontree[1],results_tdecisiontree[2],results_tdecisiontree[3]])
    if results_tdecisiontree[1]>r2score_ref:
        r2score_ref=results_tdecisiontree[1]
        ref_model=results_tdecisiontree
print('-------------FINAL_BM-------------',ref_model)        
df_resultsdecisiontree=pd.DataFrame(list_results,columns=["models","r2_score","mean_abs_error","execution_time"])
print(df_resultsdecisiontree.head())

#%%K-Nearest-Neighbor
list_results=[]
r2score_ref=-1*np.inf
for neighbors in range(1,30):
    print("NUMBER OF NEIGHBORS:",neighbors)
    for weights in ['uniform','distance']:
        print("WEIGHTS",weights)
        results_knearest=get_bestknearestmodel(X_train, X_test, y_train, y_test,kf,[neighbors,weights])
        list_results.append(["KNN:{}/{}".format(neighbors,weights),results_knearest[1],results_knearest[2],results_knearest[3]])
        if results_knearest[1]>r2score_ref:
            r2score_ref=results_knearest[1]
            ref_model=results_knearest
print('-------------FINAL_BM-------------',ref_model) 
df_resultsknearest=pd.DataFrame(list_results,columns=["models","r2_score","mean_abs_error","execution_time"])
print(df_resultsknearest.head())

#%% Neural Network MLP
list_results=[]
r2score_ref=-1*np.inf
for hidden_layers in [2,4,8,16,32,64,128,256]:
    for activation in ["identity","logistic","tanh","relu"]:
        for solver in ["lbfgs","adam"]:
            print(hidden_layers,activation,solver)
            results_mlp=get_bestmlp(X_train, X_test, y_train, y_test,kf,[hidden_layers,activation,solver])
            list_results.append(["NN_MLP:{}/{}/{}".format(hidden_layers,activation,solver),results_mlp[1],results_mlp[2],results_mlp[3]])
            if results_mlp[1]>r2score_ref:
                r2score_ref=results_mlp[1]
                ref_model=results_mlp
print('-------------FINAL_BM-------------',ref_model) 
df_resultsmlp=pd.DataFrame(list_results,columns=["models","r2_score","mean_abs_error","execution_time"])
print(df_resultsmlp.head())


#%%Prediction-Reality
fig, ax = plt.subplots(figsize=(12,12))

model=KNeighborsRegressor(28, weights="uniform")
model.fit(X_train, y_train)
y_pred=model.predict(X_test)

ax.scatter(x=y_test,y=y_pred)

ax.set_xlabel("Reality", fontsize=15)
ax.set_ylabel("Prediction", fontsize=15)
# plt.legend(prop={'size': 15})
plt.show()

#ax.figure.savefig("../../reports/pictures/freeform_vis.png")

#%% fit model

