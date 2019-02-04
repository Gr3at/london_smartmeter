
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.gridspec as gridspec
from sklearn import linear_model
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
import pickle
import os
import scipy.stats as st

df = pd.read_csv('hourly_datasets_to_use/energy_weather_global_data.csv',parse_dates=[0],index_col=0)
df=df.resample('30T').ffill().bfill().reindex(pd.date_range(df.index[0],df.index[-1],freq='H'))

#check autocorelation --> first 4 hours are highly corelated
pd.plotting.autocorrelation_plot(df['energy'].dropna())
plt.xlim(0,24)
plt.title('Auto-correlation of hourly demand over a 24 hour period')
plt.show()

def generate_lagged_features(df, var, max_lag):
    for t in range(1, max_lag+1):
        df[var+'_lag'+str(t)] = df[var].shift(t, freq='1H')

generate_lagged_features(df, 'temperature', 3)
generate_lagged_features(df, 'windSpeed', 3)
generate_lagged_features(df, 'humidity', 3)
generate_lagged_features(df, 'energy', 3)
df.dropna(how='any', inplace=True)

df.insert(loc=0, column='timeStamp', value=df.index) #insert new column in the begining of the DataFrame
train, test = (df[:'2013-09-01 00:00:00'], df['2013-09-01 00:00:00':])

#%%Ridge Regresion Model
model_name = "ridge_poly2"
X = train.drop(['energy'], axis=1)

cat_cols = ['hour', 'month', 'day_of_week']
cat_cols_idx = [X.columns.get_loc(c) for c in X.columns if c in cat_cols]
onehot = OneHotEncoder(categorical_features=cat_cols_idx, sparse=False)
regr = Ridge(fit_intercept=False)
poly = PolynomialFeatures(2)
tscv = TimeSeriesSplit(n_splits=3)

param_dist = {'alpha': st.uniform(1e-4, 5.0)}
regr_cv = RandomizedSearchCV(estimator=regr,
                            param_distributions=param_dist,
                            n_iter=20,
                            scoring='mean_squared_error',
                            iid=False,
                            cv=tscv,
                            verbose=2,
                            n_jobs=1)
regr_pipe = Pipeline([('onehot', onehot), ('poly', poly), ('regr_cv', regr_cv)])
regr_pipe.fit(X, y=train['energy'])

cv_results = pd.DataFrame(regr_pipe.named_steps['regr_cv'].cv_results_)
cv_results.sort_values(by='rank_test_score').head()

#%% Linear regression with recursive feature elimination
model_name = "linear_regression"
X = train.drop(['timeStamp','energy'], axis=1)
cat_cols = ['hour', 'month', 'day_of_week']
cat_cols_idx = [X.columns.get_loc(c) for c in X.columns if c in cat_cols]
onehot = OneHotEncoder(categorical_features=cat_cols_idx, sparse=False)
regr = linear_model.LinearRegression(fit_intercept=True)
tscv = TimeSeriesSplit(n_splits=10) # cross-validation number of folds used

demand_ts = train[['timeStamp', 'energy']].copy()
demand_ts.reset_index(drop=True, inplace=True)

for split_num, split_idx  in enumerate(tscv.split(demand_ts)):
    split_num = str(split_num)
    train_idx = split_idx[0]
    test_idx = split_idx[1]
    demand_ts['fold' + split_num] = "not used"
    demand_ts.loc[train_idx, 'fold' + split_num] = "train"
    demand_ts.loc[test_idx, 'fold' + split_num] = "test"

gs = gridspec.GridSpec(5,1)
fig = plt.figure(figsize=(15, 10), tight_layout=True)

ax = fig.add_subplot(gs[0])
ax.plot(demand_ts.loc[demand_ts['fold0']=="train", "timeStamp"], demand_ts.loc[demand_ts['fold0']=="train", "energy"], color='b')
ax.plot(demand_ts.loc[demand_ts['fold0']=="test", "timeStamp"], demand_ts.loc[demand_ts['fold0']=="test", "energy"], 'r')
ax.plot(demand_ts.loc[demand_ts['fold0']=="not used", "timeStamp"], demand_ts.loc[demand_ts['fold0']=="not used", "energy"], 'w')

ax = fig.add_subplot(gs[1], sharex=ax)
plt.plot(demand_ts.loc[demand_ts['fold1']=="train", "timeStamp"], demand_ts.loc[demand_ts['fold1']=="train", "energy"], 'b')
plt.plot(demand_ts.loc[demand_ts['fold1']=="test", "timeStamp"], demand_ts.loc[demand_ts['fold1']=="test", "energy"], 'r')
plt.plot(demand_ts.loc[demand_ts['fold1']=="not used", "timeStamp"], demand_ts.loc[demand_ts['fold1']=="not used", "energy"], 'w')

ax = fig.add_subplot(gs[2], sharex=ax)
plt.plot(demand_ts.loc[demand_ts['fold2']=="train", "timeStamp"], demand_ts.loc[demand_ts['fold2']=="train", "energy"], 'b')
plt.plot(demand_ts.loc[demand_ts['fold2']=="test", "timeStamp"], demand_ts.loc[demand_ts['fold2']=="test", "energy"], 'r')
plt.plot(demand_ts.loc[demand_ts['fold2']=="not used", "timeStamp"], demand_ts.loc[demand_ts['fold2']=="not used", "energy"], 'w')

ax = fig.add_subplot(gs[3], sharex=ax)
plt.plot(demand_ts.loc[demand_ts['fold3']=="train", "timeStamp"], demand_ts.loc[demand_ts['fold3']=="train", "energy"], 'b')
plt.plot(demand_ts.loc[demand_ts['fold3']=="test", "timeStamp"], demand_ts.loc[demand_ts['fold3']=="test", "energy"], 'r')
plt.plot(demand_ts.loc[demand_ts['fold3']=="not used", "timeStamp"], demand_ts.loc[demand_ts['fold3']=="not used", "energy"], 'w')

ax = fig.add_subplot(gs[4], sharex=ax)
plt.plot(demand_ts.loc[demand_ts['fold4']=="train", "timeStamp"], demand_ts.loc[demand_ts['fold4']=="train", "energy"], 'b')
plt.plot(demand_ts.loc[demand_ts['fold4']=="test", "timeStamp"], demand_ts.loc[demand_ts['fold4']=="test", "energy"], 'r')
plt.plot(demand_ts.loc[demand_ts['fold4']=="not used", "timeStamp"], demand_ts.loc[demand_ts['fold4']=="not used", "energy"], 'w')
plt.show()

regr_cv = RFECV(estimator=regr,
             cv=tscv,
             scoring='neg_mean_squared_error',
             verbose=2,
             n_jobs=-1)

regr_pipe = Pipeline([('onehot', onehot), ('rfecv', regr_cv)])

regr_pipe.fit(X, y=train['energy'])
with open(os.path.join(model_name + '.pkl'), 'wb') as f:
    pickle.dump(regr_pipe, f)

max(regr_pipe.named_steps['rfecv'].grid_scores_)


cv_results = pd.DataFrame.from_dict({'cv_score': regr_pipe.named_steps['rfecv'].grid_scores_})
cv_results['mean_squared_error'] = cv_results['cv_score']
plt.figure(figsize=(15, 5))
plt.plot(cv_results.index, cv_results['mean_squared_error'])
plt.xlabel('number of features')
plt.title('CV negative mean squared error')
plt.show()

regr_pipe.named_steps['rfecv'].n_features_

def get_onehot_cols(X):
    X_dummy_cols = list(pd.get_dummies(X.copy()[cat_cols], columns=cat_cols).columns)
    other_cols = list(X.columns.drop(cat_cols))
    return X_dummy_cols + other_cols

supported_features = pd.DataFrame.from_dict(
    {'feature':get_onehot_cols(X), 
     'supported':regr_pipe.named_steps['rfecv'].support_}
)
supported_features

coefs = supported_features.loc[supported_features['supported'], ].copy()
coefs['coefficients'] = regr_pipe.named_steps['rfecv'].estimator_.coef_
coefs.plot.bar('feature', 'coefficients', figsize=(15, 3), legend=False)
plt.show()

