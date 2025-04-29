import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor

from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatter, LogLocator

import math
from math import sqrt
from sklearn.cross_decomposition import CCA
import pickle 

import time
start = time.time()
from matplotlib import rcParams

rcParams['font.family']='Arial'


def nse_score(obs, sim):
    mean_obs = np.mean(obs)
    nse =  1- np.sum(np.power(obs-sim,2)) / np.sum(np.power(obs-mean_obs,2))
    return nse

def r2_score(x, y):
    xBar = np.mean(x)
    ybar = np.mean(y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(x)): 
        diffxxBar = x[i] - xBar
        diffyyBar = y[i] - ybar
        SSR += (diffxxBar * diffyyBar)
        varX += diffxxBar ** 2 
        varY += diffyyBar ** 2 
        SST = math.sqrt(varX * varY)
    return (SSR / SST)**2


def mdsa_score(obs, sim):
    symmetric_errors = np.abs((sim - obs) / obs)
    median_symmetric_error = np.median(symmetric_errors)
    mdsa = 100 * (1 - median_symmetric_error)
    return mdsa


input_c = np.load(r'./input_c.npy')
input_x = np.load(r'./input_x.npy')
input_y = np.load(r'./input_y.npy')

merged_matrix = np.concatenate((input_c, input_x, input_y), axis=1)
merged_matrix = merged_matrix[~np.isnan(merged_matrix).any(axis=1)]
X = merged_matrix[:,:-1]
y = np.log(merged_matrix[:,-1])


### 3 sigma
def remove_outliers(X, y, n_sigma = 5):
    mean = np.mean(y)
    std = np.std(y)
    z_scores = np.abs((y - mean) / std)
    mask = z_scores < n_sigma
    X_cleaned = X[mask]
    y_cleaned = y[mask]
    return X_cleaned, y_cleaned

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 42)


### GBR
# color='#7fc97f', s=100, alpha=0.5, edgecolor='#71b371'
gbr = GradientBoostingRegressor(n_estimators = 100, 
                          max_depth = 10,
                          min_samples_split = 3,
                          min_samples_leaf = 3,
                          max_features = 50,
                          warm_start = False)

gbr.fit(X_train, y_train)
y_train_ETR = gbr.predict(X_train)
y_predict_ETR = gbr.predict(X_test)


### XGB
# color='#beaed4', s=100, alpha=0.5, edgecolor='#a99bbc',
import xgboost as xgb
xgb_model = xgb.XGBRegressor(
    n_estimators=100,       
    max_depth=15,           
    learning_rate=0.1,      
    subsample=0.8,          
    colsample_bytree=0.8,   
    min_child_weight=3,     
    n_jobs=-1,             
    random_state=42         
)

xgb_model.fit(X_train, y_train)
y_train_XGB = xgb_model.predict(X_train)
y_predict_XGB = xgb_model.predict(X_test)


### ETR
# color='#386cb0', s=100, alpha=0.5, edgecolor='#32609c',
etr = ExtraTreesRegressor(n_estimators = 100, 
                          max_depth = 50,
                          min_samples_split = 2,
                          min_samples_leaf = 2,
                          n_jobs = -1,
                          max_features = 200,
                          warm_start = False,
                          bootstrap = True,
                          oob_score = True)

etr.fit(X_train, y_train)
y_train_XGB = etr.predict(X_train)
y_predict_XGB = etr.predict(X_test)



### RFR
# color='#fdc086', s=100, alpha=0.5, edgecolor='#e1ab77', linewidth=0.5 
rf = RandomForestRegressor(n_estimators = 200, 
                          max_depth = 50,
                          min_samples_split = 2,
                          min_samples_leaf = 2,
                          n_jobs = -1,
                          max_features = 200,
                          warm_start = False,
                          bootstrap = True,
                          oob_score = True)

rf.fit(X_train, y_train)
y_train_RFR = rf.predict(X_train)
y_predict_RFR = rf.predict(X_test)



print(time.time() - start)
