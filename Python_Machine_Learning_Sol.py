#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def load_raw(filepath):
    dataset = pd.read_csv(filepath)
    return dataset
    
def set_target(self,dataset):
    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2].values
    return X,y

def train_model(X,y):
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X, y)
    return regressor

def predict(self,reg):
        y_pred = reg.predict(X)
        return y_pred

