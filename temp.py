import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
 
from scipy import stats
from scipy.stats import norm, skew

from sklearn.preprocessing import RobustScaler , StandardScaler
from sklearn.linear_model import LinearRegression, Ridge , Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



import xgboost as xgb


import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()



column_name= ["MPG", "Cylinders", "Dispacement", "Horsepower", "Weight", 
              "Accleration", "Model Year", "Origin"]


data= pd.read_csv("C:/Users/AYGENN/OneDrive/Masaüstü/staj/auto-mpg.data",
                  names=column_name, na_values="?", comment="\t", sep= " ",
                  skipinitialspace= True)

data= data.rename (columns = {"MPG":"target"})

print(data.head())
print("Data shape: " , data.shape)

data.info()


describe= data.describe()

