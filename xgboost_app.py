import streamlit as st
import requests
import datetime

import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np # linear algebra
import pandas as pd #

import pandas as pd
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

import os
import gc
import numpy as np
import pandas as pd
import pickle 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import xgboost as xgb

import os

import warnings
warnings.filterwarnings("ignore")


pd.set_option('display.max_columns', 200)


from datetime import datetime as dt

from itertools import product
import warnings

import os
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')


plt.style.use('seaborn-darkgrid')


import pandas as pd
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)



from sklearn.model_selection import train_test_split

import xgboost as xgb

# %matplotlib inline

import warnings
warnings.filterwarnings("ignore")


pd.set_option('display.max_columns', 200)

#from datetime import datetime

import warnings

import os
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')

from datetime import datetime
plt.style.use('seaborn-darkgrid')

from datetime import date



from helper import *




df=get('datas-Hourly.csv', period=PERIOD, market=MARKET)


#df = pd.read_csv('datas-Hourly.csv', sep=',')
df=dropna(df)

#df.head()

df['Timestamp']=[datetime.fromtimestamp(x) for x in df['Timestamp']]

#df

df.set_index("Timestamp").Weighted_Price.plot(figsize=(14,7), title="Bitcoin Weighted Price")

#print(find_missing(df))

indx_df = df.set_index("Timestamp")
indx_df.reset_index(drop=False, inplace=True)

indx_df=rolling_feats(indx_df)

indx_df['Timestamp'] = pd.to_datetime(indx_df['Timestamp'])

indx_df['Year'] = indx_df['Timestamp'].dt.year
indx_df['Month'] = indx_df['Timestamp'].dt.month
indx_df['Week'] = indx_df['Timestamp'].dt.weekofyear
indx_df['Weekday'] = indx_df['Timestamp'].dt.weekday
indx_df['Day'] = indx_df['Timestamp'].dt.day
indx_df['Hour'] = indx_df['Timestamp'].dt.hour

#indx_df

#indx_df[indx_df.Year<=2020]

train_df,valid_df=split_dft(indx_df)

exogenous_features = ['Open_mean_lag3',
       'Open_mean_lag7', 'Open_mean_lag30', 'Open_std_lag3', 'Open_std_lag7',
       'Open_std_lag30', 'High_mean_lag3', 'High_mean_lag7', 'High_mean_lag30',
       'High_std_lag3', 'High_std_lag7', 'High_std_lag30', 'Low_mean_lag3',
       'Low_mean_lag7', 'Low_mean_lag30', 'Low_std_lag3', 'Low_std_lag7',
       'Low_std_lag30', 'Close_mean_lag3', 'Close_mean_lag7',
       'Close_mean_lag30', 'Close_std_lag3', 'Close_std_lag7',
       'Close_std_lag30', 'Volume_BTC_mean_lag3', 'Volume_BTC_mean_lag7',
       'Volume_BTC_mean_lag30', 'Volume_BTC_std_lag3',
       'Volume_BTC_std_lag7', 'Volume_BTC_std_lag30', 'Year','Month', 'Week',
       'Weekday','Day', 'Hour']

X_train, y_train = train_df[exogenous_features], train_df.Weighted_Price
X_test, y_test = valid_df[exogenous_features], valid_df.Weighted_Price

def build_model(X_train,y_train):
  reg = xgb.XGBRegressor()
  params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
 "max_depth"        : [1, 3, 4, 5, 6, 7],
 "n_estimators"     : [int(x) for x in np.linspace(start=100, stop=2000, num=10)],
 "min_child_weight" : [int(x) for x in np.arange(3, 15, 1)],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
 "subsample"        : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
 "colsample_bytree" : [0.5, 0.6, 0.7, 0.8, 0.9, 1],
 "colsample_bylevel": [0.5, 0.6, 0.7, 0.8, 0.9, 1],  
  }

  model  = RandomizedSearchCV(    
                reg,
                param_distributions=params,
                n_iter=10,
                n_jobs=-1,
                cv=5,
                verbose=3,)
  model.fit(X_train, y_train)
  best_pars = model.best_params_
  # Best XGB model that was found based on the metric score you specify
  best_model = model.best_estimator_

  best_model.save_model('mode_best.bin')

def eval_model(model_path,X_train,X_test,train_df,valid_df):
  model = xgb.XGBRegressor()
  model.load_model(model_path)

  #print(f"Model Best Score : {model.best_score_}")
  #print(f"Model Best Parameters : {model.best_estimator_.get_params()}")  
  
  train_df['Predicted_Weighted_Price'] = model.predict(X_train)
  ax=train_df[['Weighted_Price','Predicted_Weighted_Price']].plot(figsize=(15, 5))
  ax.figure.savefig('train_forecast.pdf') 
  valid_df['Forecast_XGBoost'] = model.predict(X_test)
   
  ax2=valid_df[['Close','Forecast_XGBoost']].plot(figsize=(15, 5))
  st.write(ax2.figure)
  ax2.figure.savefig('val_forcast.pdf')

  return X_train,X_test,model





model_path='/home/za3balawi/forecasting/mode_best.bin'
#build_model(X_train,y_train)
eval_model(model_path,X_train,X_test,train_df,valid_df)






selected = st.selectbox("Select a prediction frequency:", ("hourly","day","minute"))


if selected=="hourly":

   #df =get_historical_day('BTC', 'USD',limit=2000)
   print('loading hourly data')

elif selected=="mintue":

   #df =get_historical_minute('BTC', 'USD',limit=2000)
   print('Not available currently')
else :


  #df =get_historical_hour('BTC', 'USD',limit=2000)


     print('Not available currently')


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1,1,1)
ax.plot(indx_df["Close"])

st.write(fig)









def score(df_train):
  train_mae = mean_absolute_error(df_train['Close'], df_train['Predicted_Close_Price'])
  train_rmse = np.sqrt(mean_squared_error(df_train['Close'], df_train['Predicted_Close_Price']))
  train_r2 = r2_score(df_train['Close'], df_train['Predicted_Close_Price'])

  print(f"train MAE : {train_mae}")
  print(f"train RMSE : {train_rmse}")
  print(f"train R2 : {train_r2}")



int_forcast_window = st.number_input(str(selected)+'forcast window ', min_value=1, max_value=10, value=6, step=1)

start_date =  dt.now()
end_date = start_date + timedelta(hours=int_forcast_window)
print(valid_df.columns)
st.write(valid_df[(valid_df.Hour >= start_date.hour) & (valid_df.Hour <= end_date.hour)]['Forecast_XGBoost'])


