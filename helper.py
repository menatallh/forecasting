from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import RandomizedSearchCV

# Commented out IPython magic to ensure Python compatibility.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pylab as pylab
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
#import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import xgboost as xgb

# %matplotlib inline

import warnings
warnings.filterwarnings("ignore")


pd.set_option('display.max_columns', 200)

#from datetime import datetime

from itertools import product
import warnings
#import statsmodels.api as sm
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

NTESTS = 1
PREV_DAYS = 10
PERCENT_UP = 0.01
PERCENT_DOWN = 0.01
PERIOD = 'Hourly' # [5-min, 15-min, 30-min, Hourly, 2-hour, 6-hour, 12-hour, Daily, Weekly]
MARKET = 'bitstampUSD'

# DATE START
YEAR_START = 2011
MONTH_START = 9
DAY_START = 13
DATE_START = date(YEAR_START, MONTH_START, DAY_START)

# DATE END
DATE_END = date.today()

URL_DATA_BASE = 'http://bitcoincharts.com/charts/chart.json?'

# -*- coding: utf-8 -*-

"""Get bitcoin historic data.

Works with python 3
"""

from datetime import timedelta, datetime
import csv
import requests



# Get and write data
def get(path_file='datas.csv', period='6-hour', market='bitstampUSD'):

    print("Loading.....")
    header = ["Timestamp", "Open", "High", "Low", "Close", "Volume_BTC",
                "Volume_Currency", "Weighted_Price"]

    with open(path_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)

        period_to_call = False

        # 1 API CALL
        if period == 'Weekly' or period == 'Daily' or period == '12-hour':

            url = URL_DATA_BASE + 'm='+ market + \
            '&i=' + period + '&c=1' + '&s=' + DATE_START.isoformat()+ \
            '&e=' + DATE_END.isoformat()

            # print url
            data = requests.get(url).json()
            for d in data:
                writer.writerow(d)

        elif period == '6-hour' or period == '2-hour':
            period_to_call = 365 # 1 API CALL per year
        elif period == 'Hourly' or period == '30-min' or period == '15-min':
            period_to_call = 30 # 1 API CALL per month
        elif period == '5-min':
            period_to_call = 7 # 1 API CALL per week
        else:
            period_to_call = 1 # 1 API CALL per day

        if period_to_call:
            delta = DATE_END - DATE_START
            i = 0
            while i <= delta.days:
                try:
                    date_start = DATE_START + timedelta(days=i)
                    date_end = DATE_START + timedelta(days=i+period_to_call)
                    url = URL_DATA_BASE + 'm='+ market + \
                    '&i=' + period + '&c=1' + '&s=' + date_start.isoformat() + \
                    '&e=' + date_end.isoformat()
                    # print url
                    data = requests.get(url).json()
                    for d in data:
                        writer.writerow(d)
                except:
                    print('Url not available (date): ' + url)
                i += period_to_call + 1
                print(str(i) + ' of ' + str(delta.days+1) + ' days loaded...')

    print("Last Timestamp: " + \
        datetime.fromtimestamp(int(data[-1][0])).strftime('%Y-%m-%d %H:%M:%S'))

#get('datas-Hourly.csv', period=PERIOD, market=MARKET)

import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt


# Set a float format as we'll always be looking at USD monetary values
pd.options.display.float_format = '${:,.2f}'.format



# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

import pandas as pd
#from fbprophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
#from fbprophet.plot import plot_plotly

#import AlphaVantage 


historical_day_url = 'https://min-api.cryptocompare.com/data/histoday'
historical_hour_url = 'https://min-api.cryptocompare.com/data/histohour'
historical_minute_url = 'https://min-api.cryptocompare.com/data/histominute'
def get_historical_day(pair_symbol, base_symbol,
        exchange='BitTrex', aggregate=1, limit=None, all_data=False):
        '''
        Method for returning open, high, low, close, 
        volumefrom and volumeto daily historical data
        ** Parameters**
        pair_symbol <str>: Currency pair, EG ETH
        base_symbol <str>: Base currency, Eg BTC
        exchange <str>: Target exchange, EG BitTrex
        aggregate <int>: Aggregation multipler, Eg 1 = 1day, 2 = 2days
        limit <int>: Limit the number of ohlcv records returned
        all_data <bool>: Get all data
        ** Example call **
        crypto_compare_api = CryptoCompareAPI()
        data = crypto_compare_api.get_historical_day('ETH', 'BTC', aggregate=2, limit=1)
        '''


        url='{}?fsym={}&tsym={}&e={}&aggregate={}'.format(
            historical_day_url, pair_symbol.upper(),
            base_symbol.upper(), exchange, aggregate)

        if all_data:
            url += '&allData=true'

        if limit:
            url += '&limit={}'.format(limit)

        r = requests.get(url)
        data = r.json()['Data']

        df = pd.DataFrame(data)
        df[['Close', 'High', 'Low', 'Open']]=df[['close', 'high', 'low', 'open']]
        df.drop(columns=['close', 'high', 'low', 'open','conversionType' ,'conversionSymbol' ],inplace=True)

        df['Timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]        
        return data

def get_historical_hour(  pair_symbol, base_symbol,
        exchange='BitTrex', aggregate=1, limit=2000, all_data=True):
        '''
        Method for returning open, high, low, close, 
        volumefrom and volumeto hourly historical data
        ** Parameters**
        pair_symbol <str>: Currency pair, EG ETH
        base_symbol <str>: Base currency, Eg BTC
        exchange <str>: Target exchange, EG BitTrex
        aggregate <int>: Aggregation multipler, Eg 1 = 1day, 2 = 2days
        limit <int>: Limit the number of ohlcv records returned
        all_data <bool>: Get all data
        ** Example call **
        crypto_compare_api = CryptoCompareAPI()
        data = crypto_compare_api.get_historical_hour('ETH', 'BTC', aggregate=2, limit=1)
        '''

        url='{}?fsym={}&tsym={}&e={}&aggregate={}'.format(
            historical_hour_url, pair_symbol.upper(),
            base_symbol.upper(), exchange, aggregate)

        if all_data:
            url += '&allData=true'

        if limit:
            url += '&limit={}'.format(limit)

        r = requests.get(url)
        data = r.json()['Data']
        df = pd.DataFrame(data)
        print(df)
        df[['Close', 'High', 'Low', 'Open']]=df[['close', 'high', 'low', 'open']]
        df.drop(columns=['close', 'high', 'low', 'open','conversionType' ,'conversionSymbol' ],inplace=True)
        df['Timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
        print(df.columns)
        df.to_csv('hour_btc_data.csv')
        return df



def get_historical_minute(pair_symbol, base_symbol,
        exchange='BitTrex', aggregate=1, limit=2000, all_data=False):
        '''
        Method for returning open, high, low, close, 
        volumefrom and volumeto minute historical data
        ** Parameters**
        pair_symbol <str>: Currency pair, EG ETH
        base_symbol <str>: Base currency, Eg BTC
        exchange <str>: Target exchange, EG BitTrex
        aggregate <int>: Aggregation multipler, Eg 1 = 1day, 2 = 2days
        limit <int>: Limit the number of ohlcv records returned
        all_data <bool>: Get all data
        ** Example call **
        crypto_compare_api = CryptoCompareAPI()
        data = crypto_compare_api.get_historical_minute('ETH', 'BTC', aggregate=2, limi
        '''

        url='{}?fsym={}&tsym={}&e={}&aggregate={}'.format(
            historical_minute_url, pair_symbol.upper(),
            base_symbol.upper(), exchange, aggregate)

        if all_data:
            url += '&allData=true'

        if limit:
            url += '&limit={}'.format(limit)

        r = requests.get(url)
        data = r.json()['Data']

        df = pd.DataFrame(data)
        df[['Close', 'High', 'Low', 'Open']]=df[['close', 'high', 'low', 'open']]
        df.drop(columns=['close', 'high', 'low', 'open','conversionType' ,'conversionSymbol' ],inplace=True)

        df['Timestamp'] = [datetime.fromtimestamp(d) for d in df.time]     
        return data

import math
import pandas as pd
from sklearn.metrics import log_loss, cohen_kappa_score, accuracy_score, confusion_matrix, hinge_loss, classification_report
from datetime import datetime
from sklearn.metrics import roc_auc_score

# split dataset (train and test) in 2 pieces.
# start piece to train, end piece to test.
def split_df(dframe):
    test = dframe.tail(settings.NTESTS)
    train = dframe[:-settings.NTESTS]
    return train, test

# Split dataset (train and test)
# splitea 1 de cada 4 de forma salpicada
def split_df2(dframe):
    trainfilter = [False if i%4 == 0 else True for i in range(dframe.shape[0])]
    testfilter = [True if i%4 == 0 else False for i in range(dframe.shape[0])]
    return dframe[trainfilter], dframe[testfilter]

# drop rows with "Nans" values
def dropna(df):
    df = df[df < math.exp(709)] # big number
    df = df[df != 0.0]
    df = df.dropna()
    return df

def split_dft(df):

  df_train = df.iloc[:82150-1255,:]
  #df_valid = df[df.Year > 2020]
  df_valid=df.loc[(df['Year'] >= 2021) & (df['Month'] >= 8)]
  return df_train,df_valid

# show metrics
def metrics(y_true, y_pred, y_pred_proba=False):
    target_names = ['KEEP', 'UP', 'DOWN']

    if y_pred_proba is not False:
        print('Cross Entropy: {}'.format(log_loss(y_true, y_pred_proba)))
    print('Accuracy: {}'.format(accuracy_score(y_true, y_pred)))
    print('Coefficient Kappa: {}'.format(cohen_kappa_score(y_true, y_pred)))
    print('Classification Report:')
    print(classification_report(y_true.values, y_pred, target_names=target_names))
    print("Confussion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# show metrics
def metrics2(y_true, y_pred):
    print('Accuracy: {}'.format(accuracy_score(y_true, y_pred)))
    print('Coefficient Kappa: {}'.format(cohen_kappa_score(y_true, y_pred)))
    print("Confussion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def timestamptodate(timestamp):
    return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
def find_missing(df):
  missing_values = df.isnull().sum()
  missing_per = (missing_values/df.shape[0])*100
  missing_table = pd.concat([missing_values,missing_per], axis=1, ignore_index=True) 
  missing_table.rename(columns={0:'Total Missing Values',1:'Missing %'}, inplace=True)
  return missing_table       


def  rolling_feats(df):



  lag_features = ["Open", "High", "Low", "Close","Volume_BTC"]
  window1 = 3
  window2 = 7
  window3 = 30

  df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
  df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
  df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)

  df_mean_3d = df_rolled_3d.mean().shift(1).reset_index()
  df_mean_7d = df_rolled_7d.mean().shift(1).reset_index()
  df_mean_30d = df_rolled_30d.mean().shift(1).reset_index()

  df_std_3d = df_rolled_3d.std().shift(1).reset_index()
  df_std_7d = df_rolled_7d.std().shift(1).reset_index()
  df_std_30d = df_rolled_30d.std().shift(1).reset_index()

  for feature in lag_features:
    df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
    df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
    df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]
    
    df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
    df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
    df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]

  df.fillna(df.mean(), inplace=True)

  df.set_index("Timestamp", drop=False, inplace=True)


  return df

