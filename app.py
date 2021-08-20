import streamlit as st
import pandas as pd 

from fbprophet import Prophet
from datetime import datetime as dt
from datetime import timedelta



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
from fbprophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from fbprophet.plot import plot_plotly

#import AlphaVantage 




st.title('btc forcasting app')

def rem_timezone(time):
    return str(time).split(' ')[0]


def price(symbol, comparison_symbols=['USD'], exchange=''):
    url = 'https://min-api.cryptocompare.com/data/price?fsym={}&tsyms={}'\
            .format(symbol.upper(), ','.join(comparison_symbols).upper())
    if exchange:
        url += '&e={}'.format(exchange)
    page = requests.get(url)
    data = page.json()
    return data

def daily_price_historical(symbol, comparison_symbol, all_data=True, limit=1, aggregate=1, exchange=''):
    url = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit={}&aggregate={}'\
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
    if exchange:
        url += '&e={}'.format(exchange)
    if all_data:
        url += '&allData=true'
    page = requests.get(url)
    data = page.json()['Data']
    df = pd.DataFrame(data)

    df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
    print(df.head())
    return df

def minute_price_historical(symbol, comparison_symbol, limit, aggregate, exchange=''):
    url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}'\
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
    if exchange:
        url += '&e={}'.format(exchange)
    page = requests.get(url)
    data = page.json()['Data']
    df = pd.DataFrame(data)
    print(df.head())
    df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
    return df


time_delta=1

df =daily_price_historical('BTC', 'USD', 9999, time_delta)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1,1,1)
ax.plot(df["close"])

st.write(fig)


df['ds'] = df.timestamp
df['ds'] = df['ds'].apply(rem_timezone)
selected_series = st.selectbox("Select a prediction feature:", ("close","open","min","max"))
df['y'] = df[selected_series]

forecast_data = df[['ds', 'y']].copy()
forecast_data.reset_index(inplace=True)
#del forecast_data['timestamp']
st.write(forecast_data.head())



m = Prophet()
m.fit(forecast_data);

future = m.make_future_dataframe(periods=96, freq='H')
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


st.write(forecast.tail())

fig = plot_plotly(m, forecast)
fig.update_layout(
        title='Btc_forecast', yaxis_title='btc_avg_price', xaxis_title="Date",
    )

int_forcast_hours = st.number_input('hours forcast window ', min_value=1, max_value=10, value=1, step=1)

start_date =  dt.now()
end_date = start_date + timedelta(hours=int_forcast_hours)

st.write(forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)].head()[['ds','yhat']])

st.plotly_chart(fig)



