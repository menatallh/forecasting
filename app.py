import streamlit as st
import pandas as pd 

from fbprophet import Prophet
from datetime import datetime
from datetime import timedelta






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



df = pd.read_csv('http://data.bitcoinity.org/export_data.csv?currency=USD&data_type=price&exchange=coinbase&r=hour&t=l&timespan=30d', parse_dates=['Time'])

# Set the date/time to be the index for the dataframe
df.set_index('Time', inplace=True)
df.head()


ax = df['avg'].plot(title="Bitcoin daily price USD")
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))




df['ds'] = df.index
df['ds'] = df['ds'].apply(rem_timezone)

df['y'] = df['avg']

forecast_data = df[['ds', 'y']].copy()
forecast_data.reset_index(inplace=True)
del forecast_data['Time']
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


start_date =  datetime.now()
end_date = start_date + timedelta(hours=1)

st.write(forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)].head()[['ds','yhat']])





