# import yfinance as yf
# import streamlit as st
#
# st.write("""
# # Simple Stock Price App
# Shown are the stock **closing price** and ***volume*** of Google!
# """)
#
# # https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
# #define the ticker symbol
# tickerSymbol = 'GOOGL'
# #get data on this ticker
# tickerData = yf.Ticker(tickerSymbol)
# #get the historical prices for this ticker
# tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')
# # Open	High	Low	Close	Volume	Dividends	Stock Splits
#
# st.write("""
# ## Closing Price
# """)
# st.line_chart(tickerDf.Close)
# st.write("""
# ## Volume Price
# """)
# st.line_chart(tickerDf.Volume)
#
#

import streamlit as st
import numpy as np
import pandas as pd
# import plotly.plotly as py
import chart_studio.plotly
import plotly.graph_objs as go
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
import sys
import os


st.title('Stock Price Movement Dashboard')


st.write("""
**This is simple dashboard that shows the daily price movement over the
last six months of the stock of any companies shown in Yahoo! Finance**
""")

ticker_symbol = st.text_input("Enter Valid Ticker Symbol", "TSLA")

@st.cache
def load_data():
    stock_df = pdr.get_data_yahoo(ticker_symbol.strip(), period='6mo', interval='1d', prepost = True)#yf.Ticker(str(ticker_symbol).upper())
    return stock_df

df = load_data().reset_index()
# restart = True
# while restart:
# # def catcher():
if len(df) == 0:
    st.error("Ticker may be delisted or invalid. Enter a valid ticker")
    st.stop()
else:
    st.write("Valid ticker symbol entered; here goes the rest of the app")

# try:
#     catcher()
# except SystemExit:
#     st.write("Enter a valid ticker")
#

columns = st.multiselect(
    "Choose Price To Display", list(df.drop(['Adj Close', 'Volume'], axis=1).columns), ['Close']
)

columns.extend(['Date'])

start_date = st.date_input('Start date', value=df['Date'].min())
end_date = st.date_input('End date', value=df['Date'].max())

data = df[columns][(df['Date'].dt.date>=start_date) & (df['Date'].dt.date<=end_date)]

st.write(data)

st.subheader(f'Line chart of {ticker_symbol.upper()} daily stock price')
chart = st.line_chart(data.set_index('Date'))

if st.checkbox('Show summaries'):
    st.subheader('Summaries:')
    st.write(data.describe())

    week_df = data.groupby(data['Date'].dt.day_name()).mean()

    traces = [go.Bar(
        x = week_df.index,
        y = data[col],
        name = col,
        marker = dict(
            line = dict(
                color = 'rgb(0, 0, 0)',
                width = 2
            )
        )
    ) for col in data.drop(['Date'], axis=1).columns]

    layout = go.Layout(
        title = 'Stockprice over days',
        xaxis = dict(
            title = 'Weekday',
        ),
        yaxis = dict(
            title = 'Average Price'
        )
    )

    fig = go.Figure(data=traces, layout=layout)

    st.plotly_chart(fig)
