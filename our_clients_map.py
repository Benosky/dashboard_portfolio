
#Created by Benjamin Umeh
import streamlit as st
import numpy as np
import pandas as pd
# import plotly.plotly as py
# import chart_studio.plotly
import plotly.express as px
from pandas_datareader import data as pdr
import sys
import os
import pycountry_convert as pc
import pydeck as pdk


st.title('Geographical Distribution of Clients')


st.write("""
**This dashboard shows the distribution of our clients across several regions
and broken down to various age brackets. With the possibility of narrowing down to specific continents**
""")

#define function to get continent from country name
def get_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name

@st.cache
def load_data():
    clients_df = pd.read_csv("clients_data1.csv")
    clients_df = clients_df.dropna()
    clients_df['Continent'] = clients_df['Country'].apply(lambda x: get_continent(x))
    return clients_df.dropna()

df = load_data().dropna()
# st.write(df.head())
continents = st.sidebar.multiselect(
    "You Can Choose More Continents", list(df.Continent.unique()), ['North America']
)

# selecting rows based on chosen continents
data = df[df['Continent'].isin(continents)]

age_range = st.sidebar.multiselect(
    "Choose Specific Age Range?", list(df.Age_range.unique()), list(df.Age_range.unique())
)

data = data[data['Age_range'].isin(age_range)]
# st.write(data.head())
map_data = data[["lat", "lon"]]

st.map(map_data)

# midpoint = (np.average(map_data["lat"]), np.average(map_data["lon"]))
#
# st.pydeck_chart(pdk.Deck(
#         map_style='mapbox://styles/mapbox/light-v9',
#         initial_view_state=pdk.ViewState(
#             latitude=midpoint[0],
#             longitude=midpoint[1],
#             zoom=11,
#             pitch=50,
#             ),
#             layers=[
#             pdk.Layer(
#                 'HexagonLayer',
#                 data=map_data,
#                 get_position='[lon, lat]',
#                 radius=200,
#                 elevation_scale=4,
#                 elevation_range=[0, 1000],
#                 pickable=True,
#                 extruded=True,
#             ),
#                 ],
#                 ))


# Set the viewport location


# # Map to show the physical locations of Crime for the selected day.
# midpoint = (np.average(map_data["lat"]), np.average(map_data["lon"]))
#
# st.pydeck_chart(viewport={ "latitude": midpoint[0],
#                             "longitude": midpoint[1],
#                             "zoom": 11, "pitch": 40,},
#                 layers=[{"type": "HexagonLayer",
#                             "data": map_data,
#                             "radius": 80,
#                             "elevationScale": 4,
#                             "elevationRange": [0, 1000],
#                             "pickable": True,
#                             "extruded": True,
#                             }
#                             ],
#                             )
#
#


# columns.extend(['Date'])
#
# start_date = st.date_input('Start date', value=df['Date'].min())
# end_date = st.date_input('End date', value=df['Date'].max())
#
# data = df[columns][(df['Date'].dt.date>=start_date) & (df['Date'].dt.date<=end_date)]

st.write(data)

st.subheader('Clients Age_Range Distribution Chart')

grp_df = data.copy()
grp_df['Count'] = 1
d = pd.DataFrame(grp_df.groupby('Age_range')['Age_range'].count())
d = d.rename(columns={"Age_range": "Age_range_counts"})#.reset_index()

age_range = pd.Series(d.index[:])
count = list(d['Age_range_counts'][:])
age_count = pd.DataFrame(list(zip(age_range, count)),
                           columns=['Age_Range', 'Count'])
fig = px.bar(age_count, x='Age_Range', y='Count', color='Count',
             labels={'Age_Range': 'Clients Age Range', 'Count': 'Count'})
st.plotly_chart(fig)
st.success("     ")

# st.subheader(f'Line chart of {ticker_symbol.upper()} daily stock price')
# chart = st.line_chart(data.set_index('Date'))

if st.checkbox('Show summaries'):
    st.subheader('Summaries:')
    st.write(data.describe())

    # st.subheader('Histogram of Age_Range Distribution')

    # st.bar_chart(list(data.Age_range.values))

    # st.subheader(" Top 10 Crimes ")
    # grp_df = data.copy()
    # grp_df['Count'] = 1
    #
    # d = pd.DataFrame(grp_df.groupby('Age_range')['Age_range'].count())
    #
    # d = d.rename(columns={"Age_range": "Age_range_counts"})#.reset_index()
    #
    #
    # # data['Age_range'].value_counts()
    # # k = pd.DataFrame(grp_data.groupby(['C_Descrip'], sort=False)['C_Address'].count().rename_axis(["Type of Crime"]))
    # age_range = pd.Series(d.index[:])
    # count = list(d['Age_range_counts'][:])
    # age_count = pd.DataFrame(list(zip(age_range, count)),
    #                            columns=['Age_Range', 'Count'])
    # fig = px.bar(age_count, x='Age_Range', y='Count', color='Count',
    #              labels={'Age_Range': 'Clients Age Range', 'Count': 'Count'})
    # st.plotly_chart(fig)
    # st.write('--------------------------------- Clients Age_Range Distribution ---------------------------------')
    # st.success("     ")

    #
    #
    #
    # week_df = data.groupby(data['Date'].dt.day_name()).mean()
    #
    # traces = [go.Bar(
    #     x = week_df.index,
    #     y = data[col],
    #     name = col,
    #     marker = dict(
    #         line = dict(
    #             color = 'rgb(0, 0, 0)',
    #             width = 2
    #         )
    #     )
    # ) for col in data.drop(['Date'], axis=1).columns]
    #
    # layout = go.Layout(
    #     title = 'Stockprice over days',
    #     xaxis = dict(
    #         title = 'Weekday',
    #     ),
    #     yaxis = dict(
    #         title = 'Average Price'
    #     )
    # )
    #
    # fig = go.Figure(data=traces, layout=layout)
    #
    # st.plotly_chart(fig)
