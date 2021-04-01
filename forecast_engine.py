import os
import pandas as pd

import numpy as np
import pandas as pd

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from fbprophet import Prophet
from pyspark.sql import SQLContext
from pyspark import SparkContext

from pyspark.sql.functions import pandas_udf, PandasUDFType, current_date
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from scipy.special import boxcox1p

import numpy
import pandas
import pyarrow
import sklearn #.preprocessing import PowerTransformer, MinMaxScaler
from pyspark.sql.functions import unix_timestamp, from_unixtime, to_timestamp
from pyspark.sql.functions import current_date

import datetime

st.title('Branches Sales Forecasts Dashboard')

@st.cache
def load_data():
    sales_df = pd.read_csv('sales_data.csv')
    return sales_df
# train2 = spark.createDataFrame(stock_df2)
# train2.createOrReplaceTempView('train2')

df = load_data()

branch = st.sidebar.selectbox(
    "Select Branch", list(df.facilities.unique()), ['3A HEALTH']
)

# Select the chosen branch sales data
data = df[df['facilities']== branch]

product = st.sidebar.selectbox(
    "Select Product", list(data.drug_name.unique()), ['ATENOLOL 50MG x28']
)

prod_data = data[data['drug_name']== product]

# def facility_names():
#   facilty_name_list_1 = sorted(list(stock_df.facilities.unique()))
#   facilty_name_list_2 = facilty_name_list_1  #Remember that choices that go into a drop-down list, check boxes, radio buttons, etc. need to be in tuples in Django forms.
#   facilty_name_tuples =  list(zip(facilty_name_list_1, facilty_name_list_2))
#   return facilty_name_tuples

# def facility_drug_combo():
#     raw_dt = pd.read_pickle("stock_data.pkl")
#     # stock_df["facility_drug"] = stock_df.facilities.astype(str).str.cat(stock_df.drug_name.astype(str), sep=' - ')
#     facility_drug_combo_1 = sorted(list(raw_dt.facility_drug.unique()))
#     facility_drug_combo_2 = facility_drug_combo_1  #Remember that choices that go into a drop-down list, check boxes, radio buttons, etc. need to be in tuples in Django forms.
#     facility_drug_combo_tuples =  list(zip(facility_drug_combo_1, facility_drug_combo_2))
#     return facility_drug_combo_tuples

# def facility_drug_names(selected_facility):
#   facility_name = selected_facility #query_params_1['facility_name'] # to be derived from the facility_names() through the form
#   facility_drugs_df =  stock_df[stock_df["facilities"]==facility_name]
#   facility_drug_names_list = sorted(list(facility_drug_df.drug_name.unique()))
#   return facility_drug_name_list

@st.cache
def train_data():
    # structure of the training data set
    train_schema = StructType([
    StructField('sale_week', DateType()),
    StructField('facilities', StringType()),
    StructField('drug_name', StringType()),
    StructField('total_units_sold', IntegerType())
    ])

    spark = SparkSession.builder.appName('StockLevelModel').getOrCreate()

    # Enable Arrow-based columnar data transfers
    spark.conf.get("spark.sql.execution.arrow.enabled", "true")

    # Create a Spark DataFrame from a pandas DataFrame using Arrow
    train = spark.createDataFrame(prod_data) # We will uncomment this during the final deployment

    # make the dataframe queriable as a temporary view
    train.createOrReplaceTempView('train')

    # Scaling the model training and forecasting
    sc = SparkContext.getOrCreate()
    sql_statement = '''
    SELECT
      facilities,
      drug_name,
      CAST(sale_week as date) as ds,
      SUM(total_units_sold) as y
    FROM train
    GROUP BY facilities, drug_name, ds
    ORDER BY facilities, drug_name, ds
    '''
    branch_product_history = (
    spark
      .sql(sql_statement)
      .repartition(sc.defaultParallelism, ['facilities', 'drug_name'])
    ).cache()

    #Convert the ds column to datetimevm
    branch_product_history = branch_product_history.select(
      'facilities', 'drug_name',
      from_unixtime(unix_timestamp('ds', 'yyyy-MM-dd')).alias('ds'), 'y'
      )
    return branch_product_history


trnData = train_data()


@st.cache
def generate_forecast():
    # structure of the forecast result data set
    result_schema =StructType([
    StructField('ds',DateType()),
    StructField('facilities',StringType()),
    StructField('drug_name',StringType()),
    StructField('y',FloatType()),
    StructField('yhat',FloatType()),
    StructField('yhat_upper',FloatType()),
    StructField('yhat_lower',FloatType())
    ])

    # Define the forecasting model
    @pandas_udf( result_schema, PandasUDFType.GROUPED_MAP )
    def forecast_store_item(history_pd ):

        # remove missing values (more likely at day-store-item level)
        history_pd = history_pd.fillna(0)
        # --------------------------------------
        # TRANSFORM THE TIMESERIES
        # --------------------------------------
        #Power transform the data to resolve any inherent heteroscedasticity (non-constant variance)
        scaler = MinMaxScaler()
        transf = scaler.fit_transform(history_pd[['y']].values) #Best for now

        #Replace the untransformed 'y' with the transformed 'y'
        history_pd['y'] = transf
        # --------------------------------------
        # TRAIN MODEL
        # --------------------------------------
        # configure the model
        model = Prophet(
        interval_width=0.95,
        growth='linear',
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
  #     seasonality_mode='multiplicative'
        )

        # train the model
        model.fit( history_pd )
        # --------------------------------------

        # BUILD FORECAST
        # --------------------------------------
        # make predictions
        future_pd = model.make_future_dataframe(
        periods=5,
        freq='W-MON',
        include_history=True
        )
        forecast_pd = model.predict(future_pd)
        # --------------------------------------
        # INVERSE TRANSFORM 'y' AND THE FORECASTS
        # --------------------------------------
        history_pd['y'] = scaler.inverse_transform(history_pd[['y']])
        forecast_pd['yhat'] = scaler.inverse_transform(forecast_pd[['yhat']])
        forecast_pd['yhat_upper'] = scaler.inverse_transform(forecast_pd[['yhat_upper']])
        forecast_pd['yhat_lower'] = scaler.inverse_transform(forecast_pd[['yhat_lower']])
        # --------------------------------------
        # ASSEMBLE EXPECTED RESULT SET
        # --------------------------------------
        # get relevant fields from forecast
        f_pd = forecast_pd[['ds','yhat', 'yhat_upper', 'yhat_lower']].set_index('ds')

        # get relevant fields from history
        h_pd = history_pd[['ds','facilities','drug_name', 'y']].set_index('ds')#,'y']].set_index('ds')

        # join history and forecast
        results_pd = f_pd.join( h_pd, how='left' )
        results_pd.reset_index(level=0, inplace=True)

        # get store & item from incoming data set (that is, each store-item combination)
        results_pd['facilities'] = history_pd['facilities'].iloc[0]
        results_pd['drug_name'] = history_pd['drug_name'].iloc[0]
        # --------------------------------------
        results_pd = results_pd[['ds', 'facilities', 'drug_name', 'y', 'yhat', 'yhat_upper', 'yhat_lower']].tail(2)
        # return expected dataset
        return (results_pd, model, forecast_pd)

    # store_item_df = train_data(rawDf, facilityDrug)

    results = (
    trnData.groupBy('facilities', 'drug_name')
      .apply(forecast_store_item[0])
      .withColumn('training_date', current_date())
      )

    results.createOrReplaceTempView('results')

    # forecast_result = results.drop('y')

    # Convert the Spark DataFrame back to a Pandas DataFrame using Arrow
    spark_forecast_pdf = results.select("*").toPandas()

    #Convert all negative forecasts to zero
    spark_forecast_pdf[['yhat','yhat_upper','yhat_lower']] = spark_forecast_pdf[['yhat','yhat_upper','yhat_lower']].clip(lower=0)

    #Replace all zeroes with nan to prevent us from converting it to 1 later
    spark_forecast_pdf = spark_forecast_pdf.replace(0.0, np.nan)

    #Convert all forecasts between 0 and 1 to 1
    spark_forecast_pdf[['yhat','yhat_upper','yhat_lower']] = spark_forecast_pdf[['yhat','yhat_upper','yhat_lower']].round(0).clip(lower=1)#

    #Convert all nan forecasts back to zero and convert the rest to integer
    spark_forecast_pdf[['Average_Forecast','Upper_Forecast','Lower_Forecast']] = spark_forecast_pdf[['yhat','yhat_upper','yhat_lower']].fillna(0).astype(int)

return (spark_forecast_pdf, forecast_store_item[1], forecast_store_item[2])


forecasted_demand = generate_forecast()

st.write(forecasted_demand[0])

st.write("Forecast components")
fig2 = generate_forecast()[1].plot_components(forecasted_demand[2])
st.write(fig2)
