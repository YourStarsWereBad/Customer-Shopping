import streamlit as st
import pandas as pd
import joblib
from prophet.plot import plot_plotly
from prophet.diagnostics import performance_metrics
from prophet import Prophet
from datetime import datetime
import plotly.graph_objs as go

# --- Load data and model ---
@st.cache_data
def load_model():
    return joblib.load('prophet_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('monthly_sales.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    return df

model = load_model()
df = load_data()

# --- App Layout ---
st.title("📈 Monthly Sales Forecast")

# User input: how many months to forecast
n_months = st.slider("Months to forecast into future", 1, 24, 6)

# Forecast
future = model.make_future_dataframe(periods=n_months, freq='M')
forecast = model.predict(future)

# Merge actuals with forecast for display
forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
latest_date = df['ds'].max()
forecast_display = forecast_display[forecast_display['ds'] > latest_date]

# Show forecast table
st.subheader("📊 Forecasted Sales")
st.dataframe(forecast_display.set_index('ds').round(2))

# Plot
st.subheader("📉 Forecast Visualization")
plot_fig = plot_plotly(model, forecast)
st.plotly_chart(plot_fig)

# Optional: actual vs predicted on training set
if st.checkbox("Show actual vs predicted on training data"):
    actual_vs_pred = df.merge(forecast[['ds', 'yhat']], on='ds', how='left')
    st.line_chart(actual_vs_pred.set_index('ds')[['y', 'yhat']])
