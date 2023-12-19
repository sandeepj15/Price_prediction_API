# bitcoin_prediction.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import timedelta, date
import plotly.express as px
import json


# Function to get historical Bitcoin price data using Yahoo Finance
def get_bitcoin_data():
    btc_data = yf.download('BTC-USD', start='2012-01-01', end= date.today())
    return btc_data

# Function to preprocess the data
def preprocess_data(data):
    data['Date'] = pd.to_datetime(data.index)  # Convert 'Date' to Timestamp
    data.reset_index(drop=True, inplace=True)
    return data

# Function to create features
def create_features(data):
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    return data


# Function to train a Random Forest model
def train_model(data):
    X = data[['Day', 'Month', 'Year']]
    y = data['Close']

    # Filter data for training till the end of 2022
    train_data = data[data['Date'] <= pd.to_datetime('2022-12-31')]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

    model = RandomForestRegressor(n_estimators=100, random_state=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    return model

# Function to extend the data by predicting future dates
def extend_data(data, model, start_date, end_date):
    future_dates = pd.date_range(start=start_date, end=end_date)

    future_data = pd.DataFrame({'Date': future_dates})
    future_data = create_features(future_data)

    future_data['Predictions'] = model.predict(future_data[['Day', 'Month', 'Year']])

    data_with_future = pd.concat([data, future_data], ignore_index=True)

    return data_with_future


# Function to visualize predictions using Plotly Express
def visualize_predictions(data):
    fig = px.line(data, x='Date', y=['Close', 'Predictions'], labels={'value': 'Price (USD)'})
    fig.update_layout(title_text='Bitcoin Price Prediction',
                      xaxis_title='Date',
                      yaxis_title='Price (USD)')

    # Convert NumPy arrays to lists before serializing
    fig_json = json.loads(fig.to_json())

    return fig_json

# Function to integrate Bitcoin price prediction into FastAPI
def integrate_prediction():
    # Get Bitcoin data
    bitcoin_data = get_bitcoin_data()

    # Preprocess the data
    preprocessed_data = preprocess_data(bitcoin_data)

    # Create features
    featured_data = create_features(preprocessed_data)

    # Train the Random Forest model
    model = train_model(featured_data)

    # Extend data and visualize predictions using Plotly
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2024-03-01')
    extended_data = extend_data(featured_data, model, start_date, end_date)

    # Serialize the Plotly figure to JSON
    fig_json = visualize_predictions(extended_data)

    return fig_json