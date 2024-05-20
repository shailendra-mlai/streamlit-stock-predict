import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

# Streamlit App
st.title("Stock Price Prediction App")

# Inputting the Stock symbol for prediction
TICK = st.text_input("Enter Stock symbol for prediction:", "AAPL")

# Fetch the data for the last 5 years from Yahoo Finance
@st.cache
def load_data(ticker):
    ticker_data = yf.Ticker(ticker).history(period="5y")
    ticker_data['Company'] = ticker
    return ticker_data

ticker_data = load_data(TICK)

# Visualize the stock portfolio performance of holding
st.subheader("Stock Prices Over Last 5 Years")
fig, visual_plot = plt.subplots(figsize=(15,8))
visual_plot = sns.lineplot(data=ticker_data, x=ticker_data.index, y=ticker_data["Close"], hue=ticker_data["Company"])
visual_plot.set_title(f"Stock Prices of {TICK}")
visual_plot.set_xlabel('Date', fontsize=16)
visual_plot.set_ylabel('Closing Price (USD)', fontsize=16)
st.pyplot(fig)

st.write("Thank you for your patience. The app is currently retrieving the last 5 years of stock data for the entered symbol, building an LSTM model, and predicting the stock price for the next 30 days. This process may take a few moments.")

# Define functions
def close_prices_only(data):
    return data[['Close']]

def min_max_scale(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    return scaler, scaled_data

def split_and_reshape_data(dataframe, pred_days):
    prediction_days = pred_days
    train_size = int(np.ceil(len(dataframe) * 0.95))
    test_size = len(dataframe) - train_size
    train_data = dataframe[0:int(train_size), :]
    test_data = dataframe[train_size - prediction_days:, :]
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(prediction_days, len(train_data)):
        X_train.append(train_data[i - prediction_days: i, 0])
        y_train.append(train_data[i, 0])
    for i in range(prediction_days, len(test_data)):
        X_test.append(test_data[i - prediction_days: i, 0])
        y_test.append(test_data[i, 0])
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_train, y_train, X_test, y_test

def lstm_model(X_train, y_train, epochs=100, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def predictions_inverse_scaler(scaler, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    return y_pred, y_test

def plot_predictions(y_pred, y_true, ticker_label):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label='Predicted Values')
    plt.title(f'True vs Predicted Values for {ticker_label}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot()

# Get only the close prices for the selected ticker
df_close = close_prices_only(ticker_data)

# Scaling close prices for stock ticker
df_scaler, df_scaled = min_max_scale(df_close)

# Split and reshape data for each stock ticker
X_train, y_train, X_test, y_test = split_and_reshape_data(df_scaled, 30)

# Build the LSTM model
model_lstm = lstm_model(X_train, y_train)

# Make predictions and inverse scale for each stock ticker
y_pred, y_test_inv = predictions_inverse_scaler(df_scaler, model_lstm, X_test, y_test)

# Plot the predictions against the true values
st.subheader("True vs Predicted Stock Prices")
plot_predictions(y_pred, y_test_inv, TICK)

# Prepare data for future prediction
def prepare_future_data(data, prediction_days):
    X_future = data[-prediction_days:].reshape(1, -1, 1)
    return X_future

def predict_future_prices(model, data, future_days):
    predictions = []
    current_batch = data[-future_days:].reshape(1, future_days, 1)
    for _ in range(future_days):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    return np.array(predictions).reshape(-1, 1)

def inverse_transform_predictions(scaler, predictions):
    return scaler.inverse_transform(predictions)

# Number of days to predict into the future
future_days = 30

# Prepare data for future prediction
X_future = prepare_future_data(df_scaled, future_days)

# Predict future prices
future_predictions = predict_future_prices(model_lstm, df_scaled, future_days)

# Inverse transform the predictions
future_predictions_inverse = inverse_transform_predictions(df_scaler, future_predictions)

# Plot the future predictions
st.subheader("Future Stock Price Predictions")
def plot_future_predictions(ticker, future_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(future_pred, color='red', label='Future Predictions')
    plt.title(f'Future Predicted Values for {ticker}')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot()

# Usage example for plotting future predictions for the specified ticker
plot_future_predictions(TICK, future_predictions_inverse)

# %%
