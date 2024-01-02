from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


def get_stock_data(days, stock_name):
    """
    Fetches historical close prices for a given stock over a specified number of days.

    :param days: Number of days to fetch data for.
    :param stock_name: The ticker symbol of the stock.
    :return: A pandas Series with date as index and close price.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    # Fetch data from Yahoo Finance API
    df = yf.download(stock_name, start=start_date, end=end_date)

    # Convert the 'Close' column to a pandas Series
    close_prices = pd.Series(df['Close'].values, index=df.index)

    return close_prices


# Create sequences for LSTM
def create_sequences(data, sequence_length):
    """
    Creates sequences for the LSTM model with a given sequence length.
    :param data: The data to create sequences from.
    :param sequence_length: The length of the sequences.
    :return: A tuple of the input sequences and the corresponding labels.
    """
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(x), np.array(y)


def preprocess_data(data, scaler):
    """
    Preprocesses the data by scaling the data using the MinMaxScaler.

    :param data: A  numpy array of the input data.
    :return: The preprocessed data as a numpy array.
    """
    # Reshape the data into a 2D array
    data = data.reshape(-1, 1)
    # Fit and transform the data using MinMaxScaler
    scaled_data = scaler.fit_transform(data)

    return scaled_data


def reverse_preprocess_data(scaled_data, scaler):
    """
    Reverses the preprocessing by using the inverse transformation of MinMaxScaler.

    :param scaled_data: The scaled data as a numpy array.
    :param scaler: The MinMaxScaler instance used for scaling.
    :return: The original data as a numpy array.
    """
    original_data = scaler.inverse_transform(scaled_data)
    return original_data

# if __name__ == '__main__':
#     stock_data = get_stock_data(30, 'MSFT')
#     dfa,_ = create_sequences(stock_data, 1)
#     ed = preprocess_data(dfa)
#     print(ed)
