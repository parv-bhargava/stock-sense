import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
def get_stock_data(days, stock_name):
    """
    Fetches historical close prices for a given stock over a specified number of days.

    :param days: Number of days to fetch data for.
    :param stock_name: The ticker symbol of the stock.
    :return: A pandas DataFrame with date and close price.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    # Fetch data from Yahoo Finance API
    df = yf.download(stock_name, start=start_date, end=end_date)

    # Keeping only the 'Close' column for now
    df = df[['Close']]

    return df

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


def preprocess_data(df):
    """
    Preprocesses the data by scaling the data using the MinMaxScaler.

    :param df: The DataFrame to preprocess.
    :return: The preprocessed DataFrame.
    """
    scaler = MinMaxScaler()
    return scaler.transform(df.reshape(-1, 1)).reshape(df.shape)

if __name__ == '__main__':
    stock_data = get_stock_data(30, 'AAPL')  # Fetches the last 30 days of Apple's stock
    dfa,_ = create_sequences(stock_data, 7)
    ed = preprocess_data(dfa)
    print(ed)