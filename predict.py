from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from stock import get_stock_data, create_sequences, preprocess_data, reverse_preprocess_data

model = load_model('best_model.h5')
scaler = MinMaxScaler()

def predict(days, stock_name):
    """
    Predicts the stock price for the next number of days using the trained LSTM model.
    :param days: Number of days to predict the stock price for.
    :param stock_name: The ticker symbol of the stock.
    :return: A numpy array of the predicted stock prices.
    """

    # Get the stock data from Yahoo Finance API.
    stock_data = get_stock_data(days, stock_name)

    # Get the stock data from Yahoo Finance API.
    input_sequences, _ = create_sequences(stock_data, 1)

    # Preprocess the data.
    input_data = preprocess_data(input_sequences, scaler)

    # Make predictions using the model.
    future_predictions_scaled = model.predict(input_data)

    # Reverse the preprocessing.
    future_predictions = reverse_preprocess_data(future_predictions_scaled, scaler)

    return future_predictions
