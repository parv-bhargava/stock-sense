from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from stock import get_stock_data, create_sequences, preprocess_data, reverse_preprocess_data

model = load_model('best_model.h5')
scaler = MinMaxScaler()


def predict_price(days, stock_name):
    """
    Predicts the stock price for the next number of days using the trained LSTM model.
    :param days: Number of days to predict the stock price for.
    :param stock_name: The ticker symbol of the stock.
    :returns: A numpy array of the predicted stock prices and the stock data.
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

    return future_predictions, stock_data


def create_graph(predictions):
    """
    Generates a graph from the prediction data and saves it as an image.
    :param predictions: Array of predicted stock prices.
    :return: returns Path to the saved graph image.
    """
    # Generating the graph
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=6)
    plt.title('Stock Price Predictions')
    plt.xlabel('Days')
    plt.ylabel('Predicted Price')
    plt.grid(True)

    # Saving the graph as an image
    graph_image_path = 'static/data/stock_predictions_graph.png'
    plt.savefig(graph_image_path)

    return graph_image_path

# if __name__ == '__main__':
#     predictions = predict_price(30, 'MSFT')
#     print(predictions)
#     create_graph(predictions)
