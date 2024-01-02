import pandas as pd
from flask import Flask, request, render_template
from predict import predict
from stock import get_stock_data

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the stock name from the POST request.
        stock_name = 'MSFT'
        # stock_name = request.form['stock_name']
        # Get the number of days from the POST request.
        days = int(request.form['Number of Days'])

        # Get the stock data from Yahoo Finance API.
        stock_data = get_stock_data(days, stock_name)

        # Make predictions using the model.
        future_predictions = predict(days, stock_name)

        # Get the next day's date.
        next_day = stock_data.index[-1] + pd.DateOffset(days=1)

        # Add the predictions and the corresponding dates to a dictionary.
        output = {}
        output['future_predictions'] = future_predictions.flatten().tolist()
        output['dates'] = pd.date_range(start=next_day, periods=days + 1).strftime('%Y-%m-%d').tolist()
        print(output)
        return render_template('index.html', output=output)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
