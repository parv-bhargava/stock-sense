import pandas as pd
from flask import Flask, request, render_template
from predict import predict_price, create_graph

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
        days = int(request.form['days'])

        # Make predictions using the model.
        future_predictions, stock_data = predict_price(days, stock_name)

        # Get the next day's date.
        next_day = stock_data.index[-1] + pd.DateOffset(days=1)

        # Add the predictions and the corresponding dates to a dictionary.
        output = {}
        output['future_predictions'] = future_predictions.flatten().tolist()
        output['dates'] = pd.date_range(start=next_day, periods=days + 1).strftime('%Y-%m-%d').tolist()
        last_day_value = output['future_predictions'][-1]
        graph_image_path = create_graph(future_predictions)
        return render_template('index.html',
                               prediction=f'Price after {days} days is {float(last_day_value).__round__(2)}',
                               graph_path=graph_image_path)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
