from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from stock import get_stock_data
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# Load your trained model
model = load_model('best_model.h5')


@app.route('/predict', methods=['POST'])
def predict():
    pass

if __name__ == '__main__':
    app.run()
