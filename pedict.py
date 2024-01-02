from tensorflow.keras.models import load_model
from stock import get_stock_data , create_sequences, preprocess_data ,reverse_preprocess_data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
stock_data = get_stock_data(30, 'AAPL')
dfa, _ = create_sequences(stock_data, 1)
ed = preprocess_data(dfa,scaler)
model = load_model('best_model.h5')

future_predictions_scaled = model.predict(ed)
future_predictions = reverse_preprocess_data(future_predictions_scaled, scaler)
print(future_predictions)