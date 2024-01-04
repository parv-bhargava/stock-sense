# StockSense
<p align="center">
<img src="static/logo.png" width="400" height="350" title="StockSense" alt="StockSense Logo">
</p>

We use sentiment analysis to gauge the market mood and make predictions based on time series data.

**Curretnt Status:** Have used LSTM to predict the stock price and initially deployed on AWS.

**Future Work:** Include sentiment analysis part and add multiple tickers (Stocks).

## Data Collection

The data has been collected from **Yahoo Finance**. This data is the whole historic data that dates back to the start of the stock. The data consists of some core features which are:

  1. **Date**
     - The date associated with the stock market data.
  
  2. **Open**
     - The opening price of the stock on a given trading day.
  
  3. **High**
     - The highest price reached by the stock during the trading day.
  
  4. **Low**
     - The lowest price reached by the stock during the trading day.
  
  5. **Close**
     - The closing price of the stock on a given trading day.
  
  6. **Volume**
     - The total number of shares traded for the stock on a particular day.
  
  7. **Dividends**
     - Payments made by a company to its shareholders, typically in the form of cash or additional shares.
  
  8. **Stock Splits**
     - A corporate action in which a company divides its existing shares into multiple shares. This does not impact the overall value of the investment but increases the number of shares outstanding.


## Data preprocessing:

In our model, we employed Linear Imputation as a technique to address missing values during weekends. This method involves estimating the values for the weekends by considering the data from both preceding and subsequent dates. The imputed data is then organized into sequences of 90 days, which serve as input for the LSTM model. Subsequently, we utilized MinMaxScaler to scale the data before feeding it into the model for training.

## Data Modelling:

The preprocessed data undergoes input to an LSTM network comprised of four layers with node counts of 200, 150, 100, and 10 in sequential order. To mitigate overfitting concerns, a dropout layer is incorporated. For optimization, the Adam optimizer is employed with a learning rate of 0.0001, chosen to prevent convergence to local minima and encourage reaching the global minimum. The model yields a Mean Absolute Error of **0.96** and a Root Mean Squared Error of **1.15**.

Here is the forecasting of Apple stock that we have conducted.

<p align="center">
<img src="static/Forcasting.png" width="400" height="350" title="Apple stock" alt="Forcasting of apple stock">
</p>

## Deployment:

The Stock Sense platform is launched as a website with a user-friendly design. Utilizing Flask and the accessibility features of AWS, our platform presents a responsive interface. This facilitates seamless navigation through input forms, allowing users to obtain precise predictions for their final stock price, accompanied by a graphical representation.
