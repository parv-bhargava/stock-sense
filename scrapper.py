import yfinance as yf
import pandas as pd
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period='max', interval='3mo', prepost=True, actions=True, auto_adjust=True, back_adjust=False)

def multi_stock_data(tickers, start, end):
    """
    :param tickers: list of tickers
    :param start: start date
    :param end: end date

    :return: dataframe of stock data
    """
    data = yf.download(tickers, start=start, end=end)
    return pd.DataFrame(data)
# def get_stock_info(ticker):
#     stock = yf.Ticker(ticker)
#     return stock.info
#
# def get_stock_calendar(ticker):
#     stock = yf.Ticker(ticker)
#     return stock.calendar
#
# def get_stock_dividends(ticker):
#     stock = yf.Ticker(ticker)
#     return stock.dividends

# apple = get_stock_data("AAPL")

#Apple stock data
# AAPL: Invalid input - interval=3m is not supported. Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
# pd.DataFrame(get_stock_data("AAPL")).to_csv("data/apple_quaterly_max.csv")

#Multi Stock Data
# multi_stock_data(["AAPL", "MSFT", "GOOG"], "2019-01-01", "2020-01-01").to_csv("data/apple_msft_goog_2019.csv")
