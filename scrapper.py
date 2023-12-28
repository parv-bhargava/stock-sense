import yfinance as yf
import pandas as pd
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period="1mo")

def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    return stock.info

def get_stock_recommendations(ticker):
    stock = yf.Ticker(ticker)
    return stock.recommendations
def get_stock_calendar(ticker):
    stock = yf.Ticker(ticker)
    return stock.calendar

def get_stock_dividends(ticker):
    stock = yf.Ticker(ticker)
    return stock.dividends

data = get_stock_data("AAPL")
pd.DataFrame(data).to_csv("AAPL.csv")