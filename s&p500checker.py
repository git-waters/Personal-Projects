import pickle
import bs4 as bs
import requests
import pandas as pd
import datetime as dt
import os
import seaborn as sns
import numpy as np
import pandas_datareader as web
import matplotlib.pyplot as plt
from matplotlib import style

# Note, had to use yfinance to fix a Date bug.

import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()
plt.style.use('ggplot')


#def save_sp500_tickers():
    #df_companies = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    #tickers = df_companies[0]['Symbol']
    #with open("sp500tickers.pickle", "wb") as f:
        #pickle.dump(tickers, f)
    #print(tickers)
    #return tickers


def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'id': 'constituents'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.find_all('td')[0].text.replace('\n', '')
        if "." in ticker:
            ticker = ticker.replace('.', '-')
            print('ticker replaced to', ticker)
        tickers.append(ticker)
        with open("s&p500tickers.pickle", "wb") as f:
            pickle.dump(tickers, f)

        return tickers


def get_pricing_data(reload_sp500=False):
    print("Please enter the year, month and day for the beginning and end of the search")
    starty = int(input("Start Year?"))
    startm = int(input("Start Month?"))
    startd = int(input("Start Day?"))
    endy = int(input("End Year?"))
    endm = int(input("End Month?"))
    endd = int(input("End Day?"))
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("s&p500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(starty, startm, startd)
    end = dt.datetime(endy, endm, endd)

    for ticker in tickers:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker, start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


def compile_stock_data():
    with open("s&p500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
        print("stock data loaded.")

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        # will get error if 1 is 0 which is default value
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        # make df compatible
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        # divide count by 10, remainder is 0, we print the count
        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv('s&p500_joined_closes.csv')


# Function to see the correlation between stock returns
# Stock returns tend to follow normal distribution and prices do not


def visualize_data():
    df = pd.read_csv('s&p500_joined_closes.csv')
    # df['AAPL'].plot()
    # plt.show()
    # Make a correlation between stock returns
    # Produces a heat map of the correlation between stock returns
    df.set_index('Date', inplace=True)
    df_corr = df.pct_change().corr()
    print(df_corr.head())
    sns.heatmap(df_corr, annot=False, cmap='RdYlGn')
    plt.show()


save_sp500_tickers()
get_pricing_data()
compile_stock_data()
visualize_data()

