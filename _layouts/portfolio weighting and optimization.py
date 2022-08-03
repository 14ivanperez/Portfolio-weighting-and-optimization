import pandas as pd
import time
import yfinance as yf
import matplotlib.pyplot as plt #for graphs
import numpy as np #for math operations
import seaborn as sns #for visualisation
import pandas_datareader.data as web #for reading financial data
import matplotlib.ticker as ticker #to change graph axes
from datetime import datetime #date manipulation
import pypfopt as pypfopt
from pandas_datareader import data

# Download the daily close prices since 2017 from the 10 highest market cap SP500 companies in 2022
tickers = ['AAPL','MSFT', 'AMZN', 'TSLA', 'GOOGL', 'GOOG', 'BRK-A', 'UNH', 'JNJ', 'NVDA']
data = pd.DataFrame(yf.download(tickers, '2017-01-01', progress=False)['Close'])

#heatmap of the stocks correlation
returns = data.pct_change()[1:]
sns.heatmap(returns.corr(), annot=True, cmap='OrRd')
plt.show()

#calculate daily portfolio returns with a weighted portfolio
returns = data.pct_change()
weights = np.full((10, 1), 0.1, dtype=float)

portfolio = returns.dot(weights)
portfolio_annual_return = portfolio.mean()*250

#COV,VAR,STD 
covariance_matrix = returns.cov()*250 
portfolio_var = np.dot(weights.T,np.dot(covariance_matrix,weights))
portfolio_std = np.sqrt(portfolio_var)

#RISK-TRADE OFF
df=pd.concat([returns.mean()*250,np.sqrt(250)*returns.std()],axis=1)
df.columns = ['returns', 'std']
ax = sns.scatterplot(x=df['std'],y=df['returns'],marker="o", color='#a63287')
for line in range(0,df.shape[0]):
     ax.text(df.iloc[line,1], df.iloc[line,0], df.index[line])
#graph
plt.xlabel('Standart Deviation',size=20)
plt.title('Return and Risk Tradeoff',size=20)
plt.ylabel('Annual Return',size=20)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()

# CAGR portfolio
cagrportfolio = ((1+portfolio).cumprod().fillna(1))**(250/len(portfolio))

# DRAWDOWN portfolio
ddcgr = (cagrportfolio/cagrportfolio.cummax()-1)
ddcgr.plot()
plt.show()

# DRAWDOWN of each stock
dd = (data/data.cummax()-1)
mdd = dd.min()
dd.plot()
plt.show()

#Optimizing portfolio with minimun volatility (std)

#We need an expected value to solve the optimization problem, we use CAPM
import statistics
from pandas_datareader import data
from pulp import *
from pypfopt import expected_returns
from pypfopt import EfficientFrontier

today = datetime.today().strftime('%Y-%m-%d')
df_prices = pd.DataFrame()
assets = ['AAPL','MSFT', 'AMZN', 'TSLA', 'GOOGL', 'GOOG', 'BRK-A', 'UNH', 'JNJ', 'NVDA','^GSPC']
def dataYahoo(dataframe,asset_list,start,finish):
    for i in asset_list:
        dataframe[i] = data.DataReader(i,data_source='yahoo',start= start , end=finish)["Close"]
    return dataframe
df = dataYahoo(df_prices,assets,'2017-01-01',today)
df

#we need log returns
df = np.log(df).diff()
df = df.dropna()

#Separate stock prices from benchmark
df_assets =  df.loc[:, df.columns != '^GSPC']
df_assets
df_benchmark1 =  df.loc[:, df.columns == '^GSPC']

covariance_matrix = df_assets.cov()*250
returns1 = expected_returns.capm_return(df_assets, market_prices = df_benchmark1, returns_data= True, risk_free_rate=0.07/100, frequency=250)
returns1
ef = EfficientFrontier(returns1, covariance_matrix, weight_bounds=(-1,1))
weights = ef.min_volatility()
cleaned_weights = ef.clean_weights() 
print(cleaned_weights) 
ef.portfolio_performance(verbose=True)



