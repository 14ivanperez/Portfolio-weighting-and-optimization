import  yahoo_fin.stock_info as si

# EXAMPLE portfolio

tickers = ['FB','AMZN', 'SAN.MC', 'SLV']
data = pd.DataFrame(yf.download(tickers, '2010-01-01', progress=False)['Close'])

returns = data.pct_change()
weights = [0.4, 0.2, 0.2, 0.2]

portfolio = returns.dot(weights)

# CAGR portfolio
cagrportfolio = ((1+portfolio).cumprod().fillna(1))**(252/len(portfolio))

## DRAWDOWN portfolio
ddcgr = (cagrportfolio/cagrportfolio.cummax()-1)
ddcgr.plot()
plt.show()

# DRAWDOWN of each stock
dd = (data/data.cummax()-1)
mdd = dd.min()
dd.plot()
plt.show()


#ETFs US portoflio

import yfinance as yf
import investpy            
import yahoo_fin.stock_info as si
import pandas as pd
import matplotlib.pyplot as plt

etfs = list(investpy.etfs.etfs_as_df()['symbol'])
etfs
etfs = etfs['symbol'][etfs['country'] == 'united states']


tickers = ['FB','AMZN', 'PEZ', 'DOG']
data = pd.DataFrame(yf.download(tickers, '2010-01-01', progress=False)['Close']).dropna()

returns = data.pct_change()
weights = [0.4, 0.2, 0.2, 0.2] 
            
portfolio1 = returns.dot(weights)        
cagrportfolio1 = ((1+portfolio1).cumprod().fillna(1))**252/len(portfolio1)
cagrportfolio1.plot(title = 'CAGR portfolio', xlabel= 'Dates', ylabel='Prices', color='r', lw=1.7)
plt.show

ddcgr = (cagrportfolio1/cagrportfolio1.cummax()-1)
ddcgr.plot(title = 'CAGR portfolio', progress=False, xlabel= 'Dates', ylabel='Prices', color='r', lw=1.7)
plt.show()
mdd = dd.min()

### STOCKS PORTFOLIO
import random

data = investpy.stocks.stocks_as_df()
data.columns
spain = ['SAN.MC', 'BBVA.MC', 'IBE.MC', 'ITX.MC', 'ACS.MC', 'COL.MC','NUMV', 'IAGG']
data = yf.download(spain, '2021-01-01', progress=False)['Close'].dropna()
etf = investpy.etfs.etfs_as_df()
etf = etf['symbol'][etf['country'] == 'united states']
etf = random.sample(list(etf),k=2)
weights = [0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2]
returns = data.pct_change()
portfolio2 = returns.dot(weights)
cagrportfolio2 = ((1+portfolio2).cumprod().fillna(1))**(252/len(portfolio2))

fig, (ax1,ax2) = plt.subplots(2, figsize=(22,12))
cagrportfolio2.plot(title="CAGR portfolio 60/40", xlabel="Dates", ylabel="Prices", color='r', lw=1.2, ax=ax1)
ddcgr = (cagrportfolio2/cagrportfolio2.cummax()-1)
mdd = ddcgr.min()
ax1.get_xaxis().set_visible(False)
ddcgr.plot(title="DRAWDOWN", xlabel="Dates", ylabel="Downs", color='b',linestyle="--", lw=1.2, ax=ax2)
plt.show()

