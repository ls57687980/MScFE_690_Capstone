import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import datetime
import warnings

import talib
from hmmlearn.hmm import GaussianHMM

import quantstats as qs

import warnings
warnings.filterwarnings("ignore")

class StrategyPerformance():
    stg_df : pd.DataFrame = None
    position_tbl : pd.DataFrame = None

    stg_name : str = None
    stg_type : str = None
    stg_period : str = None
    numOfTrade : int = 0
    winTrade : int = 0
    loseTrade : int = 0
    winRate : float = 0
    avgPnL : float = 0
    profitFactor : float = 0
    avgBarsHeld : float = 0
    CAGR : float = 0
    vol : float = 0
    maxDD : float = 0
    sharpe : float = 0

    rf : float = None
    timeToYear = {'1m': 243*252, '5m': np.ceil(243*252/5), '15m': np.ceil(243*252/15),
                  '30m': np.ceil(243*252/30), '1H': 5*252, '1D':252, 'M':12, 'Q':4, 'Y':1}

    def __init__(self, dataFrame, name, stg_type, period = "1D", riskFree = 0, pnl_exist = False, pnl_col='PnL'):
        self.stg_df = dataFrame.copy()
        self.stg_period = period
        self.rf = riskFree
        self.stg_name = name
        self.stg_type = stg_type
        if pnl_exist == False:
            self.stg_df['PnL'] = self.stg_df['Position'] * self.stg_df['Returns']
        else:
            self.stg_df.rename(columns={pnl_col: 'PnL'}, inplace=True)

        self.position_summary()
        self.profit_factor()
        self.max_dd()
        self.volatility()
        self.CAGR()
        self.sharpe_ratio()


    def position_summary(self):
        data = self.stg_df.copy()
        data.fillna(0, inplace=True)

        cols = ['Open Date', 'Open Price', 'Close Date', 'Close Price', 'Type', 'Expired']
        self.position_tbl = pd.DataFrame(columns=cols)
        total_bar = 0

        if self.stg_type == 'Daily':
            tmp_df = data[data.Position != 0]

            self.position_tbl['Open Date'] = tmp_df.index.values
            self.position_tbl['Close Date'] = tmp_df.index.values
            self.position_tbl['Open Price'] = tmp_df['Open'].values
            self.position_tbl['Close Price'] = tmp_df['Close'].values
            self.position_tbl['Type'] = tmp_df['Position'].values
            self.position_tbl['Expired'] = False

            total_bar = len(self.position_tbl)
        else:
            pre_pos = 0
            cur_pos = 0
            num = 0
            open_bar = 0

            for bar in range(len(self.stg_df)):
                idx = data.index[bar]
                cur_date = idx.date()
                cur_date_str = cur_date.strftime("%d/%m/%Y")
                cur_time = idx.time()
                try:
                    pre_pos = data.iloc[bar-1]['Position']
                    cur_pos = data.iloc[bar]['Position']
                    if cur_pos == pre_pos:
                        continue

                    # Open Position
                    if (cur_pos != 0) and (pre_pos != cur_pos):
                        open_date = idx
                        open_price = data.iloc[bar]['Open']

                        close_date = None
                        close_price = np.nan
                        expired = False
                        # Type 1 is long position, type -1 is short position
                        pos_type =  data.iloc[bar]['Position']
                        new_post = pd.DataFrame([[open_date, open_price, close_date, close_price, pos_type, expired]], index=[num], columns=cols)
                        self.position_tbl = pd.concat([self.position_tbl, new_post])
                        open_bar = bar

                    # Close position
                    if (pre_pos != 0) and (cur_pos != pre_pos):
                        close_date = idx
                        close_price = data.loc[idx, 'Close']
                        self.position_tbl.loc[num, 'Close Date'] = close_date
                        self.position_tbl.loc[num, 'Close Price'] = close_price
                        """
                        eod_time = datetime.time(14, 45, 59)
                        if (cur_date_str in expire_date) and (cur_time == eod_time):
                            self.position_tbl.loc[num, 'Expired'] = True
                        """
                        num = num + 1
                        total_bar += bar - open_bar
                        open_bar = 0

                except Exception as e:
                    print(e)

        try:
            self.position_tbl['Result'] = (self.position_tbl['Close Price'] - self.position_tbl['Open Price']) * self.position_tbl['Type']
            self.position_tbl['Returns'] = self.position_tbl['Result'] / self.position_tbl['Open Price']

            self.winTrade = self.position_tbl[self.position_tbl['Result'] > 0]['Open Date'].count()
            self.loseTrade = self.position_tbl[self.position_tbl['Result'] <= 0]['Open Date'].count()
            self.numOfTrade = len(self.position_tbl)
            self.winRate = self.winTrade / self.numOfTrade
            self.avgPnL = self.position_tbl['Returns'].sum() / self.numOfTrade
            self.avgBarsHeld = total_bar / self.numOfTrade
        except Exception as e:
            print(e)

    def profit_factor(self):
        df = self.position_tbl.copy()
        try:
            loseRate = self.loseTrade / self.numOfTrade
            avgWinReturns = df[df['Result'] > 0]['Returns'].sum() / self.winTrade
            avgLoseReturns = df[df['Result'] <= 0]['Returns'].sum() / self.loseTrade
            self.profitFactor = (avgWinReturns * self.winRate) / (abs(avgLoseReturns) * loseRate)
        except Exception as e:
            print(e)

    def max_dd(self):
        self.stg_df['Cum Returns'] = (1 + self.stg_df['PnL']).cumprod()
        self.stg_df['Cum Max'] = self.stg_df['Cum Returns'].cummax()
        self.stg_df['Drawdown'] = self.stg_df['Cum Max'] - self.stg_df['Cum Returns']
        self.stg_df['Drawdown Pct'] = self.stg_df['Drawdown'] / self.stg_df['Cum Max']
        self.maxDD = self.stg_df['Drawdown Pct'].max()

    def CAGR(self):
        self.stg_df['Cum Returns'] = (1 + self.stg_df['PnL']).cumprod()
        n = len(self.stg_df) / self.timeToYear[self.stg_period]
        self.CAGR = self.stg_df['Cum Returns'].tolist()[-1]**(1/n) - 1

    def volatility(self):
        self.vol = self.stg_df['PnL'].std() * np.sqrt(self.timeToYear[self.stg_period])

    def sharpe_ratio(self):
        self.sharpe = (self.CAGR - self.rf) / self.vol

    def strategy_performance(self):
        print(f'------------------------{self.stg_name} Performance--------------------')
        print(f'Number of Trades: {self.numOfTrade}')
        print(f'Win Trades: {self.winTrade}')
        print(f'Lose Trades: {self.loseTrade}')
        print(f'Win Rate: {self.winRate * 100} %')
        print(f'Average Bars Held: {self.avgBarsHeld}')
        print(f'Strategy Cumulative Annual Growth Rate: {self.CAGR * 100} %')
        print(f'Average Return Per trade: {self.avgPnL * 100} %')
        print(f'Profit Factor: {self.profitFactor}')
        print(f'Strategy Volatility: {self.vol}')
        print(f'Strategy Sharpe Ratio: {self.sharpe}')
        print(f'Stragegy Max Drawdown: {self.maxDD * 100} %')
        print('--------------------------------------------------------------')

    def stg_cum_returns_plot(self):
        fig, ax = plt.subplots()
        self.stg_df['Cum Returns'].plot(figsize=(12,8))
        plt.title(f'{self.stg_name} strategy Cummulative Returns')
        plt.ylabel('Cum Return')
        plt.xlabel('Date')
        fig.autofmt_xdate()
        ax.legend([self.stg_name])
        plt.show()

    def stg_cum_returns_benchmark_plot(self):
        fig, ax = plt.subplots()
        plt.plot(self.stg_df['Cum Returns'])
        plt.plot((1 + self.stg_df['Returns']).cumprod())
        plt.plot((1 - self.stg_df['Returns']).cumprod())
        plt.title(f'{self.stg_name} strategy Cummulative Returns')
        plt.ylabel('Cum Return')
        plt.xlabel('Date')
        fig.autofmt_xdate()
        ax.legend([self.stg_name, 'VN30F1M Long Only', 'VN30F1M Short Only'])
        plt.show()

    def performance_report(self, benchmark=None, benchmark_col='Returns'):
        if benchmark == 'LO':
            benchmark_data = self.stg_df[benchmark_col]
        elif benchmark == 'SO':
            benchmark_data = -self.stg_df[benchmark_col]

        qs.reports.full(self.stg_df['PnL'], benchmark=benchmark_data)

    def performance_report_html(self, benchmark=None, benchmark_col='Returns'):
        benchmark_data = None
        folder = 'TestResult/'
        filename =  folder + self.stg_name.replace(' ', '') + benchmark + '_tearsheet.html'

        if benchmark == 'LO':
            benchmark_data = self.stg_df[benchmark_col]
        elif benchmark == 'SO':
            benchmark_data = -self.stg_df[benchmark_col]

        qs.reports.html(self.stg_df['PnL'], benchmark=benchmark_data , output=folder, download_filename=filename, title=self.stg_name)