import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import datetime
import warnings

import talib
from hmmlearn.hmm import GaussianHMM

import warnings
warnings.filterwarnings("ignore")

class MarketRegimes():
    """
    A Class used to represent an trend and volatility regime of data series
    """
    # Trend level
    StrongBull = 2
    Bull = 1
    Neutral = 0
    Bear = -1
    StrongBear = -2

    # Volatility level
    VeryVolatily = 3
    Volatility = 2
    Normal = 1
    Quiet = 0

    hmm_model = None
    data : pd.DataFrame = None

    def __init__(self, dataFrame):
        self.data = dataFrame.copy()

    def market_trend_regimes_bb(self, LongPeriod1=200, LongPeriod2=50):
        self.data['UpperBBLong1'] = self.data['Close'].rolling(LongPeriod1).mean() + \
                            0.5 * self.data['Close'].rolling(LongPeriod1).std()
        self.data['UpperBBLong2'] = self.data['Close'].rolling(LongPeriod2).mean() + \
                            0.5 * self.data['Close'].rolling(LongPeriod2).std()
        self.data['LowerBBLong1'] = self.data['Close'].rolling(LongPeriod1).mean() - \
                            0.5 * self.data['Close'].rolling(LongPeriod1).std()
        self.data['LowerBBLong2'] = self.data['Close'].rolling(LongPeriod2).mean() - \
                            0.5 * self.data['Close'].rolling(LongPeriod2).std()

        self.data['LongTrend'] = np.nan

        # Long Term Trend
        self.data['LongTrend'] = np.where(
                (self.data.Close > self.data.UpperBBLong1),
                self.Bull,
                self.data.LongTrend
            )

        self.data['LongTrend'] = np.where(
                (self.data.Close > self.data.UpperBBLong1) & (self.data.Close > self.data.UpperBBLong2),
                self.StrongBull,
                self.data.LongTrend
            )

        self.data['LongTrend'] = np.where(
                (self.data.Close < self.data.UpperBBLong1) & (self.data.Close > self.data.LowerBBLong1),
                self.Neutral,
                self.data.LongTrend
            )

        self.data['LongTrend'] = np.where(
                (self.data.Close < self.data.LowerBBLong1),
                self.Bear,
                self.data.LongTrend
            )

        self.data['LongTrend'] = np.where(
                (self.data.Close < self.data.LowerBBLong1) & (self.data.Close < self.data.LowerBBLong2),
                self.StrongBear,
                self.data.LongTrend
            )

        bb = ['UpperBBLong1', 'UpperBBLong2', 'LowerBBLong1', 'LowerBBLong2']
        self.data.drop(columns=bb, inplace=True)


    def market_trend_regimes_ROC(self, LongPeriod=63, threshold1=7, threshold2=4, threshold3=-2, threshold4=-5):
        self.data['ROCLong'] = (self.data.Close / self.data.Close.shift(LongPeriod) - 1) * 100
        self.data['ROCLongTrend'] = np.nan
        self.data['ROCLongTrend'] = np.where(
                self.data.ROCLong > threshold1,
                self.StrongBull,
                self.data.ROCLongTrend
                )
        self.data['ROCLongTrend'] = np.where(
                (self.data.ROCLong <= threshold1) & (self.data.ROCLong > threshold2),
                self.Bull,
                self.data.ROCLongTrend
                )
        self.data['ROCLongTrend'] = np.where(
                (self.data.ROCLong <= threshold2) & (self.data.ROCLong > threshold3),
                self.Neutral,
                self.data.ROCLongTrend
                )
        self.data['ROCLongTrend'] = np.where(
                (self.data.ROCLong <= threshold3) & (self.data.ROCLong > threshold4),
                self.Bear,
                self.data.ROCLongTrend
                )
        self.data['ROCLongTrend'] = np.where(
                self.data.ROCLong <= threshold4,
                self.StrongBear,
                self.data.ROCLongTrend
                )
        self.data.drop(columns=['ROCLong'], inplace=True)

    def market_trend_regimes_hmm(self, nState):
        df = self.data.copy()
        df.dropna(inplace=True)

        # Train returns
        X = np.column_stack([df['Returns']])

        # Create the Gaussian Hidden markov Model and fit
        self.hmm_model = GaussianHMM(
            n_components=nState, covariance_type="full", random_state=1, n_iter=1000
        ).fit(X)
        print(f'Converged: {self.hmm_model.monitor_.converged}\t\t Model Score: {self.hmm_model.score(X)}')

        hidden_states = self.hmm_model.predict(X)

        offset = len(self.data) - len(df)
        hidden_states = np.append(hidden_states, offset*[100]) # Add 100 to make array length same data length

        self.data['HMM Trend'] = hidden_states
        self.data['HMM Trend'] = self.data['HMM Trend'].shift(offset)

    def market_volatility_regimes_atrp(self, PERIOD=21, Mean=504, Std=1):
        self.data['ATRP'] = talib.ATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=PERIOD) / self.data['Close'] * 100
        self.data['ATRPThreshold1'] = self.data.ATRP.rolling(Mean).mean() + 3*self.data.ATRP.rolling(Mean).std()
        self.data['ATRPThreshold2'] = self.data.ATRP.rolling(Mean).mean() + self.data.ATRP.rolling(Mean).std()
        self.data['ATRPThreshold3'] = self.data.ATRP.rolling(Mean).mean() - self.data.ATRP.rolling(Mean).std()
        self.data['Volatility'] = np.nan
        self.data['Volatility'] = np.where(
                self.data.ATRP >= self.data.ATRPThreshold1,
                self.VeryVolatily,
                self.data.Volatility
                )
        self.data['Volatility'] = np.where(
                (self.data.ATRP >= self.data.ATRPThreshold2) & (self.data.ATRP < self.data.ATRPThreshold1),
                self.Volatility,
                self.data.Volatility
                )
        self.data['Volatility'] = np.where(
                (self.data.ATRP > self.data.ATRPThreshold3) & (self.data.ATRP < self.data.ATRPThreshold2),
                self.Normal,
                self.data.Volatility
                )
        self.data['Volatility'] = np.where(
                self.data.ATRP <= self.data.ATRPThreshold3,
                self.Quiet,
                self.data.Volatility
                )
        self.data.drop(columns=['ATRP', 'ATRPThreshold1', 'ATRPThreshold2', 'ATRPThreshold3'], inplace=True)

    def market_volatility_regimes_hmm(self, nState, atr_period):
        df = self.data.copy()
        df.dropna(inplace=True)

        df['ATRP'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_period) / df['Close'] * 100
        df.dropna(inplace=True)

        X = np.column_stack([df['ATRP']])

        # Create the Gaussian Hidden markov Model and fit
        self.hmm_model = GaussianHMM(
            n_components=nState, covariance_type="full", random_state=1, n_iter=1000
        ).fit(X)
        print(f'Converged: {self.hmm_model.monitor_.converged}\t\t Model Score: {self.hmm_model.score(X)}')

        hidden_states = self.hmm_model.predict(X)

        offset = len(self.data) - len(df)
        hidden_states = np.append(hidden_states, offset*[100]) # Add 100 to make array length same data length
        self.data['HMM Volatility'] = hidden_states
        self.data['HMM Volatility'] = self.data['HMM Volatility'].shift(offset)


    def market_regimes_plot_color(self, dataname, trend, fromDate, toDate, kind='color'):
        """
        Refer: https://stackoverflow.com/questions/31590184/plot-multicolored-line-based-on-conditional-in-python
        """
        df = self.data.loc[fromDate:toDate,:].copy()
        df = df.dropna(axis=0, how='any')

        # Create plot
        fig, ax = plt.subplots(nrows=2, ncols=1)

        if kind == 'color':
            ## Convert Trend to colors
            if trend == 'HMM Volatility' or trend == 'Volatility':
                trend2color = {0:'green', 1:'red', 2:'y', 3:'cyan'}
            else:
                trend2color = {-2: 'cyan', -1:'red', 0:'y', 1:'green', 2:'magenta'}
            df['color'] = df[trend].apply(lambda trend: trend2color[trend])

            def gen_repeating(s):
                """Generator: groups repeated elements in an iterable
                E.g.
                    'abbccc' -> [('a', 0, 0), ('b', 1, 2), ('c', 3, 5)]
                """
                i = 0
                while i < len(s):
                    j = i
                    while j < len(s) and s[j] == s[i]:
                        j += 1
                    yield (s[i], i, j-1)
                    i = j

            ## Add Close Price lines
            for color, start, end in gen_repeating(df['color']):
                if start > 0: # make sure lines connect
                    start -= 1
                idx = df.index[start:end+1]
                df.loc[idx, 'Close'].plot(ax=ax[0],figsize=(12,10), color=color, label='')

            ## Get artists and labels for legend and chose which ones to display
            handles, labels = ax[0].get_legend_handles_labels()

            ## Create custom artists
            c_line = plt.Line2D((0,1),(0,0), color='cyan')
            r_line = plt.Line2D((0,1),(0,0), color='red')
            y_line = plt.Line2D((0,1),(0,0), color='y')
            g_line = plt.Line2D((0,1),(0,0), color='green')
            v_line = plt.Line2D((0,1),(0,0), color='magenta')

            ## Create legend from custom artist/label lists
            if trend == 'HMM Volatility':
                ax[0].legend(
                    handles + [g_line, r_line, y_line, c_line],
                    labels + [
                        'State 1',
                        'State 2',
                        'State 3',
                        'State 4'
                    ],
                    loc='best',
                )
            elif trend == 'Volatility':
                ax[0].legend(
                    handles + [g_line, r_line, y_line, c_line],
                    labels + [
                        'Quiet',
                        'Normal',
                        'Volatily',
                        'Very Volatily'
                    ],
                    loc='best',
                )
            elif trend == 'HMM Trend':
                ax[0].legend(
                    handles + [g_line, r_line, y_line, c_line, v_line],
                    labels + [
                        'State 1',
                        'State 2',
                        'State 3',
                        'State 4',
                        'State 5'
                    ],
                    loc='best',
                )
            else:
                ax[0].legend(
                    handles + [c_line, r_line, y_line, g_line, v_line],
                    labels + [
                        'Strong Bear',
                        'Bear',
                        'Neutral',
                        'Bull',
                        'Strong Bull'
                    ],
                    loc='best',
                )
        elif kind == 'line':
            df[['Close', trend]].plot(ax=ax[0], figsize=(12,10), secondary_y = trend)
        else:
            print(f'Doesnt support this kind: {kind} of plot')

        if trend == 'Volatility' or trend == 'HMM Volatility':
            df['Returns'].plot(ax=ax[1])
        else:
            fig.delaxes(ax[1])

        fromDate_str = df.index[0].strftime("%d/%m/%Y")
        toDate_str = df.index[-1].strftime("%d/%m/%Y")
        ax[0].set_title(dataname + ' ' + trend + ' ' + fromDate_str + ' ' + toDate_str)

        plt.show()