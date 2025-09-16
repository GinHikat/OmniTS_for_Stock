import pandas as pd
import numpy as np
import sys, os

import ta

from PyEMD.EMD import EMD
from scipy.signal import hilbert


def EMA(series: pd.Series, period: int = 12) -> pd.Series:
    ema = series.ewm(span=period, adjust=False).mean()
    ema.iloc[:period-1] = np.nan  # mask first values
    return ema

def stochastic_oscillator(df, k_period=14, d_period=3):
    # Highest high & lowest low over the lookback window
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    
    # %K line
    k_percent = 100 * (df['Close'] - low_min) / (high_max - low_min)
    
    # %D line (moving average of %K)
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent, d_percent

class Feature:
    def __init__(self):
        pass
    
    def technical_analysis(self, df) -> pd.DataFrame:
        ''''
        Extract Technical Analysis features from the given columns: OHLC, Date and Volume
        
        Parameters:
        df: df with OHLC, Date and Volume columns
        
        Returns:
        output: Currently RSI, MACD, MFI, Stochastic Oscillator, Adjusted Close and Log Return
        
        Note: 
        1. RSI: Relative Strength Index
        2. MACD: Moving Average Convergence Divergence
        3. MFI: Money Flow Index
        4. Stochastic Oscillator: Stochastic Oscillator
        5. Adjusted Close: Adjusted Close Price
        6. Log Return: Log Return
        7. Realized Volatility: Parkinson volatility, Rogers-Satchell volatility
        
        With the given basic Stock attribute OHLC, we can assume Adjusted Close Price to be approximately similar to Close Price
        When further attributed are present in the datasets like Intradays, Stock Splits/ Reverse Splits, Dividends, spin-offs, rights offering, mergers....
        '''
        
        #RSI
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        
        #MACD
        df['EMA_12'] = EMA(df['Close'], period=12)
        df['EMA_26'] = EMA(df['Close'], period=26)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = EMA(df['MACD'], period=9)
        
        #MFI
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3

        df['MF'] = df['TP'] * df['Volume']

        df['PMF'] = np.where(df['TP'] > df['TP'].shift(1), df['MF'], 0)
        df['NMF'] = np.where(df['TP'] < df['TP'].shift(1), df['MF'], 0)

        df['PMF_sum'] = df['PMF'].rolling(window=14).sum()
        df['NMF_sum'] = df['NMF'].rolling(window=14).sum()  

        df['MR'] = df['PMF_sum'] / df['NMF_sum']
        df['MFI'] = 100 - (100 / (1 + df['MR']))
        
        #Stochastic Oscillator
        df['%K'], df['%D'] = stochastic_oscillator(df, k_period=14, d_period=3)
        
        #Boilinger Bands
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['STD20'] = df['Close'].rolling(window=20).std()
        
        df['boilinger_up'] = df['SMA20'] + 2* df['STD20']
        df['boilinger_down'] = df['SMA20'] - 2* df['STD20']
            
        #Adjusted Close
        df['Adjusted_Close_Price'] = df['Close']
        
        # Log Return
        df["LogRet"] = np.log(df["Adjusted_Close_Price"] / df["Adjusted_Close_Price"].shift(1))
        
        #Realized Volatility
            #Parkinson volatility
        df["ParkinsonVol"] = (1 / (4*np.log(2))) * (np.log(df["High"] / df["Low"]))**2

            # Rogers-Satchell volatility
        df["RSVol"] = (np.log(df["High"]/df["Close"]) * np.log(df["High"]/df["Open"]) +
                    np.log(df["Low"]/df["Close"]) * np.log(df["Low"]/df["Open"]))

        return df.drop(['TP', 'MF', 'PMF', 'NMF', 'MR'], axis = 1)
        
    def candle_plot(self, df) -> pd.DataFrame:
        '''
        Treat the OHLC data as 1 daily Candle in the Japanese Candle Model. The function identifies and extract special Sign Candles based on predefined rules
        
        Parameter: 
        df: Input dataframe, contains OHCL data, Date and Volume data for 1 individual Stock Index
        
        Return: 
        pd.DataFrame: dataframe with Corresponding binary signals for each Candle column (expected), treating these binary Candle signals as features for Models
        '''
        
        
        return df
    
    def date_extraction(self, df) -> pd.DataFrame:
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Weekday'] = df['Date'].dt.weekday 
        return df
    
    def pseudo_hht(self, df, col, t_col) -> pd.DataFrame:
        ''''
        A brief application of Hilbert-Huang Transform, applying the EMD-HSA framework and take the resulting series as features for Models
        
        Parameter: 
        df: Input dataframe, contains OHCL data, Date and Volume data for 1 individual Stock Index
        col: Column name of the Close data, this is treated as the Noisy input signal

        Return:
        pd.DataFrame: dataframe with EMD-HSA ouput, containing
        - The Hilbert Transform of each IMFs (from EMD or other variants)
        - The Instantaneous Frequency (IF) of each IMFs
        - The Instantaneous Amplitude (IA) of each IMFs
        
        each of these will increase in number according to level of decompositions (or number of extracted IMF from EMD phase)
        
        Note: As other variants of EMD (like CEEMDAN, ICEEMDAN, etc) are not implementable in Python PyEMD, we will only use the basic EMD here.
        '''
        signal = df[col].values 

        emd = EMD()
        imfs = emd(signal)
        
        df[t_col] = pd.to_datetime(df[t_col])
        
        analytic_imfs = hilbert(imfs, axis=1)
        amplitude_envelope = np.abs(analytic_imfs)
        instantaneous_phase = np.unwrap(np.angle(analytic_imfs))
        
        t = (df[t_col] - df[t_col].iloc[0]).dt.total_seconds().to_numpy()

        instantaneous_frequency = np.diff(instantaneous_phase, axis=1) / (2.0*np.pi*np.diff(t)[0])
        
        for i in range(imfs.shape[0]):
            df[f'IMF_{i+1}'] = imfs[i]
            df[f'IMF_{i+1}_Hilbert'] = analytic_imfs[i]
            df[f'IMF_{i+1}_Instant_Amplitude'] = amplitude_envelope[i]
            df[f'IMF_{i+1}_Instant_Freq'] = np.append(instantaneous_frequency[i], np.nan)
        
        return df
    
    def apply_all(self, df) -> pd.DataFrame:
        df = self.technical_analysis(df)
        df = self.date_extraction(df)
        df = self.pseudo_hht(df, 'Close', 'Date')
        return df