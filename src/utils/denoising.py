import pandas as pd
import numpy as np

from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import savgol_filter
import pywt 
from PyEMD.EMD import EMD 

class Dimensional_Denoise:
    def __init__(self):
        pass
    
    def moving_window(self, df, col, period=12) -> pd.DataFrame:
        df[f'MA_{period}'] = df[col].ewm(span=period, adjust=False).mean()
        return df
    
    def gaussian_filter(self, df, col, sigma=2) -> pd.DataFrame:
        df[f'Gaussian_{sigma}'] = gaussian_filter1d(df[col], sigma)
        return df
    
    def seasonal_decompose(self, df, col, period=12) -> pd.DataFrame:
        df['Seasonal_Dec'] = seasonal_decompose(df[col], model='additive', period=period).trend
        return df
    
    def stl(self, df, col, period=12) -> pd.DataFrame:
        stl = STL(df[col], period=period, robust=True)
        res = stl.fit()
        df['STL'] = res.trend
        return df
    
    def hp_filter(self, df, col, lamb=1600) -> pd.DataFrame:
        df['HP'] = hpfilter(df[col], lamb=lamb)[1]
        return df
    
    def savgol_filter(self, df, col, window_length=10, polyorder=2) -> pd.DataFrame:
        df['Savgol'] = savgol_filter(df[col], window_length=window_length, polyorder=polyorder)
        return df

    def fourier(self, df, col, cutoff_freq=0.1) -> pd.DataFrame:
        fft = np.fft.fft(df[col])
        freqs = np.fft.fftfreq(len(df[col]))
        fft[np.abs(freqs) > cutoff_freq] = 0
        filtered = np.real(np.fft.ifft(fft))
        df['Fourier'] = filtered
        return df
    
    def wavelet(self, df, col, wavelet='db4', level=4) -> pd.DataFrame:
        coeffs = pywt.wavedec(df[col], wavelet, level=level)
        threshold = np.std(coeffs[-1]) * np.sqrt(2*np.log(len(df[col]))) 
        new_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
        df['Wavelet'] = pywt.waverec(new_coeffs, wavelet)[:len(df[col])]
        return df
    
    def EMD(self, df, col) -> pd.DataFrame:
        signal = df[col].values
        emd = EMD()
        imfs = emd(signal)
        df['EMD'] = np.sum(imfs[2:], axis=0)
        return df
    
    def apply_all(self, df, col) -> pd.DataFrame: 
        df = self.moving_window(df, col, 12)
        df = self.gaussian_filter(df, col)
        df = self.seasonal_decompose(df, col)
        df = self.stl(df, col)
        df = self.hp_filter(df, col)
        df = self.savgol_filter(df, col)
        df = self.fourier(df, col)
        df = self.wavelet(df, col)
        df = self.EMD(df, col)
        return df
