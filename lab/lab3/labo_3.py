import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def load_data(filepath):
    return loadmat(filepath)


def separate_signals(signal):
    signal1 = signal['SIG1'].squeeze()
    signal2 = signal['SIG2'].squeeze()
    signal3 = signal['SIG3'].squeeze()
    return signal1, signal2, signal3


def plot_correlated_signal(lags, correlated_signal):
    plt.figure(figsize=(10, 6))
    plt.plot(lags, correlated_signal)
    plt.title('Correlated Signal')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.xlim(-100, 100)
    plt.grid(True)
    plt.show()


def correlation(signal1, signal2, max_lag=100):
    # R_xy[m] = sum_n x[n+m] * y[n], assumes equal-length signals
    N = len(signal1)
    lags = np.arange(-max_lag, max_lag + 1)
    result = np.zeros(len(lags))
    for i, m in enumerate(lags):
        if m >= 0:
            total = np.sum(signal1[m:] * signal2[:N - m])
        else:
            total = np.sum(signal1[:N + m] * signal2[-m:])
        result[i] = total / N
    return lags, result

def unbiased_correlation(signal1, signal2, max_lag=100):
    # R_xy[m] = sum_n x[n+m] * y[n], assumes equal-length signals
    N = len(signal1)
    lags = np.arange(-max_lag, max_lag + 1)
    result = np.zeros(len(lags))
    for i, m in enumerate(lags):
        if m >= 0:
            total = np.sum((signal1[m:] - np.mean(signal1[m:])) * (signal2[:N - m] - np.mean(signal2[:N - m])))
        else:
            total = np.sum((signal1[:N + m] - np.mean(signal1[:N + m])) * (signal2[-m:] - np.mean(signal2[-m:])))
        result[i] = total / (N - abs(m))  # Normalize by the number of overlapping samples
    return lags, result

# Part 2 - Windowed correlation

def windowed_correlation(signal1, signal2, ind1=0, L=64, max_lag=100):
    lags = np.arange(-max_lag, max_lag + 1)
    result = np.zeros(len(lags))
    x = signal1[ind1:ind1 + L]
    x = x - x.mean()

    for i, d in enumerate(lags):
        start = ind1 + d
        if start < 0 or start + L > len(signal2):
            continue  # out of signal, stays 0


        y = signal2[start:start + L]
        y = y - y.mean()  # mean over the shifted window
        result[i] = np.sum(x * y) / L

    return lags, result

def normalized_windowed_correlation(signal1, signal2, ind1=0, L=64, max_lag=100):
    lags = np.arange(-max_lag, max_lag + 1)
    result = np.zeros(len(lags))

    x = signal1[ind1:ind1 + L]
    x = x - x.mean()
    x = x / np.sqrt(np.sum(x ** 2) / L)  # divide by sqrt(C_s1s1(0))

    for i, d in enumerate(lags):
        start = ind1 + d
        if start < 0 or start + L > len(signal2):
            continue  # out of signal, stays 0

        y = signal2[start:start + L]
        y = y - y.mean()
        y = y / np.sqrt(np.sum(y ** 2) / L)  # divide by sqrt(C_s2s2(0)) of the shifted window
        result[i] = np.sum(x * y) / L

    return lags, result

def plot_window_comparison(signal1, signal2, window_sizes, ind1=500, max_lag=100):
    plt.figure(figsize=(12, 5))
    for L in window_sizes:
        lags, C = normalized_windowed_correlation(signal1, signal2, ind1=ind1, L=L, max_lag=max_lag)
        plt.plot(lags, C, label=str(L))

    plt.xlim(-max_lag, max_lag)
    plt.xlabel('d (échantillon)')
    plt.ylabel("C'(d)")
    plt.legend(loc='lower left')
    plt.show()


def main():
    x = load_data('BGM724_LABO3.mat')
    signal1, signal2, signal3 = separate_signals(x)
    plot_window_comparison(signal1, signal2, [256, 128, 64, 32])

if __name__ == "__main__":
    main()