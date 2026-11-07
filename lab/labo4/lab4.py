import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def load_data(filepath):
    data = loadmat(filepath)
    DEPL = data['DEPL']
    DEPL_INTERP = data['DEPL_INTERP']
    x = data['x']
    z = data['z']
    return DEPL, DEPL_INTERP, x, z

import numpy as np

def windowed_correlation(signal1, signal2, ind1, L, max_lag=50, normalize=True):
    lags = np.arange(max_lag + 1)
    result = np.zeros(len(lags))

    x = signal1[ind1:ind1 + L]
    x = x - x.mean()

    for i, d in enumerate(lags):
        y = signal2[ind1 + d:ind1 + d + L]
        y = y - y.mean()
        result[i] = np.sum(x * y) / L
        if normalize:
            result[i] /= (x.std() * y.std() + 1e-12)  # Pearson coefficient

    return lags, result

def find_d_optimal(signal1, signal3, max_lag=50):
    ind1_list = np.arange(1, 901)
    d_opt_list = np.zeros(len(ind1_list))
    for j, i in enumerate(ind1_list):
        lags, c = windowed_correlation(signal1, signal3, ind1=i, L=30, max_lag=max_lag)
        d_opt_list[j] = lags[np.argmax(c)]
    return d_opt_list

def interpolate_signal(signal):
    ind1 = np.arange(1, 901)
    mask = (signal > 0) & (signal < 50)  # drop values stuck on the search bounds
    p = np.polyfit(ind1[mask], signal[mask], deg=8)
    return np.polyval(p, ind1), np.polyval(np.polyder(p), ind1)


def find_youngs_modulus(stress, strain, eps_min=1e-3):
    # Stress is constant (e.g. 100); strain is the derivative of the fitted displacement
    strain = np.asarray(strain, dtype=float)
    E = np.full_like(strain, np.nan)
    valid = np.abs(strain) > eps_min
    E[valid] = stress / np.abs(strain[valid])
    return E


def plot_youngs_modulus(E, deformation):
    z = np.arange(1, 901) * 0.1  # mm
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(z, 100 * deformation); ax[0].set_ylabel("deformation (%)")
    ax[1].plot(z, E);            ax[1].set_ylabel("E (Pa)"); ax[1].set_xlabel("z (mm)")
    plt.show()

def plot_d_optimal(d_opt_list):
    plt.figure(figsize=(10, 6))
    plt.plot(d_opt_list)
    plt.title('Optimal Delay d_opt as a function of ind1')
    plt.xlabel('ind1')
    plt.ylabel('d_opt')
    plt.grid(True)
    plt.show()

def numerical_derivative(signal, dt):
    derivative = np.zeros_like(signal)
    derivative[1:-1] = (signal[2:] - signal[:-2]) / (2 * dt)
    derivative[0] = (signal[1] - signal[0]) / dt  # Forward difference for the first point
    derivative[-1] = (signal[-1] - signal[-2]) / dt  # Backward difference for the last point
    return derivative

def plot_signals_with_derivative(signal, derivative, dt):
    time = np.arange(len(signal)) * dt
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, signal, label='Signal')
    plt.title('Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(1,901), derivative, label='Derivative', color='orange')
    plt.title('Numerical Derivative of Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Derivative Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_interpolated_signal(original_signal, interpolated_signal):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, 901), original_signal, label='Original Signal')
    plt.plot(np.linspace(1, 900, len(interpolated_signal)), interpolated_signal, label='Interpolated Signal', linestyle='--')
    plt.title('Original and Interpolated Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def main():
    DEPL, DEPL_INTERP, x, z = load_data('LAB4_DISPDATA_A2025.mat')
    data = loadmat('../lab3/BGM724_LABO3.mat')
    signal1 = data['SIG1'].squeeze()
    signal2 = data['SIG2'].squeeze()
    signal3 = data['SIG3'].squeeze()

    d_opt_list = find_d_optimal(signal1, signal3, max_lag=50)
    #plot_d_optimal(d_opt_list)
    #plot_signals_with_derivative(d_opt_list, numerical_derivative(d_opt_list, dt=0.01), dt=0.01)
    deplacement, deformation = interpolate_signal(d_opt_list)
    #plot_interpolated_signal(d_opt_list, interpolated_signal)
    E = find_youngs_modulus(100, deformation)
    plot_youngs_modulus(E, deformation)

if __name__ == "__main__":
    main()