from dataclasses import dataclass

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

# parameters from the lab handout
F0 = 3e6
FS = 50e6
C = 1540
SPACING = 0.1e-3
N_ELEMENTS = 128
N_SAMPLES = 4096


@dataclass
class ImagingGrid:
    x: np.ndarray
    y: np.ndarray


def load_rf_signals(filepath, var_name=None):
    raw = loadmat(filepath)
    keys = [k for k in raw if not k.startswith('__')]
    key = var_name if var_name else keys[0]
    data = np.squeeze(raw[key])
    return data


# Q1: A-scan for a single emitter/receiver pair
def plot_ascan(data, emitter, receiver, fs=FS):
    signal = data[:, emitter - 1, receiver - 1]
    t = np.arange(len(signal)) / fs

    plt.figure()
    plt.plot(t, signal)
    plt.xlabel("Time (s)")
    plt.ylabel(f"RF({emitter},{receiver})")
    plt.title(f"A-Scan - Emitter {emitter}, Receiver {receiver}")
    plt.show()


# Q2: imaging grid (imaging domain) and array element positions (physical geometry)
def build_imaging_grid(x_range=(-15e-3, 15e-3), y_range=(0, 30e-3), step=0.2e-3,
                        spacing=SPACING, n_elements=N_ELEMENTS):
    x = np.arange(x_range[0], x_range[1] + step, step)
    y = np.arange(y_range[0], y_range[1] + step, step)
    grid = ImagingGrid(*np.meshgrid(x, y))

    array_positions = np.linspace(-(n_elements - 1) / 2 * spacing,
                                   (n_elements - 1) / 2 * spacing,
                                   n_elements)

    plt.figure()
    plt.scatter(grid.x, grid.y, s=1, c='k')
    plt.scatter(array_positions, np.zeros_like(array_positions), c='r', s=10)
    plt.gca().invert_yaxis()
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Imaging grid and array elements")
    plt.show()

    return grid, array_positions


# time-of-flight -> sample index, for one emitter/receiver physical position
def time_of_flight_index(grid, emitter_pos, receiver_pos, c, fs, n_time):
    de = np.sqrt((emitter_pos - grid.x) ** 2 + grid.y ** 2)
    dr = np.sqrt((receiver_pos - grid.x) ** 2 + grid.y ** 2)
    tof = (de + dr) / c
    idx = np.round(tof * fs).astype(int)
    return np.clip(idx, 0, n_time - 1)


# Q3: reconstruction using a single emitter/receiver pair
def reconstruct_single_pair(data, grid, array_positions, emitter, receiver, c=C, fs=FS):
    signal = data[:, emitter - 1, receiver - 1]
    idx = time_of_flight_index(grid, array_positions[emitter - 1],
                                array_positions[receiver - 1], c, fs, len(signal))
    return signal[idx]


# Q4: full DAS / TFM reconstruction over all emitter/receiver pairs
def reconstruct_das(data, grid, array_positions, c=C, fs=FS, n_elements=N_ELEMENTS):
    n_time = data.shape[0]
    image = np.zeros_like(grid.x)

    for e in range(n_elements):
        for r in range(n_elements):
            idx = time_of_flight_index(grid, array_positions[e], array_positions[r],
                                        c, fs, n_time)
            image += data[idx, e, r]

    return image


def plot_image(image, grid, title="", dynamic_range_db=40):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    extent = [grid.x.min(), grid.x.max(), grid.y.max(), grid.y.min()]

    im0 = axes[0].imshow(image, extent=extent, aspect='auto', cmap='viridis')
    axes[0].set_title(f"{title} - linear")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.colorbar(im0, ax=axes[0])

    image_db = 20 * np.log10(np.abs(image) / np.max(np.abs(image)) + 1e-12)
    im1 = axes[1].imshow(image_db, extent=extent, aspect='auto', cmap='viridis',
                          vmin=-dynamic_range_db, vmax=0)
    axes[1].set_title(f"{title} - dB")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()


# Q5: approximate PSF width (-6 dB lateral width through the peak)
def measure_psf(image, grid, threshold_db=-6):
    intensity_db = 20 * np.log10(np.abs(image) / np.max(np.abs(image)) + 1e-12)
    row, _ = np.unravel_index(np.argmax(np.abs(image)), image.shape)

    lateral_profile = intensity_db[row, :]
    x_vals = grid.x[row, :]
    above = np.where(lateral_profile >= threshold_db)[0]
    width = x_vals[above[-1]] - x_vals[above[0]]

    plt.figure()
    plt.plot(x_vals * 1e3, lateral_profile)
    plt.axhline(threshold_db, color='r', linestyle='--')
    plt.xlabel("x (mm)")
    plt.ylabel("Amplitude (dB)")
    plt.title(f"PSF lateral profile - width = {width*1e3:.2f} mm at {threshold_db} dB")
    plt.show()

    return width


def main():
    data = load_rf_signals("RF_SIGNALS.mat")
    print(f"Data shape (time, emitter, receiver): {data.shape}")

    plot_ascan(data, emitter=64, receiver=64)

    grid, array_positions = build_imaging_grid()

    single_pair_image = reconstruct_single_pair(data, grid, array_positions, 64, 64)
    plot_image(single_pair_image, grid, title="Single pair (E=64, R=64)")

    das_image = reconstruct_das(data, grid, array_positions)
    plot_image(das_image, grid, title="Full DAS (128x128 pairs)")

    measure_psf(das_image, grid)


if __name__ == "__main__":
    main()