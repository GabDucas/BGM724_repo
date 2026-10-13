from scipy.io import loadmat
from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath):
    data = loadmat(filepath)

    FMC = data['FMC'].real
    Shifted_FMC = data['shifted_FMC'].real
    Fe = float(data['Fe'].squeeze())
    N_elements = int(data['N_elements'].squeeze())
    Nt = int(data['Nt'].squeeze())
    c0 = float(data['c0'].squeeze())
    x_array = data['x_array'].squeeze()

    return FMC, Shifted_FMC, Fe, N_elements, Nt, c0, x_array


def create_imaging_grid(largeur=0.08, resolution_largeur=0.0008,
                        profondeur=0.08, resolution_profondeur=0.00008):
    x = np.arange(-largeur / 2, largeur / 2, resolution_largeur)
    z = np.arange(0, profondeur, resolution_profondeur)
    return np.meshgrid(x, z)


def calculate_distance(x, x_i, z, z_i):
    return np.sqrt((x - x_i)**2 + (z - z_i)**2)


def calculate_time_of_flight(distance, c0):
    return distance / c0


def find_sample_index(tof, Fe):
    return tof * Fe


def reconstruct_saft_image(FMC, N_elements, x_array, X, Z, c0, Fe):
    image = np.zeros(X.shape)

    # Element-to-pixel distances, computed once per element
    dist = [calculate_distance(X, x_array[k], Z, 0) for k in range(N_elements)]

    for emetteur in range(N_elements):
        for recepteur in range(N_elements):
            tof = calculate_time_of_flight(dist[emetteur] + dist[recepteur], c0)
            idx = np.round(find_sample_index(tof, Fe)).astype(int)
            valid = (idx >= 0) & (idx < FMC.shape[2])
            image[valid] += FMC[emetteur, recepteur, idx[valid]]

    return image


def compute_envelope(image):
    return np.abs(hilbert(image, axis=0))


def to_db(env):
    return 20 * np.log10(env / env.max() + 1e-12)


def disk_mask(X, Z, xc, zc, r):
    return (X - xc)**2 + (Z - zc)**2 <= r**2


def compute_cr_cnr(env, mask_in, mask_bg):
    mu_in, mu_bg = env[mask_in].mean(), env[mask_bg].mean()
    sig_in, sig_bg = env[mask_in].std(), env[mask_bg].std()
    cr = 20 * np.log10(mu_in / mu_bg)
    cnr = abs(mu_in - mu_bg) / np.sqrt(sig_in**2 + sig_bg**2)
    return cr, cnr


def report_metrics(env, X, Z, rois, r):
    mask_bg = disk_mask(X, Z, *rois['background'], r)

    for name in ('mass_left', 'mass_right'):
        mask_in = disk_mask(X, Z, *rois[name], r)
        cr, cnr = compute_cr_cnr(env, mask_in, mask_bg)
        print(f"{name:11s} CR = {cr:6.2f} dB   CNR = {cnr:5.2f}")


def show_image(env_db, rois, r):
    fig, ax = plt.subplots()
    im = ax.imshow(env_db, extent=[-40, 40, 80, 0], aspect='auto',
                   vmin=-50, vmax=0, cmap='gray')
    for xc, zc in rois.values():
        ax.add_patch(plt.Circle((xc * 1e3, zc * 1e3), r * 1e3,
                                fill=False, color='r'))
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    fig.colorbar(im, label='dB')
    plt.show()


def main():
    FMC, Shifted_FMC, Fe, N_elements, Nt, c0, x_array = load_data('BGM_724_DATA_2025.mat')

    X, Z = create_imaging_grid()
    image = reconstruct_saft_image(FMC, N_elements, x_array, X, Z, c0, Fe)
    env = compute_envelope(image)

    # Centers in meters, same units as X and Z
    rois = {
        'mass_left': (-0.021, 0.040),
        'mass_right': (0.021, 0.040),
        'background': (0.0, 0.055),
    }
    r = 0.004

    report_metrics(env, X, Z, rois, r)
    show_image(to_db(env), rois, r)


if __name__ == "__main__":
    main()