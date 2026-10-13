import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def load_data(filepath):
    raw = loadmat(filepath)
    x = raw['x'].squeeze()
    y = raw['y'].squeeze()
    IMAGE = raw['IMAGE']
    IMAGE_dB = raw['IMAGE_dB']
    return x, y, IMAGE, IMAGE_dB


def build_circular_mask(x, y, xc, yc, r):
    X, Y = np.meshgrid(x, y)
    return ((X - xc) ** 2 + (Y - yc) ** 2) <= r ** 2


def zone_stats(IMAGE, mask):
    # always on linear scale, never on dB
    valeurs = IMAGE[mask]
    return valeurs.mean(), valeurs.std()


def snr_cnr(mu_roi, mu_fond, sigma_fond):
    snr_db = 20 * np.log10(mu_roi / sigma_fond)
    cnr_db = 20 * np.log10(abs(mu_roi - mu_fond) / sigma_fond)
    return snr_db, cnr_db


def main():
    filepath = 'labo2/IMAGE_DATA.mat'
    x, y, IMAGE, IMAGE_dB = load_data(filepath)

    # ================= PART 1 - Q1: axial resolution =================
    # wires are all at x = -15mm, at 5 different depths -> one column, 5 peaks
    idx_fils = np.argmin(np.abs(x - (-15)))
    colonne_fils = IMAGE_dB[:, idx_fils]

    plt.figure(figsize=(10, 6))
    plt.plot(y, colonne_fils)
    plt.xlabel("y (mm)")
    plt.ylabel("IMAGE_dB")
    plt.title("Q1 - Profil axial des fils (x = -15 mm)")
    plt.ylim(-60, 0)
    plt.grid(True, alpha=0.3)
    plt.show()
    # TODO (you): read the -3dB width of each of the 5 peaks graphically

    # ================= PART 1 - Q2: lateral resolution =================
    # each wire is at a fixed depth -> one row per wire, plotted vs x
    profondeurs = [40, 50, 60, 70, 80]

    plt.figure(figsize=(10, 6))
    for prof in profondeurs:
        idx_y = np.argmin(np.abs(y - prof))
        ligne = IMAGE_dB[idx_y, :]
        plt.plot(x, ligne, label=f"{prof} mm")
    plt.xlabel("x (mm)")
    plt.ylabel("IMAGE_dB")
    plt.title("Q2 - Profil lateral des fils")
    plt.legend()
    plt.ylim(-60, 0)
    plt.xlim(-20, -10)
    plt.grid(True, alpha=0.3)
    plt.show()
    # TODO (you): read the -3dB width of each curve, compare to Q1

    # ================= PART 2 - Q3: mask for the largest hyper-echoic zone =================
    plt.figure(figsize=(5, 8))
    plt.imshow(IMAGE_dB, extent=[x.min(), x.max(), y.max(), y.min()],
               aspect='auto', cmap='gray', vmin=-60, vmax=0)
    plt.xlim(-10, 0)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Zoom zones hyper-echoiques")
    plt.colorbar(label="dB")
    plt.show()
    # visually: the y=80mm blob is the widest -> largest zone

    xc, yc, r = -5, 80, 3.0  # radius ~ half the 6mm diameter given in the statement
    masque_hyper = build_circular_mask(x, y, xc, yc, r)

    plt.imshow(masque_hyper, extent=[x.min(), x.max(), y.max(), y.min()], aspect='auto')
    plt.title("Q3 - Masque de la plus grosse zone hyper-echoique")
    plt.colorbar()
    plt.show()

    # ================= PART 2 - Q4: mean/std inside that mask =================
    moyenne_hyper, ecart_hyper = zone_stats(IMAGE, masque_hyper)
    print(f"Zone hyper-echoique: moyenne={moyenne_hyper:.3f}, std={ecart_hyper:.3f}")

    # ================= PART 2 - Q5: background mask, tried at 3 positions =================
    fonds = [
        (0, 45, 2, "Fond1"),
        (-20, 80, 5, "Fond2"),
        (20, 40, 5, "Fond3"),
    ]
    for xc_bg, yc_bg, r_bg, label in fonds:
        masque_fond = build_circular_mask(x, y, xc_bg, yc_bg, r_bg)
        m, s = zone_stats(IMAGE, masque_fond)
        print(f"{label}: moyenne={m:.3f}, std={s:.3f}")
    # TODO (you): compare the 3 results -> do they depend on position/size?

    # reference background used for SNR/CNR (pick one of the 3 tested above)
    xc_bg, yc_bg, r_bg = fonds[0][:3]
    masque_fond_ref = build_circular_mask(x, y, xc_bg, yc_bg, r_bg)
    mu_fond, sigma_fond = zone_stats(IMAGE, masque_fond_ref)

    # ================= PART 2 - Q6/Q7: SNR/CNR for all hyper-echoic zones =================
    # (yc, diameter_mm) for each of the 5 hyper-echoic zones, xc always -5mm
    zones_hyper = [(40, 2), (50, 3), (60, 4), (70, 5), (80, 6)]

    print("\nY(mm)  D(mm)  SNR(dB)  CNR(dB)")
    for yc, diametre in zones_hyper:
        r = diametre / 2
        masque = build_circular_mask(x, y, -5, yc, r)
        mu_roi, _ = zone_stats(IMAGE, masque)
        snr_db, cnr_db = snr_cnr(mu_roi, mu_fond, sigma_fond)
        print(f"{yc:5d}  {diametre:5d}  {snr_db:7.1f}  {cnr_db:7.1f}")
    # TODO (you): compare these values to the expected +20dB, comment on the smaller zones

    # ================= PART 2 - Q8: SNR/CNR for all hypo-echoic zones =================
    # xc = 10mm this time; diameter decreases with depth (opposite pattern from hyper zones)
    zones_hypo = [(40, 6), (50, 5), (60, 4), (70, 3), (80, 2)]

    print("\nY(mm)  D(mm)  SNR(dB)  CNR(dB)")
    for yc, diametre in zones_hypo:
        r = diametre / 2
        masque = build_circular_mask(x, y, 10, yc, r)
        mu_roi, _ = zone_stats(IMAGE, masque)
        snr_db, cnr_db = snr_cnr(mu_roi, mu_fond, sigma_fond)
        print(f"{yc:5d}  {diametre:5d}  {snr_db:7.1f}  {cnr_db:7.1f}")
    # TODO (you): compare to the theoretical value (-infinity dB) -> why can't it be reached in practice?


if __name__ == "__main__":
    main()