from scipy.io import loadmat
from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
 
 
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
   
 
def decalage_optimal(sig1, sig2, L, dmax, pas):
    """
    Pour un signal (colonne) : fenêtres de taille L sur sig1, glissées sur sig2
    avec des décalages d = 0..dmax. Corrélation normalisée non biaisée (1/L).
    Retourne les centres des fenêtres (indices), le décalage optimal et la corrélation max.
    """
    Nz = len(sig1)
    starts = np.arange(0, Nz - L - dmax, pas)
 
    # Toutes les fenêtres de sig2 (une par position de départ possible)
    W2 = sliding_window_view(sig2, L)           # shape (Nz-L+1, L)
 
    d_opt = np.zeros(len(starts), dtype=int)
    c_max = np.zeros(len(starts))
 
    for n, i0 in enumerate(starts):
        x = sig1[i0:i0 + L]
        x = x - x.mean()
        Y = W2[i0:i0 + dmax + 1]                # fenêtres décalées de d = 0..dmax
        Y = Y - Y.mean(axis=1, keepdims=True)
 
        num = (Y @ x) / L
        den = np.sqrt(np.sum(x**2) / L) * np.sqrt(np.sum(Y**2, axis=1) / L)
        C = np.where(den > 0, num / np.maximum(den, 1e-20), 0)
 
        d_opt[n] = C.argmax()
        c_max[n] = C.max()
 
    centres = starts + L // 2
    return centres, d_opt, c_max
 
 
def champ_deplacement(img_ref, img_comp, L, dmax, pas=None):
    """
    Applique decalage_optimal à chaque colonne des deux images.
    Retourne les champs de décalage (en échantillons/pixels) et de corrélation,
    de forme (n_fenêtres, n_colonnes), ainsi que les indices des centres.
    """
    if pas is None:
        pas = L // 2                            # 50 % de recouvrement
    n_cols = img_ref.shape[1]
 
    D, C = [], []
    for j in range(n_cols):
        centres, d, c = decalage_optimal(img_ref[:, j], img_comp[:, j], L, dmax, pas)
        D.append(d)
        C.append(c)
 
    return centres, np.array(D).T, np.array(C).T
 
 
def show_displacement(D_mm, centres_z_mm, x_mm, title='Déplacement axial (mm)'):
    fig, ax = plt.subplots()
    im = ax.imshow(D_mm, extent=[x_mm[0], x_mm[-1], centres_z_mm[-1], centres_z_mm[0]],
                   aspect='auto', cmap='jet')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_title(title)
    fig.colorbar(im, label='mm')
    plt.show()
   
   
def deformation_poly(D_mm, dz_fen, deg=3):
    """Ajuste un polynôme sur δ(z) pour chaque colonne, puis dérive : ε = dδ/dz."""
    n_z, n_x = D_mm.shape
    z = np.arange(n_z) * dz_fen
    eps = np.zeros_like(D_mm, dtype=float)
    for j in range(n_x):
        coef = np.polyfit(z, D_mm[:, j], deg)
        eps[:, j] = np.polyval(np.polyder(coef), z)
    return eps
 
 
def rigidite_lumped(D_mm, Nwin=3, N_pix_ress=15, F=1.0, eps_min=1e-6):
    """
    Cartographie de rigidité relative par moindres carrés.
    Chaque bloc = N_pix_ress ressorts en série x Nwin colonnes.
    Retourne K de forme (n_blocs_z, n_blocs_x) et les centres des blocs (indices).
    """
    n_z, n_x = D_mm.shape
    starts_z = np.arange(0, n_z - N_pix_ress, N_pix_ress)
    starts_x = np.arange(0, n_x - Nwin + 1, Nwin)
 
    K = np.zeros((len(starts_z), len(starts_x)))
    for a, s in enumerate(starts_z):
        # allongement total des N_pix_ress ressorts, pour chaque colonne
        delta = D_mm[s + N_pix_ress, :] - D_mm[s, :]
        for b, c0 in enumerate(starts_x):
            d = delta[c0:c0 + Nwin]
            den = np.sum(d**2)
            # k = F * sum(d) / sum(d^2)  (solution des moindres carrés)
            K[a, b] = F * np.sum(d) / den if den > eps_min else np.nan
 
    zc = starts_z + N_pix_ress / 2
    xc = starts_x + Nwin / 2
    return K, zc, xc
 
 
def show_map(M, extent, title='', label='', cmap='jet'):
    fig, ax = plt.subplots()
    im = ax.imshow(M, extent=extent, aspect='auto', cmap=cmap)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_title(title)
    fig.colorbar(im, label=label)
    plt.show()
   
   
def young_lumped(D_mm, dz_fen, Nwin=3, N_pix_ress=15, sigma=50.0, eps_min=1e-12):
    """
    Module de Young (Pa) par moindres carrés, modèle de ressorts en série.
    E = sigma * sum(eps) / sum(eps^2) sur les Nwin colonnes de chaque bloc.
    """
    n_z, n_x = D_mm.shape
    L0 = N_pix_ress * dz_fen                      # longueur d'un bloc (mm)
    starts_z = np.arange(0, n_z - N_pix_ress, N_pix_ress)
    starts_x = np.arange(0, n_x - Nwin + 1, Nwin)
 
    E = np.full((len(starts_z), len(starts_x)), np.nan)
    for a, s in enumerate(starts_z):
        eps = (D_mm[s + N_pix_ress, :] - D_mm[s, :]) / L0   # déformation par colonne
        for b, c0 in enumerate(starts_x):
            e = np.abs(eps[c0:c0 + Nwin])         # abs : évite les E négatifs
            den = np.sum(e**2)
            if den > eps_min:
                E[a, b] = sigma * np.sum(e) / den
    return E, starts_z, starts_x
   
   
   
def stats_E(E, z_top_mm, N_pix_ress, dz_fen, Nwin, rois, r):
    """E moyen (Pa) dans chaque masse et dans le fond, + rapport E_masse/E_fond."""
    nz, nx = E.shape
    xc = (-40 + (np.arange(nx) + 0.5) * Nwin * 0.8) * 1e-3
    zc = (z_top_mm + (np.arange(nz) + 0.5) * N_pix_ress * dz_fen) * 1e-3
    Xb, Zb = np.meshgrid(xc, zc)
    ok = ~np.isnan(E)
 
    bg = E[disk_mask(Xb, Zb, *rois['background'], r) & ok]
    out = {'bg': (bg.mean(), bg.std())}
    for name in ('mass_left', 'mass_right'):
        m = E[disk_mask(Xb, Zb, *rois[name], r) & ok]
        out[name] = (m.mean(), m.std(), m.mean() / bg.mean())
    return out    
 
   
 
def main():
    FMC, Shifted_FMC, Fe, N_elements, Nt, c0, x_array = load_data('C:/Users/Camilo/OneDrive - USherbrooke/APP-S8/BGM724/problématique/BGM_724_DATA_2025.mat')
 
    X, Z = create_imaging_grid()
    image_FMC = reconstruct_saft_image(FMC, N_elements, x_array, X, Z, c0, Fe)
    env_FMC = compute_envelope(image_FMC)
   
    image_SFMC = reconstruct_saft_image(Shifted_FMC, N_elements, x_array, X, Z, c0, Fe)
    env_SFMC = compute_envelope(image_SFMC)
   
   
 
    # Centers in meters, same units as X and Z
    rois = {
        'mass_left': (-0.021, 0.040),
        'mass_right': (0.021, 0.040),
        'background': (0.0, 0.055),
    }
    r = 0.004
   
    print(len(FMC))
 
    #report_metrics(env_FMC, X, Z, rois, r)
    #show_image(to_db(env_FMC), rois, r)
    #show_image(to_db(env_SFMC), rois,r)
   
   
    # Trouver le decalage optimale
    # Images RF (avant enveloppe) : meilleure précision pour la corrélation
    L = 30         # à ajuster, voir plus bas
    dmax = 80       # décalage max en pixels
    dz = 0.00008    # résolution en profondeur (m)
 
    centres, D, Ccorr = champ_deplacement(image_FMC, image_SFMC, L, dmax)
 
    D_mm = D * dz * 1e3                          # décalage en mm
    z_mm = centres * dz * 1e3
    x_mm = X[0, :] * 1e3
    show_displacement(D_mm, z_mm, x_mm)
   
   
    pas = 10
    centres, D, Ccorr = champ_deplacement(image_FMC, image_SFMC, L, dmax, pas=pas)
    D_mm = D * dz * 1e3            # déplacement en mm
    dz_fen = pas * dz * 1e3        # pas axial de D en mm (0.8)
    z_mm = centres * dz * 1e3
   
    Nwin, N_pix_ress = 3, 15
 
     # Approche polynomiale
    eps = deformation_poly(D_mm, dz_fen, deg=3)
    show_map(eps, [-40, 40, z_mm[-1], z_mm[0]],
             'Élastogramme (polynôme)', 'ε')
 
    # Éléments distribués
    K, zc, xc = rigidite_lumped(D_mm, Nwin, N_pix_ress)
    z_top = z_mm[0]
    z_bot = z_top + K.shape[0] * N_pix_ress * dz_fen
    x_right = -40 + K.shape[1] * Nwin * 0.8
    show_map(K, [-40, x_right, z_bot, z_top],
             'Rigidité relative (Lumped)', 'k (u.a.)')
   
   
    # Calcul du module de young
    E, sz, sx = young_lumped(D_mm, dz_fen, Nwin, N_pix_ress, sigma=50.0)
    E_kPa = E / 1e3
 
    z_top = z_mm[0]
    z_bot = z_top + E.shape[0] * N_pix_ress * dz_fen
    x_right = -40 + E.shape[1] * Nwin * 0.8
 
    show_map(E_kPa, [-40, x_right, z_bot, z_top],
             'Module de Young (Lumped)', 'E (kPa)')
   
    stats = stats_E(E, z_mm[0], N_pix_ress, dz_fen, Nwin, rois, r)
    print(f"Fond        : E = {stats['bg'][0]/1e3:6.1f} ± {stats['bg'][1]/1e3:.1f} kPa")
    for name in ('mass_left', 'mass_right'):
        moy, std, ratio = stats[name]
        print(f"{name:11s} : E = {moy/1e3:6.1f} ± {std/1e3:.1f} kPa   E_masse/E_fond = {ratio:.2f}")
 
 
if __name__ == "__main__":
    main()