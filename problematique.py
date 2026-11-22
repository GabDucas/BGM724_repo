from scipy.io import loadmat
from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view


##################### Première partie : Formation et caractérisation de l'image ultrasonere #################################

def load_data(filepath):
    data = loadmat(filepath)
    FMC = data['FMC'].real

    Shifted_FMC = data['shifted_FMC'].real
    Fe = float(data['Fe'].squeeze())

    # Nombre élements du traducteur (piezoélectrique)
    N_elements = int(data['N_elements'].squeeze())

    # Nombre d'échantillons temporelles
    Nt = int(data['Nt'].squeeze())

    # Vitesse du son
    c0 = float(data['c0'].squeeze())

    # Position spatiale des éléments
    x_array = data['x_array'].squeeze()

    return FMC,Shifted_FMC,Fe,N_elements,Nt,c0,x_array


def create_imaging_grid(largeur=0.08, resolution_largeur=0.0008,
                        profondeur=0.08, resolution_profondeur=0.00008):

    # la largeur
    x = np.arange(-largeur / 2, largeur / 2, resolution_largeur)

    # La profondeur
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


def to_db(env):
    return 20 * np.log10(env / env.max() + 1e-12)


def compute_envelope(image):
    return np.abs(hilbert(image, axis=0))


def disk_mask(X, Z, xc, zc, r):
    #selection de la région d'intérêt
    return (X - xc)**2 + (Z - zc)**2 <= r**2


def compute_cr_cnr_snr(env, mask_in, mask_bg):
    mu_in, mu_bg = env[mask_in].mean(), env[mask_bg].mean()
    sig_in, sig_bg = env[mask_in].std(), env[mask_bg].std()
    # All three are amplitude ratios -> 20*log10
    cr = 20 * np.log10(mu_in / mu_bg)
    cnr = 20 * np.log10(np.abs(mu_in - mu_bg) / np.sqrt(sig_in**2 + sig_bg**2))
    snr = 20 * np.log10(mu_in / sig_in)   # speckle SNR (≈ 5.6 dB, i.e. 1.91, for Rayleigh speckle)
    return cr, cnr, snr


def report_metrics(env, X, Z, masses, r, bg_centres, titre=""):
    # Fond = union de plusieurs disques à la même profondeur que les masses
    mask_bg = np.zeros(X.shape, dtype=bool)
    for xb, zb in bg_centres:
        mask_bg |= disk_mask(X, Z, xb, zb, r)
    print(f"--- {titre} ---")
    for name, (xc, zc) in masses.items():
        mask_in = disk_mask(X, Z, xc, zc, r)
        cr, cnr, snr = compute_cr_cnr_snr(env, mask_in, mask_bg)
        print(f"{name:11s} CR = {cr:6.2f} dB   CNR = {cnr:6.2f} dB   SNR = {snr:6.2f} dB")


def show_image(env_db, masses, r, bg_centres, titre=""):
    fig, ax = plt.subplots()
    im = ax.imshow(env_db, extent=[-40, 40, 80, 0], aspect='auto',
                   vmin=-50, vmax=0, cmap='gray')
    for xc, zc in masses.values():
        ax.add_patch(plt.Circle((xc * 1e3, zc * 1e3), r * 1e3,
                                fill=False, color='r'))
    for xc, zc in bg_centres:
        ax.add_patch(plt.Circle((xc * 1e3, zc * 1e3), r * 1e3,
                                fill=False, color='c'))
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_title(titre)
    fig.colorbar(im, label='dB')
    plt.show()



########################## Deuxième partie : élastographie et modélisation des tissus ######################################

def estimate_displacement(img_ref, img_cmp, L, step, max_shift, dz):
    """
    Corrélation normalisée fenêtre par fenêtre, colonne par colonne.
    img_ref : image sans compression (0 Pa), RF
    img_cmp : image avec compression (50 Pa), RF
    L : taille de la fenêtre (échantillons)
    step : pas entre deux fenêtres (échantillons)
    max_shift : décalage maximal testé (échantillons)
    dz : pas d'échantillonnage en profondeur (m)
    """
    Nz, Nx = img_ref.shape
    shifts = np.arange(-max_shift, max_shift + 1)
    starts = np.arange(max_shift, Nz - L - max_shift + 1, step)

    displacement = np.zeros((len(starts), Nx))
    corr_max = np.zeros((len(starts), Nx))

    for i, s in enumerate(starts):
        a = img_ref[s:s + L, :]                      # fenêtre de référence
        energy_a = np.sum(a**2, axis=0)
        corr = np.zeros((len(shifts), Nx))

        for k, d in enumerate(shifts):
            b = img_cmp[s + d:s + d + L, :]          # fenêtre décalée
            energy_b = np.sum(b**2, axis=0)
            corr[k] = np.sum(a * b, axis=0) / np.sqrt(energy_a * energy_b + 1e-20)

        k_best = np.argmax(corr, axis=0)             # décalage optimal par colonne
        displacement[i] = shifts[k_best] * dz        # en mètres
        corr_max[i] = np.max(corr, axis=0)

    z_centers = (starts + L / 2) * dz                # profondeur du centre de chaque fenêtre
    return displacement, corr_max, z_centers


def show_displacement_m(displacement, z_centers, x_axis, titre=""):
    fig, ax = plt.subplots()
    im = ax.imshow(displacement * 1e3, aspect='auto', cmap='jet',
                   extent=[x_axis[0] * 1e3, x_axis[-1] * 1e3,
                           z_centers[-1] * 1e3, z_centers[0] * 1e3])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_title(titre)
    fig.colorbar(im, label='Déplacement (mm)')
    plt.show()


def compute_strain(displacement, z_centers):
    """Déformation ε = dδ/dz (sans unité), colonne par colonne."""
    return np.gradient(displacement, z_centers, axis=0)


def build_lumped_matrix(n_pix, n_win, dz_pix):
    """
    Modèle : u[j, c] = u0[c] + dz_pix * somme_{i<j} eps_i
    Inconnues : eps_0..eps_(n_pix-2), puis u0[0..n_win-1]
    """
    n_sp = n_pix - 1
    A = np.zeros((n_pix * n_win, n_sp + n_win))
    for j in range(n_pix):
        for c in range(n_win):
            row = j * n_win + c
            A[row, :j] = dz_pix          # ressorts au-dessus du noeud j
            A[row, n_sp + c] = 1         # décalage de la colonne c
    return A


def compute_strain_lumped(displacement, dz_pix, n_win=3, n_pix=15):
    """Déformation par éléments distribués (moindres carrés sur fenêtres glissantes)."""
    n_z, n_x = displacement.shape
    A_pinv = np.linalg.pinv(build_lumped_matrix(n_pix, n_win, dz_pix))  # calculé une seule fois

    acc = np.zeros((n_z, n_x))
    cnt = np.zeros((n_z, n_x))
    for i in range(n_z - n_pix + 1):
        for j in range(n_x - n_win + 1):
            d = displacement[i:i + n_pix, j:j + n_win].ravel()   # ordre ligne par ligne
            eps = (A_pinv @ d)[:n_pix - 1]                       # un ε par ressort
            acc[i + 1:i + n_pix, j:j + n_win] += eps[:, None]
            cnt[i + 1:i + n_pix, j:j + n_win] += 1
    return acc / np.maximum(cnt, 1)       # moyenne des fenêtres qui se chevauchent


def compute_strain_polynomial(displacement, z_centers, degre=3):
    n_z, n_x = displacement.shape
    strain = np.zeros_like(displacement, dtype=float)

    for c in range(n_x):

        # Ajustement polynomial du déplacement
        coefficients = np.polyfit(
            z_centers,
            displacement[:, c],
            degre
        )

        # Dérivée du polynôme
        derivee = np.polyder(coefficients)

        # Calcul de la déformation
        strain[:, c] = np.polyval(derivee, z_centers)

    return strain


def show_strain(strain, z_centers, x_axis, titre=""):
    fig, ax = plt.subplots()
    im = ax.imshow(strain, aspect='auto', cmap='jet',
                   extent=[x_axis[0] * 1e3, x_axis[-1] * 1e3,
                           z_centers[-1] * 1e3, z_centers[0] * 1e3])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_title(titre)
    fig.colorbar(im, label='Déformation ε')
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

    ######### Première partie ############

    FMC, Shifted_FMC, Fe, N_elements, Nt, c0, x_array = load_data('BGM_724_DATA_2025.mat')
    #FMC, Shifted_FMC, Fe, N_elements, Nt, c0, x_array = load_data('C:/Users/Camilo/OneDrive - USherbrooke/APP-S8/BGM724/problématique/BGM_724_DATA_2025.mat')

    X, Z = create_imaging_grid()
    image_FMC = reconstruct_saft_image(FMC, N_elements, x_array, X, Z, c0, Fe)
    env_FMC = compute_envelope(image_FMC)

    image_SFMC = reconstruct_saft_image(Shifted_FMC, N_elements, x_array, X, Z, c0, Fe)
    env_SFMC = compute_envelope(image_SFMC)

    # Position des masses et de l'échantillon du background (en m, mêmes unités que X et Z)
    rois = {
        'mass_left': (-0.021, 0.040),
        'mass_right': (0.021, 0.040),
        'background': (0.0, 0.055),
    }
    r = 0.004

    print(len(FMC))

    # Métriques CR / CNR / SNR
    # Les masses descendent d'environ 2 mm dans l'image Shifted (z = 42 mm)
    for env, z0, titre in ((env_FMC, 0.040, "image FMC"),
                           (env_SFMC, 0.042, "image Shifted FMC")):
        masses = {'mass_left': (-0.021, z0), 'mass_right': (0.021, z0)}
        bg_centres = [(x, z0) for x in (-0.032, -0.010, 0.0, 0.010, 0.032)]
        report_metrics(env, X, Z, masses, r, bg_centres, titre)
        show_image(to_db(env), masses, r, bg_centres, titre)


    ################# Deuxième partie #########################

    # Pipeline A: estimate_displacement + compute_strain*
    dz = Z[1, 0] - Z[0, 0]          # pas en profondeur (m)
    x_axis = X[0, :]                # axe x (m)

    L = 100             # fenêtre de L échantillons
    step = 50           # le pas entre chaque fenetre
    max_shift = 44      # décalage max testé pour retrouver la fenêtre 0 Pa dans l'image 50 Pa

    # image_FMC = 0 Pa, image_SFMC = 50 Pa
    disp, corr_max, z_centers = estimate_displacement(image_FMC, image_SFMC, L, step, max_shift, dz)
    show_displacement_m(disp, z_centers, x_axis, titre="Champ de déplacement (0 → 50 Pa)")

    strain = compute_strain(disp, z_centers)
    show_strain(strain, z_centers, x_axis, titre="Champ de déformation (dérivée directe)")

    strain_poly = compute_strain_polynomial(disp, z_centers)
    show_strain(strain_poly, z_centers, x_axis, titre="Champ de déformation (interpolation polynomiale)")

    #Nwin = 3
    #N_pix_ress = 15
    #dz_pix = step * dz          # espacement en z entre deux points de disp (0,8 mm avec step = 10)

    #strain = compute_strain_lumped(disp, dz_pix, Nwin, N_pix_ress)
    #show_strain(strain, z_centers, x_axis, titre="Champ de déformation (éléments distribués)")


    # Pipeline B: champ_deplacement + deformation_poly + rigidite/young_lumped
    # Trouver le decalage optimale
    # Images RF (avant enveloppe) : meilleure précision pour la corrélation
    L = 30          # à ajuster
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