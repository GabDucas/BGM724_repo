import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

CHEMIN_DONNEES = 'BGM_724_DATA_2025.mat'

# Grille de l'image
DZ = 0.08            # taille d'un pixel en profondeur (mm)
DX = 0.8             # taille d'un pixel en largeur (mm)

# Estimation du déplacement
L = 40               # longueur de la fenêtre de corrélation (échantillons)
PAS = 10             # pas entre deux fenêtres (échantillons)
R_SUIVI = 6          # plage de recherche autour du décalage précédent (échantillons)
Z_MIN = 10.0         # on ignore le champ proche, trop bruité (mm)

# Estimation de la rigidité
NWIN = 3             # colonnes par bloc
N_PIX_RESS = 15      # points de déplacement (ressorts) par bloc
SIGMA = 50.0         # contrainte appliquée (Pa)

# Régions d'intérêt : (x, z) en mm
ROIS = {'masse_gauche': (-21, 40), 'masse_droite': (21, 40), 'fond': (0, 55)}
R_ROI = 4.0          # rayon des régions (mm)


# ---------------------------------------------------------------- Étape 1 : images

def charger_donnees(chemin):
    donnees = loadmat(chemin)
    FMC = donnees['FMC'].real
    FMC_decalee = donnees['shifted_FMC'].real
    Fe = float(donnees['Fe'].squeeze())
    c0 = float(donnees['c0'].squeeze())
    x_elements = donnees['x_array'].squeeze()
    return FMC, FMC_decalee, Fe, c0, x_elements


def creer_grille():
    x = np.arange(-0.04, 0.04, DX * 1e-3)
    z = np.arange(0, 0.08, DZ * 1e-3)
    return np.meshgrid(x, z)


def reconstruire_image(FMC, x_elements, X, Z, c0, Fe):
    """Delay-and-sum : pour chaque pixel, on additionne l'échantillon au temps aller-retour."""
    image = np.zeros(X.shape)
    nb_echantillons = FMC.shape[2]
    distance = [np.sqrt((X - x_elem)**2 + Z**2) for x_elem in x_elements]

    for emetteur in range(len(x_elements)):
        for recepteur in range(len(x_elements)):
            indice = (distance[emetteur] + distance[recepteur]) / c0 * Fe   # indice fractionnaire
            bas = np.floor(indice).astype(int)
            frac = indice - bas
            valide = (bas >= 0) & (bas < nb_echantillons - 1)

            # Interpolation linéaire entre les deux échantillons voisins
            avant = FMC[emetteur, recepteur, bas[valide]]
            apres = FMC[emetteur, recepteur, bas[valide] + 1]
            image[valide] += (1 - frac[valide]) * avant + frac[valide] * apres

    return image


# ---------------------------------------------------------------- Étape 2 : déplacement

def correlation(x, y):
    """Corrélation normalisée entre deux fenêtres (1 = même motif)."""
    x = x - x.mean()
    y = y - y.mean()
    denominateur = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if denominateur == 0:
        return 0.0
    return np.sum(x * y) / denominateur


def courbe_correlation(sig1, sig2, debut, decalages):
    """Corrélation entre une fenêtre de sig1 et la fenêtre de sig2 décalée de chaque valeur."""
    reference = sig1[debut:debut + L]
    scores = []
    for decalage in decalages:
        fenetre = sig2[debut + decalage:debut + decalage + L]
        scores.append(correlation(reference, fenetre))
    return np.array(scores)


def affiner_pic(scores, k):
    """Position du pic sous l'échantillon (parabole sur 3 points), entre -0,5 et 0,5."""
    if k == 0 or k == len(scores) - 1:
        return 0.0
    gauche, milieu, droite = scores[k - 1], scores[k], scores[k + 1]
    denominateur = gauche - 2 * milieu + droite
    if denominateur >= 0:
        return 0.0
    return 0.5 * (gauche - droite) / denominateur


def suivre_colonne(sig1, sig2):
    """
    Déplacement le long d'une colonne. On part du bas, où le tissu ne bouge pas,
    et on remonte : chaque recherche est centrée sur le résultat précédent.
    """
    n = len(sig1)
    debuts = range(n - L - R_SUIVI, -1, -PAS)
    centres, deplacements = [], []
    precedent = 0.0

    for debut in debuts:
        centre = int(round(precedent))
        plus_bas = max(centre - R_SUIVI, -debut)
        plus_haut = min(centre + R_SUIVI, n - L - debut)
        centres.append(debut + L // 2)

        if plus_haut < plus_bas:
            deplacements.append(precedent)
            continue

        decalages = np.arange(plus_bas, plus_haut + 1)
        scores = courbe_correlation(sig1, sig2, debut, decalages)
        k = int(np.argmax(scores))
        decalage = decalages[k] + affiner_pic(scores, k)

        if scores[k] > 0.5:         # on ne suit que les estimations fiables
            precedent = decalage
        deplacements.append(decalage)

    # On retourne de haut en bas
    return np.array(centres[::-1]), np.array(deplacements[::-1])


def champ_deplacement(image1, image2):
    """Déplacement (mm) pour chaque colonne. Forme : (nb_fenetres, nb_colonnes)."""
    nb_colonnes = image1.shape[1]
    colonnes = []
    for j in range(nb_colonnes):
        centres, decalage = suivre_colonne(image1[:, j], image2[:, j])
        colonnes.append(decalage)

    z = centres * DZ                                  # profondeur de chaque fenêtre (mm)
    D = np.array(colonnes).T * DZ                     # décalage en échantillons -> mm

    garder = z >= Z_MIN
    return z[garder], D[garder]


# ---------------------------------------------------------------- Étape 3 : déformation et rigidité

def deformation_bloc(D, pas_mm):
    """Déformation d'un bloc = compression de ses ressorts / hauteur initiale du bloc."""
    hauteur = N_PIX_RESS * pas_mm
    haut = D[:-N_PIX_RESS]
    bas = D[N_PIX_RESS:]
    return (haut - bas) / hauteur         # positive quand le tissu est comprimé


def module_young(deformation):
    """
    Pour chaque bloc, moindres carrés sur sigma = E * déformation, sur NWIN colonnes :
    E = sigma * somme(déformation) / somme(déformation^2)
    """
    nb_lignes, nb_colonnes = deformation.shape
    deformation_moy = np.full(deformation.shape, np.nan)
    E = np.full(deformation.shape, np.nan)

    for i in range(nb_lignes):
        for j in range(nb_colonnes):
            premiere = min(max(j - NWIN // 2, 0), nb_colonnes - NWIN)
            e = deformation[i, premiere:premiere + NWIN]
            somme_carres = np.sum(e**2)
            if somme_carres > 0:
                deformation_moy[i, j] = np.mean(e)
                E[i, j] = SIGMA * np.sum(e) / somme_carres

    E[E <= 0] = np.nan
    return deformation_moy, E


# ---------------------------------------------------------------- Étape 4 : résultats

def afficher_carte(ax, M, x, z, titre, etiquette, vmax=None, dessiner_rois=False):
    image = ax.imshow(M, extent=[x[0], x[-1], z[-1], z[0]], aspect='auto',
                      cmap='jet', vmin=0, vmax=vmax)
    if dessiner_rois:
        for x_c, z_c in ROIS.values():
            ax.add_patch(plt.Circle((x_c, z_c), R_ROI, fill=False, color='w'))
    ax.set_title(titre)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    plt.colorbar(image, ax=ax, label=etiquette)


def afficher_valeurs_roi(E, x, z):
    X, Z = np.meshgrid(x, z)
    for nom, (x_c, z_c) in ROIS.items():
        dedans = (X - x_c)**2 + (Z - z_c)**2 <= R_ROI**2
        valeurs = E[dedans & np.isfinite(E)]
        print(f"{nom:13s} E = {np.median(valeurs):.2f} kPa (médiane), écart-type {valeurs.std():.2f}")


def main():
    FMC, FMC_decalee, Fe, c0, x_elements = charger_donnees(CHEMIN_DONNEES)
    X, Z = creer_grille()

    image1 = reconstruire_image(FMC, x_elements, X, Z, c0, Fe)
    image2 = reconstruire_image(FMC_decalee, x_elements, X, Z, c0, Fe)

    z_depl, D = champ_deplacement(image1, image2)

    pas_mm = PAS * DZ
    deformation = deformation_bloc(D, pas_mm)
    deformation_moy, E = module_young(deformation)

    x_mm = X[0, :] * 1e3
    z_blocs = z_depl[:-N_PIX_RESS] + N_PIX_RESS * pas_mm / 2
    deformation_pct = 100 * deformation_moy
    E_kpa = E / 1e3

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    afficher_carte(axes[0], D, x_mm, z_depl, 'Déplacement', 'mm')
    afficher_carte(axes[1], deformation_pct, x_mm, z_blocs, 'Déformation', '%',
                   np.nanpercentile(deformation_pct, 98), dessiner_rois=True)
    afficher_carte(axes[2], E_kpa, x_mm, z_blocs, "Module d'Young", 'kPa',
                   np.nanpercentile(E_kpa, 98), dessiner_rois=True)
    fig.tight_layout()

    afficher_valeurs_roi(E_kpa, x_mm, z_blocs)
    plt.show()


if __name__ == "__main__":
    main()