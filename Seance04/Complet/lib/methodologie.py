"""Description.

Tout ce qui concerne la façon de sélectionner le bon modèle.
"""
import numpy as np
from .data import Cible, Echantillon
from .modeles import Modele
from random import random

def erreur_quadratique(residus: np.ndarray) -> float:
    """Calcul de l'erreur quadratique moyenne."""
    return np.sum(residus ** 2) / len(residus)


def erreur_empirique(modele: Modele, echantillon: Echantillon) -> float:
    """Erreur quadratique moyenne calculée sur l'échantillon."""
    return erreur_quadratique(modele(echantillon.abcisses) - echantillon.ordonnees)


def erreur_objective(
    modele: Modele, cible: Cible, nb_points: int = 500
) -> float:
    """Erreur quadratique moyenne calculée directement sur la cible."""
    x = np.linspace(cible.gauche, cible.droite, nb_points)
    return erreur_quadratique(modele(x) - cible.fonction(x))

def train_test_split(
        echantillon: Echantillon, 
        proba_train: float
) -> tuple[Echantillon, Echantillon]:
    """Découpage d'un échantillon en deux."""
    train_x, train_y, test_x, test_y = [[] for _ in range(4)]
    for x,y in echantillon:
        if random() < proba_train:
            train_x.append(x)
            train_y.append(y)
        else:
            test_x.append(x)
            test_y.append(y)
    train = Echantillon(
        abcisses=np.array(train_x), 
        ordonnees=np.array(train_y),
    )
    test = Echantillon(
        abcisses=np.array(test_x), 
        ordonnees=np.array(test_y),
    )
    return (train, test)

def cross_validation(
    echantillon: Echantillon, 
    modeles: list[Modele]
) -> Modele:
    """Sélection du meilleur modéle par cross-validation sur l'échantillon."""
    a_decouper, ech3 = train_test_split(
        echantillon, 
        proba_train=0.66,
    )
    ech1, ech2 = train_test_split(a_decouper, proba_train=0.5)
    resultats = dict()
    for modele in modeles:
        erreurs = list()
        modele.entraine(ech1 + ech2)
        erreurs.append(erreur_empirique(modele, ech3))
        modele.entraine(ech1 + ech3)
        erreurs.append(erreur_empirique(modele, ech2))
        modele.entraine(ech3 + ech2)
        erreurs.append(erreur_empirique(modele, ech1))
        resultats[modele] = sum(erreurs)
    
    meilleur, _ = min(resultats.items(), key=lambda tu: tu[1])
    meilleur.entraine(echantillon)
    return meilleur
    
    
        
        