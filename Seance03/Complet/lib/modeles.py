#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Description.

Objets relatifs aux modèles.
"""
import numpy as np
import sys
from matplotlib.axes import Axes
from scipy.optimize import least_squares
from .data import Cible, Echantillon


class ModelePolynomial:
    """Représente les polynômes d'un degrés donné."""

    def __init__(self, degre: int):
        self.degre = degre
        self._parametres = np.zeros(shape=(degre + 1,))

    def __repr__(self):
        return "self.ModelePolynomial(degres={self.degres})"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluation du polynôme par la méthode de Horner pour plus de précision."""
        resultat = np.zeros_like(x)
        for coef in self._parametres[::-1]:
            resultat = resultat * x + coef
        return resultat

    def entraine(self, echantillon: Echantillon):
        """Entrainement par minimisation des erreurs au sens des moindre carrés."""

        def calcule_residus(parametres: np.ndarray) -> np.ndarray:
            """Calcule les résidus pour résoudre le problème de moindre carrés ensuite."""
            self._parametres = parametres
            return self(echantillon.abcisses) - echantillon.ordonnees

        resultat = least_squares(
            fun=calcule_residus, x0=np.zeros_like(self._parametres)
        )
        if not resultat.success:
            print("Impossible de faire converger le solveur.")
            sys.exit(1)

    def affichage(self, cible: Cible, repere: Axes):
        x_aff = np.linspace(cible.gauche, cible.droite, 500)
        y_aff = self(x_aff)
        repere.plot(x_aff, y_aff, linewidth=3.0, label=f"degre={self.degre}")


def erreur_quadratique(residus: np.ndarray) -> float:
    """Calcul de l'erreur quadratique moyenne."""
    return np.sum(residus ** 2) / len(residus)


def erreur_empirique(modele: ModelePolynomial, echantillon: Echantillon) -> float:
    """Erreur quadratique moyenne calculée sur l'échantillon."""
    return erreur_quadratique(modele(echantillon.abcisses) - echantillon.ordonnees)


def erreur_objective(
    modele: ModelePolynomial, cible: Cible, nb_points: int = 500
) -> float:
    """Erreur quadratique moyenne calculée directement sur la cible."""
    x = np.linspace(cible.gauche, cible.droite, nb_points)
    return erreur_quadratique(modele(x) - cible.fonction(x))
