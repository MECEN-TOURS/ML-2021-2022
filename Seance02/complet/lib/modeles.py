#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Description.

Objects relatifs aux modèles.
"""
import numpy as np
from matplotlib.axes import Axes
from scipy.optimize import minimize
from .data import Cible, Echantillon


class ModelePolynomial:
    """Représente les polynômes d'un degrés donné."""

    def __init__(self, degres: int):
        self.degres = degres
        self._parametres = np.zeros(shape=(degres,))

    def __repr__(self):
        return "self.ModelePolynomial(degres={self.degres})"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluation du polynôme par la méthode de Horner pour plus de précision."""
        resultat = np.zeros_like(x)
        for coef in reversed(self._parametres):
            resultat = resultat * x + coef
        return resultat

    def entraine(self, echantillon: Echantillon):
        def a_minimiser(parametres: np.ndarray) -> float:
            self._parametres = parametres
            predictions = self(echantillon.abcisses)
            residus = predictions - echantillon.ordonnees
            return erreur_quadratique(residus)

        resultat = minimize(fun=a_minimiser, x0=self._parametres)
        if not resultat.success:
            raise ValueError("Impossible de déterminer le meilleur prédicteur.")

    def affichage(self, cible: Cible, repere: Axes):
        x_aff = np.linspace(cible.gauche, cible.droite, 500)
        y_aff = self(x_aff)
        repere.plot(x_aff, y_aff, linewidth=3.0, label=f"degres={self.degres}")


def erreur_quadratique(residus: np.ndarray) -> float:
    """Calcul de l'erreur quadratique moyenne."""
    return np.sum(residus ** 2) / len(residus)
