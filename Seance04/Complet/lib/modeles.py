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
from abc import ABC, abstractmethod

class Modele(ABC):
    @abstractmethod
    def __repr__(self) -> str:
        pass
    
    def __hash__(self):
        return hash(repr(self))
    
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass
        
    def affichage(self, cible: Cible, repere: Axes):
        x_aff = np.linspace(cible.gauche, cible.droite, 500)
        y_aff = self(x_aff)
        repere.plot(x_aff, y_aff, linewidth=3.0, label=f"degre={self.degre}")
        
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


class ModeleTrigonometrique(Modele):
    """Représente les polynômes trigonométriques."""
    def __init__(self, degre: int, gauche: float, droite: float):
        self.degre = degre
        self._gauche = gauche
        self._droite = droite
        self._parametres = np.zeros(shape=(2 * degre + 1,))

    def __repr__(self):
        return f"self.ModeleTrigonometrique(degre={self.degre}, gauche={self._gauche}, droite={self._droite})"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluation standard du polynôme trigonométrique."""
        resultat = 0
        temp = 2 * np.pi * (x - self._gauche) / (self._droite - self._gauche)
        for k, coef in enumerate(self._parametres[::2]):
            resultat = resultat +  coef * np.cos( k * temp)
        for k, coef in enumerate(self._parametres[1::2]):
            resultat = resultat +  coef * np.sin( (k+1) * temp)
         
        return resultat

    
    
class ModelePolynomial(Modele):
    """Représente les polynômes d'un degrés donné."""

    def __init__(self, degre: int):
        self.degre = degre
        self._parametres = np.zeros(shape=(degre + 1,))

    def __repr__(self):
        return f"self.ModelePolynomial(degre={self.degre})"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluation du polynôme par la méthode standard."""
        resultat = 0
        for k, coef in enumerate(self._parametres):
            resultat = resultat +  coef * x ** k
        return resultat
    



