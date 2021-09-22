#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Description.

Objets relatifs aux données.
TODO:
    - Documenter
    - Tester
"""
import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass
from matplotlib.axes import Axes
from enum import Enum, auto


@dataclass
class Cible:
    """Représente une fonction mathématiques sur un segment."""

    fonction: Callable[[np.ndarray], np.ndarray]
    gauche: float
    droite: float

    def __post_init__(self):
        if self.gauche > self.droite:
            raise ValueError(
                "La borne droite de l'intervalle doit être au moins égale à celle de gauche"
            )
        valeurs_temporaires = self.fonction(np.linspace(self.gauche, self.droite, 500))
        m = valeurs_temporaires.min()
        M = valeurs_temporaires.max()
        self.min = m - 0.1 * (M - m)
        self.max = M + 0.1 * (M - m)

    def affichage(self, repere: Axes):
        x_aff = np.linspace(self.gauche, self.droite, 500)
        y_aff = self.fonction(x_aff)

        repere.plot(x_aff, y_aff, linewidth=2.0, linestyle="--", label=r"$y=f(x)$")


@dataclass
class Echantillon:
    """Echantillonage bruitée de la fonction cible.

    On laisse les données réelles pour les afficher même si ce n'est pas réaliste.
    """

    abcisses: np.ndarray
    ordonnees: np.ndarray

    def __post_init__(self):
        if self.abcisses.shape != self.ordonnees.shape:
            raise ValueError("Les tableaux doivent avoir la même forme.")

    def affichage(self, repere: Axes):
        repere.scatter(
            self.abcisses,
            self.ordonnees,
            color="red",
            label="échantillon",
            s=49,
        )


class ChoixPoints(Enum):
    equirepartis = auto()
    uniforme = auto()


class ChoixBruit(Enum):
    normale = auto()
    uniforme = auto()


def genere_echantillon(
    cible: Cible,
    nb_points: int,
    choix_points: ChoixPoints,
    choix_bruit: ChoixBruit,
    amplitude: float,
) -> Echantillon:
    """Echantillonne la cible suivant les options passées."""
    if choix_points == ChoixPoints.uniforme:
        x = np.random.uniform(low=cible.gauche, high=cible.droite, size=(nb_points,))
    elif choix_points == ChoixPoints.equirepartis:
        x = np.linspace(start=cible.gauche, stop=cible.droite, num=nb_points)
    y = cible.fonction(x)
    if choix_bruit == ChoixBruit.uniforme:
        p = y + np.random.uniform(low=-amplitude, high=amplitude, size=y.size)
    elif choix_bruit == ChoixBruit.normale:
        p = y + np.random.normal(loc=0.0, scale=amplitude, size=y.size)
    return Echantillon(
        abcisses=x,
        ordonnees=p,
    )
