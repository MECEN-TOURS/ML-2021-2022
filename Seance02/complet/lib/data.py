#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Description.

Objets relatifs aux données.
TODO:
    - Rajouter des options pour l'échantillonage
    - Documenter
    - Tester
"""
import numpy as np
from typing import Callable
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


def affichage_cible(cible: Cible, repere: Axes):
    x_aff = np.linspace(cible.gauche, cible.droite, 500)
    y_aff = cible.fonction(x_aff)

    repere.plot(x_aff, y_aff, linewidth=2.0, linestyle="--", label=r"$y=f(x)$")


class Points(Enum):
    equirepartis = auto()
    uniforme = auto()


class Bruit(Enum):
    normale = auto()
    uniforme = auto()


def genere_echantillon(
    cible: Cible,
    nb_points: int,
    choix_points: Points,
    choix_bruit: Bruit,
    amplitude: float,
) -> Echantillon:
    """Echantillonne la cible suivant les options passées."""
    if choix_points == Points.uniforme:
        x = np.random.uniform(low=cible.gauche, high=cible.droite, size=(nb_points,))
    elif choix_points == Points.equirepartis:
        x = np.linspace(start=cible.gauche, stop=cible.droite, num=nb_points)
    y = cible.fonction(x)
    if choix_bruit == Bruit.uniforme:
        p = y + np.random.uniform(low=-amplitude, high=amplitude, size=y.size)
    elif choix_bruit == Bruit.normale:
        p = y + np.random.normal(loc=0.0, scale=amplitude, size=y.size)
    return Echantillon(
        abcisses=x,
        ordonnees=p,
    )


def affichage_echantillon(echantillon: Echantillon, repere: Axes):
    repere.scatter(
        echantillon.abcisses,
        echantillon.ordonnees,
        color="red",
        label="échantillon",
        s=49,
    )
