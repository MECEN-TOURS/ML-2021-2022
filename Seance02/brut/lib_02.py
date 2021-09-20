"""Description.

Librairie pour l'apprentissage polynomial.

TODO:
    - résoudre le type de "Axes" dans matplotlib
"""
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Callable
from dataclasses import dataclass

@dataclass
class Cible:
    fonction: Callable[[np.ndarray], np.ndarray]
    a: float
    b: float
    
    def __post_init__(self):
        if self.a > self.b:
            raise ValueError("a doit être à gauche de b")

    
@dataclass
class Echantillon:
    x: np.ndarray
    y: np.ndarray # vraie valeur! à garder?
    p: np.ndarray
    # éventuellement post_init pour vérifier qu'ils ont même taille
  

    
@dataclass
class Modele:
    """Représente les polynômes d'un degrés donné."""
    degres: int
    
@dataclass
class Predicteur:
    modele: Modele
    parametres: np.ndarray
    
    def __post_init__(self):
        if len(self.parametres) != self.modele.degres + 1:
            raise ValueError('Le nombre de coefficients doit être égal au degrés plus un.')
            
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return sum(
            coef * x ** k 
            for k, coef in enumerate(self.parametres)
        )

    
def affichage_cible(
    cible: Cible, 
    repere: "Axes"
):
    x_aff = np.linspace(cible.a, cible.b, 500)
    y_aff = cible.fonction(x_aff)

    repere.plot(
        x_aff, 
        y_aff, 
        linewidth=3., 
        label=r"$y=f(x)$"
    )

def genere_echantillon(
    cible: Cible, 
    nb_points: int, 
    sigma: float
) -> Echantillon:
    x = np.random.uniform(low=cible.a, high=cible.b, size=(nb_points,)) 
    y = cible.fonction(x)
    p = y + np.random.normal(loc=0., scale=sigma, size=y.size)
    return Echantillon(
        x=x,
        y=y,
        p=p,
    )

def affichage_echantillon(
    echantillon: Echantillon, 
    repere: "Axes"
):
    repere.scatter(
        echantillon.x, 
        echantillon.y, 
        label="données exactes", 
        color="magenta",
        s=49
    )
    repere.scatter(
        echantillon.x, 
        echantillon.p, 
        color="red", 
        label="données bruitées", 
        s=49
    )

def entraine(echantillon: Echantillon, modele: Modele) -> Predicteur:
    def erreur(parametres: np.ndarray) -> float:
        predicteur = Predicteur(modele, parametres)
        predictions = predicteur(echantillon.x)
        return np.sum((predictions - echantillon.p)**2) / len(predictions)
    
    resultat = minimize(
        fun=erreur,
        x0=np.zeros(modele.degres + 1)
    )
    if not resultat.success:
        raise ValueError("Impossible de déterminer le meilleur prédicteur.")
    return Predicteur(
        modele=modele,
        parametres=resultat.x,
    )
    
def affichage_predicteur(pred: Predicteur, cible: Cible, repere: "Axes"):
    x_aff = np.linspace(cible.a, cible.b, 500)
    y_aff = pred(x_aff)

    repere.plot(
        x_aff, 
        y_aff, 
        linewidth=3., 
        label=f"degres={pred.modele.degres}"
    )