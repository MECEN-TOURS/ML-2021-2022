#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Description.

Script permettant de planifier un entrainement par crossvalidation via un fichier de configuration en yaml.
"""
from dataclasses import dataclass
from enum import Enum
from rich import print
from serde import serialize, deserialize
from serde.yaml import to_yaml, from_yaml
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from typing import Optional, Union
import typer


class Strategie(Enum):
    PRIOR = "prior"
    UNIFORM = "uniform"
    STRATIFIED = "stratified"


@serialize
@deserialize
@dataclass
class Benet:
    strategies: Optional[list[Strategie]] = None


@serialize
@deserialize
@dataclass
class Foret:
    nombre_estimateurs: Optional[list[int]] = None


@serialize
@deserialize
@dataclass
class KVoisins:
    nombre_voisins: Optional[list[int]] = None


@serialize
@deserialize
@dataclass
class Neurones:
    architecture: Optional[list[list[int]]] = None


class Solveur(Enum):
    LBFGS = "lgbgs"
    SAG = "sag"
    SAGA = "saga"


@serialize
@deserialize
@dataclass
class RegressionLogistique:
    solveurs: Optional[list[Solveur]] = None
    max_iter: int = 1000


@serialize
@deserialize
@dataclass
class SupportVecteurs:
    C: Optional[list[float]] = None


@serialize
@deserialize
@dataclass
class Config:
    arbre_decision: bool = False
    bayesien_naif: bool = False
    benet: Optional[Benet] = None
    foret: Optional[Foret] = None
    k_voisins: Optional[KVoisins] = None
    neurones: Optional[Neurones] = None
    regression_logistique: Optional[RegressionLogistique] = None
    support_vecteurs: Optional[SupportVecteurs] = None


app = typer.Typer()


@app.command()
def genere(nom_fichier: str = "config.yaml"):
    """Genere un fichier de configuration par défaut."""
    test = Config(
        benet=Benet(strategies=[Strategie.PRIOR, Strategie.STRATIFIED]),
        foret=Foret(),
        k_voisins=KVoisins(),
        neurones=Neurones(
            architecture=[
                [
                    100,
                ],
                [
                    50,
                    50,
                ],
            ]
        ),
        regression_logistique=RegressionLogistique(solveurs=[Solveur.SAG]),
        support_vecteurs=SupportVecteurs(),
    )

    with open(nom_fichier, "w") as fichier:
        fichier.write(to_yaml(test))


def chargement(nom_fichier: str) -> Config:
    """Chargement du fichier de configuration."""
    with open(nom_fichier, "r") as fichier:
        data = fichier.read()

    config = from_yaml(Config, data)
    return config


@app.command()
def affichage(nom_fichier: str):
    """Affiche l'objet généré par le fichier de configuration sans apprentissage."""
    config = chargement(nom_fichier)
    print(config)


@app.command()
def apprentissage(nom_fichier: str):
    """Lance un apprentissage paramétré par le fichier de configuration."""
    config = chargement(nom_fichier)
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    resultats = dict()
    if config.arbre_decision:
        p = DecisionTreeClassifier()
        score = cross_val_score(estimator=p, X=X_tr, y=y_tr).mean()
        resultats[p] = score
    if config.bayesien_naif:
        p = MultinomialNB()
        score = cross_val_score(estimator=p, X=X_tr, y=y_tr).mean()
        resultats[p] = score
    if config.benet is not None:
        p = DummyClassifier()
        if config.benet.strategies is None:
            score = cross_val_score(estimator=p, X=X_tr, y=y_tr).mean()
            resultats[p] = score
        else:
            g = GridSearchCV(
                p,
                param_grid={
                    "strategy": [
                        strategie.value for strategie in config.benet.strategies
                    ]
                },
            )
            g.fit(X_tr, y_tr)
            resultats[g.best_estimator_] = g.best_score_
    if config.foret is not None:
        p = RandomForestClassifier()
        if config.foret.nombre_estimateurs is None:
            score = cross_val_score(estimator=p, X=X_tr, y=y_tr).mean()
            resultats[p] = score
        else:
            g = GridSearchCV(
                p, param_grid={"n_estimators": config.foret.nombre_estimateurs}
            )
            g.fit(X_tr, y_tr)
            resultats[g.best_estimator_] = g.best_score_

    if config.k_voisins is not None:
        p = KNeighborsClassifier()
        if config.k_voisins.nombre_voisins is None:
            score = cross_val_score(estimator=p, X=X_tr, y=y_tr).mean()
            resultats[p] = score
        else:
            g = GridSearchCV(
                p, param_grid={"n_neighbors": config.k_voisins.nombre_voisins}
            )
            g.fit(X_tr, y_tr)
            resultats[g.best_estimator_] = g.best_score_
    if config.neurones is not None:
        p = MLPClassifier()
        if config.neurones.architecture is None:
            score = cross_val_score(estimator=p, X=X_tr, y=y_tr).mean()
            resultats[p] = score
        else:
            g = GridSearchCV(
                p, param_grid={"hidden_layer_sizes": config.neurones.architecture}
            )
            g.fit(X_tr, y_tr)
            resultats[g.best_estimator_] = g.best_score_
    if config.regression_logistique:
        p = LogisticRegression(max_iter=config.regression_logistique.max_iter)
        if config.regression_logistique.solveurs is None:
            score = cross_val_score(estimator=p, X=X_tr, y=y_tr).mean()
            resultats[p] = score
        else:
            g = GridSearchCV(
                p,
                param_grid={
                    "solver": [
                        solveur.value
                        for solveur in config.regression_logistique.solveurs
                    ]
                },
            )
            g.fit(X_tr, y_tr)
            resultats[g.best_estimator_] = g.best_score_
    if config.support_vecteurs is not None:
        p = SVC()
        if config.support_vecteurs.C is None:
            score = cross_val_score(estimator=p, X=X_tr, y=y_tr).mean()
            resultats[p] = score
        else:
            g = GridSearchCV(p, param_grid={"C": config.support_vecteurs.C})
            g.fit(X_tr, y_tr)
            resultats[g.best_estimator_] = g.best_score_

    print(resultats)


if __name__ == "__main__":
    app()
