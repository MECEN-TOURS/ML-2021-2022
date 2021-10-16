#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Description.

Démonstration de
    - pyserde pour la sérialisation,
    - typer pour la génération d'un CLI.
"""
from enum import Enum
from dataclasses import dataclass
from serde import serialize, deserialize
from serde.yaml import from_yaml, to_yaml
from rich import print
import typer


class Cote(Enum):
    GAUCHE = "gauche"
    DROIT = "droit"


@serialize
@deserialize
@dataclass
class Etat:
    berger: Cote
    loup: Cote
    mouton: Cote
    chou: Cote


Arrete = tuple[Etat, Etat]


@serialize
@deserialize
@dataclass
class Graphe:
    sommets: list[Etat]
    arretes: list[Arrete]


def genere_voisins(depart: Etat, graphe: Graphe) -> list[Etat]:
    return [droite for gauche, droite in graphe.arretes if gauche == depart]


def ablation(a_enlever: Etat, graphe: Graphe) -> Graphe:
    sommets = [sommet for sommet in graphe.sommets if sommet != a_enlever]
    arretes = [
        (gauche, droite)
        for gauche, droite in graphe.arretes
        if gauche != a_enlever and droite != a_enlever
    ]
    return Graphe(sommets, arretes)


def a_solution(depart: Etat, arrivee: Etat, graphe: Graphe) -> bool:
    if depart == arrivee:
        return True
    for voisin in genere_voisins(depart=depart, graphe=graphe):
        if a_solution(
            depart=voisin,
            arrivee=arrivee,
            graphe=ablation(a_enlever=depart, graphe=graphe),
        ):
            return True
    return False


@serialize
@deserialize
@dataclass
class Donnees:
    depart: Etat
    arrivee: Etat
    graphe: Graphe


app = typer.Typer()


@app.command()
def test():
    """Effectue les tests interne du solver."""
    depart = Etat(
        berger=Cote.GAUCHE,
        loup=Cote.GAUCHE,
        mouton=Cote.GAUCHE,
        chou=Cote.GAUCHE,
    )
    arrivee = Etat(
        berger=Cote.DROIT,
        loup=Cote.DROIT,
        mouton=Cote.DROIT,
        chou=Cote.DROIT,
    )
    graphe_vide = Graphe([depart, arrivee], [])
    assert not a_solution(depart, arrivee, graphe_vide)

    graphe_bateau = Graphe([depart, arrivee], [(depart, arrivee)])
    assert a_solution(depart, arrivee, graphe_bateau)


@app.command()
def exemple(nom_fichier: str = "config.yaml"):
    """Génère un fichier de DONNEES minimal."""
    depart = Etat(
        berger=Cote.GAUCHE,
        loup=Cote.GAUCHE,
        mouton=Cote.GAUCHE,
        chou=Cote.GAUCHE,
    )
    arrivee = Etat(
        berger=Cote.DROIT,
        loup=Cote.DROIT,
        mouton=Cote.DROIT,
        chou=Cote.DROIT,
    )
    graphe_bateau = Graphe([depart, arrivee], [(depart, arrivee)])
    donnees = Donnees(depart, arrivee, graphe_bateau)
    with open(nom_fichier, "w") as fichier:
        fichier.write(to_yaml(donnees))


@app.command()
def solver(fichier_donnees: str):
    """Décide si il existe un chemin reliant DEPART et ARRIVEE dans le GRAPHE."""
    with open(fichier_donnees, "r") as fichier:
        data = fichier.read()

    donnees = from_yaml(Donnees, data)
    print(f"DEPART:")
    print(donnees.depart)
    print(f"ARRIVEE:")
    print(donnees.arrivee)
    print(f"GRAPHE:")
    print(donnees.graphe)
    if a_solution(donnees.depart, donnees.arrivee, donnees.graphe):
        message = "Il existe un chemin!"
    else:
        message = "Il n'y a pas de chemin!"
    total = typer.style(
        message, bg=typer.colors.BRIGHT_BLACK, fg=typer.colors.BRIGHT_RED, bold=True
    )
    typer.echo(total)


if __name__ == "__main__":
    app()
