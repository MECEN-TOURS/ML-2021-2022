from .data import (
    Cible,
    Echantillon,
    ChoixBruit,
    ChoixPoints,
    genere_echantillon,
)

from .modeles import (
    Modele, 
    ModeleTrigonometrique,
    ModelePolynomial,
)

from .methodologie import (
    erreur_objective,
    erreur_empirique,
    train_test_split,
    cross_validation,
)
