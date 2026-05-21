"""llmlego_scuola — helpers didattici per la lezione king − man + woman."""

from .tokenizer import mostra_token
from .vocab import mostra_dizionario
from .embeddings import (
    carica_modello,
    vettore,
    parola_piu_vicina,
    mostra_vettore,
    widget_vicini,
)
from .geometria2d import plotta_2d
from .golf import (
    WordGolf, mostra_parole_operatore, classifica, PAROLE_OPERATORE,
    test_dashboard, errori_dashboard,
)
from .bias import mostra_bias_professioni

__all__ = [
    "mostra_token",
    "mostra_dizionario",
    "carica_modello",
    "vettore",
    "parola_piu_vicina",
    "mostra_vettore",
    "widget_vicini",
    "plotta_2d",
    "WordGolf",
    "mostra_parole_operatore",
    "classifica",
    "PAROLE_OPERATORE",
    "test_dashboard",
    "errori_dashboard",
    "mostra_bias_professioni",
]
