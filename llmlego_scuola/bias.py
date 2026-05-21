"""Modulo 5: visualizzazione del bias di genere negli embeddings."""
import numpy as np
import plotly.graph_objects as go

from .embeddings import _ensure_loaded, vettore


_PROFESSIONI = [
    "doctor", "nurse", "engineer", "teacher", "programmer",
    "secretary", "scientist", "receptionist", "pilot", "librarian",
    "surgeon", "dancer", "mathematician", "hairdresser", "professor",
    "nanny", "lawyer", "maid", "architect", "chef",
]


def mostra_bias_professioni():
    """Proietta una lista di professioni sulla direzione man -> woman in GloVe.

    Restituisce un bar chart orizzontale: valori positivi = professione
    'tira' verso woman, valori negativi = verso man.
    """
    _ensure_loaded()

    # Direzione di genere
    direzione = vettore("woman") - vettore("man")
    direzione_norm = direzione / np.linalg.norm(direzione)

    proiezioni = []
    for p in _PROFESSIONI:
        try:
            v = vettore(p)
        except KeyError:
            continue
        proiezioni.append((p, float(np.dot(v, direzione_norm))))
    proiezioni.sort(key=lambda x: x[1])

    colori = ["#e74c3c" if s < 0 else "#3498db" for _, s in proiezioni]
    fig = go.Figure(go.Bar(
        y=[p for p, _ in proiezioni],
        x=[s for _, s in proiezioni],
        orientation="h",
        marker_color=colori,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Bias di genere nelle professioni — proiezione sull'asse <i>man → woman</i>",
        xaxis_title="←  man                                          woman  →",
        height=600, width=750,
        template="plotly_white",
        margin=dict(l=120),
    )
    fig.show()
