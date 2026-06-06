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
    """Misura il bias di genere di una lista di professioni come *cosine
    similarity* tra il vettore della professione e la direzione di genere
    `woman - man`. Per costruzione i valori sono in [-1, +1]: negativi =
    professione "tira" verso man, positivi = verso woman.
    """
    _ensure_loaded()

    # Direzione di genere normalizzata
    direzione = vettore("woman") - vettore("man")
    direzione_norm = direzione / np.linalg.norm(direzione)

    proiezioni = []
    for p in _PROFESSIONI:
        try:
            v = vettore(p)
        except KeyError:
            continue
        # Cosine similarity: il valore vive in [-1, +1]
        cos_sim = float(np.dot(v, direzione_norm) / np.linalg.norm(v))
        proiezioni.append((p, cos_sim))
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
        title="Bias di genere: cosine similarity con la direzione "
              "<i>woman − man</i>",
        xaxis=dict(
            title="−1  ←  man                                 woman  →  +1",
            range=[-1, 1],
            zeroline=True, zerolinewidth=1, zerolinecolor="#888",
        ),
        height=600, width=750,
        template="plotly_white",
        margin=dict(l=120),
    )
    fig.show()
