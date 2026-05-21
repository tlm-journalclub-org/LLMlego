"""Modulo 2: visualizzazione di parole in 2D (vettori inventati a mano)."""
from typing import Dict, Optional, Tuple
import numpy as np
import plotly.graph_objects as go


_COLORI_BASE = "#1f77b4"
_COLORE_EVIDENZA = "#d62728"


def plotta_2d(
    spazio: Dict[str, Tuple[float, float]],
    evidenzia: Optional[Dict[str, Tuple[float, float]]] = None,
    frecce_dall_origine: bool = True,
    titolo: str = "Le parole come vettori 2D",
):
    """Scatter Plotly delle parole con frecce dall'origine.

    Parametri
    ---------
    spazio
        Dizionario parola -> (x, y). Mostrate in blu.
    evidenzia
        Dizionario opzionale di punti aggiuntivi da evidenziare in rosso
        (es. il risultato di un'operazione `re - uomo + donna`).
    frecce_dall_origine
        Se True (default), disegna ogni vettore come freccia (0,0) -> (x,y).
    """
    fig = go.Figure()

    # Frecce dall'origine, se richiesto
    if frecce_dall_origine:
        for parola, (x, y) in spazio.items():
            fig.add_annotation(
                x=x, y=y, ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.2,
                arrowwidth=1.5, arrowcolor="rgba(31,119,180,0.45)",
            )
        if evidenzia:
            for parola, (x, y) in evidenzia.items():
                fig.add_annotation(
                    x=x, y=y, ax=0, ay=0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1.4,
                    arrowwidth=2, arrowcolor=_COLORE_EVIDENZA,
                )

    # Punti delle parole
    parole = list(spazio.keys())
    xs = [spazio[p][0] for p in parole]
    ys = [spazio[p][1] for p in parole]
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        text=parole,
        textposition="top center",
        marker=dict(size=12, color=_COLORI_BASE),
        name="parole",
        hovertemplate="<b>%{text}</b><br>(%{x:.2f}, %{y:.2f})<extra></extra>",
    ))

    # Punti evidenziati
    if evidenzia:
        ev_parole = list(evidenzia.keys())
        ex = [evidenzia[p][0] for p in ev_parole]
        ey = [evidenzia[p][1] for p in ev_parole]
        fig.add_trace(go.Scatter(
            x=ex, y=ey,
            mode="markers+text",
            text=ev_parole,
            textposition="top center",
            marker=dict(size=14, color=_COLORE_EVIDENZA, symbol="star"),
            name="evidenziato",
            hovertemplate="<b>%{text}</b><br>(%{x:.2f}, %{y:.2f})<extra></extra>",
        ))

    # Origine + assi
    fig.add_shape(type="line", x0=-1.2, x1=1.2, y0=0, y1=0,
                  line=dict(color="#999", width=1, dash="dot"))
    fig.add_shape(type="line", x0=0, x1=0, y0=-1.2, y1=1.2,
                  line=dict(color="#999", width=1, dash="dot"))

    # Calcola range adattivo
    tutti_x = xs + (list(ex) if evidenzia else []) + [0]
    tutti_y = ys + (list(ey) if evidenzia else []) + [0]
    rx = max(abs(min(tutti_x)), abs(max(tutti_x))) * 1.3 + 0.2
    ry = max(abs(min(tutti_y)), abs(max(tutti_y))) * 1.3 + 0.2

    fig.update_layout(
        title=titolo,
        xaxis=dict(range=[-rx, rx], zeroline=False, title="dimensione 1"),
        yaxis=dict(range=[-ry, ry], zeroline=False, title="dimensione 2",
                   scaleanchor="x", scaleratio=1),
        width=650, height=550,
        template="plotly_white",
        showlegend=False,
    )
    fig.show()
