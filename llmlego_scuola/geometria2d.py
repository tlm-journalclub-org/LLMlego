"""Modulo 2: visualizzazione di parole in 2D (vettori inventati a mano)."""
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML, display


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


def _cosine_sim_2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Cosine similarity tra due vettori 2D."""
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(np.dot(va, vb) / norm)


def mostra_similarita_2d(
    spazio: Dict[str, Tuple[float, float]],
    parola_a: str,
    parola_b: str,
    titolo: Optional[str] = None,
):
    """Visualizza due vettori 2D dall'origine, l'angolo θ tra loro,
    e il valore di cos(θ) = cosine similarity.

    Pensato per il Modulo 2: far vedere intuitivamente che
    cos(θ) = 1 (stessa direzione), 0 (perpendicolari), -1 (opposti).
    """
    if parola_a not in spazio or parola_b not in spazio:
        raise KeyError(
            f"parole non nello spazio: {parola_a!r} o {parola_b!r}"
        )
    a = spazio[parola_a]
    b = spazio[parola_b]
    cos_sim = _cosine_sim_2d(a, b)
    # angolo in gradi
    theta_rad = math.acos(max(-1.0, min(1.0, cos_sim)))
    theta_deg = math.degrees(theta_rad)

    fig = go.Figure()
    # vettori dall'origine
    for (x, y), nome, colore in [
        (a, parola_a, "#1f77b4"),
        (b, parola_b, "#d62728"),
    ]:
        fig.add_annotation(
            x=x, y=y, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.4,
            arrowwidth=2.5, arrowcolor=colore,
        )
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            text=[nome],
            textposition="top center",
            marker=dict(size=12, color=colore),
            showlegend=False,
            hovertemplate=f"<b>{nome}</b><br>(%{{x:.2f}}, %{{y:.2f}})<extra></extra>",
        ))

    # arco che indica l'angolo θ, vicino all'origine
    raggio_arco = 0.25
    angolo_a = math.atan2(a[1], a[0])
    angolo_b = math.atan2(b[1], b[0])
    # disegna l'arco minore tra i due angoli
    a_start, a_end = min(angolo_a, angolo_b), max(angolo_a, angolo_b)
    # arco minore: se la differenza > π, prendi il complementare
    if a_end - a_start > math.pi:
        a_start, a_end = a_end, a_start + 2 * math.pi
    arc_t = np.linspace(a_start, a_end, 40)
    arc_x = raggio_arco * np.cos(arc_t)
    arc_y = raggio_arco * np.sin(arc_t)
    fig.add_trace(go.Scatter(
        x=arc_x, y=arc_y,
        mode="lines",
        line=dict(color="#888", width=1.5, dash="dot"),
        showlegend=False,
        hoverinfo="skip",
    ))
    # label θ al centro dell'arco
    arc_mid = (a_start + a_end) / 2
    fig.add_annotation(
        x=(raggio_arco + 0.08) * math.cos(arc_mid),
        y=(raggio_arco + 0.08) * math.sin(arc_mid),
        text=f"θ ≈ {theta_deg:.0f}°",
        showarrow=False,
        font=dict(size=14, color="#444"),
    )

    # Origine + assi tratteggiati
    fig.add_shape(type="line", x0=-1.2, x1=1.2, y0=0, y1=0,
                  line=dict(color="#bbb", width=1, dash="dot"))
    fig.add_shape(type="line", x0=0, x1=0, y0=-1.2, y1=1.2,
                  line=dict(color="#bbb", width=1, dash="dot"))

    # Range
    xs = [a[0], b[0], 0]
    ys = [a[1], b[1], 0]
    r = max(max(abs(v) for v in xs + ys), 1.0) * 1.25 + 0.1

    titolo_finale = titolo or (
        f"cos(θ) tra <b>{parola_a}</b> e <b>{parola_b}</b> = "
        f"<b>{cos_sim:+.3f}</b>"
    )
    fig.update_layout(
        title=titolo_finale,
        xaxis=dict(range=[-r, r], zeroline=False, title="dimensione 1"),
        yaxis=dict(range=[-r, r], zeroline=False, title="dimensione 2",
                   scaleanchor="x", scaleratio=1),
        width=550, height=500,
        template="plotly_white",
        showlegend=False,
    )
    fig.show()


def tabella_similarita_2d(
    spazio: Dict[str, Tuple[float, float]],
    coppie: list,
):
    """Tabella HTML con cosine similarity per una lista di coppie.

    `coppie` è una lista di tuple (parola_a, parola_b).
    """
    righe = []
    for a, b in coppie:
        if a not in spazio or b not in spazio:
            continue
        sim = _cosine_sim_2d(spazio[a], spazio[b])
        # barra orizzontale proporzionale (sim può essere negativa)
        pct = int(abs(sim) * 100)
        colore = "#3498db" if sim >= 0 else "#e74c3c"
        barra = (
            f'<div style="display:inline-block;height:10px;width:{pct}px;'
            f'background:{colore};vertical-align:middle;"></div>'
        )
        righe.append(
            f'<tr>'
            f'<td style="padding:3px 8px;font-family:monospace;">{a}</td>'
            f'<td style="padding:3px 8px;font-family:monospace;">{b}</td>'
            f'<td style="padding:3px 8px;text-align:right;font-family:monospace;">'
            f'{sim:+.3f}</td>'
            f'<td style="padding:3px 8px;">{barra}</td>'
            f'</tr>'
        )
    html = (
        f'<div style="font-family:sans-serif;">'
        f'<table style="border-collapse:collapse;font-size:13px;">'
        f'<thead><tr style="border-bottom:1px solid #999;">'
        f'<th style="padding:3px 8px;">parola A</th>'
        f'<th style="padding:3px 8px;">parola B</th>'
        f'<th style="padding:3px 8px;">cos(θ)</th>'
        f'<th></th></tr></thead>'
        f'<tbody>{"".join(righe)}</tbody></table>'
        f'</div>'
    )
    display(HTML(html))
