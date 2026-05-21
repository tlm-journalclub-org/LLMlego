"""Modulo 3: accesso ai word embeddings GloVe (inglese, 100D).

API minimale pensata per la lezione:
- `carica_modello()`: una volta sola, carica GloVe
- `vettore(parola)`: ritorna l'array numpy (100,)
- `parola_piu_vicina(v)`: ritorna la parola più vicina al vettore v
- `mostra_vettore(parola)`: heatmap visuale delle 100 componenti
- `widget_vicini()`: campo di testo + lista vicini, in tempo reale
"""
import os
import numpy as np
from typing import Optional, Union, List, Tuple
from IPython.display import HTML, display

# Stato globale del modulo (caricamento pigro)
_model = None
_common_words: Optional[List[str]] = None
_common_vecs_norm: Optional[np.ndarray] = None

# Default: usiamo le top 30k parole più frequenti come spazio "snappabile"
# (GloVe è ordinato per frequenza). Evita di finire su parole obsolete.
_N_COMMON_DEFAULT = 30000

# Possibili percorsi del file GloVe (cwd quando si lancia da repo o da setup)
_GLOVE_CANDIDATES = [
    "glove-wiki-gigaword-100.kv",
    "../glove-wiki-gigaword-100.kv",
    os.path.expanduser("~/glove-wiki-gigaword-100.kv"),
]


def carica_modello(path: Optional[str] = None, n_common: int = _N_COMMON_DEFAULT):
    """Carica GloVe (chiamare una sola volta). Successivi `vettore()` etc.
    funzioneranno senza ricaricare.

    Idempotente: se già caricato, no-op.
    """
    global _model, _common_words, _common_vecs_norm
    if _model is not None:
        return _model

    from gensim.models import KeyedVectors

    if path is not None:
        candidates = [path]
    else:
        candidates = _GLOVE_CANDIDATES

    last_err = None
    for p in candidates:
        if os.path.exists(p):
            _model = KeyedVectors.load(p)
            break
        last_err = f"Non trovato: {p}"
    if _model is None:
        raise FileNotFoundError(
            f"Non riesco a trovare glove-wiki-gigaword-100.kv. {last_err}\n"
            f"Esegui prima `python setup_colab.py`."
        )

    # Precalcola matrice normalizzata delle parole "comuni" (top-N per frequenza)
    keys = list(_model.key_to_index.keys())
    _common_words = keys[:n_common]
    vecs = np.array([_model[w] for w in _common_words], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    _common_vecs_norm = vecs / norms
    return _model


def _ensure_loaded():
    if _model is None:
        carica_modello()


def vettore(parola: str) -> np.ndarray:
    """Ritorna il vettore (100,) della parola in GloVe.

    Solleva KeyError se la parola non è nel vocabolario, con messaggio chiaro.
    """
    _ensure_loaded()
    if parola not in _model.key_to_index:
        raise KeyError(
            f"La parola '{parola}' non è nel vocabolario di GloVe (400k parole inglesi).\n"
            f"Suggerimento: prova in inglese, e in minuscolo (es. 'king' invece di 'King')."
        )
    return _model[parola]


def parola_piu_vicina(
    v: np.ndarray,
    escludi: Optional[Union[str, List[str]]] = None,
    topn: int = 1,
) -> Union[str, List[Tuple[str, float]]]:
    """Trova la parola del vocabolario "comune" (top-30k) più vicina al vettore v.

    - Con topn=1 ritorna una stringa (la parola più vicina).
    - Con topn>1 ritorna lista di (parola, cosine_sim).
    - `escludi`: parola/e da non considerare (utile per evitare di "ritornare" su se stessa).
    """
    _ensure_loaded()
    if isinstance(escludi, str):
        escludi_set = {escludi}
    elif escludi is None:
        escludi_set = set()
    else:
        escludi_set = set(escludi)

    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        v_norm = v
    else:
        v_norm = v / norm
    sims = _common_vecs_norm @ v_norm  # (N,)

    # Prendi piu' del necessario per poter scartare gli esclusi
    k = topn + len(escludi_set) + 3
    if k > len(_common_words):
        k = len(_common_words)
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    risultati: List[Tuple[str, float]] = []
    for i in top_idx:
        w = _common_words[i]
        if w in escludi_set:
            continue
        risultati.append((w, float(sims[i])))
        if len(risultati) >= topn:
            break

    if topn == 1:
        return risultati[0][0]
    return risultati


def mostra_vettore(parola: str):
    """Mostra il vettore di una parola come heatmap orizzontale di 100 celle."""
    _ensure_loaded()
    v = vettore(parola)
    norma = float(np.linalg.norm(v))

    # Normalizza per visualizzazione (per palette colori a banda fissa)
    vmin, vmax = float(v.min()), float(v.max())
    span = max(abs(vmin), abs(vmax)) + 1e-9

    celle = []
    for i, val in enumerate(v):
        # Mappa val in [-1, 1] tramite span; poi a colore blu->bianco->rosso
        t = float(val) / span  # [-1, 1]
        if t >= 0:
            r = 255
            g = int(255 * (1 - t))
            b = int(255 * (1 - t))
        else:
            r = int(255 * (1 + t))
            g = int(255 * (1 + t))
            b = 255
        celle.append(
            f'<div title="dim {i}: {val:+.3f}" '
            f'style="display:inline-block;width:14px;height:30px;'
            f'background:rgb({r},{g},{b});border:1px solid #ddd;'
            f'margin:0;"></div>'
        )

    html = (
        f'<div style="font-family:sans-serif;">'
        f'<div style="margin-bottom:6px;"><b>"{parola}"</b> in GloVe — '
        f'vettore a <b>100 dimensioni</b>, norma {norma:.2f}</div>'
        f'<div style="white-space:nowrap;line-height:0;">{"".join(celle)}</div>'
        f'<div style="font-size:11px;color:#666;margin-top:4px;">'
        f'Rosso = componente positiva, blu = negativa. '
        f'Passa il mouse sopra una cella per vedere il valore.'
        f'</div></div>'
    )
    display(HTML(html))


def widget_vicini(parola_iniziale: str = "king", topn: int = 10):
    """Campo di testo + lista interattiva dei vicini.
    L'utente cambia parola e la lista si aggiorna."""
    _ensure_loaded()
    try:
        import ipywidgets as widgets
    except ImportError:
        raise ImportError("ipywidgets non installato. Esegui setup_colab.py")

    txt = widgets.Text(
        value=parola_iniziale,
        description="Parola:",
        continuous_update=False,
    )
    out = widgets.Output()

    def aggiorna(_=None):
        with out:
            out.clear_output()
            p = txt.value.strip().lower()
            if not p:
                return
            try:
                v = vettore(p)
            except KeyError as e:
                print(str(e))
                return
            vicini = parola_piu_vicina(v, escludi=p, topn=topn)
            righe = []
            for w, s in vicini:
                # barra proporzionale alla similarità
                w_pct = max(0, min(100, int(s * 100)))
                barra = (
                    f'<div style="display:inline-block;background:#3498db;'
                    f'height:10px;width:{w_pct * 2}px;margin-right:8px;'
                    f'vertical-align:middle;"></div>'
                )
                righe.append(
                    f'<tr>'
                    f'<td style="padding:2px 8px;font-family:monospace;">{w}</td>'
                    f'<td style="padding:2px 8px;font-family:monospace;text-align:right;">'
                    f'{s:.3f}</td>'
                    f'<td>{barra}</td>'
                    f'</tr>'
                )
            html = (
                f'<table style="border-collapse:collapse;font-size:13px;">'
                f'<thead><tr style="border-bottom:1px solid #999;">'
                f'<th style="padding:3px 8px;text-align:left;">vicino</th>'
                f'<th style="padding:3px 8px;">cos. sim.</th>'
                f'<th></th></tr></thead>'
                f'<tbody>{"".join(righe)}</tbody></table>'
            )
            display(HTML(html))

    txt.observe(aggiorna, names="value")
    aggiorna()
    display(txt, out)
