"""Modulo 4: Word Golf.

Regole:
- Si parte da `start`, si vuole raggiungere `target`.
- A ogni mossa l'utente sceglie `+ parola` o `- parola` da una lista curata.
- La nuova parola corrente è quella più vicina al vettore risultante (escludendo
  la parola attuale per evitare loop).
- Vince quando il `target` compare nei top-5 vicini del vettore corrente.
"""
from typing import Optional, List, Dict, Tuple
import numpy as np
from IPython.display import HTML, display
import plotly.graph_objects as go

from .embeddings import _ensure_loaded, vettore, parola_piu_vicina


# Le 57 parole-operatore validate. Coprono: genere, dimensione, animali/natura,
# paesi, cibo, tempo/meteo, astratto, azioni, tech.
PAROLE_OPERATORE = [
    # genere / persone
    "man", "woman", "boy", "girl", "king", "queen",
    # dimensione / fisicita'
    "big", "small", "tall", "short",
    # animali / natura
    "bird", "fish", "dog", "horse", "tree", "flower",
    "mountain", "sea", "river", "sky", "fire", "stone",
    # paesi / luoghi
    "italy", "france", "japan", "america", "city", "country", "rome", "paris",
    # cibo
    "bread", "fruit", "meat", "water", "wine",
    # tempo / meteo
    "summer", "spring", "day", "night", "sun", "rain", "snow",
    # astratto
    "love", "fear", "music", "war", "peace",
    "money", "power", "knowledge", "death", "life",
    # azione / movimento
    "run", "fly", "walk",
    # tech
    "computer", "machine",
]

# Categorie per la visualizzazione
_CATEGORIE = {
    "👥 persone": ["man", "woman", "boy", "girl", "king", "queen"],
    "📏 dimensione": ["big", "small", "tall", "short"],
    "🐾 animali e natura": ["bird", "fish", "dog", "horse", "tree", "flower",
                            "mountain", "sea", "river", "sky", "fire", "stone"],
    "🌍 luoghi": ["italy", "france", "japan", "america", "city", "country",
                  "rome", "paris"],
    "🍞 cibo": ["bread", "fruit", "meat", "water", "wine"],
    "🌤️ tempo": ["summer", "spring", "day", "night", "sun", "rain", "snow"],
    "💭 astratto": ["love", "fear", "music", "war", "peace",
                   "money", "power", "knowledge", "death", "life"],
    "🏃 azione": ["run", "fly", "walk"],
    "💻 tech": ["computer", "machine"],
}

# Quanti top-K considerare per dichiarare "vittoria"
_TOP_K_VITTORIA = 5

# Registro classifiche multi-partita (per usare classifica() in fondo al modulo)
_RECORD: List[Dict] = []


def mostra_parole_operatore():
    """Mostra la griglia delle parole disponibili come operatori."""
    sezioni = []
    for cat, parole in _CATEGORIE.items():
        chips = []
        for p in parole:
            chips.append(
                f'<span style="display:inline-block;padding:3px 9px;'
                f'background:#eef;border:1px solid #ccd;border-radius:12px;'
                f'margin:2px;font-family:monospace;font-size:12px;">{p}</span>'
            )
        sezioni.append(
            f'<div style="margin-bottom:6px;">'
            f'<span style="font-weight:600;font-size:13px;">{cat}</span><br>'
            f'{"".join(chips)}'
            f'</div>'
        )
    html = (
        f'<div style="font-family:sans-serif;">'
        f'<div style="margin-bottom:8px;"><b>{len(PAROLE_OPERATORE)} parole disponibili</b> '
        f'come operatori (puoi aggiungerle o sottrarle ad ogni mossa).</div>'
        f'{"".join(sezioni)}'
        f'</div>'
    )
    display(HTML(html))


class WordGolf:
    """Una partita di Word Golf."""

    def __init__(self, start: str, target: str, squadra: Optional[str] = None):
        _ensure_loaded()
        try:
            self._start_v = vettore(start)
            self._target_v = vettore(target)
        except KeyError as e:
            raise KeyError(f"Parola non in vocabolario: {e}") from e

        self.start = start
        self.target = target
        self.squadra = squadra or "anonima"

        # Stato interno: parola corrente + sua storia.
        # Ogni mossa parte dal VETTORE della parola attuale (snap-reset),
        # non dal vettore accumulato: questo rende il gioco trasparente
        # ("sono qui, applico questa operazione, arrivo li'") e coerente
        # con il modo in cui interpretiamo l'aritmetica vettoriale nel Modulo 3.
        self.parola_corrente = start
        self.mosse: List[Tuple[str, str, str]] = []  # (segno, op, risultato)
        self.vinto = False

        self._intro()

    # ---- API utente ----

    def aggiungi(self, parola: str):
        """Esegue cur += vettore(parola), poi snappa al vicino più prossimo."""
        return self._mossa("+", parola)

    def sottrai(self, parola: str):
        """Esegue cur -= vettore(parola), poi snappa al vicino più prossimo."""
        return self._mossa("-", parola)

    def stato(self):
        """Mostra parola attuale, mosse fatte, distanza dal target."""
        self._stampa_stato()

    def visualizza_percorso(self):
        """Plot 2D del percorso: parole visitate + target, ridotte in 2D via PCA."""
        from sklearn.decomposition import PCA
        parole_path = [self.start] + [m[2] for m in self.mosse]
        # Aggiungiamo il target per averlo nello stesso piano PCA
        tutte = parole_path + [self.target]
        vecs = np.array([vettore(p) for p in tutte])
        pca = PCA(n_components=2).fit(vecs)
        coords = pca.transform(vecs)

        path_xy = coords[:-1]
        target_xy = coords[-1]

        fig = go.Figure()
        # Path
        fig.add_trace(go.Scatter(
            x=path_xy[:, 0], y=path_xy[:, 1],
            mode="lines+markers+text",
            text=parole_path,
            textposition="top center",
            line=dict(color="#3498db", width=2),
            marker=dict(size=10, color="#3498db"),
            name="il tuo cammino",
        ))
        # Target
        fig.add_trace(go.Scatter(
            x=[target_xy[0]], y=[target_xy[1]],
            mode="markers+text",
            text=[f"🎯 {self.target}"],
            textposition="top center",
            marker=dict(size=18, color="#d62728", symbol="star"),
            name="target",
        ))
        # Frecce tra step consecutivi
        for i in range(len(parole_path) - 1):
            fig.add_annotation(
                x=path_xy[i + 1, 0], y=path_xy[i + 1, 1],
                ax=path_xy[i, 0], ay=path_xy[i, 1],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.4,
                arrowcolor="#3498db",
            )
        fig.update_layout(
            title=f"{self.squadra}: {self.start} → {self.target} ({len(self.mosse)} mosse)",
            template="plotly_white",
            width=750, height=550,
            xaxis=dict(title="PCA 1"),
            yaxis=dict(title="PCA 2"),
        )
        fig.show()

    # ---- interno ----

    def _mossa(self, segno: str, parola: str):
        if self.vinto:
            self._html_msg(
                "Hai già vinto! Per giocare un altro round crea un nuovo "
                "<code>WordGolf(...)</code>.", colore="#888"
            )
            return
        if parola not in PAROLE_OPERATORE:
            self._html_msg(
                f"<b>'{parola}' non è una parola-operatore valida.</b><br>"
                f"Usa <code>mostra_parole_operatore()</code> per vedere la lista.",
                colore="#c00",
            )
            return

        # Snap-reset: ogni mossa parte dal vettore della parola corrente
        base_v = vettore(self.parola_corrente)
        delta = vettore(parola)
        if segno == "+":
            nuovo_v = base_v + delta
        else:
            nuovo_v = base_v - delta

        # Snap: cerca i top-5 vicini per controllare se target è dentro.
        # Escludiamo solo la parola corrente (per evitare no-op visivi);
        # non escludiamo l'operatore perche' a volte serve come destinazione
        # naturale (es. `pizza + king -> king`).
        topk = parola_piu_vicina(
            nuovo_v,
            escludi=[self.parola_corrente],
            topn=_TOP_K_VITTORIA,
        )
        # topk è list of (word, sim)
        top_words = [w for w, _ in topk]

        nuova_parola = top_words[0]
        self.mosse.append((segno, parola, nuova_parola))
        self._ultimo_v = nuovo_v  # per la riga di diagnostica "distanza dal target"
        precedente = self.parola_corrente
        self.parola_corrente = nuova_parola

        # Check vittoria
        if self.target in top_words:
            self.vinto = True
            _RECORD.append({
                "squadra": self.squadra,
                "start": self.start,
                "target": self.target,
                "mosse": len(self.mosse),
                "ultimo_topk": top_words,
            })
            self._html_msg(
                f"🎉 <b>VINTO!</b> '{self.target}' è nei top-{_TOP_K_VITTORIA} "
                f"vicini di <code>{precedente} {segno} {parola}</code>.<br>"
                f"Top-{_TOP_K_VITTORIA}: {', '.join(top_words)}<br>"
                f"<b>Mosse totali: {len(self.mosse)}</b>",
                colore="#2c662d",
                sfondo="#dff0d8",
            )
        else:
            # mossa normale: mostra dove si è atterrati e quanto manca
            sim_target = float(np.dot(
                vettore(nuova_parola) / (np.linalg.norm(vettore(nuova_parola)) + 1e-9),
                self._target_v / np.linalg.norm(self._target_v),
            ))
            self._html_msg(
                f"Mossa {len(self.mosse)}: <code>{precedente} {segno} {parola}</code> → "
                f"<b>{nuova_parola}</b><br>"
                f"<small>Top-5 vicini: {', '.join(top_words)}</small><br>"
                f"<small>Distanza dal target (cosine sim): {sim_target:.3f} "
                f"— più alta = più vicino</small>"
            )

    def _intro(self):
        self._html_msg(
            f"🏁 Nuova partita: <b>{self.start} → {self.target}</b><br>"
            f"Squadra: <i>{self.squadra}</i><br>"
            f"Usa <code>g.aggiungi('parola')</code> o <code>g.sottrai('parola')</code>."
        )

    def _stampa_stato(self):
        righe = []
        cur = self.start
        for i, (segno, op, ris) in enumerate(self.mosse, 1):
            righe.append(
                f'<tr><td style="padding:2px 8px;">{i}</td>'
                f'<td style="padding:2px 8px;font-family:monospace;">{cur}</td>'
                f'<td style="padding:2px 8px;text-align:center;">{segno}</td>'
                f'<td style="padding:2px 8px;font-family:monospace;">{op}</td>'
                f'<td style="padding:2px 8px;text-align:center;">→</td>'
                f'<td style="padding:2px 8px;font-family:monospace;"><b>{ris}</b></td>'
                f'</tr>'
            )
            cur = ris
        tabella = (
            f'<table style="border-collapse:collapse;font-size:13px;">'
            f'<thead><tr style="border-bottom:1px solid #999;">'
            f'<th style="padding:2px 8px;">#</th>'
            f'<th style="padding:2px 8px;">da</th>'
            f'<th></th><th>op</th><th></th><th>a</th></tr></thead>'
            f'<tbody>{"".join(righe) or "<tr><td colspan=6><i>nessuna mossa ancora</i></td></tr>"}</tbody>'
            f'</table>'
        )
        v_cur = vettore(self.parola_corrente)
        sim_target = float(np.dot(
            v_cur / (np.linalg.norm(v_cur) + 1e-9),
            self._target_v / np.linalg.norm(self._target_v),
        ))
        html = (
            f'<div style="font-family:sans-serif;">'
            f'<div><b>{self.squadra}</b> — {self.start} → 🎯 {self.target}</div>'
            f'<div>Mosse fatte: {len(self.mosse)}. Parola attuale: '
            f'<code>{self.parola_corrente}</code>. '
            f'Sim. con target: {sim_target:.3f}</div>'
            f'<div style="margin-top:6px;">{tabella}</div>'
            f'</div>'
        )
        display(HTML(html))

    def _html_msg(self, msg: str, colore: str = "#333", sfondo: str = "#f5f5f5"):
        display(HTML(
            f'<div style="font-family:sans-serif;color:{colore};'
            f'background:{sfondo};padding:8px 12px;border-radius:4px;'
            f'margin:4px 0;">{msg}</div>'
        ))


def classifica():
    """Stampa la classifica di tutti i record di WordGolf giocati in questo notebook."""
    if not _RECORD:
        display(HTML(
            '<div style="font-family:sans-serif;color:#888;">'
            'Nessuna partita vinta ancora. Quando una squadra vince un round, '
            'compare qui automaticamente.</div>'
        ))
        return
    # Raggruppa per (start, target), poi ordina per mosse asc
    da_target: Dict[Tuple[str, str], List[Dict]] = {}
    for r in _RECORD:
        key = (r["start"], r["target"])
        da_target.setdefault(key, []).append(r)

    sezioni = []
    for (s, t), recs in da_target.items():
        recs_ord = sorted(recs, key=lambda x: x["mosse"])
        righe = []
        for i, r in enumerate(recs_ord, 1):
            medaglia = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
            righe.append(
                f'<tr><td style="padding:2px 10px;">{medaglia}</td>'
                f'<td style="padding:2px 10px;"><b>{r["squadra"]}</b></td>'
                f'<td style="padding:2px 10px;font-family:monospace;">{r["mosse"]} mosse</td>'
                f'</tr>'
            )
        tabella = (
            f'<table style="border-collapse:collapse;font-size:13px;margin-bottom:10px;">'
            f'<thead><tr style="border-bottom:1px solid #999;">'
            f'<th colspan=3 style="padding:3px 10px;text-align:left;">'
            f'{s} → {t}</th></tr></thead>'
            f'<tbody>{"".join(righe)}</tbody></table>'
        )
        sezioni.append(tabella)
    html = (
        f'<div style="font-family:sans-serif;">'
        f'<h3 style="margin-top:0;">🏆 Classifica Word Golf</h3>'
        f'{"".join(sezioni)}'
        f'</div>'
    )
    display(HTML(html))
