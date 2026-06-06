"""Modulo 4: Word Golf.

Regole:
- Si parte da `start`, si vuole raggiungere `target`.
- A ogni mossa l'utente sceglie `+ parola` o `- parola` da una lista curata.
- La nuova parola corrente è quella più vicina al vettore risultante (escludendo
  la parola attuale per evitare loop).
- Vince quando il `target` compare nei top-5 vicini del vettore corrente.
"""
from typing import Optional, List, Dict, Tuple
import json
import os
import threading
import time
import urllib.request
import urllib.error

import numpy as np
from IPython.display import HTML, display
import plotly.graph_objects as go

from .embeddings import _ensure_loaded, vettore, parola_piu_vicina


# URL della dashboard condivisa (Apps Script Web App). Se None o vuoto,
# la classifica resta puramente locale al notebook. Si puo' settare:
#   - via variabile globale del modulo:  llmlego_scuola.golf.DASHBOARD_URL = "..."
#   - oppure via env var:                os.environ["WORDGOLF_DASHBOARD_URL"] = "..."
DASHBOARD_URL: Optional[str] = None

# Buffer per gli errori delle POST asincrone (vuoto se tutto OK).
# Si puo' ispezionare con `llmlego_scuola.golf.errori_dashboard()`.
_DASHBOARD_ERRORS: List[str] = []


def _get_dashboard_url() -> Optional[str]:
    url = DASHBOARD_URL or os.environ.get("WORDGOLF_DASHBOARD_URL", "")
    return url.strip() or None


def _do_post(url: str, payload: dict, timeout: int = 10):
    """Esegue il POST e ritorna (status, final_url, body_text). Solleva eccezioni."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.url, resp.read().decode("utf-8", errors="replace")


def _post_record_async(payload: dict):
    """POST asincrono a DASHBOARD_URL. Non blocca il notebook.
    Eventuali errori vengono accumulati in _DASHBOARD_ERRORS."""
    url = _get_dashboard_url()
    if not url:
        return

    def _runner():
        try:
            status, final_url, body = _do_post(url, payload, timeout=8)
            # Apps Script risponde sempre 200 anche su errore applicativo,
            # quindi controlliamo il body
            if '"ok":true' not in body:
                _DASHBOARD_ERRORS.append(
                    f"HTTP {status} body inatteso: {body[:200]}"
                )
        except Exception as e:
            _DASHBOARD_ERRORS.append(f"{type(e).__name__}: {e}")

    threading.Thread(target=_runner, daemon=True).start()


def _post_event_async(event_type: str, base_payload: dict):
    """Helper: POST un evento di tipo `event_type` (start | win) alla dashboard.
    Apps Script fa dedup: per ogni (squadra, start, target, type) tiene solo
    il PRIMO evento. Cosi' anche se gli studenti rieseguono la cella per
    'barare' col tempo, il loro start_time ufficiale resta quello iniziale."""
    payload = dict(base_payload)
    payload["type"] = event_type
    _post_record_async(payload)


def errori_dashboard():
    """Stampa gli eventuali errori di POST verso la dashboard.
    Utile se 'vinto' e' apparso nel notebook ma il record non e' nel Sheet."""
    if not _DASHBOARD_ERRORS:
        print("Nessun errore registrato.")
        return
    print(f"{len(_DASHBOARD_ERRORS)} errori:")
    for i, e in enumerate(_DASHBOARD_ERRORS, 1):
        print(f"  {i}. {e}")


def test_dashboard(squadra: str = "test-diag"):
    """Diagnostica sincrona: prova a postare un record di test e stampa cosa
    risponde il server. Usalo per capire se la URL e la configurazione del
    Web App di Apps Script sono corrette."""
    url = _get_dashboard_url()
    print("=" * 60)
    print(f"URL dashboard: {url!r}")
    if not url:
        print("❌ DASHBOARD_URL non impostato.")
        print("   Setta: llmlego_scuola.golf.DASHBOARD_URL = 'https://...'")
        return

    payload = {
        "squadra": squadra,
        "start": "test",
        "target": "test",
        "mosse": 0,
        "timestamp": int(time.time() * 1000),
        "dettaglio": ["test diagnostico"],
    }
    print(f"Payload: {json.dumps(payload)}")
    print("-" * 60)

    try:
        status, final_url, body = _do_post(url, payload, timeout=15)
        print(f"HTTP status: {status}")
        if final_url != url:
            print(f"(redirect finale a: {final_url})")
        print(f"Body: {body[:500]}")
        if '"ok":true' in body:
            print("\n✅ POST riuscito. Controlla il Google Sheet:")
            print("   dovresti vedere una nuova riga col timestamp di adesso.")
            print(f"   squadra: '{squadra}'")
        else:
            print("\n⚠️  Il server ha risposto, ma il body non contiene 'ok:true'.")
            print("   Probabilmente il deploy ha problemi o l'URL non e' la Web App.")
    except urllib.error.HTTPError as e:
        print(f"❌ HTTPError {e.code}: {e.reason}")
        try:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"Body: {err_body[:500]}")
        except Exception:
            pass
        if e.code in (401, 403):
            print("\n   Il deploy probabilmente NON ha 'Who has access: Anyone'.")
            print("   Vai su Deploy -> Manage deployments -> edit -> setta Anyone.")
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")


# ~50 parole-operatore semplici e narrative. Categorie evocative pensate
# per studenti delle superiori (in inglese perche' GloVe e' in inglese).
PAROLE_OPERATORE = [
    # persone / ruoli
    "man", "woman", "boy", "girl", "child", "baby",
    "king", "queen", "soldier", "doctor", "farmer", "teacher", "student",
    # luoghi
    "italy", "france", "japan", "america", "paris", "tokyo",
    "school", "hospital", "farm", "city", "mountain", "sea", "river",
    # cibo
    "bread", "fish", "fruit", "meat", "water", "wine",
    # oggetti
    "book", "pen", "crown", "weapon", "money", "computer",
    # natura / animali
    "sun", "snow", "fire", "tree", "flower", "dog", "cat", "bird",
    # tempo
    "summer", "winter", "day", "night",
    # astratti
    "love", "war", "peace", "fear", "life", "death",
    "power", "knowledge", "music", "age", "math", "movement",
]

# Categorie per la visualizzazione (mostra_parole_operatore)
_CATEGORIE = {
    "persone e ruoli": [
        "man", "woman", "boy", "girl", "child", "baby",
        "king", "queen", "soldier", "doctor", "farmer", "teacher", "student",
    ],
    "luoghi": [
        "italy", "france", "japan", "america", "paris", "tokyo",
        "school", "hospital", "farm", "city", "mountain", "sea", "river",
    ],
    "cibo": ["bread", "fish", "fruit", "meat", "water", "wine"],
    "oggetti": ["book", "pen", "crown", "weapon", "money", "computer"],
    "natura e animali": [
        "sun", "snow", "fire", "tree", "flower", "dog", "cat", "bird",
    ],
    "tempo": ["summer", "winter", "day", "night"],
    "astratti": [
        "love", "war", "peace", "fear", "life", "death",
        "power", "knowledge", "music", "age", "math", "movement",
    ],
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

    def __init__(
        self,
        start: str,
        target: Optional[str] = None,
        squadra: Optional[str] = None,
    ):
        """
        - Con `target` impostato: partita classica, vincita quando il target
          entra nei top-5.
        - Senza `target` (None): modalità **esplorazione libera**. Niente
          obiettivo, niente classifica: gli studenti sommano/sottraggono
          parole per vedere dove finiscono nello spazio vettoriale.
        """
        _ensure_loaded()
        try:
            self._start_v = vettore(start)
            self._target_v = vettore(target) if target is not None else None
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
        # Ogni mossa: (segno, operatore, risultato, top5)
        self.mosse: List[Tuple[str, str, str, List[str]]] = []
        self.vinto = False
        # Tempo di inizio della partita: lo manderemo come evento "start"
        # alla dashboard. La dashboard ha dedup per (squadra, round) quindi
        # eseguire piu' volte la cella NON resetta il tempo ufficiale.
        self._start_ts = int(time.time() * 1000)

        # POST "start" event (solo modalita' sfida)
        if self.target is not None:
            _post_event_async("start", {
                "squadra": self.squadra,
                "start": self.start,
                "target": self.target,
                "timestamp": self._start_ts,
            })

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
        """Plot 2D del percorso: parole visitate (e target se c'e'), ridotte
        in 2D via PCA."""
        from sklearn.decomposition import PCA
        parole_path = [self.start] + [mossa[2] for mossa in self.mosse]
        if self.target is not None:
            tutte = parole_path + [self.target]
        else:
            tutte = parole_path
        if len(tutte) < 2:
            display(HTML(
                '<div style="color:#888;font-family:sans-serif;">'
                'Almeno una mossa serve per disegnare il percorso.'
                '</div>'
            ))
            return
        vecs = np.array([vettore(p) for p in tutte])
        pca = PCA(n_components=2).fit(vecs)
        coords = pca.transform(vecs)

        if self.target is not None:
            path_xy = coords[:-1]
            target_xy = coords[-1]
        else:
            path_xy = coords
            target_xy = None

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
        # Target (se presente)
        if target_xy is not None:
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
        if self.target is not None:
            titolo = (
                f"{self.squadra}: {self.start} → {self.target} "
                f"({len(self.mosse)} mosse)"
            )
        else:
            titolo = (
                f"Esplorazione libera da {self.start} "
                f"({len(self.mosse)} mosse)"
            )
        fig.update_layout(
            title=titolo,
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
        # Vietato usare start o target come operatore: sarebbe banale.
        if parola == self.start:
            self._html_msg(
                f"❌ Non puoi usare la parola di <b>partenza</b> "
                f"(<code>{self.start}</code>) come operatore: "
                f"sarebbe una mossa banale. Mossa annullata.",
                colore="#c00",
            )
            return
        if self.target is not None and parola == self.target:
            self._html_msg(
                f"❌ Non puoi usare la parola <b>target</b> "
                f"(<code>{self.target}</code>) come operatore: "
                f"sarebbe come barare. Mossa annullata.",
                colore="#c00",
            )
            return

        if parola not in PAROLE_OPERATORE:
            # Warning ma proseguiamo: la parola va comunque cercata in
            # GloVe; se non esiste nel vocabolario fermiamo li' con un
            # messaggio chiaro.
            self._html_msg(
                f"⚠️ <b>'{parola}'</b> non è tra le parole-operatore "
                f"consigliate. Procedo comunque, ma se non è in GloVe "
                f"la mossa fallirà. "
                f"Usa <code>mostra_parole_operatore()</code> per "
                f"vedere la lista consigliata.",
                colore="#9a6300",
                sfondo="#fff3cd",
            )

        # Snap-reset: ogni mossa parte dal vettore della parola corrente
        try:
            base_v = vettore(self.parola_corrente)
            delta = vettore(parola)
        except KeyError as e:
            self._html_msg(
                f"❌ La parola <code>{parola}</code> non è nel "
                f"vocabolario di GloVe: la mossa è annullata. "
                f"<small>{e}</small>",
                colore="#c00",
            )
            return
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
        self.mosse.append((segno, parola, nuova_parola, list(top_words)))
        self._ultimo_v = nuovo_v  # per la riga di diagnostica "distanza dal target"
        precedente = self.parola_corrente
        self.parola_corrente = nuova_parola

        # Check vittoria (solo se c'e' un target — modalita' sfida)
        if self.target is not None and self.target in top_words:
            self.vinto = True
            win_ts = int(time.time() * 1000)
            record = {
                "squadra": self.squadra,
                "start": self.start,
                "target": self.target,
                "mosse": len(self.mosse),
                "ultimo_topk": top_words,
                "timestamp": win_ts,
                "dettaglio": [
                    f"{s}{op}->{r}" for s, op, r, _ in self.mosse
                ],
            }
            _RECORD.append(record)
            # Manda alla dashboard condivisa come evento "win".
            # Apps Script fa dedup: solo la prima vittoria di una squadra
            # per round viene conservata.
            _post_event_async("win", record)
            badge_dash = (
                ' <span style="background:#2c662d;color:white;padding:2px 8px;'
                'border-radius:10px;font-size:11px;">→ dashboard</span>'
                if _get_dashboard_url() else ""
            )
            self._html_msg(
                f"🎉 <b>VINTO!</b> '{self.target}' è nei top-{_TOP_K_VITTORIA} "
                f"vicini di <code>{precedente} {segno} {parola}</code>.{badge_dash}<br>"
                f"Top-{_TOP_K_VITTORIA}: {', '.join(top_words)}<br>"
                f"<b>Mosse totali: {len(self.mosse)}</b>",
                colore="#2c662d",
                sfondo="#dff0d8",
            )
        else:
            # mossa normale: mostra dove si è atterrati
            base_msg = (
                f"Mossa {len(self.mosse)}: "
                f"<code>{precedente} {segno} {parola}</code> → "
                f"<b>{nuova_parola}</b><br>"
                f"<small>Top-5 vicini: {', '.join(top_words)}</small>"
            )
            if self._target_v is not None:
                # In modalita' sfida aggiungiamo la distanza dal target
                sim_target = float(np.dot(
                    vettore(nuova_parola) / (
                        np.linalg.norm(vettore(nuova_parola)) + 1e-9
                    ),
                    self._target_v / np.linalg.norm(self._target_v),
                ))
                base_msg += (
                    f"<br><small>Distanza dal target (cosine sim): "
                    f"{sim_target:.3f} — più alta = più vicino</small>"
                )
            self._html_msg(base_msg)

    def _intro(self):
        if self.target is None:
            self._html_msg(
                f"🧪 <b>Esplorazione libera</b> partendo da "
                f"<code>{self.start}</code>.<br>"
                f"Nessun obiettivo, niente classifica: somma e sottrai "
                f"parole per vedere dove finisci nello spazio vettoriale.<br>"
                f"Usa <code>g.aggiungi('parola')</code> o "
                f"<code>g.sottrai('parola')</code>."
            )
        else:
            self._html_msg(
                f"🏁 Nuova partita: <b>{self.start} → {self.target}</b><br>"
                f"Squadra: <i>{self.squadra}</i><br>"
                f"Usa <code>g.aggiungi('parola')</code> o "
                f"<code>g.sottrai('parola')</code>."
            )

    def _stampa_stato(self):
        righe = []
        cur = self.start
        for i, mossa in enumerate(self.mosse, 1):
            segno, op, ris, top5 = mossa
            # Evidenzia il target se presente nei top-5 (solo modalita' sfida)
            top5_html = ", ".join(
                f'<b style="color:#2c662d;">{w}</b>'
                if (self.target is not None and w == self.target) else w
                for w in top5
            )
            righe.append(
                f'<tr>'
                f'<td style="padding:2px 8px;text-align:right;">{i}</td>'
                f'<td style="padding:2px 8px;font-family:monospace;">{cur}</td>'
                f'<td style="padding:2px 8px;text-align:center;">{segno}</td>'
                f'<td style="padding:2px 8px;font-family:monospace;">{op}</td>'
                f'<td style="padding:2px 8px;text-align:center;">→</td>'
                f'<td style="padding:2px 8px;font-family:monospace;">'
                f'<b>{ris}</b></td>'
                f'<td style="padding:2px 8px;font-family:monospace;'
                f'font-size:11px;color:#666;">top-5: {top5_html}</td>'
                f'</tr>'
            )
            cur = ris
        tabella = (
            f'<table style="border-collapse:collapse;font-size:13px;">'
            f'<thead><tr style="border-bottom:1px solid #999;">'
            f'<th style="padding:2px 8px;">#</th>'
            f'<th style="padding:2px 8px;">da</th>'
            f'<th></th><th>op</th><th></th><th>a</th>'
            f'<th style="padding:2px 8px;text-align:left;">vicini</th>'
            f'</tr></thead>'
            f'<tbody>{"".join(righe) or "<tr><td colspan=7><i>nessuna mossa ancora</i></td></tr>"}</tbody>'
            f'</table>'
        )
        if self._target_v is not None:
            v_cur = vettore(self.parola_corrente)
            sim_target = float(np.dot(
                v_cur / (np.linalg.norm(v_cur) + 1e-9),
                self._target_v / np.linalg.norm(self._target_v),
            ))
            header = (
                f'<div><b>{self.squadra}</b> — '
                f'{self.start} → 🎯 {self.target}</div>'
                f'<div>Mosse fatte: {len(self.mosse)}. Parola attuale: '
                f'<code>{self.parola_corrente}</code>. '
                f'Sim. con target: {sim_target:.3f}</div>'
            )
        else:
            header = (
                f'<div>🧪 <b>Esplorazione libera</b> partita da '
                f'<code>{self.start}</code></div>'
                f'<div>Mosse fatte: {len(self.mosse)}. Parola attuale: '
                f'<code>{self.parola_corrente}</code>.</div>'
            )
        html = (
            f'<div style="font-family:sans-serif;">'
            f'{header}'
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
