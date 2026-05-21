"""Visualizzazione tokenizer GPT-4 per il Modulo 1."""
import tiktoken
from IPython.display import HTML, display

_enc = None

# Palette di colori pastello, riutilizzabile per evidenziare token consecutivi
_COLORI = [
    "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF",
    "#D4BAFF", "#FFBAF0", "#C9C9C9", "#FFC4A3", "#A3E4D7",
]


def _get_encoder():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def mostra_token(testo: str):
    """Tokenizza `testo` col tokenizer di GPT-4 (cl100k_base) e mostra una
    tabella HTML con i pezzi colorati, il loro ID e un riepilogo.

    Pensato per uso in Jupyter/Colab: usa IPython.display.HTML.
    """
    enc = _get_encoder()
    ids = enc.encode(testo)
    pezzi = [enc.decode([i]) for i in ids]

    # Riga 1: i pezzi colorati uno accanto all'altro (preserva spazi)
    spans = []
    for i, p in enumerate(pezzi):
        colore = _COLORI[i % len(_COLORI)]
        # Sostituisco gli spazi con un placeholder visibile
        p_visibile = p.replace(" ", "·").replace("\n", "↵")
        spans.append(
            f'<span style="background:{colore};padding:3px 5px;'
            f'border-radius:4px;margin:1px;font-family:monospace;'
            f'font-size:14px;">{p_visibile}</span>'
        )
    riga_pezzi = "".join(spans)

    # Riga 2: tabella dettagliata
    righe_tabella = []
    for i, (id_, p) in enumerate(zip(ids, pezzi)):
        colore = _COLORI[i % len(_COLORI)]
        p_vis = p.replace(" ", "·").replace("\n", "↵")
        righe_tabella.append(
            f'<tr>'
            f'<td style="padding:3px 8px;text-align:center;">{i+1}</td>'
            f'<td style="padding:3px 8px;background:{colore};font-family:monospace;">{p_vis}</td>'
            f'<td style="padding:3px 8px;text-align:right;font-family:monospace;">{id_}</td>'
            f'</tr>'
        )
    tabella = (
        '<table style="border-collapse:collapse;font-size:13px;margin-top:10px;">'
        '<thead><tr style="border-bottom:1px solid #999;">'
        '<th style="padding:3px 8px;">#</th>'
        '<th style="padding:3px 8px;">pezzo (token)</th>'
        '<th style="padding:3px 8px;">ID</th></tr></thead>'
        '<tbody>' + "".join(righe_tabella) + '</tbody></table>'
    )

    html = (
        f'<div style="font-family:sans-serif;">'
        f'<div style="margin-bottom:6px;"><b>Testo:</b> "{testo}"</div>'
        f'<div style="margin-bottom:6px;"><b>{len(ids)} token</b> '
        f'(il punto · indica uno spazio):</div>'
        f'<div>{riga_pezzi}</div>'
        f'{tabella}'
        f'</div>'
    )
    display(HTML(html))
