"""Helper per il Modulo 1: visualizzazione del dizionario costruito a mano."""
from IPython.display import HTML, display


def mostra_dizionario(dizionario: dict):
    """Mostra un dizionario parola->ID come tabella ordinata per ID.

    Utile per la cella in cui gli studenti hanno appena finito di
    costruire `dizionario` partendo da un corpus.
    """
    if not dizionario:
        display(HTML(
            '<div style="color:#c00;font-family:sans-serif;">'
            'Il dizionario è vuoto. Hai eseguito la cella precedente?'
            '</div>'
        ))
        return

    coppie = sorted(dizionario.items(), key=lambda kv: kv[1])
    righe = []
    for parola, id_ in coppie:
        righe.append(
            f'<tr>'
            f'<td style="padding:3px 12px;text-align:right;font-family:monospace;">{id_}</td>'
            f'<td style="padding:3px 12px;font-family:monospace;">"{parola}"</td>'
            f'</tr>'
        )
    tabella = (
        '<table style="border-collapse:collapse;font-size:13px;">'
        '<thead><tr style="border-bottom:1px solid #999;">'
        '<th style="padding:3px 12px;">ID</th>'
        '<th style="padding:3px 12px;text-align:left;">parola</th>'
        '</tr></thead><tbody>' + "".join(righe) + '</tbody></table>'
    )
    html = (
        '<div style="font-family:sans-serif;">'
        f'<div style="margin-bottom:6px;"><b>Dizionario</b> '
        f'({len(dizionario)} parole uniche)</div>'
        f'{tabella}</div>'
    )
    display(HTML(html))
