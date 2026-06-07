# 🧱 LLM Lego — versione scuole superiori

Branch `scuole`: adattamento del lab del journal club per una **lezione di 2 ore**
in classe di liceo (target principale: scientifico, 3°–5°).

## Cosa contiene

| File | Cos'è |
|---|---|
| [`laboratorio.ipynb`](laboratorio.ipynb) | Il notebook studente da aprire in Colab |
| [`llmlego_scuola/`](llmlego_scuola/) | Libreria didattica (funzioni in italiano, complessità nascosta) |
| [`setup_colab.py`](setup_colab.py) | Setup one-shot per Colab (installa deps + scarica GloVe ~130MB) |
| [`requirements.txt`](requirements.txt) | Dipendenze Python minime |

I materiali "admin" (presentazione del prof, dashboard live per la sfida Word Golf) sono nel **repo privato `LLMlego-admin`** — non sono qui per non esporli agli studenti.

## Come si apre in Colab

Una sola cella di setup nel notebook fa tutto:

```python
!git clone -q -b scuole https://github.com/tlm-journalclub-org/LLMlego.git
%cd LLMlego
!python setup_colab.py
from llmlego_scuola import *
import numpy as np
```

## Struttura della lezione (~120 min)

1. **Hook** — pipeline di un LLM (10')
2. **Modulo 1**: tokenizzazione e vocabolario finito (20')
3. **Modulo 2**: vettori 2D, features, cosine similarity (25')
4. **Modulo 3**: embedding veri (100D), aritmetica `king − man + woman = queen` (30')
5. **Modulo 4**: Word Golf — sfida a squadre (30')
6. **Modulo 5**: bias di genere (10')

### Round del Word Golf

| # | Round | Mosse | Path tipo |
|---|---|---|---|
| 1 | `doctor → farmer` | 1 | `+farm` |
| 2 | `paris → tokyo` | 1 | `+japan` |
| 3 | `boy → queen` | 2 | `+age +crown` |
| 4 (bonus) | `cat → eagle` | 2 | `-love +mountain` |
| 5 (finale) | `science → dance` | 2-3 | `+music +movement` |

## Note didattiche

- BERT/contextual embeddings rimossi rispetto al lab universitario: niente `transformers`/`torch` in `requirements.txt`.
- Il **Word Golf** ha ~60 parole-operatore curate. Tutti i round sono stati validati con beam search.
- Convenzione cromatica uniforme in tutta la lezione: **blu = positivo**, **rosso = negativo**.
- `WordGolf` supporta sia modalità sfida (con `target`) sia esplorazione libera (senza `target`).

## Stato

- Branch `scuole` è destinato a restare separato da `main`.
- Aggiornamenti compatibili con `main` vengono cherry-pickati al volo.
