# LLMlego — versione scuole superiori

Branch `scuole`: adattamento del lab del journal club per una **lezione di 2 ore**
in una classe di liceo (target principale: scientifico, 3°-5°).

## Cosa contiene

| File | Cos'è |
|---|---|
| [`lezione.ipynb`](lezione.ipynb) | Il notebook studente da aprire in Colab |
| [`llmlego_scuola/`](llmlego_scuola/) | Libreria didattica (funzioni in italiano, complessità nascosta) |
| [`setup_colab.py`](setup_colab.py) | Setup one-shot per Colab (installa + scarica GloVe ~130MB) |
| [`build_notebook.py`](build_notebook.py) | Sorgente del notebook (lo rigenera) |

## Struttura della lezione (120 min)

1. **Hook** — pipeline di un LLM (10')
2. **Dal testo ai numeri** — dizionario costruito a mano + tokenizer GPT-4 (25')
3. **Vettori 2D** — l'algebra dei vettori intuita in 2 dimensioni (20')
4. **Embedding veri (100D)** — `king − man + woman = queen` (30')
5. **Word Golf** — sfida a coppie (25')
6. **Wrap-up + bias** — discussione (10')

Messaggi chiave:
- Un LLM lavora su **vettori**, che obbediscono alla **stessa algebra** dei vettori 2D del liceo.
- La pipeline: **testo → dizionario → embedding → algebra → testo**.
- Le **relazioni semantiche** (genere, geografia, ecc.) sono **direzioni** nello spazio.

## Stato lavoro

Lavoro in corso — il notebook gira end-to-end ma non è ancora stato provato in aula.

## Come riaprire il lavoro

```bash
git checkout scuole
# Lavoro locale:
python build_notebook.py     # rigenera lezione.ipynb da build_notebook.py
```

## Note didattiche

- Il **modulo BERT/polisemia** del lab universitario è stato **rimosso** per ragioni di tempo.
- Il **Word Golf** ha 57 parole-operatore curate. I 3 round (`cat→eagle`, `pizza→emperor`,
  `winter→happiness`) sono stati verificati con una beam search: tutti risolvibili in
  **3-4 mosse** con la regola "target nei top-5".
- **Tip per il docente:** la sottrazione genera mosse più significative dell'addizione
  (perché togliere riduce la dominanza del vettore di origine — il classico
  `king − man + woman` funziona proprio così).
