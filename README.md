# LLMlego

Lab pratico sugli embeddings per il journal club [Trento Local Minimum](https://github.com/tlm-journalclub-org).

## Setup

Python >= 3.12

```bash
pip install -r requirements.txt
python download_models.py
```

Il secondo comando scarica i modelli (~1.1GB totali). Eseguilo **prima del lab** per evitare attese.

## Modelli usati

| Modello | Uso | Peso |
|---------|-----|------|
| `glove-wiki-gigaword-100` (gensim) | Word embeddings, analogie, Olympics | ~130MB |
| `distilbert-base-multilingual-cased` (transformers) | Contextual embeddings, polisemia | ~520MB |
| `paraphrase-multilingual-MiniLM-L12-v2` (transformers) | Sentence embeddings, RAG, cross-lingua | ~470MB |
