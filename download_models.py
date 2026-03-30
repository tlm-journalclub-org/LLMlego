"""
Scarica tutti i modelli necessari per il lab.
Esegui PRIMA della sessione per evitare attese durante il lab.

    python download_models.py
"""

import os

from transformers import AutoModel, AutoTokenizer


def download_hf_model(name, cache_dir, label, size):
    print("=" * 50)
    print(f"  {label} (~{size})")
    print("=" * 50)
    if os.path.exists(cache_dir):
        print(f"Gia presente in '{cache_dir}'\n")
        return
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModel.from_pretrained(name)
    tok.save_pretrained(cache_dir)
    mod.save_pretrained(cache_dir)
    print(f"Salvato in '{cache_dir}'\n")


# 1. GloVe word embeddings
print("=" * 50)
print("  GloVe word embeddings (~130MB)")
print("=" * 50)
import gensim.downloader as api

model = api.load("glove-wiki-gigaword-100")
model.save("glove-wiki-gigaword-100.kv")
print("OK\n")

# 2. DistilBERT multilingual (contextual embeddings, nb1)
download_hf_model(
    "distilbert/distilbert-base-multilingual-cased",
    "distilbert-multilingual-cased",
    "DistilBERT multilingual",
    "520MB",
)

# 3. Sentence model (sentence embeddings, nb2)
download_hf_model(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-model",
    "Sentence model (MiniLM)",
    "470MB",
)

print("=" * 50)
print("Tutti i modelli sono pronti!")
print("=" * 50)
print("=" * 50)
print("Tutti i modelli sono pronti!")
print("=" * 50)
print("=" * 50)
print("Tutti i modelli sono pronti!")
print("=" * 50)
