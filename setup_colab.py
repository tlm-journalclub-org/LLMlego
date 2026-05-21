"""Setup per Colab — versione scuola.

Installa solo il necessario per la lezione di 2 ore:
- gensim (word embeddings GloVe)
- tiktoken (tokenizer GPT-4)
- plotly + ipywidgets (visualizzazioni interattive)

Scarica GloVe (~130MB) ma NON i modelli BERT (~1GB risparmiato).

Idempotente: se eseguito due volte, salta i passi gia' fatti.
"""
import importlib
import os
import subprocess
import sys

GLOVE_PATH = "glove-wiki-gigaword-100.kv"

PIP_PACKAGES = [
    "gensim>=4.3.0",
    "tiktoken>=0.5.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",
    "plotly>=5.18.0",
    "ipywidgets>=8.0",
    "numpy>=1.24.0",
]


def _print(msg):
    print(f"[setup] {msg}", flush=True)


def installa_pip():
    """Installa solo i package mancanti."""
    da_installare = []
    pkg_to_module = {
        "gensim>=4.3.0": "gensim",
        "tiktoken>=0.5.0": "tiktoken",
        "scikit-learn>=1.3.0": "sklearn",
        "scipy>=1.11.0": "scipy",
        "plotly>=5.18.0": "plotly",
        "ipywidgets>=8.0": "ipywidgets",
        "numpy>=1.24.0": "numpy",
    }
    for spec, modname in pkg_to_module.items():
        try:
            importlib.import_module(modname)
        except ImportError:
            da_installare.append(spec)
    if not da_installare:
        _print("Dipendenze gia' installate.")
        return
    _print(f"Installo {len(da_installare)} pacchetti: {', '.join(da_installare)}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *da_installare]
    )


def scarica_glove():
    """Scarica GloVe via gensim.downloader, lo salva localmente come KeyedVectors."""
    if os.path.exists(GLOVE_PATH):
        _print(f"GloVe gia' presente in {GLOVE_PATH}, salto.")
        return
    _print("Scarico GloVe (~130MB)... pazienza, 1-2 minuti.")
    import gensim.downloader as api
    model = api.load("glove-wiki-gigaword-100")
    model.save(GLOVE_PATH)
    _print(f"Salvato in {GLOVE_PATH}")


def warmup_tiktoken():
    """Forza il download della cache di tiktoken cl100k_base."""
    _print("Warm-up tokenizer GPT-4...")
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    enc.encode("ciao")  # forza eventuale cache
    _print("OK.")


def abilita_widget_colab():
    """In Colab serve abilitare i custom widget manager."""
    try:
        from google.colab import output  # type: ignore
        output.enable_custom_widget_manager()
        _print("Widget Colab abilitati.")
    except ImportError:
        # non siamo in Colab, niente da fare
        pass


def main():
    installa_pip()
    scarica_glove()
    warmup_tiktoken()
    abilita_widget_colab()
    _print("Setup completato.")


if __name__ == "__main__":
    main()
