"""Provided model-loading utilities. Models are initialized on first use."""

from functools import lru_cache
import os


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )


@lru_cache(maxsize=1)
def get_llm():
    from llama_cpp import Llama

    model_path = os.environ.get(
        "RAG_LLM_PATH",
        "models/model.gguf",
    )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"GGUF model missing: {model_path}. Set RAG_LLM_PATH.")

    return Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=min(os.cpu_count() or 2, 4),
        logits_all=True,
        verbose=False,
    )
