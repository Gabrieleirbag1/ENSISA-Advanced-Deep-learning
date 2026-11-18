"""RAG Visual Lab — student exercises.

Complete TODO 1–6 in order. Do not change function signatures or
return formats: the Flask application relies on these interfaces.
"""

import numpy as np
from sklearn.decomposition import PCA

# ============================================================
# TEXT CHUNKING
# ============================================================


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """Split into character windows with exact, inspectable overlap."""
    return [
        item["text"] for item in chunk_text_with_metadata(text, chunk_size, overlap)
    ]


def chunk_text_with_metadata(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """Split text into chunks while preserving character positions."""

    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError(
            "chunk_size must be positive and overlap smaller than chunk_size"
        )

    if not text:
        return []

    # TODO 1 — Text chunking with overlap.
    # Compute the stride (step) from chunk_size and overlap.
    # For each window, append a dictionary with these keys:
    #   index, start, end, text, overlap_length
    # The first chunk has overlap_length = 0.
    # Stop as soon as the end of the text is reached.
    # Hint: use range(...), min(...), and text[start:end].
    raise NotImplementedError("TODO 1: implement overlapping text chunks")


# ============================================================
# EMBEDDING GENERATION
# ============================================================


def vectorize_chunks(
    chunks: list[str],
    model,
) -> np.ndarray:
    """Convert document chunks into embedding vectors."""

    if not chunks:
        raise ValueError("No chunks to vectorize")

    # TODO 2 — Generate document embeddings.
    # Use model.encode(...) on the list of chunks.
    # Return a NumPy array of dtype np.float32, shape (N, D).
    # Hint: model.encode(chunks, batch_size=16, show_progress_bar=False)
    raise NotImplementedError("TODO 2: encode document chunks")



def embed_query(
    question: str,
    model,
) -> np.ndarray:
    """Embed a question using the same model as the document chunks."""

    # TODO 3 — Embed the user's question.
    # Use the SAME model as for the document chunks.
    # Return one 1D np.float32 vector of shape (D,), not (1, D).
    # Hint: model.encode([question], show_progress_bar=False)[0]
    raise NotImplementedError("TODO 3: encode the query")



# ============================================================
# COSINE SIMILARITY
# ============================================================


def cosine_similarity(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity in the original embedding space."""

    query = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(-1)

    vectors = np.asarray(
        chunk_embeddings,
        dtype=np.float32,
    )

    if vectors.ndim != 2 or vectors.shape[1] != query.shape[0]:
        raise ValueError("Embedding dimensions do not match")

    # TODO 4 — Cosine similarity in the original embedding space.
    # For every document vector v, compute:
    #   cosine(q, v) = dot(q, v) / (||q|| * ||v||)
    # Return a 1D array with one score per chunk.
    # Handle zero norms safely (similarity = 0).
    # Hint: np.linalg.norm, matrix multiplication (@), np.divide.
    raise NotImplementedError("TODO 4: compute cosine similarities")



# ============================================================
# PCA VISUALIZATION
# ============================================================


def fit_embedding_pca(
    embeddings: np.ndarray,
) -> tuple[PCA | None, np.ndarray]:
    """
    Fit PCA on document embeddings.

    Returns
    -------
    pca:
        Fitted PCA model, or None when fewer than two
        document embeddings are available.

    points:
        Document embeddings projected into 2D.

    Important
    ---------
    PCA is used only for visualization.

    Retrieval always operates in the original
    high-dimensional embedding space.
    """

    vectors = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("Expected a non-empty 2D embedding matrix")

    if len(vectors) == 1:
        return None, np.zeros((1, 2), dtype=np.float32)

    pca = PCA(n_components=2)

    points = pca.fit_transform(vectors)

    return pca, np.asarray(points, dtype=np.float32)


def project_query_pca(
    query_embedding: np.ndarray,
    pca: PCA | None,
) -> list[float]:
    """
    Project a query into the PCA space fitted on documents.

    The PCA model must NOT be refitted on the query.
    """

    if pca is None:
        return [0.0, 0.0]

    query = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(1, -1)

    point = pca.transform(query)[0]

    return [
        float(point[0]),
        float(point[1]),
    ]


# ============================================================
# PCA VS ORIGINAL EMBEDDING SPACE
# ============================================================


def compare_embedding_rankings(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    pca: PCA | None,
    k: int = 3,
) -> dict:
    """
    Compare nearest-neighbor rankings in the original
    embedding space and the 2D PCA visualization.

    Three rankings are computed:

    1. Original-space cosine similarity.
    2. PCA-space Euclidean distance.
    3. PCA-space cosine similarity.

    The third ranking helps distinguish the effect of
    dimensionality reduction from the choice of metric.

    PCA is never used for actual RAG retrieval.
    """

    vectors = np.asarray(
        chunk_embeddings,
        dtype=np.float32,
    )

    query = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(-1)

    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("Expected a non-empty embedding matrix")

    if vectors.shape[1] != query.shape[0]:
        raise ValueError("Embedding dimensions do not match")

    k = max(1, min(int(k), len(vectors)))

    # --------------------------------------------------------
    # 1. Original-space cosine similarity
    # --------------------------------------------------------

    original_scores = cosine_similarity(
        query,
        vectors,
    )

    original_order = np.argsort(
        -original_scores,
        kind="stable",
    )

    # --------------------------------------------------------
    # 2. Project embeddings into the fitted PCA space
    # --------------------------------------------------------

    if pca is None:

        # PCA cannot be fitted with fewer than two chunks.
        # In this case, there is no meaningful comparison.

        return {
            "available": False,
            "reason": (
                "At least two document chunks are required " "for a 2D PCA comparison."
            ),
        }

    projected_vectors = pca.transform(vectors)

    projected_query = pca.transform(query.reshape(1, -1))[0]

    # --------------------------------------------------------
    # 3. PCA-space Euclidean distance
    # --------------------------------------------------------

    differences = projected_vectors - projected_query

    distances_2d = np.linalg.norm(
        differences,
        axis=1,
    )

    euclidean_order = np.argsort(
        distances_2d,
        kind="stable",
    )

    # --------------------------------------------------------
    # 4. PCA-space cosine similarity
    # --------------------------------------------------------

    cosine_scores_2d = cosine_similarity(
        projected_query,
        projected_vectors,
    )

    cosine_order_2d = np.argsort(
        -cosine_scores_2d,
        kind="stable",
    )

    # --------------------------------------------------------
    # 5. Compute top-k overlap
    # --------------------------------------------------------

    original_top = set(original_order[:k].tolist())

    euclidean_top = set(euclidean_order[:k].tolist())

    cosine_top = set(cosine_order_2d[:k].tolist())

    overlap_euclidean = len(original_top & euclidean_top) / k

    overlap_cosine = len(original_top & cosine_top) / k

    # --------------------------------------------------------
    # 6. Build ranking information
    # --------------------------------------------------------

    original_ranking = [
        {
            "index": int(i),
            "rank": rank + 1,
            "score": float(original_scores[i]),
        }
        for rank, i in enumerate(original_order)
    ]

    euclidean_ranking = [
        {
            "index": int(i),
            "rank": rank + 1,
            "distance": float(distances_2d[i]),
        }
        for rank, i in enumerate(euclidean_order)
    ]

    cosine_2d_ranking = [
        {
            "index": int(i),
            "rank": rank + 1,
            "score": float(cosine_scores_2d[i]),
        }
        for rank, i in enumerate(cosine_order_2d)
    ]

    return {
        "available": True,
        "k": k,
        "dimension_original": int(vectors.shape[1]),
        "dimension_projected": 2,
        "original_cosine": {
            "ranking": original_ranking,
            "top_k": sorted(original_top),
        },
        "pca_euclidean": {
            "ranking": euclidean_ranking,
            "top_k": sorted(euclidean_top),
            "overlap": float(overlap_euclidean),
        },
        "pca_cosine": {
            "ranking": cosine_2d_ranking,
            "top_k": sorted(cosine_top),
            "overlap": float(overlap_cosine),
        },
    }


# ============================================================
# FULL VECTOR SEARCH
# ============================================================


def retrieve_with_visualization(
    question: str,
    chunks: list[str],
    embeddings: np.ndarray,
    model,
    pca: PCA | None,
    k: int = 3,
) -> dict:
    """
    Perform retrieval and return additional visualization data.

    All cosine similarities are computed in the original
    embedding space.

    The query is projected into the existing PCA space
    only for visualization.

    Returns
    -------
    dict:
        query_point:
            Query coordinates in the 2D PCA space.

        similarities:
            Every document chunk with its cosine similarity.

        results:
            The top-k retrieved chunks.

        dimension:
            Original embedding dimension.
    """

    if not chunks:
        raise ValueError("No chunks available for retrieval")

    query_embedding = embed_query(question, model)

    scores = cosine_similarity(
        query_embedding,
        embeddings,
    )

    # Stable sorting ensures deterministic ordering
    # when multiple chunks have identical scores.

    # TODO 5 — Rank the chunks by decreasing cosine similarity.
    # Return an array containing ALL chunk indices, best first.
    # Hint: np.argsort(...) sorts ascending; negate the scores.
    # Use kind="stable" to keep ties deterministic.
    raise NotImplementedError("TODO 5: rank chunks by similarity")

    similarities = [
        {
            "index": int(i),
            "score": float(scores[i]),
            "rank": rank + 1,
        }
        for rank, i in enumerate(ranked_indices)
    ]

    top_k = max(
        1,
        min(k, len(chunks)),
    )

    results = [
        {
            "index": int(i),
            "text": chunks[i],
            "score": float(scores[i]),
        }
        for i in ranked_indices[:top_k]
    ]

    query_point = project_query_pca(
        query_embedding,
        pca,
    )

    # Compare the original embedding space with PCA.

    comparison = compare_embedding_rankings(
        query_embedding=query_embedding,
        chunk_embeddings=embeddings,
        pca=pca,
        k=top_k,
    )

    return {
        "query_point": query_point,
        "similarities": similarities,
        "results": results,
        "dimension": int(query_embedding.shape[0]),
        # Pass 5: PCA vs original-space comparison.
        "comparison": comparison,
    }


# ============================================================
# STANDARD TOP-K RETRIEVAL
# ============================================================


def retrieve_top_k(
    question: str,
    chunks: list[str],
    embeddings: np.ndarray,
    model,
    k: int = 3,
) -> list[dict]:
    """
    Retrieve the top-k chunks using cosine similarity.

    Preserved for backward compatibility.
    """

    query = embed_query(
        question,
        model,
    )

    scores = cosine_similarity(
        query,
        embeddings,
    )

    indices = np.argsort(
        -scores,
        kind="stable",
    )[: max(1, min(k, len(chunks)))]

    return [
        {
            "index": int(i),
            "text": chunks[i],
            "score": float(scores[i]),
        }
        for i in indices
    ]


# ============================================================
# RAG PROMPT CONSTRUCTION
# ============================================================


def build_rag_prompt(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """Build a grounded RAG prompt for conversational question answering."""

    # TODO 6 — Construct the context from retrieved passages.
    # Each passage must be formatted as:
    #   [Chunk <index>]
    #   <text>
    # Separate consecutive passages with two newline characters.
    # Keep the prompt template below unchanged.
    # Hint: "\n\n".join(...)
    raise NotImplementedError("TODO 6: construct the RAG context")

    return f"""You are a precise question-answering assistant.

Your task is to answer the current question using the document excerpts below.

RULES:
1. Read the current question carefully.
2. Find the exact information that answers the question.
3. Use ONLY information explicitly stated in the document excerpts.
4. Do not confuse numbers describing different things.
5. Use previous conversation messages ONLY to understand references
   such as "it", "they", "he", "she", or "that system".
6. Previous assistant answers are NOT evidence. The answer must be
   supported by the current document excerpts.
7. If the answer is not present, say "Not specified in the document."
8. Keep your answer short and factual.
9. Cite the chunk containing the answer when possible.

DOCUMENT EXCERPTS:
{context}

CURRENT QUESTION:
{question}

ANSWER:
"""
