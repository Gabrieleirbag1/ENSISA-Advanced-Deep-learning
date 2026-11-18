"""Flask API and frontend for the RAG Visual Lab."""

from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import pipeline
from models import get_embedding_model, get_llm

# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# APPLICATION STATE
# ============================================================

STATE = {
    "text": "",
    "chunks": [],
    "embeddings": None,
    "pca": None,
    "pca_points": None,
    "pages": [],
    "history": [],
}


# ============================================================
# HISTORY-AWARE RETRIEVAL
# ============================================================


def build_search_query(
    question: str,
    history: list[dict],
) -> str:
    """Build a retrieval query using recent conversation context."""

    if not history:
        return question

    previous = history[-1]

    return (
        f"Previous question: {previous['question']}\n"
        f"Previous answer: {previous['answer']}\n"
        f"Current question: {question}"
    )


# ============================================================
# ERROR HANDLING
# ============================================================


def failure(exc):
    status = 501 if isinstance(exc, NotImplementedError) else 400

    return (
        jsonify(
            error=str(exc),
            type=type(exc).__name__,
        ),
        status,
    )


# ============================================================
# FRONTEND
# ============================================================


@app.get("/")
def index():
    return render_template("index.html")


# ============================================================
# APPLICATION STATUS
# ============================================================


@app.get("/api/status")
def status():
    return jsonify(
        document_loaded=bool(STATE["text"]),
        chunks=len(STATE["chunks"]),
        indexed=STATE["embeddings"] is not None,
        pca_available=STATE["pca"] is not None,
    )


# ============================================================
# PDF EXTRACTION
# ============================================================


@app.post("/api/document")
def document():
    try:
        import fitz

        uploaded = request.files.get("file")

        if (
            uploaded is None
            or Path(secure_filename(uploaded.filename or "")).suffix.lower() != ".pdf"
        ):
            return jsonify(error="Upload a PDF file."), 400

        data = uploaded.read()

        with fitz.open(stream=data, filetype="pdf") as pdf:

            page_texts = [page.get_text() for page in pdf]

            text = "\n".join(page_texts)

            pages = len(pdf)

        if not text.strip():
            return (
                jsonify(
                    error=("No extractable text found " "(scanned PDFs need OCR).")
                ),
                400,
            )

        # A new document invalidates all previous
        # embeddings, PCA projections, and history.

        STATE.update(
            text=text,
            chunks=[],
            embeddings=None,
            pca=None,
            pca_points=None,
            pages=page_texts,
            history=[],
        )

        return jsonify(
            pages=pages,
            characters=len(text),
            words=len(text.split()),
            preview=text[:1200],
            page_previews=[
                {
                    "page": i + 1,
                    "characters": len(p),
                    "preview": p[:500],
                }
                for i, p in enumerate(page_texts)
            ],
        )

    except Exception as exc:
        return failure(exc)


# ============================================================
# TEXT CHUNKING
# ============================================================


@app.post("/api/chunks")
def chunks():
    try:

        if not STATE["text"]:
            return jsonify(error="Upload a PDF first."), 400

        data = request.get_json(silent=True) or {}

        size = int(data.get("chunk_size", 500))

        overlap = int(data.get("overlap", 100))

        if not 50 <= size <= 5000 or not 0 <= overlap < size:
            return jsonify(error="Invalid chunk size or overlap."), 400

        metadata = pipeline.chunk_text_with_metadata(
            STATE["text"],
            size,
            overlap,
        )

        # Changing the chunks invalidates the existing
        # vector index and PCA projection.

        STATE.update(
            chunks=[c["text"] for c in metadata],
            embeddings=None,
            pca=None,
            pca_points=None,
            history=[],
        )

        return jsonify(
            chunks=metadata,
            count=len(metadata),
            chunk_size=size,
            overlap=overlap,
        )

    except Exception as exc:
        return failure(exc)


# ============================================================
# DOCUMENT VECTORIZATION + PCA
# ============================================================


@app.post("/api/vectorize")
def vectorize():
    try:

        if not STATE["chunks"]:
            return jsonify(error="Create chunks first."), 400

        # ----------------------------------------------------
        # 1. Generate document embeddings
        # ----------------------------------------------------

        vectors = np.asarray(
            pipeline.vectorize_chunks(
                STATE["chunks"],
                get_embedding_model(),
            )
        )

        if vectors.ndim != 2 or vectors.shape[0] != len(STATE["chunks"]):
            raise ValueError(
                "Expected embedding matrix of shape " "(number_of_chunks, dimension)."
            )

        # ----------------------------------------------------
        # 2. Fit PCA on document embeddings
        # ----------------------------------------------------

        pca, points = pipeline.fit_embedding_pca(vectors)

        # ----------------------------------------------------
        # 3. Store embeddings and fitted PCA
        # ----------------------------------------------------

        STATE.update(
            embeddings=vectors,
            pca=pca,
            pca_points=points,
        )

        # ----------------------------------------------------
        # 4. Return visualization information
        # ----------------------------------------------------

        return jsonify(
            dimension=int(vectors.shape[1]),
            points=points.tolist(),
            vectors_preview=vectors[:, :6].tolist(),
            pca_available=pca is not None,
        )

    except Exception as exc:
        return failure(exc)


# ============================================================
# HISTORY-AWARE VECTOR RETRIEVAL
# ============================================================


@app.post("/api/retrieve")
def retrieve():
    try:

        if STATE["embeddings"] is None:
            return jsonify(error="Vectorize chunks first."), 400

        data = request.get_json(silent=True) or {}

        question = str(data.get("question", "")).strip()

        if not question:
            return jsonify(error="Enter a question."), 400

        # ----------------------------------------------------
        # 1. Retrieval parameters
        # ----------------------------------------------------

        k = max(
            1,
            min(
                int(data.get("top_k", 3)),
                len(STATE["chunks"]),
            ),
        )

        history_length = max(
            0,
            min(
                int(data.get("history_length", 3)),
                5,
            ),
        )

        # ----------------------------------------------------
        # 2. Conversation history
        # ----------------------------------------------------

        history = STATE["history"][-history_length:] if history_length > 0 else []

        search_query = build_search_query(
            question,
            history,
        )

        # ----------------------------------------------------
        # 3. Vector search with visualization data
        # ----------------------------------------------------

        retrieval = pipeline.retrieve_with_visualization(
            question=search_query,
            chunks=STATE["chunks"],
            embeddings=STATE["embeddings"],
            model=get_embedding_model(),
            pca=STATE["pca"],
            k=k,
        )

        results = retrieval["results"]

        # ----------------------------------------------------
        # 4. Build the RAG prompt
        # ----------------------------------------------------

        prompt = pipeline.build_rag_prompt(
            question,
            results,
        )

        # ----------------------------------------------------
        # 5. Return retrieval + animation data
        # ----------------------------------------------------

        return jsonify(
            # Existing retrieval information
            results=results,
            prompt=prompt,
            search_query=search_query,
            history_used=len(history),
            # Pass 4: Animated vector search
            query_point=retrieval["query_point"],
            similarities=retrieval["similarities"],
            dimension=retrieval["dimension"],
            pca_available=STATE["pca"] is not None,
            # Pass 5: PCA vs. Actual Similarity
            comparison=retrieval["comparison"],
        )

    except Exception as exc:
        return failure(exc)


# ============================================================
# LOCAL LLM GENERATION
# ============================================================


@app.post("/api/chat")
def chat():
    try:

        data = request.get_json(silent=True) or {}

        prompt = str(data.get("prompt", "")).strip()

        question = str(data.get("question", "")).strip()

        if not prompt or not question:
            return jsonify(error="Prompt and question are required."), 400

        temperature = float(data.get("temperature", 0.2))

        if not 0 <= temperature <= 2:
            return jsonify(error="Temperature must be between 0 and 2."), 400

        history_length = max(
            0,
            min(
                int(data.get("history_length", 3)),
                5,
            ),
        )

        history = STATE["history"][-history_length:] if history_length > 0 else []

        # ----------------------------------------------------
        # 1. System instruction
        # ----------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise question-answering assistant. "
                    "Answer using only the retrieved document excerpts "
                    "provided in the latest user message. "
                    "Use conversation history only to understand "
                    "references in follow-up questions. "
                    "If the answer is not present in the retrieved "
                    "excerpts, say it is not specified in the document. "
                    "Keep answers short and factual."
                ),
            }
        ]

        # ----------------------------------------------------
        # 2. Add previous exchanges
        # ----------------------------------------------------

        for exchange in history:

            messages.extend(
                [
                    {
                        "role": "user",
                        "content": exchange["question"],
                    },
                    {
                        "role": "assistant",
                        "content": exchange["answer"],
                    },
                ]
            )

        # ----------------------------------------------------
        # 3. Add current RAG prompt
        # ----------------------------------------------------

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        # ----------------------------------------------------
        # 4. Generate with local Qwen model
        # ----------------------------------------------------

        result = get_llm().create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=temperature,
            logprobs=True,
            top_logprobs=5,
        )

        choice = result["choices"][0]

        answer = choice["message"]["content"]

        # Extract the token-generation trace.
        token_trace = []

        logprobs_data = choice.get("logprobs") or {}

        for step, token_info in enumerate(logprobs_data.get("content") or []):

            selected_token = token_info.get("token", "")

            selected_logprob = token_info.get("logprob")

            candidates = []

            for candidate in token_info.get("top_logprobs") or []:

                logprob = candidate.get("logprob")

                candidates.append(
                    {
                        "token": candidate.get("token", ""),
                        "logprob": float(logprob) if logprob is not None else None,
                        "probability": (
                            float(np.exp(logprob)) if logprob is not None else None
                        ),
                    }
                )

            token_trace.append(
                {
                    "step": step + 1,
                    "token": selected_token,
                    "logprob": float(selected_logprob) if selected_logprob is not None else None,
                    "probability": (
                        float(np.exp(selected_logprob))
                        if selected_logprob is not None
                        else None
                    ),
                    "candidates": candidates,
                }
            )

        # ----------------------------------------------------
        # 5. Store conversation history
        # ----------------------------------------------------

        STATE["history"].append(
            {
                "question": question,
                "answer": answer,
            }
        )

        STATE["history"] = STATE["history"][-20:]

        # ----------------------------------------------------
        # 6. Return answer
        # ----------------------------------------------------

        return jsonify(
            answer=answer,
            usage=result.get("usage", {}),
            history_used=len(history),
            messages=messages,
            # Pass 6: Token generation visualization
            generation={
                "tokens": token_trace,
                "temperature": temperature,
                "total_steps": len(token_trace),
                "top_logprobs": 5,
                "finish_reason": choice.get("finish_reason"),
            },
        )

    except Exception as exc:
        return failure(exc)


# ============================================================
# RESET ENTIRE SESSION
# ============================================================


@app.post("/api/reset")
def reset():
    """Remove the current document and all derived data."""

    STATE.update(
        text="",
        chunks=[],
        embeddings=None,
        pca=None,
        pca_points=None,
        pages=[],
        history=[],
    )

    return jsonify(success=True)


# ============================================================
# CLEAR CONVERSATION HISTORY
# ============================================================


@app.post("/api/chat/clear")
def clear_chat_history():
    """Clear conversation memory without removing the PDF."""

    STATE["history"] = []

    return jsonify(success=True)


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000,
    )
