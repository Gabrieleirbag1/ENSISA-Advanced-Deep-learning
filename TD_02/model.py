import csv
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# --------------------------------------------------
# Paths and default model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "English-to-Elvish-clean.tsv"
MODELS_DIR = BASE_DIR / "models"
PRETRAINED_MODEL = "google/t5-efficient-tiny"
DEVICE = torch.device("cpu")


# --------------------------------------------------
# Dataset
# --------------------------------------------------


def load_dataset():
    """Load the English / Quenya sentence pairs."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    return rows


# --------------------------------------------------
# Model directories
# --------------------------------------------------


def get_model_directory(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Model name may only contain letters, numbers, '-' and '_'.")
    return MODELS_DIR / name


# --------------------------------------------------
# Pretrained model
# --------------------------------------------------


def load_pretrained_model():
    """Load the pretrained T5 tokenizer and model."""

    # TODO 1: tokenizer associé au checkpoint
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    # TODO 2: modèle séquence-à-séquence pré-entraîné
    model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)

    # TODO 3: déplacer le modèle sur DEVICE
    model = model.to(DEVICE)

    return tokenizer, model


# --------------------------------------------------
# Saved fine-tuned models
# --------------------------------------------------


def list_finetuned_models():
    """Return the configurations of all fine-tuned models stored in models/."""

    if not MODELS_DIR.exists():
        return []

    models = []

    for directory in sorted(MODELS_DIR.iterdir()):
        if not directory.is_dir():
            continue

        config_path = directory / "training_config.json"
        model_config_path = directory / "config.json"

        if not config_path.exists() or not model_config_path.exists():
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            training_config = json.load(f)

        models.append(training_config)

    return models


def load_finetuned_model(name):
    """Load a previously fine-tuned model."""

    model_directory = get_model_directory(name)

    if not (model_directory / "config.json").exists():
        raise FileNotFoundError(f"Fine-tuned model '{name}' does not exist.")

    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_directory)
    model = model.to(DEVICE)

    return tokenizer, model


# --------------------------------------------------
# Training
# --------------------------------------------------


def train_model(
    name,
    epochs=50,
    batch_size=8,
    learning_rate=5e-4,
    max_length=64,
    progress_callback=None,
):
    """Fine-tune T5 on the English -> Quenya dataset."""

    tokenizer, model = load_pretrained_model()
    rows = load_dataset()
    model_directory = get_model_directory(name)

    # --------------------------------------------------
    # Build translation examples
    # --------------------------------------------------

    examples = []

    for row in rows:
        # TODO 4: préfixe de tâche + phrase anglaise
        source = "translate English to Quenya: " + row["english"]

        # TODO 5: traduction attendue
        target = row["quenya"]

        examples.append((source, target))

    # --------------------------------------------------
    # Prepare batches
    # --------------------------------------------------

    def collate_fn(batch):

        sources = [source for source, target in batch]
        targets = [target for source, target in batch]

        # TODO 6: tokenisation des sources
        inputs = tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # TODO 7: tokenisation des cibles (text_target uniquement,
        # on ne passe PAS les cibles en argument positionnel)
        labels = tokenizer(
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # TODO 8: le padding ne doit pas compter dans la loss -> -100
        # (on reste sur des tenseurs, pas des listes Python)
        label_ids = labels["input_ids"]
        label_ids[label_ids == tokenizer.pad_token_id] = -100

        # TODO 9: ajouter les labels aux entrées du modèle
        inputs["labels"] = label_ids

        return inputs

    dataloader = DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    # TODO 10
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    # --------------------------------------------------
    # Fine-tuning loop
    # --------------------------------------------------

    loss_history = []

    for epoch in range(epochs):

        # TODO 11: mode entraînement (active le dropout)
        model.train()

        total_loss = 0.0

        for batch in dataloader:

            batch = {key: value.to(DEVICE) for key, value in batch.items()}

            # TODO 12: remise à zéro des gradients
            optimizer.zero_grad()

            # TODO 13: forward pass (labels dans batch -> loss calculée par T5)
            outputs = model(**batch)

            # TODO 14: récupération de la loss
            loss = outputs.loss

            # TODO 15: rétropropagation
            loss.backward()

            # TODO 16: mise à jour des paramètres
            optimizer.step()

            total_loss += loss.item()

        mean_loss = total_loss / len(dataloader)
        loss_history.append(mean_loss)

        if progress_callback is not None:
            progress_callback(
                epoch=epoch + 1,
                total_epochs=epochs,
                loss=mean_loss,
            )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    # TODO 17: sauvegarde du modèle et du tokenizer
    model.save_pretrained(model_directory)
    tokenizer.save_pretrained(model_directory)

    # --------------------------------------------------
    # Save experiment configuration
    # --------------------------------------------------

    training_config = {
        "name": name,
        "base_model": PRETRAINED_MODEL,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "dataset_size": len(rows),
    }

    with open(
        model_directory / "training_config.json", "w", encoding="utf-8"
    ) as f:
        json.dump(training_config, f, indent=4)

    return loss_history


# --------------------------------------------------
# Tokenization, embeddings and encoder
# --------------------------------------------------


def tokenize_sentence(sentence, model_name=None):
    """Tokenize a sentence and return information used by the visualization."""

    if model_name is None:
        tokenizer, model = load_pretrained_model()
    else:
        tokenizer, model = load_finetuned_model(model_name)

    prompt = "translate English to Quenya: " + sentence

    # TODO 18: tokenisation du prompt complet
    encoding = tokenizer(prompt, return_tensors="pt")

    # TODO 19: input_ids sur DEVICE
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    # TODO 20: IDs de la première séquence en liste Python
    token_ids = input_ids[0].tolist()

    # TODO 21: IDs -> tokens SentencePiece
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # --------------------------------------------------
    # Input embeddings
    # --------------------------------------------------

    # TODO 22: couche d'embedding d'entrée
    embedding_layer = model.get_input_embeddings()

    # TODO 23: embeddings sans gradient
    with torch.no_grad():
        embeddings = embedding_layer(input_ids)

    embedding_vectors = []

    for vector in embeddings[0]:
        embedding_vectors.append(
            {
                "first": vector[0].item(),
                "second": vector[1].item(),
                "last": vector[-1].item(),
            }
        )

    # --------------------------------------------------
    # Transformer encoder
    # --------------------------------------------------

    model.eval()

    # TODO 24: passage dans l'encodeur (input_ids + attention_mask, sans gradient)
    with torch.no_grad():
        encoder_outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # TODO 25: dernier état caché
    hidden_states = encoder_outputs.last_hidden_state

    contextual_vectors = []

    for vector in hidden_states[0]:
        contextual_vectors.append(
            {
                "first": vector[0].item(),
                "second": vector[1].item(),
                "last": vector[-1].item(),
            }
        )

    return {
        "sentence": sentence,
        "prompt": prompt,
        "token_ids": token_ids,
        "tokens": tokens,
        "embedding_dimension": embeddings.shape[-1],
        "embeddings": embedding_vectors,
        "contextual_vectors": contextual_vectors,
    }


# --------------------------------------------------
# Translation
# --------------------------------------------------


def translate(sentence, model_name=None, max_new_tokens=40):
    """Translate an English sentence into Quenya."""

    if model_name is None:
        tokenizer, model = load_pretrained_model()
    else:
        tokenizer, model = load_finetuned_model(model_name)

    model.eval()

    prompt = "translate English to Quenya: " + sentence

    # TODO 26: tokenisation du prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # TODO 27: déplacement sur DEVICE
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    # TODO 28: génération autoregressive sans gradient
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # TODO 29: IDs générés en liste
    output_ids = output[0].tolist()

    # TODO 30: IDs -> tokens SentencePiece
    output_tokens = tokenizer.convert_ids_to_tokens(output_ids)

    # TODO 31: décodage en texte, sans tokens spéciaux
    translation = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "sentence": sentence,
        "translation": translation,
        "token_ids": output_ids,
        "tokens": output_tokens,
    }


# --------------------------------------------------
# Model information used by the web interface
# --------------------------------------------------


def get_model_info():

    tokenizer, model = load_pretrained_model()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    return {
        "name": PRETRAINED_MODEL,
        "parameters": parameter_count,
        "device": str(DEVICE),
        "dataset_size": len(load_dataset()),
        "finetuned_models": list_finetuned_models(),
    }