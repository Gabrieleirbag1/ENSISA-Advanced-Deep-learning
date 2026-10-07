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
    """
    Load the English / Quenya sentence pairs.

    Returns
    -------
    rows : list
        List of dictionaries containing the columns
        "english" and "quenya".
    """

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        rows = list(reader)

    return rows


# --------------------------------------------------
# Model directories
# --------------------------------------------------


def get_model_directory(name):

    if not re.fullmatch(
        r"[A-Za-z0-9_-]+",
        name,
    ):
        raise ValueError(
            "Model name may only contain letters, " "numbers, '-' and '_'."
        )

    return MODELS_DIR / name


# --------------------------------------------------
# Pretrained model
# --------------------------------------------------


def load_pretrained_model():
    """
    Load the pretrained T5 tokenizer and model.

    Returns
    -------
    tokenizer
        T5 tokenizer.

    model
        Pretrained sequence-to-sequence T5 model.
    """

    # TODO 1
    # Load the tokenizer corresponding to PRETRAINED_MODEL.
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    # TODO 2
    # Load the pretrained sequence-to-sequence model.
    model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)

    # TODO 3
    # Move the model to DEVICE.
    model = model.to(DEVICE)

    return tokenizer, model


# --------------------------------------------------
# Saved fine-tuned models
# --------------------------------------------------


def list_finetuned_models():
    """
    Return the configurations of all fine-tuned
    models stored in models/.
    """

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

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as f:

            training_config = json.load(f)

        models.append(training_config)

    return models


def load_finetuned_model(name):
    """
    Load a previously fine-tuned model.
    """

    model_directory = get_model_directory(name)

    if not (model_directory / "config.json").exists():

        raise FileNotFoundError(f"Fine-tuned model '{name}' " "does not exist.")

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
    """
    Fine-tune T5 on the English -> Quenya dataset.
    """

    # --------------------------------------------------
    # Load model and dataset
    # --------------------------------------------------

    tokenizer, model = load_pretrained_model()

    rows = load_dataset()

    model_directory = get_model_directory(name)

    # --------------------------------------------------
    # Build translation examples
    # --------------------------------------------------

    examples = []

    for row in rows:

        # TODO 4
        # T5 receives a task prefix followed by the
        # English sentence.
        #
        # Construct:
        #
        # "translate English to Quenya: <sentence>"

        source = None

        # TODO 5
        # Extract the expected Quenya translation.

        target = None

        examples.append((source, target))

    # --------------------------------------------------
    # Prepare batches
    # --------------------------------------------------

    def collate_fn(batch):

        sources = [source for source, target in batch]

        targets = [target for source, target in batch]

        # TODO 6
        # Tokenize the source sentences.
        #
        # Requirements:
        # - padding
        # - truncation
        # - max_length
        # - return PyTorch tensors

        inputs = None

        # TODO 7
        # Tokenize the target Quenya sentences.
        #
        # Hint:
        # tokenizer(..., text_target=...)

        labels = None

        # TODO 8
        # Padding tokens must not contribute to
        # the sequence-to-sequence cross-entropy loss.
        #
        # Replace padding token IDs in labels by -100.

        # TODO 9
        # Add the labels to the model inputs.

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
    # Create an AdamW optimizer using:
    #
    # - model.parameters()
    # - learning_rate

    optimizer = None

    # --------------------------------------------------
    # Fine-tuning loop
    # --------------------------------------------------

    loss_history = []

    for epoch in range(epochs):

        # TODO 11
        # Put the model in training mode.

        total_loss = 0.0

        for batch in dataloader:

            # Move every tensor in the batch
            # to the selected device.

            batch = {key: value.to(DEVICE) for key, value in batch.items()}

            # TODO 12
            # Reset gradients.

            # TODO 13
            # Perform the forward pass.
            #
            # The labels are already inside batch.
            # The T5 model will therefore compute
            # the sequence-to-sequence loss.

            outputs = None

            # TODO 14
            # Retrieve the loss produced by T5.

            loss = None

            # TODO 15
            # Backpropagate the loss.

            # TODO 16
            # Update the model parameters.

            total_loss += loss.item()

        mean_loss = total_loss / len(dataloader)

        loss_history.append(mean_loss)

        # The web application uses this callback
        # to update the loss curve in real time.

        if progress_callback is not None:

            progress_callback(
                epoch=epoch + 1,
                total_epochs=epochs,
                loss=mean_loss,
            )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # TODO 17
    # Save the fine-tuned model and tokenizer
    # inside model_directory.

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
        model_directory / "training_config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            training_config,
            f,
            indent=4,
        )

    return loss_history


# --------------------------------------------------
# Tokenization, embeddings and encoder
# --------------------------------------------------


def tokenize_sentence(
    sentence,
    model_name=None,
):
    """
    Tokenize a sentence and return information used
    by the web visualization.

    The function returns:
    - tokens
    - token IDs
    - input embeddings
    - contextual representations
    """

    if model_name is None:

        tokenizer, model = load_pretrained_model()

    else:

        tokenizer, model = load_finetuned_model(model_name)

    # --------------------------------------------------
    # Task prompt
    # --------------------------------------------------

    prompt = "translate English to Quenya: " + sentence

    # TODO 18
    # Tokenize the prompt and return PyTorch tensors.

    encoding = None

    # TODO 19
    # Extract input_ids and move them to DEVICE.

    input_ids = None

    # TODO 20
    # Convert the IDs of the first sequence
    # into a regular Python list.

    token_ids = None

    # TODO 21
    # Convert the token IDs back into their
    # SentencePiece token strings.

    tokens = None

    # --------------------------------------------------
    # Input embeddings
    # --------------------------------------------------

    # TODO 22
    # Retrieve the model's input embedding layer.

    embedding_layer = None

    # TODO 23
    # Pass input_ids through the embedding layer.
    #
    # Do not compute gradients during visualization.

    embeddings = None

    # Keep only the first two and last value of each
    # vector for visualization.

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

    # TODO 24
    # Pass the token sequence through the encoder.
    #
    # Use:
    # - input_ids
    # - attention_mask
    #
    # Do not compute gradients.

    encoder_outputs = None

    # TODO 25
    # Retrieve the last hidden state produced
    # by the encoder.

    hidden_states = None

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


def translate(
    sentence,
    model_name=None,
    max_new_tokens=40,
):
    """
    Translate an English sentence into Quenya.
    """

    if model_name is None:

        tokenizer, model = load_pretrained_model()

    else:

        tokenizer, model = load_finetuned_model(model_name)

    model.eval()

    prompt = "translate English to Quenya: " + sentence

    # TODO 26
    # Tokenize the complete translation prompt
    # and return PyTorch tensors.

    inputs = None

    # TODO 27
    # Move the tokenized inputs to DEVICE.

    # TODO 28
    # Generate the output sequence using the model.
    #
    # Use max_new_tokens.
    # Gradients are not needed.

    output = None

    # TODO 29
    # Convert the generated sequence into a list
    # of token IDs.

    output_ids = None

    # TODO 30
    # Convert the generated IDs into their
    # SentencePiece token strings.

    output_tokens = None

    # TODO 31
    # Decode the generated sequence into the final
    # human-readable Quenya sentence.
    #
    # Special tokens should not appear in the
    # final sentence.

    translation = None

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
        "finetuned_models": (list_finetuned_models()),
    }
