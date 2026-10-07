from threading import Lock, Thread

from flask import Flask, jsonify, render_template, request

import model as model

app = Flask(__name__)


# --------------------------------------------------
# Training state
# --------------------------------------------------

training_lock = Lock()

training_state = {
    "running": False,
    "finished": False,
    "error": None,
    "epoch": 0,
    "total_epochs": 0,
    "loss": None,
    "history": [],
}


# --------------------------------------------------
# Home page
# --------------------------------------------------


@app.route("/")
def index():

    return render_template("index.html")


# --------------------------------------------------
# Model information
# --------------------------------------------------


@app.route("/api/model-info")
def model_info():

    try:

        info = model.get_model_info()

        return jsonify(info)

    except Exception as error:

        return (
            jsonify(
                {
                    "error": str(error),
                }
            ),
            500,
        )


# --------------------------------------------------
# Training worker
# --------------------------------------------------


def training_worker(
    name,
    epochs,
    batch_size,
    learning_rate,
    max_length,
):

    def progress_callback(
        epoch,
        total_epochs,
        loss,
    ):

        with training_lock:

            training_state["epoch"] = epoch
            training_state["total_epochs"] = total_epochs

            training_state["loss"] = loss

            training_state["history"].append(
                {
                    "epoch": epoch,
                    "loss": loss,
                }
            )

    try:

        model.train_model(
            name=name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            progress_callback=progress_callback,
        )

        with training_lock:

            training_state["running"] = False
            training_state["finished"] = True

    except Exception as error:

        with training_lock:

            training_state["running"] = False
            training_state["finished"] = False
            training_state["error"] = str(error)


# --------------------------------------------------
# Start training
# --------------------------------------------------


@app.route(
    "/api/train",
    methods=["POST"],
)
def start_training():

    data = request.get_json()

    name = data.get(
        "name",
        "",
    ).strip()

    if not name:
        return jsonify({"error": ("A configuration name is required.")}), 400

    try:

        epochs = int(data.get("epochs", 50))

        batch_size = int(data.get("batch_size", 8))

        learning_rate = float(data.get("learning_rate", 5e-4))

        max_length = int(data.get("max_length", 64))

    except (TypeError, ValueError):

        return jsonify({"error": "Invalid training parameters."}), 400

    # Basic parameter validation.

    if epochs <= 0:

        return jsonify({"error": "Epochs must be positive."}), 400

    if batch_size <= 0:

        return jsonify({"error": "Batch size must be positive."}), 400

    if learning_rate <= 0:

        return jsonify({"error": "Learning rate must be positive."}), 400

    if max_length <= 0:

        return jsonify({"error": "Maximum length must be positive."}), 400

    # Do not allow two training jobs simultaneously.

    with training_lock:

        if training_state["running"]:

            return jsonify({"error": ("Training is already running.")}), 409

        training_state["running"] = True
        training_state["finished"] = False
        training_state["error"] = None
        training_state["epoch"] = 0
        training_state["total_epochs"] = epochs
        training_state["loss"] = None
        training_state["history"] = []

    # Start training without blocking Flask.

    thread = Thread(
        target=training_worker,
        args=(
            name,
            epochs,
            batch_size,
            learning_rate,
            max_length,
        ),
        daemon=True,
    )

    thread.start()

    return jsonify(
        {
            "message": "Training started.",
        }
    )


# --------------------------------------------------
# Training progress
# --------------------------------------------------


@app.route("/api/training-status")
def training_status():

    with training_lock:

        state = {
            "running": (training_state["running"]),
            "finished": (training_state["finished"]),
            "error": (training_state["error"]),
            "epoch": (training_state["epoch"]),
            "total_epochs": (training_state["total_epochs"]),
            "loss": (training_state["loss"]),
            "history": list(training_state["history"]),
        }

    return jsonify(state)


# --------------------------------------------------
# Tokenize a sentence
# --------------------------------------------------


@app.route(
    "/api/tokenize",
    methods=["POST"],
)
def tokenize():

    data = request.get_json()

    sentence = data.get(
        "sentence",
        "",
    ).strip()

    model_name = data.get("model_name")

    if not sentence:

        return jsonify({"error": "Sentence cannot be empty."}), 400

    try:

        result = model.tokenize_sentence(
            sentence=sentence,
            model_name=model_name,
        )

        return jsonify(result)

    except Exception as error:

        return (
            jsonify(
                {
                    "error": str(error),
                }
            ),
            500,
        )


# --------------------------------------------------
# Translate a sentence
# --------------------------------------------------


@app.route(
    "/api/translate",
    methods=["POST"],
)
def translate():

    data = request.get_json()

    sentence = data.get(
        "sentence",
        "",
    ).strip()

    model_name = data.get("model_name")

    max_new_tokens = int(
        data.get(
            "max_new_tokens",
            40,
        )
    )

    if not sentence:

        return jsonify({"error": "Sentence cannot be empty."}), 400

    try:

        result = model.translate(
            sentence=sentence,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )

        return jsonify(result)

    except Exception as error:

        return (
            jsonify(
                {
                    "error": str(error),
                }
            ),
            500,
        )


# --------------------------------------------------
# Run development server
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
