"""
app.py
======
Flask web application for real-time text emotion detection.

Loads the trained Keras model, tokenizer, label encoder, and max sequence
length, then exposes:
    GET  /            -> main UI
    GET  /about        -> about page
    POST /predict       -> JSON prediction endpoint (AJAX)
    GET  /health        -> health check endpoint

Author: Emotion Detection Project
"""

import os
import re
import pickle
import string
import logging

from flask import Flask, render_template, request, jsonify

import numpy as np

# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Flask app initialization
# --------------------------------------------------------------------------
app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# --------------------------------------------------------------------------
# Lazy globals for model artifacts (loaded once at startup)
# --------------------------------------------------------------------------
model = None
tokenizer = None
label_encoder = None
max_len = None

EMOTION_EMOJI = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲",
}

EMOTION_COLOR = {
    "joy": "#FFD166",
    "sadness": "#4C7DF0",
    "anger": "#EF476F",
    "fear": "#8D5BC7",
    "love": "#FF6B9D",
    "surprise": "#06D6A0",
}

# NLTK resources are needed for stopwords/lemmatization at inference time,
# matching the preprocessing used during training.
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    try:
        STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(stopwords.words("english"))

    try:
        LEMMATIZER = WordNetLemmatizer()
        LEMMATIZER.lemmatize("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        LEMMATIZER = WordNetLemmatizer()
except Exception as exc:  # pragma: no cover
    logger.error(f"Failed to initialize NLTK resources: {exc}")
    STOPWORDS = set()
    LEMMATIZER = None

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NUMBER_PATTERN = re.compile(r"\d+")
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z\s]")


def preprocess_text(text: str) -> str:
    """Clean raw input text using the same pipeline used during training."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_PATTERN.sub("", text)
    text = NUMBER_PATTERN.sub("", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = SPECIAL_CHAR_PATTERN.sub("", text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if LEMMATIZER is not None:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def load_artifacts():
    """Load the trained model and preprocessing artifacts into memory."""
    global model, tokenizer, label_encoder, max_len

    from tensorflow.keras.models import load_model

    try:
        logger.info("Loading model and artifacts ...")
        model = load_model(os.path.join(MODEL_DIR, "emotion_model.keras"))

        with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
            tokenizer = pickle.load(f)

        with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)

        with open(os.path.join(MODEL_DIR, "max_len.pkl"), "rb") as f:
            max_len = pickle.load(f)

        logger.info("Model and artifacts loaded successfully.")
    except Exception as exc:
        logger.error(f"Error loading model artifacts: {exc}")
        raise


@app.route("/")
def index():
    """Render the main prediction UI."""
    return render_template("index.html")


@app.route("/about")
def about():
    """Render the about page with project + model information."""
    return render_template("about.html")


@app.route("/health")
def health():
    """Simple health check endpoint for deployment platforms."""
    ready = model is not None
    return jsonify({"status": "ok", "model_loaded": ready}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the emotion of a piece of input text.

    Expects JSON: { "text": "..." }
    Returns JSON: {
        "emotion": str, "emoji": str, "color": str,
        "confidence": float, "probabilities": {label: prob, ...}
    }
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    try:
        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Input text cannot be empty."}), 400

        if model is None:
            return jsonify({"error": "Model is not loaded yet. Please try again shortly."}), 503

        clean = preprocess_text(text)
        if not clean:
            return jsonify({"error": "Input text has no meaningful content after cleaning."}), 400

        seq = tokenizer.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

        probs = model.predict(padded, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        emotion = label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])

        probabilities = {
            label_encoder.classes_[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        }

        response = {
            "emotion": emotion,
            "emoji": EMOTION_EMOJI.get(emotion, "🙂"),
            "color": EMOTION_COLOR.get(emotion, "#4C7DF0"),
            "confidence": round(confidence * 100, 2),
            "probabilities": probabilities,
        }
        logger.info(f"Prediction: '{text[:50]}...' -> {emotion} ({confidence:.2%})")
        return jsonify(response), 200

    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Internal server error: {str(exc)}"}), 500


@app.route("/predict-form", methods=["POST"])
def predict_form():
    """Non-JS fallback: classic form POST that renders result.html."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    text = request.form.get("text", "").strip()
    if not text or model is None:
        return render_template(
            "result.html", sentence=text, emotion="unknown",
            emoji="🙂", color="#4C7DF0", confidence=0
        )

    clean = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    emotion = label_encoder.inverse_transform([pred_idx])[0]
    confidence = round(float(probs[pred_idx]) * 100, 2)

    return render_template(
        "result.html",
        sentence=text,
        emotion=emotion,
        emoji=EMOTION_EMOJI.get(emotion, "🙂"),
        color=EMOTION_COLOR.get(emotion, "#4C7DF0"),
        confidence=confidence,
    )


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


# Load model artifacts at import time so the app works under both
# `python app.py` and WSGI servers such as gunicorn.
try:
    load_artifacts()
except Exception:
    logger.warning("Model artifacts not found yet - run train_pipeline.py first.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
