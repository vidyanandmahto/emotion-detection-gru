"""
train_pipeline.py
==================
End-to-end training pipeline for the Text Emotion Detection project.

Covers:
    Phase 1  - Data Understanding
    Phase 2  - Data Preprocessing
    Phase 3  - Target Encoding
    Phase 4  - Tokenization
    Phase 5  - Train/Test Split
    Phase 6  - Model Building (SimpleRNN, LSTM, GRU)
    Phase 7  - Model Evaluation
    Phase 8  - Visualization
    Phase 9  - Model Saving
    Phase 10 - Custom Sentence Testing

Run:
    python3 train_pipeline.py
"""

import os
import re
import json
import pickle
import logging
import string

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# --------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Global configuration
# --------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "data"
MODEL_DIR = "model"
ASSETS_DIR = "notebook/assets"   # generated charts / reports
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

VOCAB_SIZE = 12000
OOV_TOKEN = "<OOV>"
EMBEDDING_DIM = 64
EPOCHS = 12
BATCH_SIZE = 64

sns.set_style("whitegrid")


# ==========================================================================
# PHASE 1 : DATA UNDERSTANDING
# ==========================================================================
def load_data():
    """Load train/val/test emotion datasets (text;label format)."""
    logger.info("PHASE 1: Loading dataset ...")
    cols = ["text", "emotion"]
    train_df = pd.read_csv(f"{DATA_DIR}/train.txt", sep=";", names=cols, engine="python")
    val_df = pd.read_csv(f"{DATA_DIR}/validation.txt", sep=";", names=cols, engine="python")
    test_df = pd.read_csv(f"{DATA_DIR}/test.txt", sep=";", names=cols, engine="python")

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Validation shape: {val_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    logger.info(f"Combined dataset shape: {full_df.shape}")
    logger.info(f"Duplicate rows: {full_df.duplicated().sum()}")
    logger.info(f"Missing values:\n{full_df.isnull().sum()}")
    logger.info(f"Emotion distribution:\n{full_df['emotion'].value_counts()}")

    # Plot class distribution
    plt.figure(figsize=(8, 5))
    order = full_df["emotion"].value_counts().index
    sns.countplot(data=full_df, x="emotion", order=order, palette="viridis")
    plt.title("Emotion Class Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/class_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 7))
    full_df["emotion"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90)
    plt.title("Emotion Distribution (Pie Chart)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/class_distribution_pie.png", dpi=150)
    plt.close()

    return full_df


# ==========================================================================
# PHASE 2 : DATA PREPROCESSING
# ==========================================================================
STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NUMBER_PATTERN = re.compile(r"\d+")
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z\s]")


def preprocess_text(text: str, use_stemming: bool = False, use_lemmatization: bool = True) -> str:
    """
    Reusable NLP preprocessing function.

    Steps: lowercase -> URL removal -> number removal -> punctuation/special char
    removal -> tokenization -> stopword removal -> stemming/lemmatization.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_PATTERN.sub("", text)
    text = NUMBER_PATTERN.sub("", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = SPECIAL_CHAR_PATTERN.sub("", text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    if use_stemming:
        tokens = [STEMMER.stem(t) for t in tokens]
    if use_lemmatization:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("PHASE 2: Running text preprocessing ...")
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    sample_before = df["text"].iloc[0]
    df["clean_text"] = df["text"].apply(preprocess_text)
    sample_after = df["clean_text"].iloc[0]

    logger.info(f"BEFORE: {sample_before}")
    logger.info(f"AFTER : {sample_after}")

    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)
    return df


# ==========================================================================
# PHASE 3 : TARGET ENCODING
# ==========================================================================
def encode_labels(df: pd.DataFrame):
    logger.info("PHASE 3: Encoding target labels ...")
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["emotion"])

    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    logger.info(f"Emotion -> Integer mapping: {mapping}")

    with open(f"{MODEL_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    return df, le


# ==========================================================================
# PHASE 4 : TOKENIZATION
# ==========================================================================
def tokenize_and_pad(df: pd.DataFrame):
    logger.info("PHASE 4: Tokenizing text ...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(df["clean_text"])

    sequences = tokenizer.texts_to_sequences(df["clean_text"])
    logger.info(f"Example sequence: {sequences[0][:15]} ...")

    max_len = int(np.percentile([len(s) for s in sequences], 95))
    max_len = max(max_len, 5)
    logger.info(f"Chosen max_len (95th percentile of seq length): {max_len}")

    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    with open(f"{MODEL_DIR}/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open(f"{MODEL_DIR}/max_len.pkl", "wb") as f:
        pickle.dump(max_len, f)

    return padded, tokenizer, max_len


# ==========================================================================
# PHASE 5 : TRAIN / TEST SPLIT
# ==========================================================================
def split_data(X, y):
    logger.info("PHASE 5: Splitting data (80/20, stratified) ...")
    # stratify=y preserves the class proportions of the imbalanced emotion
    # classes in both the train and test partitions; random_state=42 makes
    # the split reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ==========================================================================
# PHASE 6 : MODEL BUILDING
# ==========================================================================
def build_model(cell_type: str, vocab_size: int, max_len: int, num_classes: int):
    model = Sequential(name=f"Emotion_{cell_type}")
    model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len))

    if cell_type == "SimpleRNN":
        model.add(SimpleRNN(64, return_sequences=False))
    elif cell_type == "LSTM":
        model.add(LSTM(64, return_sequences=False))
    elif cell_type == "GRU":
        model.add(GRU(64, return_sequences=False))

    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_all_models(X_train, X_test, y_train, y_test, vocab_size, max_len, num_classes):
    logger.info("PHASE 6: Building & training SimpleRNN, LSTM, GRU models ...")

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    results = {}
    histories = {}
    models = {}

    for cell_type in ["SimpleRNN", "LSTM", "GRU"]:
        logger.info(f"--- Training {cell_type} ---")
        model = build_model(cell_type, vocab_size, max_len, num_classes)
        model.summary(print_fn=logger.info)

        history = model.fit(
            X_train, y_train_cat,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=2,
        )

        loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
        logger.info(f"{cell_type} Test Accuracy: {acc:.4f} | Test Loss: {loss:.4f}")

        results[cell_type] = acc
        histories[cell_type] = history.history
        models[cell_type] = model

    best_model_name = max(results, key=results.get)
    logger.info(f"BEST MODEL: {best_model_name} (Test Accuracy: {results[best_model_name]:.4f})")

    return models, histories, results, best_model_name


# ==========================================================================
# PHASE 7 : MODEL EVALUATION
# ==========================================================================
def evaluate_model(model, X_test, y_test, label_encoder, model_name="model"):
    logger.info(f"PHASE 7: Evaluating {model_name} ...")
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"Accuracy : {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall   : {rec:.4f}")
    logger.info(f"F1 Score : {f1:.4f}")
    logger.info(f"\nClassification Report:\n{report}")

    with open(f"{ASSETS_DIR}/classification_report_{model_name}.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n\n")
        f.write(report)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm, "y_pred": y_pred}


# ==========================================================================
# PHASE 8 : VISUALIZATION
# ==========================================================================
def plot_training_curves(histories):
    logger.info("PHASE 8: Plotting training curves ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, hist in histories.items():
        axes[0].plot(hist["accuracy"], label=f"{name} train")
        axes[0].plot(hist["val_accuracy"], linestyle="--", label=f"{name} val")
        axes[1].plot(hist["loss"], label=f"{name} train")
        axes[1].plot(hist["val_loss"], linestyle="--", label=f"{name} val")

    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy"); axes[0].legend(fontsize=8)
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/training_curves.png", dpi=150)
    plt.close()


def plot_confusion_matrix(cm, classes, model_name="best_model"):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/confusion_matrix_{model_name}.png", dpi=150)
    plt.close()


def plot_model_comparison(results):
    plt.figure(figsize=(7, 5))
    names = list(results.keys())
    accs = [results[n] for n in names]
    bars = plt.bar(names, accs, color=["#4C72B0", "#55A868", "#C44E52"])
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.005, f"{acc:.3f}", ha="center")
    plt.title("Model Comparison - Test Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/model_comparison.png", dpi=150)
    plt.close()


def plot_prediction_distribution(y_pred, label_encoder, model_name="best_model"):
    labels = label_encoder.inverse_transform(y_pred)
    plt.figure(figsize=(8, 5))
    sns.countplot(x=labels, order=label_encoder.classes_, palette="magma")
    plt.title(f"Prediction Distribution - {model_name}")
    plt.xlabel("Predicted Emotion")
    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/prediction_distribution_{model_name}.png", dpi=150)
    plt.close()


# ==========================================================================
# PHASE 9 : MODEL SAVING
# ==========================================================================
def save_best_model(model, model_name):
    logger.info(f"PHASE 9: Saving best model ({model_name}) ...")
    model.save(f"{MODEL_DIR}/emotion_model.keras")
    with open(f"{MODEL_DIR}/model_info.json", "w") as f:
        json.dump({"best_model": model_name}, f, indent=2)
    logger.info(f"Saved to {MODEL_DIR}/emotion_model.keras")


# ==========================================================================
# PHASE 10 : CUSTOM SENTENCE TESTING
# ==========================================================================
CUSTOM_SENTENCES = [
    "I am so happy today, everything is going great!",
    "I feel like crying, nothing is working out for me.",
    "This makes me so angry, I cannot believe it.",
    "I am scared of what might happen tomorrow.",
    "I love spending time with my family.",
    "What a shocking and unexpected surprise!",
    "I feel completely hopeless about the future.",
    "You make me feel so special and loved.",
    "I am furious about how they treated me.",
    "I am terrified of the dark.",
    "Life feels wonderful and full of joy right now.",
    "I can't stop smiling, today was amazing.",
    "I feel so lonely and sad tonight.",
    "That was an absolutely delightful surprise party.",
    "I am nervous about my exam results.",
    "I appreciate everything you have done for me.",
    "This news left me speechless with shock.",
    "I feel worthless and empty inside.",
    "I trust you with all my heart.",
    "I am proud and excited about this achievement.",
]


def test_custom_sentences(model, tokenizer, max_len, label_encoder):
    logger.info("PHASE 10: Testing custom sentences ...")
    rows = []
    for sentence in CUSTOM_SENTENCES:
        clean = preprocess_text(sentence)
        seq = tokenizer.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        prob = model.predict(padded, verbose=0)[0]
        pred_idx = int(np.argmax(prob))
        emotion = label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(prob[pred_idx])
        rows.append({
            "sentence": sentence,
            "predicted_emotion": emotion,
            "confidence": round(confidence, 4),
            "probabilities": {label_encoder.classes_[i]: round(float(p), 4) for i, p in enumerate(prob)}
        })
        logger.info(f"'{sentence}' -> {emotion} ({confidence:.2%})")

    with open(f"{ASSETS_DIR}/custom_predictions.json", "w") as f:
        json.dump(rows, f, indent=2)

    return rows


# ==========================================================================
# MAIN PIPELINE
# ==========================================================================
def main():
    df = load_data()
    df = run_preprocessing(df)
    df, label_encoder = encode_labels(df)

    padded, tokenizer, max_len = tokenize_and_pad(df)
    y = df["label"].values
    num_classes = len(label_encoder.classes_)

    X_train, X_test, y_train, y_test = split_data(padded, y)

    models, histories, results, best_model_name = train_all_models(
        X_train, X_test, y_train, y_test, VOCAB_SIZE, max_len, num_classes
    )

    plot_training_curves(histories)
    plot_model_comparison(results)

    best_model = models[best_model_name]
    eval_result = evaluate_model(best_model, X_test, y_test, label_encoder, best_model_name)
    plot_confusion_matrix(eval_result["cm"], label_encoder.classes_, best_model_name)
    plot_prediction_distribution(eval_result["y_pred"], label_encoder, best_model_name)

    save_best_model(best_model, best_model_name)
    test_custom_sentences(best_model, tokenizer, max_len, label_encoder)

    logger.info("PIPELINE COMPLETE.")
    logger.info(f"Final results: { {k: round(v,4) for k,v in results.items()} }")


if __name__ == "__main__":
    main()
