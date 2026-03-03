import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)

from utils import load_sms_spam_df, train_test_split_text


def build_model(vocab_size: int, max_len: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=64),
        layers.SpatialDropout1D(0.2),
        layers.LSTM(64, dropout=0.2, recurrent_dropout=0.0),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    os.makedirs("results", exist_ok=True)

    # Reproducibility (as much as practical)
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    df = load_sms_spam_df()
    X_train, X_test, y_train, y_test = train_test_split_text(df, random_state=seed)

    # Text vectorization for deep learning
    vocab_size = 20000
    max_len = 60  # SMS messages are short

    vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len,
        standardize=None,  # already cleaned in utils.py
        split="whitespace"
    )
    vectorizer.adapt(X_train)

    X_train_seq = vectorizer(np.array(X_train))
    X_test_seq = vectorizer(np.array(X_test))

    y_train_np = np.array(y_train).astype("float32")
    y_test_np = np.array(y_test).astype("int32")

    model = build_model(vocab_size=vocab_size, max_len=max_len)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train_seq, y_train_np,
        validation_split=0.2,
        epochs=8,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Predictions
    probs = model.predict(X_test_seq, verbose=0).reshape(-1)
    y_pred = (probs >= 0.5).astype("int32")

    metrics = {
        "model": "Embedding + LSTM",
        "accuracy": float(accuracy_score(y_test_np, y_pred)),
        "precision": float(precision_score(y_test_np, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test_np, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test_np, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test_np, y_pred).tolist(),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }

    print("\n=== LSTM Deep Learning Results ===")
    print(json.dumps(metrics, indent=2))
    print("\nClassification Report:\n")
    print(classification_report(y_test_np, y_pred, digits=4, zero_division=0))

    with open("results/lstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model (optional but nice for a project)
    model.save("results/lstm_model.keras")
    print("\nSaved: results/lstm_metrics.json")
    print("Saved: results/lstm_model.keras")


if __name__ == "__main__":
    main()