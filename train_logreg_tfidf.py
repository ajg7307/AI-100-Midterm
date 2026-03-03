import os
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)

from utils import load_sms_spam_df, train_test_split_text


def main():
    os.makedirs("results", exist_ok=True)

    df = load_sms_spam_df()
    X_train, X_test, y_train, y_test = train_test_split_text(df)

    # TF-IDF features
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Logistic Regression baseline
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",  # helps with spam imbalance
        n_jobs=None
    )
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    metrics = {
        "model": "LogisticRegression + TFIDF",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    print("\n=== Logistic Regression + TF-IDF Results ===")
    print(json.dumps(metrics, indent=2))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    with open("results/logreg_tfidf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved: results/logreg_tfidf_metrics.json")


if __name__ == "__main__":
    main()