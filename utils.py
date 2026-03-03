import os
import re
import zipfile
import urllib.request
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


UCI_SMS_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "smsspamcollection.zip")
EXTRACTED_PATH = os.path.join(DATA_DIR, "SMSSpamCollection")


def ensure_dataset() -> str:
    """
    Ensures the SMS Spam Collection dataset exists locally.
    Returns the path to the extracted dataset file.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(EXTRACTED_PATH):
        return EXTRACTED_PATH

    if not os.path.exists(ZIP_PATH):
        print(f"Downloading dataset from UCI to: {ZIP_PATH}")
        urllib.request.urlretrieve(UCI_SMS_ZIP_URL, ZIP_PATH)

    print(f"Extracting dataset to: {DATA_DIR}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    if not os.path.exists(EXTRACTED_PATH):
        raise FileNotFoundError("Dataset extraction failed: SMSSpamCollection not found.")

    return EXTRACTED_PATH


def clean_text(text: str) -> str:
    """
    Light cleaning for text messages.
    Keep it simple for a class project.
    """
    text = text.lower().strip()
    # replace URLs with token
    text = re.sub(r"http\S+|www\.\S+", " <url> ", text)
    # replace numbers with token
    text = re.sub(r"\d+", " <num> ", text)
    # keep letters, basic punctuation, and tokens
    text = re.sub(r"[^a-z<>\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_sms_spam_df() -> pd.DataFrame:
    """
    Loads the SMS Spam Collection into a DataFrame with columns:
    - label (0/1)
    - text (cleaned)
    """
    path = ensure_dataset()

    # File is tab-separated: label \t message
    df = pd.read_csv(path, sep="\t", header=None, names=["raw_label", "raw_text"], encoding="utf-8")

    df["label"] = df["raw_label"].map({"ham": 0, "spam": 1}).astype(int)
    df["text"] = df["raw_text"].astype(str).apply(clean_text)

    df = df[["label", "text"]].dropna()
    return df


def train_test_split_text(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[list, list, list, list]:
    """
    Stratified split for stable evaluation.
    Returns X_train, X_test, y_train, y_test as Python lists.
    """
    X = df["text"].tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test