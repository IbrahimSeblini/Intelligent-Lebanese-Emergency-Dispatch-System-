
"""
Emergency message classifier starter script.

What this script does:
1. Loads a CSV with columns: text,label
2. Splits the data into train/test
3. Trains a TF-IDF + Logistic Regression classifier
4. Prints evaluation metrics
5. Saves the trained model
6. Exposes a classify_emergency() helper with a confidence threshold
"""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required_cols}, found {set(df.columns)}")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["text"] != ""]
    return df


def train_and_evaluate(csv_path: str, model_out: str) -> None:
    df = load_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n=== DATASET SIZE ===")
    print(f"Total rows: {len(df)}")
    print("\n=== LABEL DISTRIBUTION ===")
    print(df["label"].value_counts())

    print("\n=== TEST ACCURACY ===")
    print(f"{acc:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("\n=== CONFUSION MATRIX ===")
    labels = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    print(cm_df)

    joblib.dump(model, model_out)
    print(f"\nSaved model to: {model_out}")

    # quick demo
    examples = [
        "someone is trying to break into my house",
        "my father collapsed and is not breathing",
        "there is smoke and fire in the apartment downstairs",
        "help please come fast",
    ]
    print("\n=== SAMPLE PREDICTIONS ===")
    for text in examples:
        result = classify_emergency(model, text)
        print(f"\nInput: {text}")
        print(result)


def classify_emergency(model: Pipeline, text: str, threshold: float = 0.65) -> dict:
    probs = model.predict_proba([text])[0]
    classes = model.classes_
    best_idx = probs.argmax()
    predicted_label = classes[best_idx]
    confidence = float(probs[best_idx])

    if confidence < threshold:
        predicted_label = "uncertain"

    return {
        "text": text,
        "category": predicted_label,
        "confidence": round(confidence, 4),
        "all_scores": {label: round(float(score), 4) for label, score in zip(classes, probs)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="emergency_dataset_starter.csv", help="Path to labeled CSV dataset")
    parser.add_argument("--model_out", default="emergency_classifier.pkl", help="Where to save the trained model")
    args = parser.parse_args()

    train_and_evaluate(args.csv, args.model_out)


if __name__ == "__main__":
    main()
