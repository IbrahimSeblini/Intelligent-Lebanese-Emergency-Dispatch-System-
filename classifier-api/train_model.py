"""Train TF–IDF + LogisticRegression and write emergency_classifier.pkl."""

from __future__ import annotations

import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Bilingual-ish toy corpus — replace with real data for production accuracy
TRAINING: list[tuple[str, str]] = [
    # Fire
    ("fire in the kitchen heavy smoke", "firefighters"),
    ("building on fire need help", "firefighters"),
    ("gas leak smell gas apartment", "firefighters"),
    ("explosion heard in street", "firefighters"),
    ("حريق كبير في الطابق الثاني", "firefighters"),
    ("دخان كثير النار اشتعلت", "firefighters"),
    ("في نار بالبيت لحقونا", "firefighters"),
    # Police
    ("someone broke into my house", "police"),
    ("robbery at the store weapon", "police"),
    ("fight in the street knives", "police"),
    ("car stolen license plate", "police"),
    ("سرقة محل بالسلاح", "police"),
    ("في حدا عم يضربني بالسكين", "police"),
    ("بدي الشرطة في سرقة", "police"),
    # Medical
    ("my father unconscious not breathing", "red_cross"),
    ("chest pain ambulance please", "red_cross"),
    ("child injured bleeding lot", "red_cross"),
    ("severe allergic reaction swelling", "red_cross"),
    ("جدي توفى فجأة بدي اسعاف", "red_cross"),
    ("في حدا مغمى عليه", "red_cross"),
    ("وجع بالصدر بدي مستشفى", "red_cross"),
    # Mixed / harder
    ("car crash people hurt fire", "firefighters"),
    ("shooting victim bleeding", "police"),
    ("smoke inhalation cannot breathe", "red_cross"),
]

LABELS = ["police", "firefighters", "red_cross"]


def main() -> None:
    texts, y = zip(*TRAINING)
    pipe: Pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=8192,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(list(texts), list(y))

    out = os.path.join(os.path.dirname(__file__), "emergency_classifier.pkl")
    joblib.dump(pipe, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
