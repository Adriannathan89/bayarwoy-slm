import csv
import os
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

_base = Path(__file__).resolve().parent.parent

MODEL_PATH         = _base / "models" / "classifier.pkl"
BASE_DATA_PATH     = _base / "data" / "transactions_v3.csv"
FEEDBACK_DATA_PATH = Path(os.getenv("FEEDBACK_DIR", str(_base / "data"))) / "feedback.csv"

VALID_CATEGORIES = {
    "makanan", "minuman", "transport", "belanja",
    "hiburan", "tagihan", "kesehatan", "gaji", "hadiah",
}

INCOME_CATEGORIES = {"gaji", "hadiah"}


def type_from_category(category: str) -> str:
    return "pemasukan" if category in INCOME_CATEGORIES else "pengeluaran"

STOPWORDS_ID = {
    "yang", "dan", "di", "ini", "itu", "dengan",
    "pada", "dalam", "adalah", "atau", "juga", "sudah", "akan", "bisa",
    "ada", "tidak", "saya", "kamu", "kami", "mereka", "anda",
    # "ke" dan "dari" sengaja TIDAK di-stopword: keduanya sinyal arah kritis
    # (transfer ke = pengeluaran, transfer dari = pemasukan)
}


def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_ID and len(t) > 1]
    return " ".join(tokens)


def _load_csv(path: Path) -> tuple[list[str], list[str]]:
    titles, labels = [], []
    if not path.exists():
        return titles, labels
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("title", "").strip()
            label = row.get("category", "").strip()
            if title and label in VALID_CATEGORIES:
                titles.append(title)
                labels.append(label)
    return titles, labels


def _build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=6000,
            sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            max_features=6000,
            sublinear_tf=True,
        )),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs")),
    ])


class TransactionClassifier:
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.classes: list[str] = []

    def _fit(self, titles: list[str], labels: list[str]):
        processed = [preprocess(t) for t in titles]
        self.pipeline = _build_pipeline()
        self.pipeline.fit(processed, labels)
        self.classes = list(self.pipeline.classes_)

    def train(self, titles: list[str], labels: list[str]) -> dict:
        from sklearn.model_selection import cross_val_score

        processed = [preprocess(t) for t in titles]
        pipeline = _build_pipeline()
        scores = cross_val_score(pipeline, processed, labels, cv=5, scoring="accuracy")

        self._fit(titles, labels)
        return {
            "cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_std": float(scores.std()),
            "classes": self.classes,
            "samples": len(titles),
        }

    def retrain_with_feedback(self) -> dict:
        base_titles, base_labels = _load_csv(BASE_DATA_PATH)
        fb_titles, fb_labels = _load_csv(FEEDBACK_DATA_PATH)
        titles = base_titles + fb_titles
        labels = base_labels + fb_labels
        self._fit(titles, labels)
        self.save()
        return {
            "base_samples": len(base_titles),
            "feedback_samples": len(fb_titles),
            "total_samples": len(titles),
            "classes": self.classes,
        }

    def predict(self, title: str) -> dict:
        if self.pipeline is None:
            raise RuntimeError("Model belum dilatih. Jalankan train.py terlebih dahulu.")

        processed = preprocess(title)
        proba = self.pipeline.predict_proba([processed])[0]
        top_idx = int(np.argmax(proba))

        alternatives = [
            {"category": self.classes[i], "confidence": round(float(proba[i]), 4)}
            for i in np.argsort(proba)[::-1]
            if i != top_idx
        ][:3]

        return {
            "category": self.classes[top_idx],
            "confidence": round(float(proba[top_idx]), 4),
            "alternatives": alternatives,
        }

    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "classes": self.classes}, f)

    def load(self, path: Path = MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Model tidak ditemukan di {path}. Jalankan train.py.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.pipeline = data["pipeline"]
        self.classes = data["classes"]