import csv
import os
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

_base = Path(__file__).resolve().parent.parent

MODEL_PATH         = _base / "models" / "classifier.joblib"
BASE_DATA_PATH     = _base / "data" / "transactions_v5_clean.csv"
FEEDBACK_DATA_PATH = Path(os.getenv("FEEDBACK_DIR", str(_base / "data"))) / "feedback.csv"

VALID_CATEGORIES = {
    "makanan", "minuman", "transport", "belanja",
    "hiburan", "tagihan", "kesehatan", "gaji", "hadiah",
}

VALID_SECONDARY_CATEGORIES = {
    "makanan":   {"makanan", "jajanan"},
    "minuman":   {"minuman", "jajanan"},
    "transport": {"transport", "online", "umum", "pesawat", "pribadi"},
    "belanja":   {"belanja", "fashion", "elektronik", "kecantikan", "online_shop", "harian"},
    "hiburan":   {"hiburan", "streaming", "game", "liburan", "tontonan", "aktivitas"},
    "tagihan":   {"tagihan", "utilitas", "internet_pulsa", "asuransi", "kredit", "sewa", "gaji_pihak3", "iuran"},
    "kesehatan": {"kesehatan", "apotek", "prosedur", "obat", "konsul"},
    "gaji":      {"gaji", "bonus", "komisi"},
    "hadiah":    {"hadiah", "undian", "reward", "kado", "sumbangan", "pemberian_masuk"},
}

ALL_SECONDARY = {s for subs in VALID_SECONDARY_CATEGORIES.values() for s in subs}
INCOME_CATEGORIES = {"gaji", "hadiah"}


def type_from_category(category: str) -> str:
    return "pemasukan" if category in INCOME_CATEGORIES else "pengeluaran"


STOPWORDS_ID = {
    "yang", "dan", "di", "ini", "itu", "dengan",
    "pada", "dalam", "adalah", "atau", "juga", "sudah", "akan", "bisa",
    "ada", "tidak", "saya", "kamu", "kami", "mereka", "anda",
    # "ke" dan "dari" sengaja TIDAK di-stopword: sinyal arah kritis
}


def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_ID and len(t) > 1]
    return " ".join(tokens)


def _load_csv(path: Path, label_field: str = "category") -> tuple[list[str], list[str]]:
    titles, labels = [], []
    if not path.exists():
        return titles, labels
    valid_set = VALID_CATEGORIES if label_field == "category" else ALL_SECONDARY
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("title", "").strip()
            label = row.get(label_field, "").strip()
            if title and label in valid_set:
                titles.append(title)
                labels.append(label)
    return titles, labels


def _load_csv_with_secondary(path: Path) -> tuple[list[str], list[str], list[str]]:
    titles, primary, secondary = [], [], []
    if not path.exists():
        return titles, primary, secondary
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("title", "").strip()
            pri   = row.get("category", "").strip()
            sec   = row.get("secondary_category", "").strip()
            if title and pri in VALID_CATEGORIES and sec:
                titles.append(title)
                primary.append(pri)
                secondary.append(sec)
    return titles, primary, secondary


# ---------------------------------------------------------------------------
# Money direction features
# ---------------------------------------------------------------------------

IN_CUES = [
    "terima", "menerima", "nerima", "dapat", "dapet", "masuk", "cair", "dikasih",
    "dikirimi", "ditransfer", "dipinjamin", "ditraktir", "dibeliin", "dibawain",
    "disangui", "gajian", "bonus", "thr", "cashback", "refund", "reward", "hadiah",
    "warisan", "hibah", "dari",
]

OUT_CUES = [
    "bayar", "byr", "beli", "beliin", "bayarin", "kasih", "ngasih", "kirim",
    "transfer ke", "sumbang", "sedekah", "traktir", "top up", "langganan",
    "cicilan", "isi", "order", "pesan", "jajan", "ke ",
]


class MoneyDirectionFeatures(BaseEstimator, TransformerMixin):
    """8-column sparse feature matrix encoding income/expense direction cues.

    Columns: [n_in, n_out, starts_in, starts_out, has_ke, has_dari, has_sama, word_count]
    """

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _cue_in(cue: str, low: str) -> bool:
        """Check if cue appears as a whole-word match inside low (which is space-padded)."""
        # If cue already has a leading or trailing space, match as-is (e.g. "ke ")
        if cue.startswith(" ") or cue.endswith(" "):
            return cue in low
        # For plain tokens (no embedded spaces), require word boundaries via spaces
        if " " not in cue:
            return (" " + cue + " ") in low
        # Multi-word cues without leading/trailing space: match as substring
        return cue in low

    def transform(self, X):
        feats = []
        for s in X:
            low        = " " + s.lower() + " "
            n_in       = sum(1 for c in IN_CUES  if self._cue_in(c, low))
            n_out      = sum(1 for c in OUT_CUES if self._cue_in(c, low))
            starts_in  = int(any(low.strip().startswith(c) for c in IN_CUES))
            starts_out = int(any(low.strip().startswith(c) for c in OUT_CUES))
            has_ke     = 1 if re.search(r"\bke\b",   low) else 0
            has_dari   = 1 if re.search(r"\bdari\b", low) else 0
            has_sama   = 1 if re.search(r"\bsama\b", low) else 0
            feats.append([n_in, n_out, starts_in, starts_out,
                          has_ke, has_dari, has_sama, len(s.split())])
        return csr_matrix(np.array(feats, dtype=float))


# ---------------------------------------------------------------------------
# Unified model
# ---------------------------------------------------------------------------

class TransactionModel:
    """Unified classifier: shared features → two LR classifiers (category + secondary).

    type is derived rule-based via type_from_category(), not a trained classifier.
    """

    def __init__(self):
        self.text_feats = FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
            )),
        ])
        self.money_feats = MoneyDirectionFeatures()
        self.clf_category  = LogisticRegression(C=4.0, max_iter=1000, solver="lbfgs", class_weight="balanced")
        self.clf_secondary = LogisticRegression(C=4.0, max_iter=1000, solver="lbfgs", class_weight="balanced")
        self.classes_category:  list[str] = []
        self.classes_secondary: list[str] = []

    def _featurize(self, raw_titles: list[str], fit: bool = False):
        processed = [preprocess(t) for t in raw_titles]
        Xt = self.text_feats.fit_transform(processed) if fit else self.text_feats.transform(processed)
        Xm = self.money_feats.transform(processed)
        return hstack([Xt, Xm]).tocsr()

    def train(
        self,
        titles: list[str],
        primary_labels: list[str],
        secondary_labels: list[str],
    ) -> dict:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pri_scores: list[float] = []
        sec_scores: list[float] = []

        for tr_idx, val_idx in cv.split(titles, primary_labels):
            X_tr      = [titles[i]           for i in tr_idx]
            X_val     = [titles[i]           for i in val_idx]
            y_pri_tr  = [primary_labels[i]   for i in tr_idx]
            y_pri_val = [primary_labels[i]   for i in val_idx]
            y_sec_tr  = [secondary_labels[i] for i in tr_idx]
            y_sec_val = [secondary_labels[i] for i in val_idx]

            tmp = TransactionModel()
            Xf_tr  = tmp._featurize(X_tr,  fit=True)
            tmp.clf_category.fit(Xf_tr,  y_pri_tr)
            tmp.clf_secondary.fit(Xf_tr, y_sec_tr)
            Xf_val = tmp._featurize(X_val, fit=False)
            pri_scores.append(accuracy_score(y_pri_val, tmp.clf_category.predict(Xf_val)))
            sec_scores.append(accuracy_score(y_sec_val, tmp.clf_secondary.predict(Xf_val)))

        Xf = self._featurize(titles, fit=True)
        self.clf_category.fit(Xf,  primary_labels)
        self.clf_secondary.fit(Xf, secondary_labels)
        self.classes_category  = list(self.clf_category.classes_)
        self.classes_secondary = list(self.clf_secondary.classes_)

        return {
            "cv_primary_mean":   float(np.mean(pri_scores)),
            "cv_primary_std":    float(np.std(pri_scores)),
            "cv_secondary_mean": float(np.mean(sec_scores)),
            "cv_secondary_std":  float(np.std(sec_scores)),
            "classes_category":  self.classes_category,
            "classes_secondary": self.classes_secondary,
            "samples":           len(titles),
        }

    def retrain_with_feedback(self) -> dict:
        base_t, base_pri, base_sec = _load_csv_with_secondary(BASE_DATA_PATH)
        fb_t,   fb_pri,   fb_sec   = _load_csv_with_secondary(FEEDBACK_DATA_PATH)

        titles    = base_t   + fb_t
        primary   = base_pri + fb_pri
        secondary = base_sec + fb_sec

        Xf = self._featurize(titles, fit=True)
        self.clf_category.fit(Xf,  primary)
        self.clf_secondary.fit(Xf, secondary)
        self.classes_category  = list(self.clf_category.classes_)
        self.classes_secondary = list(self.clf_secondary.classes_)
        self.save()

        return {
            "base_samples":     len(base_t),
            "feedback_samples": len(fb_t),
            "total_samples":    len(titles),
        }

    def predict(self, title: str) -> dict:
        if not self.classes_category:
            raise RuntimeError("Model belum dilatih. Jalankan train.py terlebih dahulu.")

        Xf = self._featurize([title], fit=False)

        proba_cat = self.clf_category.predict_proba(Xf)[0]
        proba_sec = self.clf_secondary.predict_proba(Xf)[0]

        top_cat = int(np.argmax(proba_cat))
        top_sec = int(np.argmax(proba_sec))

        category           = self.classes_category[top_cat]
        secondary_category = self.classes_secondary[top_sec]

        alt_cat = [
            {"category": self.classes_category[i], "confidence": round(float(proba_cat[i]), 4)}
            for i in np.argsort(proba_cat)[::-1] if i != top_cat
        ][:3]
        alt_sec = [
            {"category": self.classes_secondary[i], "confidence": round(float(proba_sec[i]), 4)}
            for i in np.argsort(proba_sec)[::-1] if i != top_sec
        ][:3]

        return {
            "category":               category,
            "secondary_category":     secondary_category,
            "transaction_type":       type_from_category(category),
            "confidence":             round(float(proba_cat[top_cat]), 4),
            "secondary_confidence":   round(float(proba_sec[top_sec]), 4),
            "alternatives":           alt_cat,
            "secondary_alternatives": alt_sec,
        }

    def save(self, path: Optional[Path] = None):
        path = path or MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    def load(self, path: Optional[Path] = None):
        path = path or MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Model tidak ditemukan di {path}. Jalankan train.py.")
        loaded: "TransactionModel" = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
