# TransactionModel Migration (v5 Dataset) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ganti dua `TransactionClassifier` terpisah dengan satu `TransactionModel` unified yang pakai shared features (TF-IDF + MoneyDirectionFeatures) dan v5 dataset (14.945 baris).

**Architecture:** Satu `TransactionModel` memegang `FeatureUnion(word + char TF-IDF)` + `MoneyDirectionFeatures`, dua LR classifier (category + secondary), serialisasi joblib. `main.py` load satu model, panggil satu `.predict()`. `type` tetap rule-based.

**Tech Stack:** Python 3.11+, scikit-learn, FastAPI, joblib, scipy.sparse

---

## File Map

| File | Aksi | Tanggung jawab |
|------|------|----------------|
| `requirements.txt` | Modify | Tambah `joblib`, `scipy`, `pytest` |
| `.gitignore` | Modify | Tambah `models/*.joblib` |
| `tests/test_classifier.py` | Create | Unit tests untuk `preprocess` + `MoneyDirectionFeatures` |
| `src/classifier.py` | Rewrite | `MoneyDirectionFeatures`, `TransactionModel` — hapus `TransactionClassifier`, `_build_pipeline` |
| `src/train.py` | Rewrite | Train `TransactionModel` dari v5 data |
| `src/main.py` | Rewrite | Single `clf: TransactionModel`, simplify retrain schema |

---

## Task 1: Update Dependencies dan Gitignore

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Tambah joblib, scipy, pytest ke requirements.txt**

Ganti isi `requirements.txt` dengan:

```
fastapi==0.115.5
uvicorn[standard]==0.32.1
pydantic==2.10.3
scikit-learn==1.6.0
numpy==2.2.0
joblib>=1.3.0
scipy>=1.12.0
pytest>=8.0.0
```

- [ ] **Step 2: Tambah models/*.joblib ke .gitignore**

Di `.gitignore`, update baris `models/*.pkl` menjadi:

```
models/*.pkl
models/*.joblib
```

- [ ] **Step 3: Install dependencies baru**

Cari venv yang tersedia, lalu install. Coba urutan ini:

```bash
# Jika ada .venv (dari setup.sh):
.venv/bin/pip install -r requirements.txt

# Jika ada /tmp/bw-nlp-venv (dari Makefile):
/tmp/bw-nlp-venv/bin/pip install -r requirements.txt

# Fallback — buat venv baru dulu:
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Expected: `Successfully installed joblib-... scipy-...`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "chore: add joblib, scipy, pytest to requirements; gitignore joblib model"
```

---

## Task 2: Tulis Tests untuk MoneyDirectionFeatures dan preprocess

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_classifier.py`

- [ ] **Step 1: Buat direktori tests dan file kosong**

```bash
mkdir -p tests && touch tests/__init__.py
```

- [ ] **Step 2: Tulis failing tests di `tests/test_classifier.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


class TestPreprocess:
    def test_lowercase_and_strip(self):
        from classifier import preprocess
        assert preprocess("  Makan SIANG  ") == "makan siang"

    def test_removes_stopwords(self):
        from classifier import preprocess
        result = preprocess("makan yang enak")
        tokens = result.split()
        assert "yang" not in tokens
        assert "makan" in tokens
        assert "enak" in tokens

    def test_keeps_ke_dan_dari(self):
        from classifier import preprocess
        result = preprocess("transfer ke rekening dari ATM")
        tokens = result.split()
        assert "ke" in tokens
        assert "dari" in tokens

    def test_removes_punctuation(self):
        from classifier import preprocess
        assert preprocess("makan! siang.") == "makan siang"

    def test_removes_short_tokens(self):
        from classifier import preprocess
        result = preprocess("makan a b")
        tokens = result.split()
        assert "a" not in tokens
        assert "b" not in tokens
        assert "makan" in tokens


class TestMoneyDirectionFeatures:
    def setup_method(self):
        from classifier import MoneyDirectionFeatures
        self.mdf = MoneyDirectionFeatures()

    def test_output_shape(self):
        out = self.mdf.transform(["bayar listrik", "dapat gaji"])
        assert out.shape == (2, 8)

    def test_income_cue_n_in(self):
        out = self.mdf.transform(["dapat gaji bulan ini"]).toarray()
        assert out[0, 0] > 0, "kolom n_in harus > 0 untuk cue 'dapat'"

    def test_expense_cue_n_out(self):
        out = self.mdf.transform(["bayar listrik PLN"]).toarray()
        assert out[0, 1] > 0, "kolom n_out harus > 0 untuk cue 'bayar'"

    def test_has_ke(self):
        out = self.mdf.transform(["transfer ke kakak"]).toarray()
        assert out[0, 4] == 1, "has_ke harus 1"

    def test_has_dari(self):
        out = self.mdf.transform(["dikasih uang dari ayah"]).toarray()
        assert out[0, 5] == 1, "has_dari harus 1"

    def test_has_sama(self):
        out = self.mdf.transform(["makan sama teman"]).toarray()
        assert out[0, 6] == 1, "has_sama harus 1"

    def test_ambiguous_text_no_cues(self):
        out = self.mdf.transform(["komisi tomoro"]).toarray()
        assert out[0, 0] == 0 and out[0, 1] == 0, \
            "teks ambigu tanpa kata kerja arah: n_in dan n_out harus 0"

    def test_starts_in_flag(self):
        out = self.mdf.transform(["dapat bonus dari kantor"]).toarray()
        assert out[0, 2] == 1, "starts_in harus 1 jika dimulai dengan cue masuk"

    def test_starts_out_flag(self):
        out = self.mdf.transform(["bayar tagihan internet"]).toarray()
        assert out[0, 3] == 1, "starts_out harus 1 jika dimulai dengan cue keluar"
```

- [ ] **Step 3: Jalankan tests — harus FAIL dulu**

```bash
# Gunakan venv yang tersedia (pilih salah satu):
.venv/bin/pytest tests/test_classifier.py -v
# atau:
/tmp/bw-nlp-venv/bin/pytest tests/test_classifier.py -v
```

Expected: `ImportError: cannot import name 'MoneyDirectionFeatures' from 'classifier'`

- [ ] **Step 4: Commit test file**

```bash
git add tests/__init__.py tests/test_classifier.py
git commit -m "test: add unit tests for preprocess and MoneyDirectionFeatures"
```

---

## Task 3: Rewrite src/classifier.py

**Files:**
- Rewrite: `src/classifier.py`

- [ ] **Step 1: Ganti seluruh isi src/classifier.py**

```python
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
    "hadiah":    {"hadiah", "undian", "reward", "kado", "sumbangan"},
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

    def transform(self, X):
        feats = []
        for s in X:
            low        = " " + s.lower() + " "
            n_in       = sum(1 for c in IN_CUES  if c in low)
            n_out      = sum(1 for c in OUT_CUES if c in low)
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
            X_tr      = [titles[i]          for i in tr_idx]
            X_val     = [titles[i]          for i in val_idx]
            y_pri_tr  = [primary_labels[i]  for i in tr_idx]
            y_pri_val = [primary_labels[i]  for i in val_idx]
            y_sec_tr  = [secondary_labels[i] for i in tr_idx]
            y_sec_val = [secondary_labels[i] for i in val_idx]

            tmp = TransactionModel()
            Xf_tr  = tmp._featurize(X_tr,  fit=True)
            tmp.clf_category.fit(Xf_tr,  y_pri_tr)
            tmp.clf_secondary.fit(Xf_tr, y_sec_tr)
            Xf_val = tmp._featurize(X_val, fit=False)
            pri_scores.append(accuracy_score(y_pri_val, tmp.clf_category.predict(Xf_val)))
            sec_scores.append(accuracy_score(y_sec_val, tmp.clf_secondary.predict(Xf_val)))

        # Fit on full dataset
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
```

- [ ] **Step 2: Jalankan tests — harus PASS**

```bash
.venv/bin/pytest tests/test_classifier.py -v
```

Expected output (semua hijau):
```
tests/test_classifier.py::TestPreprocess::test_lowercase_and_strip PASSED
tests/test_classifier.py::TestPreprocess::test_removes_stopwords PASSED
tests/test_classifier.py::TestPreprocess::test_keeps_ke_dan_dari PASSED
tests/test_classifier.py::TestPreprocess::test_removes_punctuation PASSED
tests/test_classifier.py::TestPreprocess::test_removes_short_tokens PASSED
tests/test_classifier.py::TestMoneyDirectionFeatures::test_output_shape PASSED
tests/test_classifier.py::TestMoneyDirectionFeatures::test_income_cue_n_in PASSED
tests/test_classifier.py::TestMoneyDirectionFeatures::test_expense_cue_n_out PASSED
tests/test_classifier.py::TestMoneyDirectionFeatures::test_has_ke PASSED
tests/test_classifier.py::TestMoneyDirectionFeatures::test_has_dari PASSED
tests/test_classifier.py::TestMoneyDirectionFeatures::test_has_sama PASSED
tests/test_classifier.py::TestMoneyDirectionFeatures::test_ambiguous_text_no_cues PASSED
tests/test_classifier.py::TestMoneyDirectionFeatures::test_starts_in_flag PASSED
tests/test_classifier.py::TestMoneyDirectionFeatures::test_starts_out_flag PASSED

14 passed
```

- [ ] **Step 3: Commit**

```bash
git add src/classifier.py
git commit -m "feat: migrate to TransactionModel with MoneyDirectionFeatures and v5 dataset"
```

---

## Task 4: Update src/train.py

**Files:**
- Rewrite: `src/train.py`

- [ ] **Step 1: Ganti seluruh isi src/train.py**

```python
"""
Jalankan: python src/train.py   atau   make train

Train TransactionModel unified:
  Primary   → category (9 kelas)
  Secondary → secondary_category (~40 kelas)
  Type      → rule-based via type_from_category()

Dataset: transactions_v5_clean.csv (14.945 baris)
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from classifier import (
    BASE_DATA_PATH,
    MODEL_PATH,
    TransactionModel,
    _load_csv_with_secondary,
    type_from_category,
)


def main():
    print(f"Memuat dataset dari {BASE_DATA_PATH}...")
    titles, primary_labels, secondary_labels = _load_csv_with_secondary(BASE_DATA_PATH)
    print(f"Total sampel: {len(titles)}")

    pri_dist = Counter(primary_labels)
    sec_dist = Counter(secondary_labels)
    print(f"\nDistribusi PRIMARY ({len(pri_dist)} kelas):")
    for c, n in sorted(pri_dist.items()):
        print(f"  {c:18s}: {n}")
    print(f"\nDistribusi SECONDARY ({len(sec_dist)} kelas):")
    for c, n in sorted(sec_dist.items()):
        print(f"  {c:20s}: {n}")

    print(f"\n{'='*60}")
    print("Training TransactionModel (5-fold CV)...")
    print("=" * 60)

    model = TransactionModel()
    metrics = model.train(titles, primary_labels, secondary_labels)

    print(f"  CV Primary   : {metrics['cv_primary_mean']:.2%} ± {metrics['cv_primary_std']:.2%}")
    print(f"  CV Secondary : {metrics['cv_secondary_mean']:.2%} ± {metrics['cv_secondary_std']:.2%}")
    print(f"  Kelas primary   ({len(metrics['classes_category'])}): {metrics['classes_category']}")
    print(f"  Kelas secondary ({len(metrics['classes_secondary'])}): {metrics['classes_secondary']}")

    model.save(MODEL_PATH)
    print(f"\n  Model disimpan → {MODEL_PATH}")

    print(f"\n{'='*60}")
    print("Contoh prediksi kontras (cek arah uang):")
    print("=" * 60)

    contrast_tests = [
        ("ngasih sumbangan ke kakak",   "pengeluaran"),
        ("dikasih makan sama kakak",    "pemasukan"),
        ("dapat komisi dari tomoro",    "pemasukan"),
        ("bayar komisi ke tomoro",      "pengeluaran"),
        ("makan siang di warteg",       "pengeluaran"),
        ("grab ke kantor",              "pengeluaran"),
        ("lion air ke jakarta",         "pengeluaran"),
        ("isi bensin",                  "pengeluaran"),
        ("belanja di shopee",           "pengeluaran"),
        ("beli kuota telkomsel",        "pengeluaran"),
        ("bayar listrik",               "pengeluaran"),
        ("nonton netflix",              "pengeluaran"),
        ("beli paracetamol di apotek",  "pengeluaran"),
        ("gaji bulanan januari",        "pemasukan"),
        ("bonus akhir tahun",           "pemasukan"),
        ("hadiah dari orang tua",       "pemasukan"),
    ]

    correct = 0
    print()
    for text, expected_type in contrast_tests:
        r = model.predict(text)
        ok = r["transaction_type"] == expected_type
        correct += int(ok)
        mark = "✓" if ok else "✗"
        print(f"  {mark} '{text}'")
        print(f"     → {r['category']:12s} / {r['secondary_category']:16s} "
              f"/ {r['transaction_type']} ({r['confidence']:.0%})")

    print(f"\n{'='*60}")
    print(f"Kontras test: {correct}/{len(contrast_tests)} benar")
    print(f"CV Primary   : {metrics['cv_primary_mean']:.2%} ± {metrics['cv_primary_std']:.2%}")
    print(f"CV Secondary : {metrics['cv_secondary_mean']:.2%} ± {metrics['cv_secondary_std']:.2%}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/train.py
git commit -m "feat: update train.py to use TransactionModel with v5 data"
```

---

## Task 5: Train Model dan Verifikasi

**Files:** (tidak ada file baru — hanya menjalankan training)

- [ ] **Step 1: Jalankan training**

```bash
make train
# atau jika venv berbeda:
.venv/bin/python src/train.py
```

Training 5-fold CV dengan 14.945 baris membutuhkan ~30-60 detik.

Expected output (angka CV bisa sedikit berbeda):
```
Memuat dataset dari .../data/transactions_v5_clean.csv...
Total sampel: 14944

Distribusi PRIMARY (9 kelas):
  belanja           : ...
  gaji              : ...
  ...

Training TransactionModel (5-fold CV)...
  CV Primary   : 97.xx% ± 0.xx%
  CV Secondary : 95.xx% ± 0.xx%
  ...
  Model disimpan → .../models/classifier.joblib

Contoh prediksi kontras (cek arah uang):
  ✓ 'ngasih sumbangan ke kakak'
     → hadiah       / sumbangan         / pengeluaran (xx%)
  ✓ 'dikasih makan sama kakak'
     → makanan      / ...
  ...
Kontras test: xx/16 benar
```

Jika CV accuracy < 90% atau contoh kontras < 12/16 benar, ada masalah. Periksa distribusi data.

- [ ] **Step 2: Verifikasi file model tersimpan**

```bash
ls -lh models/classifier.joblib
```

Expected: file ada, ukuran ~5-20 MB.

---

## Task 6: Update src/main.py

**Files:**
- Rewrite: `src/main.py`

- [ ] **Step 1: Ganti seluruh isi src/main.py**

```python
import csv
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from classifier import (
    ALL_SECONDARY,
    FEEDBACK_DATA_PATH,
    MODEL_PATH,
    TransactionModel,
    VALID_CATEGORIES,
    VALID_SECONDARY_CATEGORIES,
    type_from_category,
)

# --- Config ---
AUTO_REINFORCE_THRESHOLD = 0.85
AUTO_RETRAIN_EVERY       = 100
FEEDBACK_HEADER = ["title", "category", "secondary_category", "source", "timestamp"]

# --- State ---
clf = TransactionModel()
_feedback_lock = threading.Lock()
_pending_count = 0
_is_retraining = False


# --- Feedback persistence ---

def _ensure_feedback_header():
    if not FEEDBACK_DATA_PATH.exists():
        return
    with open(FEEDBACK_DATA_PATH, encoding="utf-8") as f:
        first_line = f.readline().strip()
    if first_line == ",".join(FEEDBACK_HEADER):
        return
    backup = FEEDBACK_DATA_PATH.with_suffix(".csv.bak")
    FEEDBACK_DATA_PATH.rename(backup)
    rows = []
    with open(backup, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "title":              row.get("title", ""),
                "category":           row.get("category", ""),
                "secondary_category": row.get("secondary_category", ""),
                "source":             row.get("source", "unknown"),
                "timestamp":          row.get("timestamp", ""),
            })
    with open(FEEDBACK_DATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _append_feedback(title: str, category: str, secondary: str, source: str):
    FEEDBACK_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FEEDBACK_DATA_PATH.exists()
    with open(FEEDBACK_DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(FEEDBACK_HEADER)
        writer.writerow([title, category, secondary, source, datetime.now().isoformat()])


def _background_retrain():
    global _pending_count, _is_retraining
    _is_retraining = True
    try:
        result = clf.retrain_with_feedback()
        print(
            f"[retrain] base={result['base_samples']}, "
            f"feedback={result['feedback_samples']}, total={result['total_samples']}"
        )
        _pending_count = 0
    except Exception as e:
        print(f"[retrain] gagal: {e}")
    finally:
        _is_retraining = False


def maybe_save_and_retrain(
    title: str,
    category: str,
    secondary: str,
    confidence: float,
) -> bool:
    global _pending_count
    if confidence < AUTO_REINFORCE_THRESHOLD:
        return False

    with _feedback_lock:
        _append_feedback(title, category, secondary, "auto")
        _pending_count += 1
        should_retrain = _pending_count >= AUTO_RETRAIN_EVERY and not _is_retraining

    if should_retrain:
        print(f"[reinforce] {_pending_count} sampel baru → retrain di background...")
        threading.Thread(target=_background_retrain, daemon=True).start()
    return True


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    clf.load()
    _ensure_feedback_header()
    print(f"Category  : {len(clf.classes_category)} kelas → {clf.classes_category}")
    print(f"Secondary : {len(clf.classes_secondary)} kelas")
    yield


app = FastAPI(
    title="BayarWoy SLM Service",
    description="Klasifikasi judul transaksi → primary + secondary category (Bahasa Indonesia)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# --- Schemas ---

class ClassifyRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, example="makan siang di warteg")


class CategoryResult(BaseModel):
    category: str
    confidence: float


class ClassifyResponse(BaseModel):
    title: str
    category: str
    secondary_category: str
    transaction_type: str
    confidence: float
    secondary_confidence: float
    alternatives: list[CategoryResult]
    secondary_alternatives: list[CategoryResult]
    reinforced: bool


class BatchClassifyRequest(BaseModel):
    titles: list[str] = Field(..., min_length=1, max_length=50)


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]


class FeedbackRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    correct_category: str = Field(..., example="makanan")
    correct_secondary_category: str = Field(..., example="jajanan")


class FeedbackResponse(BaseModel):
    status: str
    message: str
    retrain_triggered: bool


class RetrainResponse(BaseModel):
    status: str
    base_samples: int
    feedback_samples: int
    total_samples: int


# --- Endpoints ---

@app.get("/health")
def health():
    fb_count = 0
    if FEEDBACK_DATA_PATH.exists():
        with open(FEEDBACK_DATA_PATH, encoding="utf-8") as f:
            fb_count = sum(1 for _ in f) - 1
    return {
        "status":                "ok",
        "model_loaded":          bool(clf.classes_category),
        "categories":            clf.classes_category,
        "secondary_categories":  clf.classes_secondary,
        "feedback_samples":      max(fb_count, 0),
        "pending_for_retrain":   _pending_count,
        "auto_retrain_threshold": AUTO_REINFORCE_THRESHOLD,
        "auto_retrain_every":    AUTO_RETRAIN_EVERY,
    }


def _classify_one(title: str) -> ClassifyResponse:
    title = title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Title tidak boleh kosong")
    try:
        result = clf.predict(title)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    reinforced = maybe_save_and_retrain(
        title,
        result["category"],
        result["secondary_category"],
        result["confidence"],
    )

    return ClassifyResponse(
        title=title,
        category=result["category"],
        secondary_category=result["secondary_category"],
        transaction_type=result["transaction_type"],
        confidence=result["confidence"],
        secondary_confidence=result["secondary_confidence"],
        alternatives=[CategoryResult(**a) for a in result["alternatives"]],
        secondary_alternatives=[CategoryResult(**a) for a in result["secondary_alternatives"]],
        reinforced=reinforced,
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    return _classify_one(req.title)


@app.post("/classify/batch", response_model=BatchClassifyResponse)
def classify_batch(req: BatchClassifyRequest):
    return BatchClassifyResponse(results=[
        _classify_one(t) for t in req.titles if t.strip()
    ])


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    if req.correct_category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Kategori tidak valid. Pilih dari: {sorted(VALID_CATEGORIES)}",
        )
    valid_sec = VALID_SECONDARY_CATEGORIES.get(req.correct_category, set())
    if req.correct_secondary_category not in valid_sec:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Secondary category tidak valid untuk '{req.correct_category}'. "
                f"Pilih dari: {sorted(valid_sec)}"
            ),
        )

    global _pending_count
    with _feedback_lock:
        _append_feedback(
            req.title.strip(),
            req.correct_category,
            req.correct_secondary_category,
            "manual",
        )
        _pending_count += 1
        should_retrain = _pending_count >= AUTO_RETRAIN_EVERY and not _is_retraining

    retrain_triggered = False
    if should_retrain:
        retrain_triggered = True
        threading.Thread(target=_background_retrain, daemon=True).start()

    return FeedbackResponse(
        status="ok",
        message=(
            f"Feedback disimpan: '{req.title}' → "
            f"{req.correct_category}/{req.correct_secondary_category}"
        ),
        retrain_triggered=retrain_triggered,
    )


@app.post("/retrain", response_model=RetrainResponse)
def retrain():
    global _is_retraining
    if _is_retraining:
        raise HTTPException(status_code=409, detail="Retrain sedang berjalan.")
    _is_retraining = True
    try:
        result = clf.retrain_with_feedback()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _is_retraining = False
    return RetrainResponse(
        status="ok",
        base_samples=result["base_samples"],
        feedback_samples=result["feedback_samples"],
        total_samples=result["total_samples"],
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/main.py
git commit -m "feat: update main.py to use unified TransactionModel"
```

---

## Task 7: Smoke Test Service

**Files:** (tidak ada perubahan — hanya verifikasi)

- [ ] **Step 1: Start service**

```bash
make run
# atau:
.venv/bin/uvicorn src.main:app --reload --host 0.0.0.0 --port 8001
```

Expected startup log:
```
Category  : 9 kelas → ['belanja', 'gaji', 'hadiah', 'hiburan', 'kesehatan', 'makanan', 'minuman', 'tagihan', 'transport']
Secondary : N kelas
INFO:     Application startup complete.
```

Jika ada `FileNotFoundError` → jalankan training dulu (Task 5).

- [ ] **Step 2: Test /health**

```bash
curl -s http://localhost:8001/health | python3 -m json.tool
```

Expected:
```json
{
  "status": "ok",
  "model_loaded": true,
  "categories": ["belanja", "gaji", "hadiah", ...],
  "secondary_categories": [...],
  "feedback_samples": 0,
  "pending_for_retrain": 0,
  "auto_retrain_threshold": 0.85,
  "auto_retrain_every": 100
}
```

- [ ] **Step 3: Test /classify — kasus pengeluaran**

```bash
curl -s -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{"title": "ngasih sumbangan ke kakak"}' | python3 -m json.tool
```

Expected: `"transaction_type": "pengeluaran"` (bukan pemasukan)

- [ ] **Step 4: Test /classify — kasus pemasukan ambigu**

```bash
curl -s -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{"title": "dikasih makan sama kakak"}' | python3 -m json.tool
```

Expected: `"transaction_type": "pemasukan"`, category `makanan` atau `hadiah`

- [ ] **Step 5: Test /classify/batch**

```bash
curl -s -X POST http://localhost:8001/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"titles": ["bayar listrik PLN", "gaji bulanan", "grab ke kantor"]}' | python3 -m json.tool
```

Expected: 3 results, `"tagihan"` / `"gaji"` / `"transport"` masing-masing.

- [ ] **Step 6: Test /feedback**

```bash
curl -s -X POST http://localhost:8001/feedback \
  -H "Content-Type: application/json" \
  -d '{"title": "grabfood ayam geprek", "correct_category": "makanan", "correct_secondary_category": "makanan"}' | python3 -m json.tool
```

Expected:
```json
{
  "status": "ok",
  "message": "Feedback disimpan: 'grabfood ayam geprek' → makanan/makanan",
  "retrain_triggered": false
}
```

- [ ] **Step 7: Test /retrain**

```bash
curl -s -X POST http://localhost:8001/retrain | python3 -m json.tool
```

Expected:
```json
{
  "status": "ok",
  "base_samples": 14944,
  "feedback_samples": 1,
  "total_samples": 14945
}
```

- [ ] **Step 8: Test /feedback — kategori tidak valid (422)**

```bash
curl -s -X POST http://localhost:8001/feedback \
  -H "Content-Type: application/json" \
  -d '{"title": "test", "correct_category": "invalid", "correct_secondary_category": "x"}' | python3 -m json.tool
```

Expected: HTTP 422 dengan detail `"Kategori tidak valid. ..."`

---

## Checklist Selesai

- [ ] `requirements.txt` sudah include joblib + scipy + pytest
- [ ] `.gitignore` sudah block `models/*.joblib`
- [ ] 14 unit tests PASS
- [ ] `models/classifier.joblib` ada dan valid
- [ ] `/health` mengembalikan `model_loaded: true`
- [ ] `/classify` membedakan "ngasih ke kakak" (keluar) vs "dikasih sama kakak" (masuk)
- [ ] `/feedback` + `/retrain` berfungsi
- [ ] Semua perubahan ter-commit

---

## Notes

- **Perubahan `/retrain` response schema**: Schema lama punya `primary` dan `secondary` nested. Schema baru flat: `{status, base_samples, feedback_samples, total_samples}`. Perbarui klien jika ada yang depend pada schema lama.
- **`classifier.pkl` / `secondary_classifier.pkl`**: Sudah dihapus. Tidak ada fallback — jika model `.joblib` belum ada, service akan gagal startup. Solusi: jalankan `make train` sebelum `make run`.
- **Venv path**: `setup.sh` buat `.venv/` di root project; `Makefile` pakai `/tmp/bw-nlp-venv`. Gunakan yang sesuai, atau unifikasi di sprint selanjutnya.
