# Design: Migrasi ke TransactionModel Unified (v5 Dataset)

**Tanggal:** 2026-05-26  
**Scope:** `src/classifier.py`, `src/main.py`, `src/train.py`  
**Dataset target:** `data/transactions_v5_clean.csv` (14.945 baris)  
**Referensi:** `Arsitektur.md`, `train_classifier.py`

---

## 1. Ringkasan

Menggantikan dua `TransactionClassifier` terpisah dengan satu `TransactionModel` unified. Model baru berbagi satu feature extractor (word TF-IDF + char TF-IDF + `MoneyDirectionFeatures`) untuk dua LR classifier (category + secondary_category). `type` tetap rule-based via `type_from_category()`. Serialisasi beralih dari dua file `.pkl` ke satu file `classifier.joblib`.

---

## 2. Arsitektur

### 2.1 Sebelum (v4)

```
main.py
  ├─ clf_primary   = TransactionClassifier(label_field="category")
  └─ clf_secondary = TransactionClassifier(label_field="secondary_category")

Tiap TransactionClassifier:
  Pipeline(FeatureUnion(word_tfidf, char_tfidf(3-4)) → LR(C=5.0))
  Serialized: classifier.pkl / secondary_classifier.pkl
```

### 2.2 Sesudah (v5)

```
main.py
  └─ clf = TransactionModel()

TransactionModel:
  text_feats  = FeatureUnion(word_tfidf(1-2), char_tfidf(3-5))
  money_feats = MoneyDirectionFeatures()   # 8 fitur biner
  _featurize() = hstack(text_feats, money_feats)
  clf_category          = LR(C=4.0, class_weight="balanced")
  clf_secondary         = LR(C=4.0, class_weight="balanced")
  predict(title) → {category, secondary_category, type (rule), confidence, ...}
  Serialized: classifier.joblib (satu file)
```

---

## 3. Perubahan per File

### 3.1 `src/classifier.py`

**Paths:**
- `BASE_DATA_PATH` → `data/transactions_v5_clean.csv`
- `MODEL_PATH` → `models/classifier.joblib`
- Hapus `SECONDARY_MODEL_PATH`

**Tambah class `MoneyDirectionFeatures(BaseEstimator, TransformerMixin)`:**
- Input: list preprocessed strings
- Output: sparse matrix (8 kolom)
- Fitur: `n_in`, `n_out`, `starts_in`, `starts_out`, `has_ke`, `has_dari`, `has_sama`, `word_count`
- Cue lists IN/OUT sesuai `train_classifier.py`
- Bekerja pada preprocessed text — cue kritis (`ke`, `dari`, `bayar`, dll.) survive `preprocess()`

**Tambah class `TransactionModel`:**

| Method | Deskripsi |
|--------|-----------|
| `__init__()` | Init `text_feats`, `money_feats`, `clf_category`, `clf_secondary` |
| `_featurize(raw_titles, fit)` | preprocess each title internally → TF-IDF+money → hstack CSR |
| `train(titles, pri, sec)` | 5-fold CV tiap label, fit full, return metrics dict |
| `retrain_with_feedback()` | Load v5 + feedback.csv, refit, save |
| `predict(title)` | Return dict: category, secondary_category, type, confidence, secondary_confidence, alternatives, secondary_alternatives |
| `save(path)` | `joblib.dump(self, path)` |
| `load(path)` | `joblib.load(path)`, update `self.__dict__` |

**`predict()` detail:**
- `_featurize([title], fit=False)` — preprocessing dilakukan di dalam `_featurize`
- `proba_cat = clf_category.predict_proba(Xf)[0]`
- `proba_sec = clf_secondary.predict_proba(Xf)[0]`
- Build alternatives (top 3, exclude top)
- `transaction_type = type_from_category(top_category)` — rule-based, bukan classifier

**Dipertahankan:**
- `preprocess()`, `STOPWORDS_ID`
- `type_from_category()`
- `VALID_CATEGORIES`, `VALID_SECONDARY_CATEGORIES`, `ALL_SECONDARY`, `INCOME_CATEGORIES`
- `_load_csv()`, `_load_csv_with_secondary()`

**Dihapus:**
- `class TransactionClassifier`
- `_build_pipeline()`
- `SECONDARY_MODEL_PATH`

### 3.2 `src/main.py`

**State:**
```python
# Sebelum
clf_primary   = TransactionClassifier(label_field="category",   model_path=MODEL_PATH)
clf_secondary = TransactionClassifier(label_field="secondary_category", model_path=SECONDARY_MODEL_PATH)

# Sesudah
clf = TransactionModel()
```

**Lifespan:** `clf.load()` satu kali.

**`_classify_one(title)`:**
```python
result = clf.predict(title)
# result sudah punya: category, secondary_category, type, confidence,
# secondary_confidence, alternatives, secondary_alternatives
reinforced = maybe_save_and_retrain(title, result["category"], result["secondary_category"], result["confidence"])
return ClassifyResponse(**result, reinforced=reinforced)
```

**`maybe_save_and_retrain()`:**
- Threshold: cukup `pri_conf >= AUTO_REINFORCE_THRESHOLD` (satu check)
- Secondary confidence tidak perlu diperiksa independen (model shared)

**`_background_retrain()`:**
- `clf.retrain_with_feedback()` satu kali (bukan dua)

**`/health` endpoint:**
- Hapus `secondary_loaded`, tambah `model_loaded: clf.clf_category is not None`
- `categories` → `clf.classes_category`
- `secondary_categories` → `clf.classes_secondary`

**Response schema tidak berubah** — `ClassifyResponse` tetap sama.

**Import cleanup:**
- Hapus `SECONDARY_MODEL_PATH`, `clf_primary`, `clf_secondary`
- Tambah `TransactionModel`

### 3.3 `src/train.py`

```python
from classifier import TransactionModel, BASE_DATA_PATH, MODEL_PATH, _load_csv_with_secondary, preprocess

titles, primary_labels, secondary_labels = _load_csv_with_secondary(BASE_DATA_PATH)
model = TransactionModel()
metrics = model.train(titles, primary_labels, secondary_labels)
model.save(MODEL_PATH)
```

- Report CV accuracy untuk kedua label
- Sample predictions seperti sebelumnya
- Remove semua referensi ke `TransactionClassifier`, `SECONDARY_MODEL_PATH`

---

## 4. Feature Engineering Detail

### MoneyDirectionFeatures

```
IN_CUES  = ["terima","menerima","nerima","dapat","dapet","masuk","cair","dikasih",
             "dikirimi","ditransfer","dipinjamin","ditraktir","dibeliin","dibawain",
             "disangui","gajian","bonus","thr","cashback","refund","reward","hadiah",
             "warisan","hibah","dari"]

OUT_CUES = ["bayar","byr","beli","beliin","bayarin","kasih","ngasih","kirim",
             "transfer ke","sumbang","sedekah","traktir","top up","langganan",
             "cicilan","isi","order","pesan","jajan","ke "]
```

8 fitur output: `[n_in, n_out, starts_in, starts_out, has_ke, has_dari, has_sama, word_count]`

### TF-IDF Hyperparameters (update dari v4)

| Parameter | v4 (lama) | v5 (baru) |
|-----------|-----------|-----------|
| word ngram | (1,2) | (1,2) |
| char ngram | (3,4) | **(3,5)** |
| max_features | 6000 | - |
| min_df | - | **2** |
| LR C | 5.0 | **4.0** |
| class_weight | - | **"balanced"** |

---

## 5. Serialisasi

| | Sebelum | Sesudah |
|--|---------|---------|
| Format | `pickle` | `joblib` |
| Files | `classifier.pkl` + `secondary_classifier.pkl` | `classifier.joblib` |
| Load | dua `TransactionClassifier.load()` | satu `TransactionModel.load()` |

File `.pkl` lama bisa dihapus setelah model baru berhasil dilatih dan diverifikasi.

---

## 6. Feedback Loop

`feedback.csv` schema tidak berubah: `title, category, secondary_category, source, timestamp`.

`retrain_with_feedback()` pada `TransactionModel`:
1. Load v5 base data → `(titles, pri, sec)`
2. Load `feedback.csv` → tambahkan ke list
3. `_featurize(all_titles, fit=True)` — refit feature extractors
4. Refit `clf_category` dan `clf_secondary`
5. `save()` → `classifier.joblib`

---

## 7. Files yang Bisa Dihapus Setelah Integrasi

- `train_classifier.py` (di root) — logic sudah diintegrasikan
- `models/classifier.pkl` — digantikan `classifier.joblib`
- `models/secondary_classifier.pkl` — digantikan `classifier.joblib`
- `src/generate_v4.py`, `src/synthesize.py` — opsional, tergantung kebutuhan

---

## 8. Error Handling & Backward Compatibility

- Jika `classifier.joblib` tidak ada saat startup, `clf.load()` raise `FileNotFoundError` → HTTP 503 seperti sebelumnya
- `ClassifyResponse` schema tidak berubah → FE tidak perlu update
- API endpoints tidak berubah

---

## 9. Testing Plan

Setelah implementasi:
1. `python src/train.py` → verifikasi CV accuracy kedua label
2. Start service → `GET /health` → cek `model_loaded: true`
3. `POST /classify` dengan contoh kontras dari `Arsitektur.md`:
   - "ngasih sumbangan ke kakak" → pengeluaran
   - "dikasih makan sama kakak" → pemasukan
   - "dapat komisi dari tomoro" → pemasukan (hadiah/gaji)
   - "bayar komisi ke tomoro" → pengeluaran (tagihan)
4. `POST /feedback` → verifikasi feedback tersimpan
5. `POST /retrain` → verifikasi retrain berjalan

---

## 10. Out of Scope

- Fine-tuning IndoBERT (Arsitektur.md section 6 — opsional)
- Probability calibration via `CalibratedClassifierCV`
- Merchant lookup dictionary
- `/friends/:id` endpoint (backend sprint 4)
- Type classifier training (rule-based sudah cukup)
