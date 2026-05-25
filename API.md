# BayarWoy NLP Service — API Docs

Base URL: `http://localhost:8001`

---

## Endpoints

### `GET /health`

Cek status service dan statistik model.

**Response**
```json
{
  "status": "ok",
  "model_loaded": true,
  "categories": ["belanja", "hiburan", "kesehatan", "makanan", "minuman", "tagihan", "transport"],
  "feedback_samples": 3,
  "pending_for_retrain": 3,
  "auto_retrain_threshold": 0.85,
  "auto_retrain_every": 20
}
```

| Field | Tipe | Keterangan |
|-------|------|------------|
| `status` | string | `"ok"` jika service berjalan normal |
| `model_loaded` | bool | `true` jika model sudah ter-load |
| `categories` | string[] | Daftar kategori yang bisa diprediksi |
| `feedback_samples` | int | Total sampel di `feedback.csv` |
| `pending_for_retrain` | int | Sampel baru sejak retrain terakhir |
| `auto_retrain_threshold` | float | Minimum confidence untuk auto-save (default: `0.85`) |
| `auto_retrain_every` | int | Jumlah sampel baru sebelum auto-retrain (default: `20`) |

---

### `POST /classify`

Klasifikasikan satu judul transaksi ke kategori pengeluaran.

Jika confidence ≥ 85%, prediksi otomatis disimpan ke `feedback.csv` sebagai data latih baru (`reinforced: true`).

**Request**
```json
{
  "title": "makan siang di warteg"
}
```

| Field | Tipe | Wajib | Keterangan |
|-------|------|-------|------------|
| `title` | string | Ya | Judul transaksi, 1–200 karakter |

**Response**
```json
{
  "title": "makan siang di warteg",
  "category": "makanan",
  "confidence": 0.9649,
  "alternatives": [
    { "category": "belanja", "confidence": 0.0083 },
    { "category": "minuman", "confidence": 0.0065 },
    { "category": "hiburan", "confidence": 0.0056 }
  ],
  "reinforced": true
}
```

| Field | Tipe | Keterangan |
|-------|------|------------|
| `category` | string | Kategori hasil prediksi |
| `confidence` | float | Skor keyakinan model (0–1) |
| `alternatives` | object[] | 3 kategori alternatif dengan confidence masing-masing |
| `reinforced` | bool | `true` jika prediksi ini disimpan untuk reinforcement |

**Contoh curl**
```bash
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{"title": "grab ke kantor"}'
```

---

### `POST /classify/batch`

Klasifikasikan banyak judul transaksi sekaligus (maks. 50).

**Request**
```json
{
  "titles": [
    "bayar listrik PLN",
    "nonton CGV",
    "beli vitamin di apotek",
    "isi bensin pertalite",
    "shopee haul baju"
  ]
}
```

| Field | Tipe | Wajib | Keterangan |
|-------|------|-------|------------|
| `titles` | string[] | Ya | Array judul transaksi, maks. 50 item |

**Response**
```json
{
  "results": [
    {
      "title": "bayar listrik PLN",
      "category": "tagihan",
      "confidence": 0.978,
      "alternatives": [...],
      "reinforced": true
    },
    {
      "title": "nonton CGV",
      "category": "hiburan",
      "confidence": 0.9124,
      "alternatives": [...],
      "reinforced": true
    }
  ]
}
```

**Contoh curl**
```bash
curl -X POST http://localhost:8001/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"titles": ["makan bakso", "grab motor", "bayar wifi"]}'
```

---

### `POST /feedback`

Koreksi manual kategori untuk sebuah transaksi. Dipakai ketika prediksi model salah.

Data yang dikirim langsung disimpan ke `feedback.csv` dan ikut menghitung ke counter auto-retrain.

**Request**
```json
{
  "title": "grabfood ayam geprek",
  "correct_category": "makanan"
}
```

| Field | Tipe | Wajib | Keterangan |
|-------|------|-------|------------|
| `title` | string | Ya | Judul transaksi yang salah diprediksi |
| `correct_category` | string | Ya | Kategori yang benar (lihat daftar kategori valid di bawah) |

**Response**
```json
{
  "status": "ok",
  "message": "Feedback disimpan: 'grabfood ayam geprek' → makanan",
  "retrain_triggered": false
}
```

| Field | Tipe | Keterangan |
|-------|------|------------|
| `retrain_triggered` | bool | `true` jika feedback ini memicu auto-retrain |

**Error** — kategori tidak valid (`422`):
```json
{
  "detail": "Kategori tidak valid. Pilih dari: ['belanja', 'hiburan', 'kesehatan', 'makanan', 'minuman', 'tagihan', 'transport']"
}
```

**Contoh curl**
```bash
curl -X POST http://localhost:8001/feedback \
  -H "Content-Type: application/json" \
  -d '{"title": "gofood pizza", "correct_category": "makanan"}'
```

---

### `POST /retrain`

Trigger retrain model secara manual menggunakan semua data (`transactions.csv` + `feedback.csv`).

Tidak perlu request body.

**Response**
```json
{
  "status": "ok",
  "base_samples": 619,
  "feedback_samples": 12,
  "total_samples": 631,
  "classes": ["belanja", "hiburan", "kesehatan", "makanan", "minuman", "tagihan", "transport"]
}
```

**Error** — retrain sedang berjalan (`409`):
```json
{
  "detail": "Retrain sedang berjalan."
}
```

**Contoh curl**
```bash
curl -X POST http://localhost:8001/retrain
```

---

## Kategori Valid

| Kategori | Contoh transaksi |
|----------|-----------------|
| `makanan` | makan siang, nasi goreng, KFC, gofood, pesan bakso |
| `minuman` | kopi kenangan, starbucks, bubble tea, es teh, aqua |
| `transport` | grab, gojek, bensin pertalite, bayar tol, KRL |
| `belanja` | shopee, indomaret, beli baju, belanja bulanan |
| `hiburan` | netflix, bioskop, top up ML, konser, liburan bali |
| `tagihan` | bayar listrik, BPJS, cicilan, pulsa, wifi indihome |
| `kesehatan` | beli obat, apotek, dokter, vitamin, cek darah |

---

## Alur Reinforcement Learning

```
POST /classify
      │
      ▼
 confidence ≥ 0.85?
      │
   Ya │                Tidak
      ▼                  ▼
 simpan ke          tidak disimpan
 feedback.csv       reinforced: false
 reinforced: true
      │
      ▼
 pending_count >= 20?
      │
   Ya │
      ▼
 retrain di background thread
 (tidak block response)
      │
      ▼
 model diperbarui + disimpan
```

- **Auto-save threshold:** `0.85` (confidence minimum)
- **Auto-retrain every:** `20` sampel feedback baru
- Retrain berjalan di background — response API tidak terhambat
- File feedback: `data/feedback.csv`
- Model tersimpan di: `models/classifier.pkl`

---

## Menjalankan Service

```bash
# Training awal
make train

# Jalankan server (port 8001)
make run

# Atau manual
/tmp/bw-nlp-venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

Interactive docs tersedia di `http://localhost:8001/docs` (Swagger UI) saat service berjalan.