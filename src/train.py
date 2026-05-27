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
