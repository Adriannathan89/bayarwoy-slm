"""
Jalankan: python src/train.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from classifier import TransactionClassifier, MODEL_PATH, BASE_DATA_PATH, type_from_category

DATA_PATH = BASE_DATA_PATH


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    titles, labels = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row["title"].strip()
            label = row["category"].strip()
            if title and label:
                titles.append(title)
                labels.append(label)
    return titles, labels


def main():
    print(f"Memuat dataset dari {DATA_PATH}...")
    titles, labels = load_dataset(DATA_PATH)

    from collections import Counter
    dist = Counter(labels)
    print(f"\nDistribusi dataset ({len(titles)} sampel total):")
    for cat, count in sorted(dist.items()):
        print(f"  {cat:15s}: {count} sampel")

    print("\nMelatih model...")
    clf = TransactionClassifier()
    metrics = clf.train(titles, labels)

    print(f"\nHasil training:")
    print(f"  CV Accuracy: {metrics['cv_accuracy_mean']:.1%} ± {metrics['cv_accuracy_std']:.1%}")
    print(f"  Kategori   : {', '.join(metrics['classes'])}")

    clf.save(MODEL_PATH)
    print(f"\nModel disimpan ke: {MODEL_PATH}")

    print("\nContoh prediksi:")
    tests = [
        "makan siang di warteg",
        "grab ke kantor",
        "starbucks cold brew",
        "belanja di shopee",
        "nonton netflix",
        "bayar bpjs",
        "beli paracetamol di apotek",
        "top up diamond ml",
        "gaji bulanan januari",
        "bonus akhir tahun",
        "hadiah dari orang tua",
    ]
    for t in tests:
        result = clf.predict(t)
        tx_type = type_from_category(result['category'])
        print(f"  '{t}' → {result['category']} ({result['confidence']:.0%}) [{tx_type}]")


if __name__ == "__main__":
    main()