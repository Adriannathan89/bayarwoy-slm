# Rekomendasi Arsitektur NLP — Klasifikasi Transaksi

Dokumen ini berisi rekomendasi yang bisa langsung dieksekusi (mis. lewat Claude Code) untuk membangun model klasifikasi transaksi dari dataset `transactions_v5_clean.csv` (14.945 baris; kolom: `title, category, secondary_category, type`). Titik awalnya adalah baseline yang sudah kamu pakai: `TfidfVectorizer + LogisticRegression` di dalam `Pipeline`/`FeatureUnion`.

Sudah disertakan skrip siap pakai: `train_classifier.py`.

## Ringkasan singkat

Pendekatan yang direkomendasikan mempertahankan stack-mu (TF-IDF + LogisticRegression) karena untuk teks pendek berbahasa Indonesia dengan label yang cukup terstruktur, model linear seperti ini biasanya sudah sangat kompetitif, cepat dilatih, ringan di-deploy, dan mudah di-debug. Yang ditambahkan hanyalah tiga hal: representasi karakter n-gram agar tahan typo, fitur leksikal eksplisit untuk arah uang, dan tiga classifier terpisah untuk tiga kolom label.

## 1. Kenapa tetap pakai TF-IDF + LogisticRegression

Untuk judul transaksi yang pendek (rata-rata 3-5 kata), model linear di atas fitur TF-IDF umumnya mengungguli atau menyamai model neural yang jauh lebih berat, dengan biaya komputasi minimal. Transformer (mis. IndoBERT) baru benar-benar unggul ketika konteksnya panjang dan ambigu secara semantik — yang bukan karakteristik utama data ini. Saran: jadikan model linear sebagai baseline produksi, dan hanya naik ke transformer kalau evaluasi pada data nyata menunjukkan kebutuhan itu.

## 2. Tiga peningkatan konkret pada baseline

Pertama, gabungkan TF-IDF kata (1-2 gram) dengan TF-IDF karakter (`char_wb`, 3-5 gram) lewat `FeatureUnion`. Karakter n-gram membuat model mengenali "netflix" dan "netfilx" sebagai mirip, sehingga tahan terhadap typo yang banyak terdapat di data — ini penting karena input pengguna nyata penuh salah ketik.

Kedua, tambahkan fitur leksikal arah-uang sebagai sinyal eksplisit. Daripada berharap model menyimpulkan sendiri bahwa "ke" menandakan uang keluar dan "dari/sama" menandakan uang masuk, berikan fitur biner yang menghitung keberadaan kata kerja masuk (terima, dapat, dikasih, ditraktir) versus keluar (bayar, kirim, transfer ke, sumbang). Ini langsung memperkuat kemampuan membedakan kasus seperti "ngasih sumbangan ke kakak" (keluar) vs "dikasih makan sama kakak" (masuk).

Ketiga, latih tiga classifier terpisah (satu untuk `category`, satu untuk `secondary_category`, satu untuk `type`) yang berbagi representasi teks yang sama. Ini lebih sederhana dan lebih mudah di-debug daripada satu model multi-label, dan memungkinkan penerapan aturan konsistensi di akhir.

## 3. Aturan konsistensi (post-processing)

Karena `type` hampir sepenuhnya dapat diturunkan dari `category` (gaji dan hadiah selalu pemasukan; sisanya pengeluaran), terapkan aturan deterministik setelah prediksi: jika kategori adalah gaji atau hadiah, paksa type menjadi pemasukan. Ini menghilangkan kemungkinan prediksi yang saling bertentangan antar kolom, dan dalam praktik menaikkan akurasi `type` tanpa biaya. Pertimbangkan juga: apakah `type` perlu diprediksi model sama sekali, atau cukup dipetakan dari `category`? Untuk banyak kasus, pemetaan langsung sudah cukup dan lebih andal.

## 4. PERINGATAN PENTING soal evaluasi

Skrip yang disertakan melaporkan akurasi sekitar 99% pada test split. **Angka ini tidak boleh ditafsirkan sebagai performa dunia nyata.** Penyebabnya: train dan test split sama-sama berasal dari data yang sebagian besar dihasilkan secara sintetis dari pola/template yang serupa, sehingga model sebagian "menghafal pola buatan" alih-alih belajar generalisasi ke bahasa alami. Test set yang berbagi distribusi buatan dengan training set akan selalu memberi skor optimistis.

Yang harus dilakukan: siapkan set evaluasi terpisah dari transaksi nyata pengguna (sekalipun jumlahnya kecil, misalnya 200-500 contoh yang dilabeli manual), dan ukur akurasi di situ. Skor itulah yang mencerminkan performa sebenarnya. Gunakan data sintetis untuk training, tetapi jangan untuk menilai keberhasilan akhir.

Hal penting kedua: ada banyak near-duplicate antar template. Pastikan tidak ada kebocoran di mana varian dari frasa yang sama muncul di train sekaligus test (mis. "makan di warteg" di train dan "makan di warteg tadi" di test). Idealnya, lakukan split berdasarkan kelompok makna (group split), bukan baris acak, agar evaluasi tidak menggelembung.

## 5. Langkah eksekusi (untuk Claude Code)

Urutan yang disarankan: pasang dependensi (`pip install scikit-learn pandas scipy joblib`), jalankan `python train_classifier.py --data transactions_v5_clean.csv` untuk melatih dan menyimpan `model.joblib`, periksa output demo untuk kasus kontras, lalu siapkan set evaluasi dari data nyata dan ukur ulang. Setelah itu, bila perlu, lakukan penyetelan hiperparameter ringan (`C` pada LogisticRegression, `min_df` dan `ngram_range` pada vectorizer) dengan validasi silang.

## 6. Peningkatan lanjutan (opsional, sesuai kebutuhan)

Kalau evaluasi pada data nyata menunjukkan kelemahan, beberapa arah pengembangan: kalibrasi probabilitas (`CalibratedClassifierCV`) agar bisa menetapkan ambang kepercayaan dan menandai transaksi yang perlu konfirmasi manual; kamus merchant eksplisit (lookup nama resto/brand ke kategori) sebagai fitur tambahan atau aturan prioritas, karena nama merchant adalah sinyal kuat dan deterministik; dan baru sebagai langkah terakhir, fine-tuning IndoBERT bila konteks panjang/ambiguitas semantik terbukti jadi penghambat utama.

## 7. Batasan yang melekat

Frasa yang benar-benar tanpa kata kerja arah (mis. "komisi tomoro" tanpa "dapat" atau "bayar") secara hakiki ambigu — model akan menebak berdasarkan kecenderungan data, dan tidak ada arsitektur yang bisa menjamin tebakan itu benar karena informasinya memang tidak cukup. Untuk kasus seperti ini di aplikasi nyata, sinyal eksplisit dari pengguna (mis. pemisahan input pemasukan/pengeluaran, atau memori merchant per pengguna) jauh lebih andal daripada mengandalkan model menebak dari teks pendek.
