# 🪻 PsikisAI — Mental Health Text Insight

Aplikasi web (Streamlit) untuk mendeteksi indikasi kondisi kesehatan mental dari teks
media sosial, ditenagai model **RoBERTa-base** yang sudah di-*fine-tune* (`roberta_best`).

> ⚠️ **Bukan alat diagnosis.** Proyek akademik untuk mendeteksi pola bahasa pada teks,
> bukan kondisi klinis seseorang.

---

## Untuk download model
Link Google Drive: https://drive.google.com/drive/folders/1PDvmo2_aHBs7KSUy9ekULQ1_zSVO5NvS?usp=sharing

## 1. Struktur folder

Letakkan model di samping `app.py` seperti ini:

```
psikisai/
├── app.py
├── test_inference.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── roberta_best/            ← folder model (dari Drive)
    ├── config.json
    ├── model.safetensors
    ├── merges.txt
    ├── vocab.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

## 2. Instalasi (sekali saja)

Disarankan pakai virtual environment.

```bash
cd psikisai
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> Tidak butuh GPU — CPU sudah cukup untuk inference. Instalasi `torch` agak besar
> (beberapa ratus MB), jadi pastikan koneksi internet stabil saat install.

## 3. (Opsional) Cek model dulu

```bash
python test_inference.py
```

Kalau muncul prediksi yang masuk akal untuk tiap kalimat contoh, model siap dipakai.

## 4. Jalankan aplikasi

```bash
streamlit run app.py
```

Browser otomatis terbuka di `http://localhost:8501`.

---

## Fitur

- **Analisis Teks** — prediksi 1 teks + tingkat keyakinan untuk ke-6 kelas.
- **Penjelasan LIME** — menyoroti kata yang paling memengaruhi keputusan model.
- **Analisis Batch** — unggah CSV/TXT, prediksi massal, distribusi, dan unduh hasil.
- **Dukungan sensitif** — saat prediksi `Suicidal`/`Depression`, app menampilkan
  sumber bantuan resmi (Healing119.id, JakCare, dll) dengan nada suportif.

## Catatan teknis penting

- **6 kelas** (bukan 5): `Anxiety, Bipolar, Depression, Normal, Stress, Suicidal`.
  PTSD tidak ada di model — sesuaikan laporan tim agar konsisten dengan output app.
- **Teks mentah** langsung di-*tokenize* (tanpa `clean_text`), karena RoBERTa di
  notebook dilatih pada teks asli; pembersihan hanya dipakai untuk baseline TF-IDF.
- Tokenizer: `truncation=True, max_length=128` (sama seperti saat training).
- Ganti lokasi model lewat env var bila perlu: `PSIKISAI_MODEL_DIR=/path/ke/roberta_best`.

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `Folder model tidak ditemukan` | Pastikan folder `roberta_best/` ada di samping `app.py`. |
| LIME lambat | Wajar di CPU (~20–60 dtk). Kurangi `num_samples` di `app.py` bila perlu. |
| Error versi transformers | Pakai `transformers==4.44.0` (sesuai versi training). |
