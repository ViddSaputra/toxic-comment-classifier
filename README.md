# 🔍 Indonesian Toxic Comment Classifier

Web app sederhana yang mendeteksi apakah sebuah komentar berbahasa Indonesia
mengandung unsur toxic/hate speech, menggunakan TF-IDF + Logistic Regression.

**🔗 Live Demo:** https://huggingface.co/spaces/Viddwny/toxic-comment-classifier

---

## 📌 Latar Belakang

Komentar toxic di media sosial Indonesia adalah masalah nyata, tapi tools
moderasi otomatis untuk Bahasa Indonesia masih sangat terbatas dibanding
Bahasa Inggris. Project ini adalah eksperimen pertama saya membangun
text classification pipeline end-to-end — dari data mentah Twitter sampai
web app yang bisa dipakai siapa saja.

## 🎯 Problem Statement

Diberikan sebuah teks komentar, klasifikasikan apakah komentar tersebut
**toxic** (mengandung hate speech) atau **non-toxic**.

## 📊 Dataset

- **Sumber:** [Indonesian Abusive and Hate Speech Twitter Text](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection) (Ibrohim & Budi, 2019)
- **Ukuran:** ~13.000 tweet berlabel
- **Fitur tambahan:** Kamus normalisasi slang (`new_kamusalay.csv`) — 15.000+ entri
  untuk mengubah kata gaul/typo (`"gw"`, `"yg"`, `"bgt"`) menjadi bentuk formal

## ⚙️ Pipeline

```
Raw Tweet
  → Text Cleaning (lowercase, hapus URL/mention/RT, hapus non-alfabet)
  → Slang Normalization (kamus kamusalay)
  → TF-IDF Vectorization (unigram + bigram, max 5000 fitur)
  → Logistic Regression (class_weight='balanced')
  → Prediksi: TOXIC / NON-TOXIC + confidence score
```

## 🛠️ Tech Stack

| Komponen | Tools |
|----------|-------|
| Data processing | Pandas, Regex |
| Modeling | scikit-learn (TF-IDF, Logistic Regression) |
| Evaluation | Cross-validation, Confusion Matrix |
| Web App | Streamlit |
| Deployment | Hugging Face Spaces |

## 📈 Hasil Model

| Metrik | Score |
|--------|-------|
| Accuracy | 0.84 |
| F1-score (toxic) | 0.82 |
| F1-score (non-toxic) | 0.86 |
| Cross-val F1 (mean ± std) | 0.82 |

Model dievaluasi menggunakan **5-fold cross-validation** untuk memastikan
hasil tidak kebetulan dari satu split data saja.

## 🔍 Apa yang Saya Pelajari

- **Data leakage**: TF-IDF vectorizer hanya boleh di-`fit` pada training set,
  bukan seluruh dataset — kesalahan umum yang terlihat sepele tapi
  mempengaruhi validitas evaluasi.
- **Spurious correlation**: model bisa belajar korelasi yang tidak relevan
  secara semantik (misalnya entitas tertentu) hanya karena kebetulan sering
  muncul di data toxic.
- **Context-dependent toxicity**: TF-IDF + Logistic Regression tidak bisa
  menangkap sarkasme atau toxic implisit karena model ini hanya melihat
  kemunculan kata, bukan konteks kalimat.
- **Error analysis lebih penting dari angka akhir**: membaca sampel false
  positive/negative memberi insight yang tidak terlihat dari classification
  report saja.

## ⚠️ Keterbatasan

- Dataset terbatas (~13K tweet) dan berasal dari satu sumber/waktu tertentu —
  model mungkin tidak generalize baik ke platform lain (TikTok, YouTube, dll)
  atau bahasa gaul yang lebih baru.
- Model tidak memahami sarkasme, konteks percakapan, atau nuansa budaya.
- TF-IDF bersifat *bag-of-words* — urutan kata tidak diperhitungkan.

## 🚀 Pengembangan Selanjutnya

- [ ] Fine-tune model berbasis transformer (IndoBERT) untuk menangkap konteks
- [ ] Tambah data dari sumber lain untuk meningkatkan generalisasi
- [ ] Multi-label classification (HS, abusive, severity level) — dataset
      sebenarnya punya 12 label, project ini hanya pakai 1
- [ ] Tambahkan explanation di UI (kata mana yang memicu prediksi)

## 🗂️ Struktur Project

```
toxic-classifier/
├── app.py                  # Streamlit app
├── requirements.txt
├── data/
│   ├── data.csv             # dataset asli
│   ├── data_clean.csv        # hasil preprocessing
│   └── new_kamusalay.csv     # kamus normalisasi slang
├── model/
│   └── pipeline_final.joblib # TF-IDF + Logistic Regression
└── notebooks/
    ├── 01_eda.ipynb
    ├── 02_preprocessing.ipynb
    ├── 03_modeling.ipynb
    └── 04_evaluation.ipynb
```

## 🧑‍💻 Cara Menjalankan Lokal

```bash
git clone <repo-url>
cd toxic-classifier
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

**Dataset citation:**
Ibrohim, M. O., & Budi, I. (2019). Multi-label Hate Speech and Abusive
Language Detection in Indonesian Twitter. *Proceedings of the Third
Workshop on Abusive Language Online*.