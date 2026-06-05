# 🧠 DataScience ChronoCanvas

ChronoCanvas merupakan aplikasi berbasis Python yang memanfaatkan teknologi Data Science untuk menghasilkan konten edukasi secara otomatis berdasarkan prompt yang diberikan pengguna.

Aplikasi ini dirancang untuk membantu guru, siswa, maupun masyarakat umum dalam memperoleh materi pengetahuan umum dan kecerdasan buatan (Artificial Intelligence) secara cepat, interaktif, dan mudah dipahami.

---

## 🚀 Fitur Utama

- 📚 Generate materi pembelajaran berdasarkan prompt
- 🤖 Pemanfaatan teknologi Artificial Intelligence
- 📊 Pengolahan dataset pengetahuan umum Indonesia
- 🧹 Data preprocessing dan cleaning
- 🌐 Antarmuka web sederhana dan mudah digunakan
- ⚡ Respon cepat dan interaktif

---

## 📂 Struktur Project

```text
DataScience_ChronoCanvas
│
├── Data Clean/
│   └── cleaned_data.csv
│
├── data/
│   ├── dataset_ai_indonesia_500.csv
│   └── dataset_ai_pengetahuan_umum.csv
│
├── README.md
├── ChronoCanvas_fix.ipynb
├── app.py
└── requirements.txt
```

---

## 📁 Deskripsi Folder

### `data/`

Berisi dataset utama yang digunakan dalam proses analisis dan pembangkitan konten.

| File | Deskripsi |
|--------|-----------|
| dataset_ai_indonesia_500.csv | Dataset pengetahuan AI berbahasa Indonesia |
| dataset_ai_pengetahuan_umum.csv | Dataset pengetahuan umum sebagai sumber informasi |

### `Data Clean/`

Berisi dataset yang telah melalui proses preprocessing dan pembersihan data.

| File | Deskripsi |
|--------|-----------|
| cleaned_data.csv | Dataset hasil cleaning dan siap digunakan |

### `app.py`

File utama aplikasi yang menjalankan sistem dan antarmuka pengguna.

### `ChronoCanvas_fix.ipynb`

Notebook yang digunakan untuk eksplorasi data, eksperimen model, dan pengembangan sistem.

---

## 🛠️ Teknologi yang Digunakan

### Bahasa Pemrograman

- Python

### Library

- Pandas
- NumPy
- Scikit-Learn
- Flask / Streamlit
- Matplotlib
- Seaborn

---

## ⚙️ Instalasi

Clone repository:

```bash
git clone https://github.com/ismaramadhani/DataScience_ChronoCanvas.git
```

Masuk ke folder project:

```bash
cd DataScience_ChronoCanvas
```

Install dependency:

```bash
pip install -r requirements.txt
```

---

## ▶️ Menjalankan Aplikasi

Jika menggunakan Flask:

```bash
python app.py
```

Jika menggunakan Streamlit:

```bash
streamlit run app.py
```

---

Link Deploy Streamlit

```bash
(https://dschronocanvas.streamlit.app/)
```


## 📊 Dataset

Project ini memanfaatkan dataset yang berisi:

- Pengetahuan umum
- Materi Artificial Intelligence
- Informasi edukatif berbahasa Indonesia

Dataset kemudian melalui tahapan:

1. Data Collection
2. Data Cleaning
3. Data Transformation
4. Data Processing
5. Knowledge Generation

---

## 🔄 Workflow Sistem

```text
User Prompt
      │
      ▼
Data Processing
      │
      ▼
Knowledge Retrieval
      │
      ▼
AI Generation
      │
      ▼
Output Materi Pembelajaran
```

---

## 🎯 Tujuan Project

Membangun platform edukasi berbasis Artificial Intelligence yang mampu membantu pengguna memperoleh materi pembelajaran secara cepat, relevan, dan mudah dipahami.

---

## 👨‍💻 Tim Pengembang

- Isma' Yafa Nur Zamzami Ramadhani
- Yahya Ahmad

---

## 📜 License

Project ini dibuat untuk kebutuhan pembelajaran, penelitian, dan pengembangan teknologi pendidikan berbasis Artificial Intelligence.
