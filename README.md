# 🔍 Matching CV and Job Description

> An NLP pipeline for automatic resume-to-job-description matching. Given a CV and a job offer, the pipeline computes a **semantic similarity score** and identifies **missing skills** through Named Entity Recognition.

---

## 📦 Datasets

| Dataset | Description |
|---|---|
| [54k Resume Dataset (structured)](https://www.kaggle.com/datasets/suriyaganesh/resume-dataset-structured) | 54,000 resumes split across 6 CSV files (people, abilities, skills, education, experience, person skills) |
| [Job Description Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) | 40,000 job postings across multiple roles |

---

## 📁 Project Structure

```
Matching-CV-and-Job-Description/
├── README.md
├── requirements.txt
├── .gitignore
├── notebook/
│   └── NLP_final.ipynb              # Main notebook — full pipeline execution
├── modules/
│   ├── model_nlp.py                 # NER pipeline (hirly-ner-multi) + SBERT model loading
│   ├── def_top_k.py                 # Top-K retrieval functions for TF-IDF
│   ├── top_k_sbert.py               # SBERT-specific top-K retrieval implementation
│   ├── vectorisation.py             # TF-IDF vectorization and matrix building
│   ├── preprocessing_cv.py          # CV reconstruction from structured CSV files
│   └── preprocessing_job_description.py  # Job description cleaning and text building
└── cv_test/
    └── cv_english.pdf               # Sample CV (fictitious) used for demo
```

---

## ⚙️ Pipeline

The pipeline is divided into two evaluation tasks:

**1. 📊 Corpus-scale retrieval (Mean top-5)**
For a given resume, retrieve the 5 most relevant job descriptions from the corpus and compute the average similarity score.

**2. 🎯 Pairwise matching**
Given a new unseen CV and a job description, compute a direct similarity score.

---

## 🤖 Models

| Model | Role |
|---|---|
| **TF-IDF** | Baseline retrieval |
| **Doc2Vec** (PV-DM, vector_size=300) | Document embeddings |
| **SBERT** (all-MiniLM-L6-v2) | Semantic similarity |
| **Cross-Encoder** (stsb-roberta-large) | Pairwise reranking |
| **NER** (hirly-ner-multi) | Skill extraction |

---

## 🧠 NER Skill Matching

The NER module extracts `SKILL` and `LANG` entities from both the CV and the job description using `feliponi/hirly-ner-multi`. A two-pass comparison is then applied:

1. **Exact matching** — after Porter stemming
2. **Semantic matching** — on original skill tokens using SBERT embeddings (cosine threshold = 0.75)

---

## 🚀 Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Main dependencies:** `transformers` · `sentence-transformers` · `gensim` · `scikit-learn` · `spacy` · `pdfplumber` · `plotly` · `nltk`

---

## 📖 Usage

1. Open `notebook/NLP_final.ipynb`
2. Set the path to your CV (PDF) and paste your job description
3. Run all cells
4. Get your similarity score and skill gap analysis

---

## 📄 Report

The full project report is available in the `report/` folder, written in **NeurIPS 2026** format.

---

## 📜 License

MIT