Matching CV and Job Description
An NLP pipeline for automatic resume-to-job-description matching. Given a CV and a job offer, the pipeline computes a semantic similarity score and identifies missing skills through Named Entity Recognition.
Datasets
54k Resume Dataset (structured) — 54,000 resumes split across 5 CSV files (people, abilities, skills, education, experience)
Job Description Dataset — 40,000 job postings across multiple roles
Project Structure
```
Matching-CV-and-Job-Description/
├── README.md

├── requirements.txt
├── .gitignore
├── notebook/
│   └── NLP\_final.ipynb              # Main notebook — full pipeline execution
├── modules/
│   ├── model\_nlp.py                 # NER pipeline (hirly-ner-multi) + SBERT model loading
│   ├── def\_top\_k.py                 # Top-K retrieval functions for TF-IDF and SBERT
│   ├── top\_k\_sbert.py               # SBERT-specific top-K retrieval implementation
│   ├── vectorisation.py             # TF-IDF vectorization and matrix building
│   ├── preprocessing\_cv.py          # CV reconstruction from structured CSV files
│   └── preprocessing\_job\_description.py  # Job description cleaning and text building
└── cv\_test/
    └── cv\_english.pdf               # Sample CV (fictitious) used for demo
```
Pipeline
The pipeline is divided into two evaluation tasks:
1. Corpus-scale retrieval (Mean top-5)
For a given resume, retrieve the 5 most relevant job descriptions from the corpus and compute the average similarity score.
2. Pairwise matching
Given a new unseen CV and a job description, compute a direct similarity score.
Models
Model	Role
TF-IDF	Baseline retrieval
Doc2Vec (PV-DM, vector_size=300)	Document embeddings
SBERT (all-MiniLM-L6-v2)	Semantic similarity
Cross-Encoder (stsb-roberta-large)	Pairwise reranking
NER (hirly-ner-multi)	Skill extraction
NER Skill Matching
The NER module extracts `SKILL` and `LANG` entities from both the CV and the job description using `feliponi/hirly-ner-multi`. A two-pass comparison is then applied:
Exact matching after Porter stemming
Semantic matching on original skill tokens using SBERT embeddings (cosine threshold = 0.75)
Installation
```bash
pip install -r requirements.txt
```
Main dependencies:
`transformers`
`sentence-transformers`
`gensim`
`scikit-learn`
`spacy`
`pdfplumber`
`plotly`
`nltk`
```bash
python -m spacy download en\_core\_web\_sm
```
Usage
Open `notebook/NLP\_final.ipynb`
Set the path to your CV (PDF) and paste your job description
Run all cells
Get your similarity score and skill gap analysis
Report
The full project report is available in the `report/` folder, written in NeurIPS 2026 format.
License
MIT