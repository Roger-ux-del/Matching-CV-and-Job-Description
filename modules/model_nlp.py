# model_nlp.py

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import pandas as pd


# SBERT 

_sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# CROSS-ENCODER 

_cross_encoder = CrossEncoder("cross-encoder/stsb-roberta-large")


def compute_crossencoder_score(text1, text2):
    """
    Score sémantique avancé entre deux textes (CV ↔ JD)
    """
    score = _cross_encoder.predict([(text1, text2)])[0]
    return float(score)



# NER
_ner_model_name = "Nucha/Nucha_SkillNER_BERT"
_ner_tokenizer = AutoTokenizer.from_pretrained(_ner_model_name)
_ner_model = AutoModelForTokenClassification.from_pretrained(_ner_model_name)
_ner_pipeline = pipeline("ner", model=_ner_model, tokenizer=_ner_tokenizer, aggregation_strategy="simple")


def load_skill_ner():
    return _ner_pipeline


def extract_skills(text, ner_pipeline):
    lines = str(text).split("\n")
    all_skills = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        entities = ner_pipeline(line[:512])

        skills = [
            e["word"].lower().strip()
            for e in entities
            if e["entity_group"] in ("HSKILL", "SSKILL")
            and e["score"] >= 0.7
        ]

        all_skills.extend(skills)

    return list(set(all_skills))



# COMPARAISON

def compare_skills(cv_skills, jd_skills):
    cv_set = set(s.lower().strip() for s in cv_skills)
    jd_set = set(s.lower().strip() for s in jd_skills)

    present = cv_set & jd_set
    missing = jd_set - cv_set

    return {
        "cv_skills": sorted(cv_set),
        "jd_skills": sorted(jd_set),
        "present": sorted(present),
        "missing": sorted(missing),
    }



# MAIN PIPELINE

def match_cv_jd(cv_text, jd_raw, method="crossencoder"):
    ner = load_skill_ner()

    #  Choix du modèle de scoring
    if method == "crossencoder":
        score = compute_crossencoder_score(cv_text, jd_raw)
    elif method == "sbert":
        cv_emb = _sentence_model.encode([cv_text])
        jd_emb = _sentence_model.encode([jd_raw])
        score = cosine_similarity(cv_emb, jd_emb)[0][0]
    else:
        raise ValueError("Méthode inconnue")

    print("Extraction des compétences du CV...")
    cv_skills_raw = extract_skills(cv_text, ner)

    print("Extraction des compétences de la JD...")
    jd_skills_raw = extract_skills(jd_raw, ner)

    # Mapping ESCO
    cv_skills = map_to_esco(cv_skills_raw)
    jd_skills = map_to_esco(jd_skills_raw)

    result = compare_skills(cv_skills, jd_skills)
    result["similarity_score"] = round(float(score), 4)

    print("\n" + "="*50)
    print("\n Compétences de l'offre :")
    print(result["jd_skills"])
    print("\n Compétences du CV :")
    print(result["cv_skills"])
    print("\n Compétences manquantes dans le CV :")
    print(result["missing"])
    print("="*50)

    return result