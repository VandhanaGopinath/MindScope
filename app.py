"""
MindScope - Streamlit app

A 25-question short-form Big Five (OCEAN) quiz. Answers are fed straight
into the trained XGBoost models (models/model_<TRAIT>.joblib), which
predict what each trait's full 10-item score would have been.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

TRAITS = ["E", "N", "A", "C", "O"]
TRAIT_NAMES = {
    "E": "Extraversion",
    "N": "Neuroticism",
    "A": "Agreeableness",
    "C": "Conscientiousness",
    "O": "Openness",
}

OPTIONS = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
OPTION_SCORES = {opt: i + 1 for i, opt in enumerate(OPTIONS)}


@st.cache_resource
def load_models():
    feature_cols = joblib.load("models/feature_columns.joblib")
    reverse_key = joblib.load("models/reverse_key.joblib")
    models = {t: joblib.load(f"models/model_{t}.joblib") for t in TRAITS}
    return models, feature_cols, reverse_key


@st.cache_data
def load_questions(path="questions_short.txt"):
    questions = []
    with open(path) as f:
        for line in f:
            qid, text = line.strip().split("\t", 1)
            questions.append({"id": qid, "trait": qid[0], "text": text})
    return questions


def score_to_percentage(score):
    """1-5 scale -> 0-100%."""
    return (score - 1) / 4 * 100


def describe(trait, pct):
    high = pct >= 50
    descriptions = {
        "E": ("Outgoing and energized by people", "Reserved and independent"),
        "N": ("Sensitive to stress, feels things intensely", "Calm and emotionally steady"),
        "A": ("Cooperative, trusting, compassionate", "Direct, skeptical, competitive"),
        "C": ("Organized, dependable, disciplined", "Flexible, spontaneous, easy-going"),
        "O": ("Curious, imaginative, open to new ideas", "Practical, conventional, focused"),
    }
    return descriptions[trait][0 if high else 1]


def main():
    st.set_page_config(page_title="MindScope", page_icon="🧠", layout="centered")
    st.title("🧠 MindScope")
    st.caption(
        "A 25-question short form of the Big Five (OCEAN) personality test. "
        "A trained ML model estimates your full-length score from these 25 answers."
    )

    models, feature_cols, reverse_key = load_models()
    questions = load_questions()

    with st.form("quiz_form"):
        answers = {}
        for q in questions:
            answers[q["id"]] = st.radio(
                q["text"], OPTIONS, index=2, horizontal=True, key=q["id"]
            )
        submitted = st.form_submit_button("See my results")

    if submitted:
        raw = {qid: OPTION_SCORES[val] for qid, val in answers.items()}
        # apply the same reverse-scoring key used at training time
        scored = {
            qid: (6 - val) if reverse_key.get(qid, False) else val
            for qid, val in raw.items()
        }
        row = pd.DataFrame([{col: scored[col] for col in feature_cols}])

        st.subheader("Your results")
        cols = st.columns(5)
        for i, t in enumerate(TRAITS):
            pred = float(models[t].predict(row)[0])
            pct = score_to_percentage(pred)
            with cols[i]:
                st.metric(TRAIT_NAMES[t], f"{pct:.0f}%")
        st.divider()
        for t in TRAITS:
            pred = float(models[t].predict(row)[0])
            pct = score_to_percentage(pred)
            st.progress(min(max(pct / 100, 0.0), 1.0), text=f"{TRAIT_NAMES[t]} — {pct:.0f}%")
            st.caption(describe(t, pct))

        st.info(
            "These are model estimates of your full 50-item score, learned from "
            "~19,700 real survey responses — not a diagnostic assessment."
        )


if __name__ == "__main__":
    main()
