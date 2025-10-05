import os
import pandas as pd
import string
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch

# ---------------------------
# Load CSV files from data folder
# ---------------------------
data_folder = "data"
files = ["train.csv", "test.csv", "validation.csv"]

df_list = []
for f in files:
    path = os.path.join(data_folder, f)
    if os.path.exists(path):
        df_list.append(pd.read_csv(path))

faq_data = pd.concat(df_list, ignore_index=True)

# Extract questions and answers
questions = faq_data["question"].astype(str).tolist()
answers = faq_data["answer"].astype(str).tolist()

# ---------------------------
# Text preprocessing
# ---------------------------
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

preprocessed_questions = [preprocess_text(q) for q in questions]

# ---------------------------
# Build Inverted Index
# ---------------------------
inverted_index = defaultdict(list)
for idx, question in enumerate(preprocessed_questions):
    for term in question.split():
        inverted_index[term].append(idx)

# ---------------------------
# TF-IDF Vectorizer
# ---------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)

# ---------------------------
# SBERT Model + Embeddings
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight, accurate model
sbert_embeddings = model.encode(preprocessed_questions, convert_to_tensor=True)

# ---------------------------
# Flask App
# ---------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_query = request.form["msg"]
    processed_query = preprocess_text(user_query)

    # --- Step 1: Candidate retrieval using inverted index ---
    candidate_indices = set()
    for term in processed_query.split():
        if term in inverted_index:
            candidate_indices.update(inverted_index[term])

    # Fallback: if no terms found
    if not candidate_indices:
        return jsonify({"response": "❌ Sorry, I don’t know the answer to that."})

    # --- Step 2: TF-IDF similarity ---
    query_vec = vectorizer.transform([processed_query])
    tfidf_similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # --- Step 3: SBERT semantic similarity ---
    query_embed = model.encode(processed_query, convert_to_tensor=True)
    sbert_similarity = util.cos_sim(query_embed, sbert_embeddings)[0].cpu().numpy()

    # --- Step 4: Combine both (weighted average) ---
    combined_scores = []
    for i in candidate_indices:
        combined_score = 0.6 * tfidf_similarity[i] + 0.4 * sbert_similarity[i]
        combined_scores.append((i, combined_score))

    # --- Step 5: Pick best answer ---
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    best_idx, best_score = combined_scores[0]

    # Confidence threshold
    if best_score < 0.25:
        return jsonify({"response": "❌ Sorry, I don’t know the answer to that."})

    best_answer = answers[best_idx]
    return jsonify({"response": best_answer})

if __name__ == "__main__":
    app.run(debug=True)
