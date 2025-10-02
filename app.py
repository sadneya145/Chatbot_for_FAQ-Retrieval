import os
import pandas as pd
import string
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

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
# Build Inverted Index (Explicit)
# ---------------------------
inverted_index = defaultdict(list)

for idx, question in enumerate(preprocessed_questions):
    for term in question.split():
        inverted_index[term].append(idx)

# (Optional) Print a small sample to verify
# print("Sample inverted index entries:")
# for word in list(inverted_index.keys())[:10]:
#     print(f"{word}: {inverted_index[word]}")

# ---------------------------
# TF-IDF vectorizer (for ranking)
# ---------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_query = request.form["msg"]
    processed_query = preprocess_text(user_query)

    # Lookup candidate docs from inverted index
    candidate_indices = set()
    for term in processed_query.split():
        if term in inverted_index:
            candidate_indices.update(inverted_index[term])

    if not candidate_indices:
        return jsonify({"response": "❌ Sorry, I don’t know the answer to that."})

    # Restrict TF-IDF similarity only to candidate docs
    query_vec = vectorizer.transform([processed_query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Filter scores to candidate docs only
    candidate_scores = [(i, similarity[i]) for i in candidate_indices]
    candidate_scores.sort(key=lambda x: x[1], reverse=True)

    best_idx, best_score = candidate_scores[0]
    if best_score < 0.2:  # threshold
        return jsonify({"response": "❌ Sorry, I don’t know the answer to that."})

    best_answer = answers[best_idx]
    return jsonify({"response": best_answer})

if __name__ == "__main__":
    app.run(debug=True)
