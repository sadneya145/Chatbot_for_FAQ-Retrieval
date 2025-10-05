import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

# Load dataset (train + validation)
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")
# Combine train + val for building FAQ base
faq_df = pd.concat([train_df, val_df])

questions = faq_df["question"].astype(str).tolist()
answers = faq_df["answer"].astype(str).tolist()

# Normalize text (lowercase, strip spaces)
questions = [q.lower().strip() for q in questions]
answers = [a.strip() for a in answers]

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode FAQs into embeddings
faq_embeddings = model.encode(questions, convert_to_tensor=True)

# Save data + embeddings
with open("models/faqs.pkl", "wb") as f:
    pickle.dump({
        "questions": questions,
        "answers": answers,
        "faq_embeddings": faq_embeddings.cpu().numpy()
    }, f)

print("✅ SBERT embeddings trained and saved!")