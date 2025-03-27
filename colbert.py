# colbert.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Initialize dual encoder using "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
dual_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dual_model = AutoModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dual_model.to(device)
dual_model.eval()

def compute_token_embeddings(text):
    inputs = dual_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = dual_model(**inputs)
    token_embs = outputs.last_hidden_state.squeeze(0)  # Shape: (seq_len, hidden_dim)
    return token_embs.cpu().numpy()

def colbert_score(query_text, doc_text):
    q_token_embs = compute_token_embeddings(query_text)  # shape: (q_len, dim)
    d_token_embs = compute_token_embeddings(doc_text)    # shape: (d_len, dim)
    scores = []
    for q_emb in q_token_embs:
        # Compute cosine similarity between one query token and all doc tokens:
        sims = np.dot(d_token_embs, q_emb) / (np.linalg.norm(d_token_embs, axis=1) * np.linalg.norm(q_emb) + 1e-10)
        scores.append(np.max(sims))
    return np.mean(scores)  # Final relevance score is the average of max similarities

def rerank_colbert(query_text, candidate_docs):
    scores = {doc_id: colbert_score(query_text, doc_text)
              for doc_id, doc_text in candidate_docs.items()}
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs
