# colbert.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# --------------------------------------------------------------------------
# Initialize dual encoder using "sentence-transformers/all-MiniLM-L6-v2"
# This model provides token-level embeddings for a simplified ColBERT-style approach.
# --------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
dual_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dual_model = AutoModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dual_model.to(device)
dual_model.eval()

def compute_token_embeddings(text):
    """
    Convert an input text into token-level embeddings.

    Args:
        text (str): The text to be encoded.

    Returns:
        np.ndarray: A 2D array of shape (seq_len, hidden_dim) representing
                    the token embeddings from the model's last hidden state.
    """
    inputs = dual_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = dual_model(**inputs)

    # outputs.last_hidden_state shape: (batch_size=1, seq_len, hidden_dim)
    token_embs = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
    return token_embs.cpu().numpy()

def colbert_score(query_text, doc_text):
    """
    Compute a ColBERT-style relevance score between a query and a document.

    This approach:
    1. Obtains token-level embeddings for both query and document.
    2. For each query token, finds the maximum similarity with any doc token.
    3. Averages these max similarities across all query tokens.

    Args:
        query_text (str): The query string.
        doc_text   (str): The document string.

    Returns:
        float: A single scalar relevance score (the higher, the more relevant).
    """
    # 1) Compute token embeddings for the query and the document
    q_token_embs = compute_token_embeddings(query_text)  # (q_len, hidden_dim)
    d_token_embs = compute_token_embeddings(doc_text)    # (d_len, hidden_dim)

    # 2) For each query token embedding, compute max cosine similarity across all doc token embeddings
    scores = []
    for q_emb in q_token_embs:
        # Dot product with each doc token
        numerator = np.dot(d_token_embs, q_emb)  # shape: (d_len,)
        # Norm product for each doc token embedding and the query token embedding
        denominator = (np.linalg.norm(d_token_embs, axis=1) * np.linalg.norm(q_emb)) + 1e-10
        sims = numerator / denominator  # shape: (d_len,)

        # Take the maximum similarity for this particular query token
        scores.append(np.max(sims))

    # 3) The final score is the mean of these max similarities
    return np.mean(scores)

def rerank_colbert(query_text, candidate_docs):
    """
    Re-rank candidate documents for a given query using the ColBERT-style scoring.

    Args:
        query_text (str): The query string.
        candidate_docs (dict): A dictionary mapping doc_id -> document text.

    Returns:
        list: A list of (doc_id, score) tuples, sorted in descending order of score.
    """
    # Compute a ColBERT-style score for each candidate doc
    scores = {
        doc_id: colbert_score(query_text, doc_text)
        for doc_id, doc_text in candidate_docs.items()
    }

    # Sort documents by descending score
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs
