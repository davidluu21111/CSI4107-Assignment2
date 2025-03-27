# e5_model.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# --------------------------------------------------------------------------
# Load the "intfloat/e5-base" model and tokenizer for embeddings
# E5 uses prefix-based embeddings:
#   "query: <text>" for queries
#   "passage: <text>" for documents
# --------------------------------------------------------------------------
model_name = "intfloat/e5-base"
e5_tokenizer = AutoTokenizer.from_pretrained(model_name)
e5_model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
e5_model.to(device)
e5_model.eval()

def embed_e5(text, is_query) :
    """
    Embed the given text using the E5 model.
    If is_query=True, prefix the text with "query: ".
    Otherwise, prefix with "passage: ".

    Args:
        text (str): The text to embed (query or document).
        is_query (bool): Indicates whether text is a query or a passage.

    Returns:
        np.ndarray: A 1D array representing the [CLS] embedding (hidden_dim,).
    """
    prefix = "query: " if is_query else "passage: "
    prefixed_text = prefix + text.strip()

    # Tokenize with truncation at 512 tokens
    inputs = e5_tokenizer(
        prefixed_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    # Inference mode (no gradients)
    with torch.no_grad():
        outputs = e5_model(**inputs)

    # outputs.last_hidden_state shape: (batch_size=1, seq_len, hidden_dim)
    # E5 typically uses the [CLS] token embedding at index 0
    cls_emb = outputs.last_hidden_state[:, 0, :]
    return cls_emb.squeeze(0).cpu().numpy()

def e5_score(query_text, doc_text) :
    """
    Compute a relevance score between query_text and doc_text using E5 embeddings.

    1) Embed the query (prefixed with "query: ")
    2) Embed the document (prefixed with "passage: ")
    3) Calculate cosine similarity between the two embeddings.

    Args:
        query_text (str): The query string.
        doc_text   (str): The document string.

    Returns:
        float: The cosine similarity score (higher => more relevant).
    """
    q_emb = embed_e5(query_text, is_query=True)
    d_emb = embed_e5(doc_text, is_query=False)
    numerator = np.dot(q_emb, d_emb)
    denominator = (np.linalg.norm(q_emb) * np.linalg.norm(d_emb)) + 1e-10
    return float(numerator / denominator)

def rerank_e5(query_text, candidate_docs):
    """
    Re-rank candidate documents for a given query using E5 embeddings.

    For each document in candidate_docs:
      1) Compute e5_score(query_text, doc_text).
      2) Sort documents in descending order of the score.

    Args:
        query_text (str): The query string.
        candidate_docs (dict): A dictionary mapping doc_id -> doc_text.

    Returns:
        list: A list of (doc_id, score) tuples, sorted by descending score.
    """
    scores = {
        doc_id: e5_score(query_text, doc_text)
        for doc_id, doc_text in candidate_docs.items()
    }

    # Sort by descending similarity
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs
