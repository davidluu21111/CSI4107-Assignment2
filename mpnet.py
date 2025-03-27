# mpnet.py
"""
mpnet.py

This module implements re-ranking functions using the
"multi-qa-mpnet-base-cos-v1" model from Sentence Transformers.

Functions:
    embed_text(text): Returns the embedding for the given text.
    mpnet_score(query_text, doc_text): Computes cosine similarity between query and doc embeddings.
    rerank_mpnet(query_text, candidate_docs): Re-ranks candidate documents using cosine similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util

# Initialize the SentenceTransformer model
MODEL_NAME = "multi-qa-mpnet-base-cos-v1"
model = SentenceTransformer(MODEL_NAME)

def embed_text(text):
    """
    Embed the input text using the multi-qa-mpnet-base-cos-v1 model.

    Args:
        text (str): The input text (query or document).

    Returns:
        np.ndarray: The embedding vector for the text.
    """

    # The model.encode method returns a NumPy array when convert_to_numpy=True.
    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding

def mpnet_score(query_text, doc_text):
    """
    Compute the cosine similarity between query and document embeddings
    using the multi-qa-mpnet-base-cos-v1 model.

    Args:
        query_text (str): The query string.
        doc_text   (str): The document string.

    Returns:
        float: Cosine similarity score (higher indicates higher relevance).
    """
    q_emb = embed_text(query_text)
    d_emb = embed_text(doc_text)
    sim = util.cos_sim(q_emb, d_emb)
    return float(sim)

def rerank_mpnet(query_text, candidate_docs):
    """
    Re-rank candidate documents using the multi-qa-mpnet-base-cos-v1 model.

    Args:
        query_text (str): The query string.
        candidate_docs (dict): A dictionary mapping doc_id -> document text.

    Returns:
        list: A list of (doc_id, score) tuples sorted in descending order by score.
    """
    scores = {
        doc_id: mpnet_score(query_text, doc_text)
        for doc_id, doc_text in candidate_docs.items()
    }
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs
