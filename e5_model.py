# e5_model.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

model_name = "intfloat/e5-base"
e5_tokenizer = AutoTokenizer.from_pretrained(model_name)
e5_model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
e5_model.to(device)
e5_model.eval()

def embed_e5(text, is_query):
    prefix = "query: " if is_query else "passage: "
    prefixed_text = prefix + text.strip()
    inputs = e5_tokenizer(prefixed_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = e5_model(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :]
    return cls_emb.squeeze(0).cpu().numpy()

def e5_score(query_text, doc_text ) :
    q_emb = embed_e5(query_text, is_query=True)
    d_emb = embed_e5(doc_text, is_query=False)
    numerator = np.dot(q_emb, d_emb)
    denominator = (np.linalg.norm(q_emb) * np.linalg.norm(d_emb)) + 1e-10
    return float(numerator / denominator)

def rerank_e5(query_text, candidate_docs) :
    scores = {doc_id: e5_score(query_text, doc_text)
              for doc_id, doc_text in candidate_docs.items()}
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs
