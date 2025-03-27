# bm25.py
import re
import math
from collections import defaultdict
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def preprocess_text(text, stopwords):
    text = re.sub(r'<[^>]+>', ' ', text)
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords]
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens

def preprocess_text_dataframe(df, text_column, stopwords):
    df[text_column + "_tokens"] = df[text_column].apply(lambda x: preprocess_text(x, stopwords))
    return df

def build_inverted_index_with_stats(corpus_df, text_column):
    inverted_index = {}
    doc_lengths = defaultdict(int)
    for _, row in corpus_df.iterrows():
        doc_id = row["_id"]
        tokens = row[text_column]
        freq_map = defaultdict(int)
        for token in tokens:
            freq_map[token] += 1
        doc_lengths[doc_id] = len(tokens)
        for token, tf in freq_map.items():
            if token not in inverted_index:
                inverted_index[token] = {"df": 0, "postings": {}}
            if doc_id not in inverted_index[token]["postings"]:
                inverted_index[token]["df"] += 1
            inverted_index[token]["postings"][doc_id] = tf
    total_length = sum(doc_lengths.values())
    avgdl = float(total_length) / len(doc_lengths) if doc_lengths else 0.0
    return inverted_index, doc_lengths, avgdl

def bm25_score(tf, df, dl, avgdl, N, k1, b):
    idf = math.log((N - df + 0.5) / (df + 0.5))
    numerator = tf * idf * (k1 + 1)
    denominator = k1 * ((1 - b) + b * (dl / avgdl)) + tf
    return numerator / denominator

def retrieve_and_rank_bm25(query_tokens, inverted_index, doc_lengths, avgdl, N, k1, b):
    scores = defaultdict(float)
    q_freq = defaultdict(int)
    for qt in query_tokens:
        q_freq[qt] += 1
    for qt, qf in q_freq.items():
        if qt not in inverted_index:
            continue
        df = inverted_index[qt]["df"]
        postings = inverted_index[qt]["postings"]
        for doc_id, tf in postings.items():
            dl = doc_lengths[doc_id]
            score = bm25_score(tf, df, dl, avgdl, N, k1, b)
            scores[doc_id] += score * qf
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def generate_results_file_bm25(queries_df, inverted_index, doc_lengths, avgdl, output_file, run_name, k1, b):
    N = len(doc_lengths)
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in queries_df.iterrows():
            query_id = row["_id"]
            query_tokens = row["text_tokens"]
            ranked_docs = retrieve_and_rank_bm25(query_tokens, inverted_index, doc_lengths, avgdl, N, k1, b)
            for rank, (doc_id, score) in enumerate(ranked_docs[:100], start=1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}\n")
