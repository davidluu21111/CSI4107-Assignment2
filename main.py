# main.py
import os
os.environ["HF_HOME"] = "D:/huggingface_cache"

import pandas as pd
import requests
from collections import defaultdict
import pytrec_eval

# Import our modules
from bm25 import preprocess_text_dataframe, build_inverted_index_with_stats, generate_results_file_bm25, retrieve_and_rank_bm25
from colbert import rerank_colbert
from e5_model import rerank_e5

# Helper function to load qrels and TREC results for evaluation
def load_qrels_tsv(filepath):
    df = pd.read_csv(filepath, sep="\t", dtype=str)
    results = {}
    for _, row in df.iterrows():
        query_id = f"q{row['query-id']}"
        doc_id = f"d{row['corpus-id']}"
        score = int(row["score"])
        if query_id not in results:
            results[query_id] = {}
        results[query_id][doc_id] = score
    return results

def load_trec_results(filepath):
    run = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            query_id, _, doc_id, _, score, _ = parts
            query_id = f"q{query_id}"
            doc_id = f"d{doc_id}"
            run.setdefault(query_id, {})[doc_id] = float(score)
    return run

# Generate neural results using a generic function
def generate_neural_results(queries_df, inverted_index, doc_lengths, avgdl, corpus_df,
                            output_file, run_name, k1, b, rerank_func, candidate_field="title_text"):
    N = len(doc_lengths)
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in queries_df.iterrows():
            query_id = row["_id"]
            query_tokens = row["text_tokens"]
            query_text = row["text"]
            bm25_ranked = retrieve_and_rank_bm25(query_tokens, inverted_index, doc_lengths, avgdl, N, k1, b)
            top_candidates = bm25_ranked[:100]
            candidate_docs = {}
            for doc_id, _ in top_candidates:
                doc_row = corpus_df[corpus_df["_id"] == doc_id].iloc[0]
                if candidate_field == "title":
                    candidate_docs[doc_id] = doc_row["title"]
                else:  # "title_text"
                    candidate_docs[doc_id] = doc_row["title"] + " " + doc_row["text"]
            neural_ranked = rerank_func(query_text, candidate_docs)
            for rank, (doc_id, score) in enumerate(neural_ranked[:100], start=1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}\n")

# --- Main Execution ---

# 1. Fetch stopwords
stopwords_url = "https://www.site.uottawa.ca/~diana/csi4107/StopWords"
r = requests.get(stopwords_url)
stopwords = set(r.text.splitlines()) if r.status_code == 200 else set()

# 2. Load corpus and queries
corpus_file_path = "./scifact/corpus.jsonl"
queries_file_path = "./scifact/queries.jsonl"
corpus_df = pd.read_json(corpus_file_path, lines=True)
queries_df = pd.read_json(queries_file_path, lines=True)

# Preprocess queries and corpus
queries_df = preprocess_text_dataframe(queries_df, "text", stopwords)
corpus_df = preprocess_text_dataframe(corpus_df, "title", stopwords)
corpus_df = preprocess_text_dataframe(corpus_df, "text", stopwords)
# Create combined field for Title+Text
corpus_df["title_text"] = corpus_df["title"] + " " + corpus_df["text"]
corpus_df = preprocess_text_dataframe(corpus_df, "title_text", stopwords)

# 3. BM25 Candidate Generation

#  BM25 using Title+Text
inverted_index_tt, doc_lengths_tt, avgdl_tt = build_inverted_index_with_stats(corpus_df, "title_text_tokens")
generate_results_file_bm25(queries_df, inverted_index_tt, doc_lengths_tt, avgdl_tt,
                           "./Results_BM25_TitleAndText.txt", "BM25_TitleAndText", k1=1.2, b=0.75)
print("BM25 (Title+Text) results generated.")

# 4. Neural Re-ranking with E5 (Title+Text)
generate_neural_results(queries_df, inverted_index_tt, doc_lengths_tt, avgdl_tt, corpus_df,
                        "./Results_E5_TitleAndText.txt", "E5_TitleAndText", k1=1.2, b=0.75,
                        rerank_func=rerank_e5, candidate_field="title_text")
print("E5 re-ranking (Title+Text) results generated.")

# 5. Neural Re-ranking with ColBERT (Title+Text)
generate_neural_results(queries_df, inverted_index_tt, doc_lengths_tt, avgdl_tt, corpus_df,
                        "./Results_ColBERT_TitleAndText.txt", "ColBERT_TitleAndText", k1=1.2, b=0.75,
                        rerank_func=rerank_colbert, candidate_field="title_text")
print("ColBERT re-ranking (Title+Text) results generated.")

# 6. Evaluation using pytrec_eval (MAP and P@10)
metrics = {"map", "P_10"}
qrels = load_qrels_tsv("./scifact/qrels/test.tsv")
evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)

results_bm25_tt = load_trec_results("./Results_BM25_TitleAndText.txt")
results_e5_tt = load_trec_results("./Results_E5_TitleAndText.txt")
results_colbert_tt = load_trec_results("./Results_ColBERT_TitleAndText.txt")

def compute_avg_results(eval_results):
    avg_results = {}
    for metric in metrics:
        values = [r[metric] for r in eval_results.values() if metric in r]
        avg_results[metric] = sum(values)/len(values) if values else 0.0
    return avg_results


eval_bm25_tt = evaluator.evaluate(results_bm25_tt)
eval_e5_tt = evaluator.evaluate(results_e5_tt)
eval_colbert_tt = evaluator.evaluate(results_colbert_tt)


print("BM25 (Title+Text) Evaluation:", compute_avg_results(eval_bm25_tt))
print("E5 (Title+Text) Evaluation:", compute_avg_results(eval_e5_tt))
print("ColBERT (Title+Text) Evaluation:", compute_avg_results(eval_colbert_tt))
