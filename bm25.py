# bm25.py
import re
import math
from collections import defaultdict
from nltk.stem import PorterStemmer

# Initialize Porter Stemmer
stemmer = PorterStemmer()


    
def preprocess_text(text, stopwords):

    '''
    This module preprocesses text by:
    1. Tokenizing the text.
    2. Converting all words to lowercase.
    3. Removing stopwords.
    4. Filtering out non-alphanumeric characters.
    5. Applying Porter Stemming.
    '''
    # Remove non-text markup (e.g., HTML tags or XML tags)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Tokenize and filter out punctuation and numbers
    tokens = re.findall(r'\b[a-z]+\b', text.lower())

    # Remove stopwords
    filtered_tokens = [token for token in tokens if token not in stopwords]

    # Apply Porter stemming
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    return stemmed_tokens

def preprocess_text_dataframe(df, text_column, stopwords):

    '''
        Input: Preprocessed tokens from documents.
        Output: 
        - An inverted index stored as a DataFrame
        - A document length dictionary which stores the number of tokens in each document
        - Average document length used for BM25 scoring

    '''

    df[text_column + "_tokens"] = df[text_column].apply(lambda x: preprocess_text(x, stopwords))# Get the list of tokens from the document
    return df

def build_inverted_index_with_stats(corpus_df, text_column):
    inverted_index = {}
    doc_lengths = defaultdict(int)
    for _, row in corpus_df.iterrows():
        doc_id = row["_id"]
        tokens = row[text_column]# Get the list of tokens from the document

        # Create a dictionary to store term frequencies in the document
        freq_map = defaultdict(int)
        for token in tokens:
            freq_map[token] += 1

        # store document length which is necessary for BM25 calculation
        doc_lengths[doc_id] = len(tokens)

        # Update the inverted index
        for token, tf in freq_map.items():
            if token not in inverted_index:
                inverted_index[token] = {"df": 0, "postings": {}}

            # if doc_id not present, increment df by 1
            if doc_id not in inverted_index[token]["postings"]:
                inverted_index[token]["df"] += 1

            inverted_index[token]["postings"][doc_id] = tf

    # Calculate the average document length (avgdl)
    total_length = sum(doc_lengths.values())
    avgdl = float(total_length) / len(doc_lengths) if doc_lengths else 0.0


    return inverted_index, doc_lengths, avgdl

def bm25_score(tf, df, dl, avgdl, N, k1, b):
    """
        Computes the BM25 score based on  
        https://www.site.uottawa.ca/~diana/csi4107/BM25.pdf
    """

    # IDF
    idf = math.log((N - df + 0.5) / (df + 0.5))

    # TF saturation
    numerator = tf * idf * (k1 + 1)
    denominator = k1 * ((1 - b) + b * (dl / avgdl)) + tf
    return numerator / denominator

def retrieve_and_rank_bm25(query_tokens, inverted_index, doc_lengths, avgdl, N,
                           k1, b):
    # Create a dictionary to accumulate BM25 scores for each doc_id.
    scores = defaultdict(float)

    # For each query token, count how often each term appears in the query.
    q_freq = defaultdict(int)
    for qt in query_tokens:
        q_freq[qt] += 1

    # This loop calculates BM25 scores for each term in the query.
    for qt, qf in q_freq.items():
        # If the query term is not in the index, skip it.
        if qt not in inverted_index:
            continue

        # Get the document frequency for the term
        df = inverted_index[qt]["df"]
        # Get the postings list (mapping doc_id -> term frequency in that doc)
        postings = inverted_index[qt]["postings"]

        # Calculate BM25 for each document where this term appears
        for doc_id, tf in postings.items():
            dl = doc_lengths[doc_id]  # The length of the current document
            score = bm25_score(tf, df, dl, avgdl, N, k1, b)
            # Multiply by qf to account for repeated query terms (if desired)
            scores[doc_id] += score * qf

    # Sort documents by their accumulated BM25 scores in descending order
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs

def generate_results_file_bm25(queries_df, inverted_index, doc_lengths, avgdl, output_file, run_name, k1, b):
    N = len(doc_lengths)
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in queries_df.iterrows():
            query_id = row["_id"]
            query_tokens = row["text_tokens"]
            ranked_docs = retrieve_and_rank_bm25(query_tokens, inverted_index, doc_lengths, avgdl, N, k1, b)
            for rank, (doc_id, score) in enumerate(ranked_docs[:100], start=1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}\n")
