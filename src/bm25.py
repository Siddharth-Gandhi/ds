import json
import os

import numpy as np
import pandas as pd
from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher
from sklearn.metrics import average_precision_score, ndcg_score

from utils import count_commits, get_combined_df, tokenize

# Configuration for tiktoken encoding
# ENCODING = 'p50k_base'
# enc = tiktoken.get_encoding(ENCODING)
# assert enc.decode(enc.encode("hello world")) == "hello world"




def search(searcher, query, query_date, k=1000):
    hits = searcher.search(tokenize(query), k)
    unix_date = query_date
    filtered_hits = [hit for hit in hits if int(json.loads(hit.raw)["commit_date"]) < unix_date]
    return filtered_hits

def evaluate(searcher, query, query_date, actual_modified_files, k=1000):
    hits = search(searcher, query, query_date, k)
    retrieved_files = [json.loads(hit.raw)['file_path'] for hit in hits]
    relevant = [1 if file in actual_modified_files else 0 for file in retrieved_files]

    metrics = {
        'MAP': average_precision_score(relevant, [1]*len(relevant)) if any(relevant) else 0,
        'P@10': precision_at_k(relevant, 10),
        'P@100': precision_at_k(relevant, 100),
        'P@1000': precision_at_k(relevant, 1000),
        'MRR': mean_reciprocal_rank(relevant),
        f'Recall@{k}': len(set(file for idx, file in enumerate(retrieved_files) if relevant[idx] == 1)) / len(actual_modified_files)
    }

    return {k: round(v, 4) for k, v in metrics.items()}

def precision_at_k(relevant, k):
    return sum(relevant[:k]) / k

def mean_reciprocal_rank(relevant):
    for idx, value in enumerate(relevant):
        if value == 1:
            return 1 / (idx + 1)
    return 0

def evaluate_sampling(repo_dir, idx_path, n=100, k=1000, output_file='bm25_metrics.txt', filter_merge_requests=False):
    if not os.path.exists(idx_path):
        print(f"Index at {idx_path} does not exist! Exiting...")
        return
    combined_df = get_combined_df(repo_dir)
    total_commits = combined_df.commit_id.nunique()
    if total_commits < n:
        print(f'Not enough commits to sample for {repo_dir}, skipping...')
        return

    searcher = LuceneSearcher(idx_path)
    index_reader = IndexReader(idx_path)
    index_stats = index_reader.stats()
    # print(index_reader.stats())
    total_commits = count_commits(repo_dir)

    # filter out commits that are merge_requests by checking is_merge_request column

    if filter_merge_requests:
        print("Filtering out merge requests during sampling...")
        combined_df = combined_df[combined_df['is_merge_request'] == False]

    sampled_commits = combined_df.drop_duplicates(subset='commit_id').sample(n, replace=False, random_state=42)
    results = [
        evaluate(searcher, row['commit_message'], row['commit_date'],
                 combined_df[combined_df['commit_id'] == row['commit_id']]['file_path'].tolist(), k)
        for _, row in sampled_commits.iterrows()
    ]

    avg_scores = {
        metric: round(np.mean([result[metric] for result in results]), 4)
        for metric in results[0]
    }

    with open(os.path.join(repo_dir, output_file), "w") as file:
        file.write(f"Repo Path: {repo_dir}\n")
        file.write(f'Total Commits Stored: {total_commits}\n')
        file.write(f'Total Rows Stored: {combined_df.shape[0]}\n')
        file.write(f"Index Path: {idx_path}\n")
        file.write(f"Sample Size: {n}\n")
        file.write(f"Index Stats: {index_stats}\n")
        file.write("Evaluation Metrics:\n")
        for key, value in avg_scores.items():
            file.write(f"{key}: {value}\n")
    return avg_scores

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("repo_path", help="Path to the repository directory")
    parser.add_argument("index_path", help="Path to the index directory")
    parser.add_argument("--n", type=int, default=100, help="Evaluation sample size (default: 100)")
    parser.add_argument("--k", type=int, default=1000, help="Top k results to be evaluated (default: 1000)")
    # parser.add_argument("--use_tokenizer", action="store_true", help="Whether to use the tokenizer.")
    # parser.add_argument("--content_option", choices=["commit", "code", "both"], required=True, help="Content option: commit, code, or both.")
    parser.add_argument("--output", default='bm25_metrics.txt', help="Output file path (default: path/to/repo/bm25_metrics.txt)")
    parser.add_argument("--filter_merge_requests", action="store_true", help="Whether to filter out merge requests.")
    args = parser.parse_args()

    # check if content_option was not passed, if so raise warning that we use commit as default
    # if args.content_option is None:
    #     print("No content option was passed, using commit as default")
    #     args.content_option = "commit"

    # idx_string = f'index_{args.content_option}'
    # if args.use_tokenizer:
    #     idx_string += '_tokenized'
    # idx_path = os.path.join(args.repo_path, idx_string)
    # parser.add_argument("--omit_merges", action='store_true', help="Omit merge commits")
    # args = parser.parse_args()
    # print(args)
    # print(idx_string)
    evaluate_sampling(args.repo_path, args.index_path, args.n, args.k, args.output, args.filter_merge_requests)