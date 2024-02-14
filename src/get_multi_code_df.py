import os
import sys

import pandas as pd

from bm25_v2 import BM25Searcher
from eval import ModelEvaluator, SearchEvaluator
from utils import get_code_df, get_combined_df

if __name__ == "__main__":
    metrics =['MAP', 'P@1', 'P@10', 'P@20', 'P@30', 'MRR', 'R@1', 'R@10', 'R@100', 'R@1000']
    repo_paths = [
        "data/2_7/apache_spark",
        "data/2_7/apache_kafka",
        "data/2_8/angular_angular",
        "data/2_8/django_django",
        "data/2_8/pytorch_pytorch",
        "data/2_7/julialang_julia",
        "data/2_7/ruby_ruby",
        "data/2_9/huggingface_transformers",
        "data/2_9/redis_redis",
        "data/2_7/facebook_react",
    ]
    code_df_list = []
    print(repo_paths)
    for repo_path in repo_paths:
        repo_name = repo_path.split('/')[-1]
        print(f'processing {repo_name}')
        index_path = os.path.join(repo_path, 'index_commit_tokenized')
        K = 1000 # args.k
        n = 100 # args.n
        combined_df = get_combined_df(repo_path)
        BM25_AGGR_STRAT = 'sump'
        eval_path = os.path.join(repo_path, 'eval')
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)

        bm25_searcher = BM25Searcher(index_path)
        evaluator = SearchEvaluator(metrics)
        model_evaluator = ModelEvaluator(bm25_searcher, evaluator, combined_df)

        gold_df_path = os.path.join('gold', repo_name, f'v2_{repo_name}_gpt4_train.parquet')

        recent_df = pd.read_parquet(gold_df_path)
        recent_df = recent_df.rename(columns={'commit_message': 'original_message', 'transformed_message_gpt4': 'commit_message'})
        # cache_path = f'{repo_name}_code_df.parquet'
        cache_path = os.path.join('merged_code_df', f'{repo_name}_code_df.parquet')
        code_df = get_code_df(recent_df, bm25_searcher, 1000, 10, 10, combined_df, cache_path, False)
        code_df_list.append(code_df)

    multi_code_df = pd.concat(code_df_list, ignore_index=True)
    multi_code_df.to_parquet(os.path.join('merged_code_df', 'multi_code_df.parquet'))