import argparse
import json
import os
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch
from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import count_commits, get_combined_df, reverse_tokenize, tokenize


class SearchResult:
    def __init__(self, commit_id, file_path, score, commit_date, commit_msg):
        self.commit_id = commit_id
        self.file_path = file_path
        self.score = score
        self.commit_date = commit_date
        self.commit_msg = commit_msg


    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(score={self.score:.5f}, file_path={self.file_path!r}, commit_id={self.commit_id!r}, commit_date={self.commit_date})"

    def is_actual_modified(self, actual_modified_files):
        return self.file_path in actual_modified_files

    @staticmethod
    def print_results(query, search_results, show_only_actual_modified=False):
        actual_modified_files = query['actual_files_modified']
        for i, result in enumerate(search_results):
            if show_only_actual_modified and not result.is_actual_modified(actual_modified_files):
                continue
            print(f"{i+1:2} {result}")

class AggregatedSearchResult:
    def __init__(self, file_path, aggregated_score, contributing_results):
        self.file_path = file_path
        self.score = aggregated_score
        self.contributing_results = contributing_results

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(file_path={self.file_path!r}, score={self.score}, " \
               f"contributing_results={self.contributing_results})"


class BM25Search:
    def __init__(self, index_path):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index at {index_path} does not exist!")
        self.searcher = LuceneSearcher(index_path)
        print(f"Loaded index at {index_path}")
        print(f'Index Stats: {IndexReader(index_path).stats()}')
        # self.ranking_depth = ranking_depth

    def search(self, query, query_date, ranking_depth):
        # TODO maybe change this to mean returning reranking_depths total results instead of being pruned by the query date
        hits = self.searcher.search(tokenize(query), ranking_depth)
        unix_date = query_date
        filtered_hits = [
            SearchResult(hit.docid, json.loads(hit.raw)['file_path'], hit.score, int(json.loads(hit.raw)["commit_date"]), reverse_tokenize(json.loads(hit.raw)['contents']))
            for hit in hits if int(json.loads(hit.raw)["commit_date"]) < unix_date
        ]
        return filtered_hits

    def search_full(self, query, query_date, ranking_depth):
        filtered_hits = []
        step_size = ranking_depth  # Initial search window
        total_hits_retrieved = 0

        while len(filtered_hits) < ranking_depth and step_size > 0:
            current_hits = self.searcher.search(tokenize(query), total_hits_retrieved + step_size)
            if not current_hits:
                break  # No more results to retrieve

            # Filter hits by query date
            for hit in current_hits:
                if int(json.loads(hit.raw)["commit_date"]) < query_date:
                    filtered_hits.append(
                        SearchResult(hit.docid, json.loads(hit.raw)['file_path'], hit.score,
                                     int(json.loads(hit.raw)["commit_date"]),
                                     reverse_tokenize(json.loads(hit.raw)['contents']))
                    )
                if len(filtered_hits) == ranking_depth:
                    break  # We have enough results

            total_hits_retrieved += step_size
            step_size = ranking_depth - len(filtered_hits)  # Decrease step size to only get as many as needed

        return filtered_hits[:ranking_depth]  # Return up to ranking_depth results

    def aggregate_file_scores(self, search_results, aggregation_method='sump'):
        file_to_results = defaultdict(list)
        for result in search_results:
            file_to_results[result.file_path].append(result)

        aggregated_results = []
        for file_path, results in file_to_results.items():
            # aggregated_score = sum(result.score for result in results)
            if aggregation_method == 'sump':
                aggregated_score = sum(result.score for result in results)
            elif aggregation_method == 'maxp':
                aggregated_score = max(result.score for result in results)
            # elif aggregation_method == 'firstp':
            #     aggregated_score = results[0].score
            elif aggregation_method == 'avgp':
                aggregated_score = np.mean([result.score for result in results])
            else:
                raise ValueError(f"Unknown aggregation method {aggregation_method}")

            aggregated_results.append(AggregatedSearchResult(file_path, aggregated_score, results))

        aggregated_results.sort(key=lambda result: result.score, reverse=True)
        return aggregated_results

    def pipeline(self, query, query_date, ranking_depth, aggregation_method):
        search_results = self.search(query, query_date, ranking_depth)
        if aggregation_method is not None:
            aggregated_results = self.aggregate_file_scores(search_results, aggregation_method)
            return aggregated_results
        return search_results


class SearchEvaluator:
    def __init__(self, metrics):
        self.metrics = metrics

    @staticmethod
    def precision_at_k(relevant, k):
        return sum(relevant[:k]) / k

    @staticmethod
    def mean_reciprocal_rank(relevant):
        for idx, value in enumerate(relevant):
            if value == 1:
                return 1 / (idx + 1)
        return 0

    @staticmethod
    def calculate_average_precision(relevant):
        pred_rel = [1] * len(relevant)
        relevant_documents_count = 0
        cumulative_precision = 0.0

        # We iterate through the predicted relevance scores
        for i in range(len(pred_rel)):
            # Check if the prediction at this rank is correct (i.e., if it is a relevant document)
            if pred_rel[i] == 1 and relevant[i] == 1:
                relevant_documents_count += 1
                precision_at_i = relevant_documents_count / (i + 1)
                cumulative_precision += precision_at_i

        # The average precision is the cumulative precision divided by the number of relevant documents
        average_precision = cumulative_precision / sum(relevant) if sum(relevant) > 0 else 0
        return average_precision

    # @staticmethod
    # def calculate_recall(relevant, total_modified_files, k):
    #   # Does not work for commit based approach as it can have multiple mentions of the same file across commits leading to a higher than 1 recall
    #     print(total_modified_files)
    #     print(relevant)
    #     return sum(relevant[:k]) / total_modified_files

    @staticmethod
    def calculate_recall(retrieved_files, actual_modified_files, relevant, k):
        # this complicated mess is required as compared to the above much simpler code to support both commit-based and file-based approaches
        # in file-based approach, this is equivalent to the above code
        # in code-based approach, duplicates could be present in retrieved_files, which is why we need to filter them out (the above code would not work in this case)

        return len({file for idx, file in enumerate(retrieved_files[:k])
                        if relevant[idx] == 1
                    }) / len(actual_modified_files) if len(actual_modified_files) > 0 else 0


    def evaluate(self, search_results, actual_modified_files):
        retrieved_files = [result.file_path for result in search_results]
        relevant = [1 if file in actual_modified_files else 0 for file in retrieved_files]

        evaluations = {}
        for metric in self.metrics:
            if metric == 'MAP':
                evaluations[metric] = self.calculate_average_precision(relevant)
            elif metric == 'MRR':
                evaluations[metric] = self.mean_reciprocal_rank(relevant)
            elif metric.startswith('P@'):
                k = int(metric.split('@')[1])
                evaluations[metric] = self.precision_at_k(relevant, k)
            elif metric.startswith('Recall@'):
                k = int(metric.split('@')[1])
                evaluations[metric] = self.calculate_recall(retrieved_files, actual_modified_files, relevant, k)

        return {k: round(v, 4) for k, v in evaluations.items()}

class ModelEvaluator:
    def __init__(self, model, eval_model, combined_df, seed=42):
        self.model = model
        self.eval_model = eval_model
        self.combined_df = combined_df
        self.seed = seed

    def sample_commits(self, n):
        if self.combined_df.commit_id.nunique() < n:
            raise ValueError(f'Not enough commits to sample. Required: {n}, available: {self.combined_df.commit_id.nunique()}')

        midpoint_date = np.median(self.combined_df['commit_date'])
        recent_df = self.combined_df[self.combined_df['commit_date'] > midpoint_date]

        return recent_df.drop_duplicates(subset='commit_id').sample(n=n, replace=False, random_state=self.seed)

    def evaluate_sampling(self, n=100, k=1000, output_file='metrics.txt', skip_existing=False, aggregation_strategy=None, rerankers=None):
        if rerankers is None:
            rerankers = []

        output_file_path = os.path.join(repo_path, output_file)
        model_name = self.model.__class__.__name__
        # output_file = f"{output_dir}/{model_name}_metrics.txt"

        if skip_existing and os.path.exists(output_file):
            print(f'Output file {output_file} already exists, skipping...')
            return

        sampled_commits = self.sample_commits(n)

        results = []
        # for _, row in sampled_commits.iterrows():
        for _, row in tqdm(sampled_commits.iterrows(), total=sampled_commits.shape[0]):
            # search_results = self.model.search(row['commit_message'], row['commit_date'], ranking_depth=k)
            # TODO: Add ChatGPT based query modification here
            cur_query = row['commit_message']
            search_results = self.model.pipeline(cur_query, row['commit_date'], ranking_depth=k, aggregation_method=aggregation_strategy)
            for reranker in rerankers:
                search_results = reranker.rerank_pipeline(cur_query, search_results)
            evaluation = self.eval_model.evaluate(search_results,
                                                       self.combined_df[self.combined_df['commit_id'] == row['commit_id']]['file_path'].tolist())
            results.append(evaluation)

        avg_scores = {metric: round(np.mean([result[metric] for result in results]), 4) for metric in results[0]}

        # os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
        with open(output_file_path, "w") as file:
            file.write(f"Model Name: {model_name}\n")
            file.write(f"Sample Size: {n}\n")
            file.write("Evaluation Metrics:\n")
            for key, value in avg_scores.items():
                file.write(f"{key}: {value}\n")

        print(f'Evaluation results written to {output_file_path}')

        return avg_scores

class BERTReranker:
    # def __init__(self, model_name, psg_len, psg_cnt, psg_stride, agggreagtion_strategy, batch_size, use_gpu=True):
    def __init__(self, parameters):
        self.parameters = parameters
        self.model_name = parameters['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() and parameters['use_gpu'] else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

        print(f'Using device: {self.device}')

        if torch.cuda.is_available() and parameters['use_gpu']:
            # print GPU info
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f'GPU Device Count: {torch.cuda.device_count()}')
            print(f"GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")


        self.psg_len = parameters['psg_len']
        self.psg_cnt = parameters.get('psg_cnt', None)
        # self.psg_stride = parameters.get('psg_stride', self.psg_len)
        self.aggregation_strategy = parameters['aggregation_strategy']
        self.batch_size = parameters['batch_size']
        # self.max_title_len = parameters.get('max_title_len', 0)
        # self.use_title = self.max_title_len > 0
        self.rerank_depth = parameters.get('rerank_depth', 100)
        # self.max_seq_length = parameters.get('max_seq_length', 512)
        self.max_seq_length = self.tokenizer.model_max_length

        print(f"Initialized BERT reranker with parameters: {parameters}")

    def rerank(self, query, aggregated_results: List[AggregatedSearchResult]):
        """
        Rerank the BM25 aggregated search results using BERT model scores.

        query: The issue query string.
        aggregated_results: A list of AggregatedSearchResult objects from BM25 search.
        """
        aggregated_results = aggregated_results[:self.rerank_depth]
        print(f'Reranking {len(aggregated_results)} results')

        # Flatten the list of results into a list of (query, passage) pairs but only keep max psg_cnt passages per file
        query_passage_pairs = []
        for agg_result in aggregated_results:
            for result in agg_result.contributing_results[:self.psg_cnt]:
                query_passage_pairs.append((query, result.commit_msg))

        print(f'Flattened query passage pairs: {len(query_passage_pairs)}')
        # query_passage_pairs = [(query, result.commit_msg) for aggregated_result in aggregated_results for result in aggregated_result.contributing_results]

        # print('Flattened query passage pairs')

        # tokenize the query passage pairs
        encoded_pairs = [self.tokenizer.encode_plus([query, passage], max_length=self.max_seq_length, truncation=True, padding='max_length', return_tensors='pt', add_special_tokens=True) for query, passage in query_passage_pairs]

        # print('Encoded query passage pairs')

        # create tensors for the input ids, attention masks, and token type ids
        input_ids = torch.cat([encoded_pair['input_ids'] for encoded_pair in encoded_pairs], dim=0)
        attention_masks = torch.cat([encoded_pair['attention_mask'] for encoded_pair in encoded_pairs], dim=0)

        # Create a dataloader for feeding the data to the model
        dataset = torch.utils.data.TensorDataset(input_ids, attention_masks)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        # print('Created dataloader')

        def get_scores(dataloader, model):
            scores = []
            with torch.no_grad():
                for batch in dataloader:
                    # Unpack the batch and move it to GPU
                    b_input_ids, b_attention_mask = batch
                    b_input_ids = b_input_ids.to('cuda')
                    b_attention_mask = b_attention_mask.to('cuda')

                    # Forward pass, get logit predictions
                    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask)
                    logits = outputs.logits.data.cpu().numpy()

                    # Move logits to CPU and convert to probabilities (optional)
                    # probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

                    # Collect the scores
                    scores.extend(logits)

            return scores

        scores = get_scores(dataloader, self.model)
        # print('Got scores')
        # original_structure_scores = reconstruct(scores, mapping_info)

        # reshape the scores to match the shape of the aggregated results

        # aggregate scores
        # Re-assemble scores back into the structure of aggregated_results
        score_index = 0
        for agg_result in aggregated_results:
            # Each aggregated result gets a slice of the scores equal to the number of contributing results it has
            end_index = score_index + len(agg_result.contributing_results)
            cur_passage_scores = scores[score_index:end_index]
            score_index = end_index

            # Aggregate the scores for the current aggregated result
            agg_score = self.aggregate_scores(cur_passage_scores)
            agg_result.score = agg_score  # Assign the aggregated score

        # print('Aggregated scores')

        # Sort by the new aggregated score
        aggregated_results.sort(key=lambda res: res.score, reverse=True)
        # print('Sorted aggregated results')

        return aggregated_results

    def aggregate_scores(self, passage_scores):
        """
        Aggregate passage scores based on the specified strategy.
        """
        if self.aggregation_strategy == 'firstp':
            return passage_scores[0]
        elif self.aggregation_strategy == 'maxp':
            return max(passage_scores)
        elif self.aggregation_strategy == 'avgp':
            return sum(passage_scores) / len(passage_scores)
        elif self.aggregation_strategy == 'sump':
            return sum(passage_scores)
        else:
            raise ValueError(f"Invalid score aggregation method: {self.aggregation_strategy}")

    def rerank_pipeline(self, query, aggregated_results):
        reranked_results = self.rerank(query, aggregated_results)
        return reranked_results

if __name__ == "__main__":
    # index_path = '../smalldata/fbr/index_commit_tokenized'
    # repo_path = '../smalldata/fbr/'
    # K=1000
    parser = argparse.ArgumentParser(description='Run BM25 and BERT Reranker evaluation.')
    parser.add_argument('index_path', type=str, help='Path to the index directory.')
    parser.add_argument('repo_path', type=str, help='Path to the repository directory.')
    parser.add_argument('-k', '--k', type=int, default=1000, help='The number of top documents to retrieve (default: 1000)')
    parser.add_argument('-n', '--n', type=int, default=100, help='The number of commits to sample (default: 100)')


    metrics = ['MAP', 'P@10', 'P@100', 'P@1000', 'MRR', 'Recall@100', 'Recall@1000']
    args = parser.parse_args()
    repo_path = args.repo_path
    index_path = args.index_path
    K = args.k
    n = args.n
    combined_df = get_combined_df(repo_path)
    bm25_aggregation_strategy = 'sump'

    bm25_searcher = BM25Search(index_path)
    evaluator = SearchEvaluator(metrics)
    model_evaluator = ModelEvaluator(bm25_searcher, evaluator, combined_df)
    bm25_baseline_eval = model_evaluator.evaluate_sampling(n=n, k=K, output_file='bm25_metrics.txt', aggregation_strategy=bm25_aggregation_strategy)

    print("BM25 Baseline Evaluation")
    print(bm25_baseline_eval)

    print('*' * 80)

    # Reranking with BERT
    parameters = {
    'model_name': 'microsoft/codebert-base',
    'psg_len': 400,
    'psg_cnt': 1,
    'psg_stride': 32,
    'aggregation_strategy': 'sump',
    'batch_size': 512,
    'use_gpu': True,
    'rerank_depth': 1000,
    # 'max_seq_length': 512,
    }
    bert_reranker = BERTReranker(parameters)
    rerankers = [bert_reranker]

    bert_reranker_eval = model_evaluator.evaluate_sampling(n=n, k=K, output_file='bert_reranker_metrics.txt', aggregation_strategy='sump', rerankers=rerankers)

    print("BERT Reranker Evaluation")
    print(bert_reranker_eval)


