import argparse
import os
import pickle
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bm25_v2 import BM25Searcher
from eval import ModelEvaluator, SearchEvaluator
from utils import AggregatedSearchResult, get_combined_df


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BERTReranker:
    # def __init__(self, model_name, psg_len, psg_cnt, psg_stride, agggreagtion_strategy, batch_size, use_gpu=True):
    def __init__(self, parameters):
        self.parameters = parameters
        self.model_name = parameters['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1)
        # self.model = AutoModel.from_pretrained(self.model_name)
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
        self.psg_cnt = parameters['psg_cnt']
        # self.psg_stride = parameters.get('psg_stride', self.psg_len)
        self.aggregation_strategy = parameters['aggregation_strategy']
        self.batch_size = parameters['batch_size']
        # self.max_title_len = parameters.get('max_title_len', 0)
        # self.use_title = self.max_title_len > 0
        self.rerank_depth = parameters['rerank_depth']
        # self.max_seq_length = parameters.get('max_seq_length', 512)
        self.max_seq_length = self.tokenizer.model_max_length

        print(f"Initialized BERT reranker with parameters: {parameters}")

        # input_dim = parameters['INPUT_DIM']  # Default BERT hidden size
        # hidden_dim = parameters['HIDDEN_DIM']  # Example hidden size
        # output_dim = parameters['OUTPUT_DIM']  # We want a single score as output

        # self.mlp = MLP(input_dim, hidden_dim, output_dim).to(self.device)

    def rerank(self, query, aggregated_results: List[AggregatedSearchResult]):
        """
        Rerank the BM25 aggregated search results using BERT model scores.

        query: The issue query string.
        aggregated_results: A list of AggregatedSearchResult objects from BM25 search.
        """
        # aggregated_results = aggregated_results[:self.rerank_depth] # already done in the pipeline
        # print(f'Reranking {len(aggregated_results)} results')

        # Flatten the list of results into a list of (query, passage) pairs but only keep max psg_cnt passages per file
        query_passage_pairs = []
        for agg_result in aggregated_results:
            query_passage_pairs.extend(
                (query, result.commit_msg)
                for result in agg_result.contributing_results[: self.psg_cnt]
            )

        if not query_passage_pairs:
            print('WARNING: No query passage pairs to rerank')
            print(query, aggregated_results, self.psg_cnt)
            return aggregated_results

        # tokenize the query passage pairs
        encoded_pairs = [self.tokenizer.encode_plus([query, passage], max_length=self.max_seq_length, truncation=True, padding='max_length', return_tensors='pt', add_special_tokens=True) for query, passage in query_passage_pairs]

        # create tensors for the input ids, attention masks
        input_ids = torch.stack([encoded_pair['input_ids'].squeeze() for encoded_pair in encoded_pairs], dim=0)
        attention_masks = torch.stack([encoded_pair['attention_mask'].squeeze() for encoded_pair in encoded_pairs], dim=0)

        # Create a dataloader for feeding the data to the model
        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        scores = self.get_scores(dataloader, self.model)

        score_index = 0
        # Now assign the scores to the aggregated results by mapping the scores to the contributing results
        for agg_result in aggregated_results:
            # Each aggregated result gets a slice of the scores equal to the number of contributing results it has which should be min(psg_cnt, len(contributing_results))
            assert score_index < len(scores), f'score_index {score_index} is greater than or equal to scores length {len(scores)}'
            end_index = score_index + len(agg_result.contributing_results[: self.psg_cnt])
            cur_passage_scores = scores[score_index:end_index]
            score_index = end_index


            # Aggregate the scores for the current aggregated result
            agg_score = self.aggregate_scores(cur_passage_scores)
            agg_result.score = agg_score  # Assign the aggregated score

        assert score_index == len(scores), f'score_index {score_index} does not equal scores length {len(scores)}, indices probably not working correctly'

        # Sort by the new aggregated score
        aggregated_results.sort(key=lambda res: res.score, reverse=True)

        return aggregated_results

    def get_scores(self, dataloader, model):
        scores = []
        # breakpoint()
        with torch.no_grad():
            for batch in dataloader:
                # Unpack the batch and move it to GPU
                b_input_ids, b_attention_mask = batch
                b_input_ids = b_input_ids.to(self.device)
                b_attention_mask = b_attention_mask.to(self.device)

                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask)
                # logits = outputs.logits.data.cpu().numpy()
                logits = outputs.logits.data.cpu().numpy().flatten()

                # Collect the scores
                scores.extend(logits)

        return scores

    def aggregate_scores(self, passage_scores):
        """
        Aggregate passage scores based on the specified strategy.
        """
        if len(passage_scores) == 0:
            return 0.0


        if self.aggregation_strategy == 'firstp':
            return passage_scores[0]
        if self.aggregation_strategy == 'maxp':
            return max(passage_scores)
        if self.aggregation_strategy == 'avgp':
            return sum(passage_scores) / len(passage_scores)
        if self.aggregation_strategy == 'sump':
            return sum(passage_scores)
        # else:
        raise ValueError(f"Invalid score aggregation method: {self.aggregation_strategy}")

    def rerank_pipeline(self, query, aggregated_results):
        if len(aggregated_results) == 0:
            return aggregated_results
        top_results = aggregated_results[:self.rerank_depth]
        bottom_results = aggregated_results[self.rerank_depth:]
        reranked_results = self.rerank(query, top_results)

        # this should also be not needed but there seems to be some non-determinism in the reranking - fixed by setting seed
        # min_top_score = min([result.score for result in reranked_results])
        min_top_score = reranked_results[-1].score


        # now adjust the scores of bottom_results
        for i, result in enumerate(bottom_results):
            result.score = min_top_score - i - 1
        # combine the results
        reranked_results.extend(bottom_results)

        assert(len(reranked_results) == len(aggregated_results))

        # NOT needed - fixed by setting seed
        # reranked_results.sort(key=lambda res: res.score, reverse=True)
        return reranked_results

def main(args):
    set_seed(42)  # or any other seed you choose
    metrics = ['MAP', 'P@10', 'P@100', 'P@1000', 'MRR', 'Recall@100', 'Recall@1000']
    repo_path = args.repo_path
    index_path = args.index_path
    K = args.k
    n = args.n
    combined_df = get_combined_df(repo_path)
    BM25_AGGR_STRAT = 'sump'

    bm25_searcher = BM25Searcher(index_path)
    evaluator = SearchEvaluator(metrics)
    model_evaluator = ModelEvaluator(bm25_searcher, evaluator, combined_df)
    bm25_baseline_eval = model_evaluator.evaluate_sampling(n=n, k=K, output_file='BM25_metrics.txt', aggregation_strategy=BM25_AGGR_STRAT, repo_path=repo_path)

    print("BM25 Baseline Evaluation")
    print(bm25_baseline_eval)

    print('*' * 80)

    # Reranking with BERT
    params = {
        'model_name': 'microsoft/codebert-base',
        'psg_len': 400,
        'psg_cnt': 5,
        # 'psg_stride': 32,
        'aggregation_strategy': 'sump',
        # 'batch_size': 32,
        'batch_size': 512,
        # 'batch_size': 1,
        'use_gpu': True,
        'rerank_depth': 100,
        'num_epochs': 4,
        'mlp_lr': 1e-3,
        'bert_lr': 2e-5,
        'INPUT_DIM': 768,
        'HIDDEN_DIM': 100,
        'OUTPUT_DIM': 1,
        'NUM_POSITIVE': 10,
        'NUM_NEGATIVE': 20,
        'train_depth': 1000,
    }


    bert_reranker = BERTReranker(params)
    rerankers = [bert_reranker]


    # print(combined_df.info())

    bert_reranker_eval = model_evaluator.evaluate_sampling(n=n, k=K, output_file=f"bert_reranker_{params['model_name']}_without_mlp_metrics.txt", aggregation_strategy='sump', rerankers=rerankers, repo_path=repo_path)

    print("BERT Reranker Evaluation")
    print(bert_reranker_eval)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BM25 and BERT Reranker evaluation.')
    parser.add_argument('--index_path', type=str, help='Path to the index directory.', required=True)
    parser.add_argument('--repo_path', type=str, help='Path to the repository directory.', required=True)
    parser.add_argument('-k', '--k', type=int, default=1000, help='The number of top documents to retrieve (default: 1000)')
    parser.add_argument('-n', '--n', type=int, default=100, help='The number of commits to sample (default: 100)')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing cache files.')
    parser.add_argument('--freeze_bert', action='store_true', help='Freeze BERT layers during training.')
    args = parser.parse_args()
    print(args)
    main(args)


