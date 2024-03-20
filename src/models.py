from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from splitter import *
from utils import AggregatedSearchResult, PatchResult, get_file_at_commit_from_git


class Reranker:
    def __init__(self, parameters, train_mode):
        self.parameters = parameters
        self.train_mode = train_mode
        self.model_name = parameters['model_name']
        self.device = torch.device("cuda" if torch.cuda.is_available() and parameters['use_gpu'] else "cpu")
        print(f'Using device: {self.device}')
        # print GPU info
        if torch.cuda.is_available() and parameters['use_gpu']:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f'GPU Device Count: {torch.cuda.device_count()}')
            print(f"GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        self.aggregation_strategy = parameters['aggregation_strategy'] # how to aggregate the scores of the psg_cnt contributing_results
        self.batch_size = parameters['batch_size'] # batch size for reranking efficiently
        self.rerank_depth = parameters['rerank_depth']

    def rerank(self, query, aggregated_results: List[AggregatedSearchResult]):
        raise NotImplementedError

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

    def get_scores(self, dataloader, model) -> List[np.ndarray]:
        scores = []
        with torch.no_grad():
            for batch in dataloader:
                # Unpack the batch and move it to GPU
                b_input_ids, b_attention_mask = batch
                b_input_ids = b_input_ids.to(self.device)
                b_attention_mask = b_attention_mask.to(self.device)

                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask)
                if self.train_mode == 'regression':
                    # Get scores from the model
                    scores.extend(outputs.logits.detach().cpu().numpy().squeeze(-1))
                elif self.train_mode == 'classification':
                    outputs = outputs.logits
                    outputs = torch.softmax(outputs, dim=1)

                    # get the score of being relevant aka label 1
                    outputs = outputs[:, 1]
                    scores.extend(outputs.detach().cpu().numpy())
        return scores


class BERTReranker(Reranker):
    def __init__(self, parameters, train_mode):
        super().__init__(parameters, train_mode)

        if train_mode == 'regression':
            print('Using regression model')
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1, problem_type='regression')
        elif train_mode == 'classification':
            print('Using classification model')
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        else:
            raise ValueError(f"Invalid train_mode: {train_mode}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)

        self.psg_cnt = parameters['psg_cnt'] # how many contributing_results to use per file for reranking
        self.max_seq_length = self.tokenizer.model_max_length # max sequence length for the model
        # self.psg_len = parameters['psg_len'] # not needed since commit messages are small
        # self.psg_stride = parameters.get('psg_stride', self.psg_len) # not needed since commit messages are small

        print(f"Initialized BERT reranker with parameters: {parameters}")


    def rerank(self, query, aggregated_results: List[AggregatedSearchResult]):
        """
        Rerank the BM25 aggregated search results using BERT model scores.

        query: The issue query string.
        aggregated_results: A list of AggregatedSearchResult objects from BM25 search.
        """
        self.model.eval()

        # Flatten the list of results into a list of (query, passage) pairs but only keep max psg_cnt passages per file
        query_passage_pairs = []
        for agg_result in aggregated_results:
            query_passage_pairs.extend(
                (query, result.commit_message)
                for result in agg_result.contributing_results[: self.psg_cnt]
            )

        if not query_passage_pairs:
            print('WARNING: No query passage pairs to rerank, returning original results from previous stage')
            print(query, aggregated_results, self.psg_cnt)
            return aggregated_results

        # tokenize the query passage pairs
        encoded_pairs = [self.tokenizer.encode_plus([query, passage], max_length=self.max_seq_length, truncation=True, padding='max_length', return_tensors='pt', add_special_tokens=True) for query, passage in query_passage_pairs]

        # create tensors for the input ids, attention masks
        input_ids = torch.stack([encoded_pair['input_ids'].squeeze() for encoded_pair in encoded_pairs], dim=0) # type: ignore
        attention_masks = torch.stack([encoded_pair['attention_mask'].squeeze() for encoded_pair in encoded_pairs], dim=0) # type: ignore

        # Create a dataloader for feeding the data to the model
        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False) # shuffle=False very important for reconstructing the results back into the original order

        scores = self.get_scores(dataloader, self.model)

        score_index = 0
        # Now assign the scores to the aggregated results by mapping the scores to the contributing results
        for agg_result in aggregated_results:
            # Each aggregated result gets a slice of the scores equal to the number of contributing results it has which should be min(psg_cnt, len(contributing_results))
            assert score_index < len(scores), f'score_index {score_index} is greater than or equal to scores length {len(scores)}'
            end_index = score_index + len(agg_result.contributing_results[: self.psg_cnt]) # only use psg_cnt contributing_results
            cur_passage_scores = scores[score_index:end_index]
            score_index = end_index


            # Aggregate the scores for the current aggregated result
            agg_score = self.aggregate_scores(cur_passage_scores)
            agg_result.score = agg_score  # Assign the aggregated score

        assert score_index == len(scores), f'score_index {score_index} does not equal scores length {len(scores)}, indices probably not working correctly'

        # Sort by the new aggregated score
        aggregated_results.sort(key=lambda res: res.score, reverse=True)

        return aggregated_results

    def rerank_pipeline(self, query, aggregated_results):
        if len(aggregated_results) == 0:
            return aggregated_results
        top_results = aggregated_results[:self.rerank_depth]
        bottom_results = aggregated_results[self.rerank_depth:]
        reranked_results = self.rerank(query, top_results)
        min_top_score = reranked_results[-1].score
        # now adjust the scores of bottom_results
        for i, result in enumerate(bottom_results):
            result.score = min_top_score - i - 1
        # combine the results
        reranked_results.extend(bottom_results)
        assert(len(reranked_results) == len(aggregated_results))
        return reranked_results




class CodeReranker(Reranker):
    def __init__(self, parameters, repo, mode, train_mode, split_strategy: SplitStrategy):
        super().__init__(parameters, train_mode)

        # specific to CodeReranker type
        self.repo = repo # git repo object

        self.train_mode = train_mode
        if self.train_mode == 'regression':
            print('Using regression model')
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1, problem_type='regression')
        elif self.train_mode == 'classification':
            print('Using classification model')
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        else:
            raise ValueError(f"Invalid train_mode: {self.train_mode}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.max_seq_length = self.tokenizer.model_max_length # max sequence length for the model


        self.psg_len = parameters['psg_len']
        self.psg_cnt = parameters['psg_cnt'] # how many contributing_results to use per file for reranking
        self.psg_stride = parameters.get('psg_stride', self.psg_len)
        print(f"Initialized Patch Code Reranker with parameters: {parameters}")

        self.mode = mode # either 'file' or 'patch'
        self.passage_splitter = PassageSplitter(split_strategy)


    def rerank(self, query, aggregated_results: List[AggregatedSearchResult], test_commit_id): # type: ignore
        """
        Rerank the BM25 aggregated search results using BERT model scores.

        query: The issue query string.
        aggregated_results: A list of AggregatedSearchResult objects from BM25 search.
        """
        self.model.eval()

        query_passage_pairs, per_result_contribution = self.split_into_query_passage_pairs(query, aggregated_results, test_commit_id)

        if not query_passage_pairs:
            print('WARNING: No query passage pairs to rerank, returning original results from previous stage')
            print(query, aggregated_results, self.psg_cnt)
            return aggregated_results

        # tokenize the query passage pairs
        encoded_pairs = [self.tokenizer.encode_plus([query, obj.passage], max_length=self.max_seq_length, truncation=True, padding='max_length', return_tensors='pt', add_special_tokens=True) for (query, file_path, obj) in query_passage_pairs]

        # create tensors for the input ids, attention masks
        input_ids = torch.stack([encoded_pair['input_ids'].squeeze() for encoded_pair in encoded_pairs], dim=0) # type: ignore
        attention_masks = torch.stack([encoded_pair['attention_mask'].squeeze() for encoded_pair in encoded_pairs], dim=0) # type: ignore

        # Create a dataloader for feeding the data to the model
        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False) # shuffle=False very important for reconstructing the results back into the original order

        scores = self.get_scores(dataloader, self.model)

        if self.mode == 'file':

            score_index = 0

            # Now assign the scores to the aggregated results by mapping the scores to the contributing results
            for i, agg_result in enumerate(aggregated_results):
                # Each aggregated result gets a slice of the scores equal to the number of contributing results it has which should be min(psg_cnt, len(contributing_results))
                assert score_index < len(scores), f'score_index {score_index} is greater than or equal to scores length {len(scores)}'
                end_index = score_index + per_result_contribution[i] # only use psg_cnt contributing_results
                cur_passage_scores = scores[score_index:end_index]
                score_index = end_index

                # Aggregate the scores for the current aggregated result
                agg_score = self.aggregate_scores(cur_passage_scores)
                agg_result.score = agg_score  # Assign the aggregated score

            assert score_index == len(scores), f'score_index {score_index} does not equal scores length {len(scores)}, indices probably not working correctly'

            # Sort by the new aggregated score
            aggregated_results.sort(key=lambda res: res.score, reverse=True)

            return aggregated_results

        elif self.mode == 'patch':
            # convert the scores to PatchResult objects
            patch_results = [PatchResult(file_path, obj, score) for (query, file_path, obj), score in zip(query_passage_pairs, scores)]

            # sort patch_results by the scores
            sorted_patch_results = sorted(patch_results, key=lambda res: res.score, reverse=True) # type: ignore

            return sorted_patch_results
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def split_into_query_passage_pairs(self, query, aggregated_results, test_commit_id):

        query_passage_pairs = []
        per_result_contribution = []

        for agg_result in aggregated_results:
            file_path = agg_result.file_path

            # get most recent version of the file BEFORE test_commit_id
            file_content = get_file_at_commit_from_git(self.repo, file_path, test_commit_id, from_parent=True)


            if pd.isna(file_content):
                # if file_content is NaN, then we can just set file_content to empty string
                print(f'WARNING: file_content is NaN for commit_id: {test_commit_id}, file_path: {file_path}, setting file_content to empty string')
                file_content = ''

            # now split the file content into psg_cnt passages - nope, doing all passages
            cur_result_passages = self.passage_splitter.split_passages(file_content)

            # now add the query, passage pairs to query_passage_pairs
            per_result_contribution.append(len(cur_result_passages)) # type: ignore
            # query_passage_pairs.extend((query, passage) for passage in cur_result_passages) # type: ignore
            query_passage_pairs.extend((query, file_path, obj) for obj in cur_result_passages) # type: ignore

        return query_passage_pairs, per_result_contribution

    def rerank_pipeline(self, query, aggregated_results, test_commit_id):
        if len(aggregated_results) == 0:
                return aggregated_results
        if self.mode == 'file':
            top_results = aggregated_results[:self.rerank_depth]
            bottom_results = aggregated_results[self.rerank_depth:]
            reranked_results = self.rerank(query, top_results, test_commit_id)
            min_top_score = reranked_results[-1].score # type: ignore
            # now adjust the scores of bottom_results
            for i, result in enumerate(bottom_results):
                result.score = min_top_score - i - 1
            # combine the results
            reranked_results.extend(bottom_results) # type: ignore
            assert(len(reranked_results) == len(aggregated_results)) # type: ignore
            return reranked_results # list of AggregatedSearchResult
        else:
            top_results = aggregated_results[:self.rerank_depth]
            reranked_results = self.rerank(query, top_results, test_commit_id)
            return reranked_results # list of PatchResult

