import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from BERTReranker_v4 import BERTReranker
from bm25_v2 import BM25Searcher
from eval import ModelEvaluator, SearchEvaluator
from utils import (
    AggregatedSearchResult,
    get_combined_df,
    get_recent_df,
    prepare_triplet_data_from_df,
    sanity_check_triplets,
    set_seed,
)

# set seed
set_seed(42)

class BERTCodeReranker:
    def __init__(self, parameters):
        self.parameters = parameters
        self.model_name = parameters['model_name']
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1, problem_type='regression')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() and parameters['use_gpu'] else "cpu")
        self.model.to(self.device)

        print(f'Using device: {self.device}')

        # print GPU info
        if torch.cuda.is_available() and parameters['use_gpu']:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f'GPU Device Count: {torch.cuda.device_count()}')
            print(f"GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")


        # self.psg_len = parameters['psg_len']
        self.psg_cnt = parameters['psg_cnt'] # how many contributing_results to use per file for reranking
        # self.psg_stride = parameters.get('psg_stride', self.psg_len)
        self.aggregation_strategy = parameters['aggregation_strategy'] # how to aggregate the scores of the psg_cnt contributing_results
        self.batch_size = parameters['batch_size'] # batch size for reranking efficiently
        self.rerank_depth = parameters['rerank_depth']
        self.max_seq_length = self.tokenizer.model_max_length # max sequence length for the model

        print(f"Initialized BERT reranker with parameters: {parameters}")


    def rerank(self, query, aggregated_results: List[AggregatedSearchResult]):
        """
        Rerank the BM25 aggregated search results using BERT model scores.

        query: The issue query string.
        aggregated_results: A list of AggregatedSearchResult objects from BM25 search.
        """
        # aggregated_results = aggregated_results[:self.rerank_depth] # already done in the pipeline
        # print(f'Reranking {len(aggregated_results)} results')

        self.model.eval()

        # Flatten the list of results into a list of (query, passage) pairs but only keep max psg_cnt passages per file
        # TODO change this to be for diffs instead of commit_messages
        query_passage_pairs = []
        for agg_result in aggregated_results:
            # get the top psg_cnt contributing_results
            contributing_results = agg_result.contributing_results[: self.psg_cnt]
            # for each contributing_result, get the file_path and commit_id
            # TODO maybe limit this with a diff_cnt
            for contributing_result in contributing_results:
                file_path = contributing_result.file_path
                commit_id = contributing_result.commit_id
                # get the diff from the df
                diff = combined_df[(combined_df['commit_id'] == commit_id) & (combined_df['file_path'] == file_path)]['diff']
                assert diff.shape[0] == 1, f"diff should only have one row, but has {diff.shape[0]} rows"
                diff = str(diff.iloc[0])
                query_passage_pairs.append((query, diff))

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

    def get_scores(self, dataloader, model):
        scores = []
        with torch.no_grad():
            for batch in dataloader:
                # Unpack the batch and move it to GPU
                b_input_ids, b_attention_mask = batch
                b_input_ids = b_input_ids.to(self.device)
                b_attention_mask = b_attention_mask.to(self.device)

                # Get scores from the model
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask)
                scores.extend(outputs.logits.detach().cpu().numpy().squeeze(-1))
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
        min_top_score = reranked_results[-1].score
        # now adjust the scores of bottom_results
        for i, result in enumerate(bottom_results):
            result.score = min_top_score - i - 1
        # combine the results
        reranked_results.extend(bottom_results)
        assert(len(reranked_results) == len(aggregated_results))
        return reranked_results


def do_training(triplet_data, code_reranker, hf_output_dir, args):
    def tokenize_hf(example):
        return code_reranker.tokenizer(example['query'], example['passage'], truncation=True, padding='max_length', max_length=code_reranker.max_seq_length, return_tensors='pt', add_special_tokens=True)
    print('Training the model...')
    print('Label distribution:')
    print(triplet_data['label'].value_counts())

    if args.sanity_check_triplets:
        print('Running sanity check on training data...')
        triplet_data = sanity_check_triplets(triplet_data)

    # Step 7: convert triplet_data to HuggingFace Dataset
    # convert triplet_data to HuggingFace Dataset
    triplet_data['label'] = triplet_data['label'].astype(float)
    train_df, val_df = train_test_split(triplet_data, test_size=0.2, random_state=42, stratify=triplet_data['label'])
    train_hf_dataset = HFDataset.from_pandas(train_df, split='train') # type: ignore
    val_hf_dataset = HFDataset.from_pandas(val_df, split='validation') # type: ignore

    # Step 8: tokenize the data
    tokenized_train_dataset = train_hf_dataset.map(tokenize_hf, batched=True)
    tokenized_val_dataset = val_hf_dataset.map(tokenize_hf, batched=True)

    # Step 9: set format for pytorch
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(['query', 'passage'])
    tokenized_val_dataset = tokenized_val_dataset.remove_columns(['query', 'passage'])

    # rename label column to labels
    tokenized_train_dataset = tokenized_train_dataset.rename_column('label', 'labels')
    tokenized_val_dataset = tokenized_val_dataset.rename_column('label', 'labels')

    # set format to pytorch
    tokenized_train_dataset = tokenized_train_dataset.with_format('torch')
    tokenized_val_dataset = tokenized_val_dataset.with_format('torch')
    print('Training dataset features:')
    print(tokenized_train_dataset.features)

    # Step 10: set up training arguments
    train_args = TrainingArguments(
        output_dir=hf_output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=args.num_epochs,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        save_total_limit=2,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=1000,
        fp16=True,
        dataloader_num_workers=args.num_workers,
        )

    small_train_dataset = tokenized_train_dataset.shuffle(seed=42).select(range(100))
    small_val_dataset = tokenized_val_dataset.shuffle(seed=42).select(range(100))

    if args.debug:
        print('Running in debug mode, using small datasets')
        tokenized_train_dataset = small_train_dataset
        tokenized_val_dataset = small_val_dataset

    # Step 11: set up trainer
    trainer = Trainer(
        model = code_reranker.model,
        args = train_args,
        train_dataset = tokenized_train_dataset, # type: ignore
        eval_dataset = tokenized_val_dataset, # type: ignore
        # compute_metrics=compute_metrics,
    )

    # Step 12: train the model
    trainer.train()

    # Step 13: save the model
    best_model_path = os.path.join(hf_output_dir, 'best_model')
    trainer.save_model(best_model_path)
    print(f'Saved model to {best_model_path}')
    print('Training complete')


def main(args):
    # print torch devices available
    print('Available devices: ', torch.cuda.device_count())
    print('Current cuda device: ', torch.cuda.current_device())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    metrics = ['MAP', 'P@10', 'P@100', 'P@1000', 'MRR', 'Recall@100', 'Recall@1000']
    repo_path = args.repo_path
    repo_name = repo_path.split('/')[-1]
    index_path = args.index_path
    # TODO remove K and n everywhere
    K = args.k
    n = args.n
    # combined_df = get_combined_df(repo_path)
    # TODO add this to params
    BM25_AGGR_STRAT = 'sump'

    bm25_searcher = BM25Searcher(index_path)
    evaluator = SearchEvaluator(metrics)
    model_evaluator = ModelEvaluator(bm25_searcher, evaluator, combined_df)

    params = {
        'model_name': args.model_path,
        'psg_cnt': args.psg_cnt,
        'aggregation_strategy': args.aggregation_strategy,
        'batch_size': args.batch_size,
        'use_gpu': args.use_gpu,
        'rerank_depth': args.rerank_depth,
        'num_epochs': args.num_epochs,
        'lr': args.learning_rate,
        'num_positives': args.num_positives,
        'num_negatives': args.num_negatives,
        'train_depth': args.train_depth,
        'num_workers': args.num_workers,
        'train_commits': args.train_commits,
        'bm25_aggr_strategy': BM25_AGGR_STRAT,
    }

    bert_reranker = BERTReranker(params)
    code_reranker = BERTCodeReranker(params)
    code_reranker.rerank_depth = 100 # TODO Magic number
    rerankers = [bert_reranker, code_reranker]
    save_model_name = params['model_name'].replace('/', '_')
    hf_output_dir = os.path.join(repo_path, f'code_{save_model_name}_model_output')
    best_model_path = os.path.join(hf_output_dir, 'best_model')


    # create eval directory to store results
    eval_path = os.path.join(repo_path, 'eval', save_model_name)
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    # training methods


    if args.do_train:
        recent_df = get_recent_df(combined_df, params)

        # Step 6: Prepare triplet data
        if not os.path.exists(os.path.join(repo_path, 'cache')):
            os.makedirs(os.path.join(repo_path, 'cache'))
        triplet_cache = os.path.join(repo_path, 'cache', 'triplet_data_cache.pkl')
        # TODO: filter eval commits from here
        triplet_data = prepare_triplet_data_from_df(recent_df, bm25_searcher, search_depth=params['train_depth'], num_positives=params['num_positives'], num_negatives=params['num_negatives'], cache_file=triplet_cache, overwrite=args.overwrite_cache)

        do_training(triplet_data, code_reranker, hf_output_dir, args)


    # load the best model from args.best_model_path for do_eval and eval_gold
    if args.do_eval or args.eval_gold:
        cur_best_model_path = args.best_model_path or best_model_path
        if not os.path.exists(cur_best_model_path):
            raise ValueError(f'Best model path {cur_best_model_path} does not exist, please train the model first')
        print(f'Loading model from {cur_best_model_path}...')
        code_reranker.model = AutoModelForSequenceClassification.from_pretrained(cur_best_model_path, num_labels=1, problem_type='regression')
        code_reranker.model.to(code_reranker.device)
        rerankers = [bert_reranker, code_reranker]


    if args.do_eval:

        bert_with_training_output_path = os.path.join(eval_path, 'multi_bert_with_training.txt')
        bert_with_training_eval = model_evaluator.evaluate_sampling(n=n, k=K, output_file_path=bert_with_training_output_path, aggregation_strategy=params['aggregation_strategy'], rerankers=rerankers, overwrite_eval=args.overwrite_eval)

        print("BERT Evaluation with training")
        print(bert_with_training_eval)

    if args.eval_gold:
        gold_dir = os.path.join('gold', repo_name)
        if not os.path.exists(gold_dir):
            raise ValueError(f'Gold directory {gold_dir} does not exist, please run openai_transform.py first')
        # check if gold data exists
        gold_data_path = os.path.join(gold_dir, f'{repo_name}_{args.openai_model}_gold.parquet')
        if not os.path.exists(gold_data_path):
            raise ValueError(f'Gold data {gold_data_path} does not exist, please run openai_transform.py first')
        print(f'Model: {args.openai_model}')
        gold_df = pd.read_parquet(gold_data_path)
        # assert all transformed_message_gpt3 are not NaN
        # rename the column transformed_message_gpt3 to transformed_message_{oai_model}
        gold_df = gold_df.rename(columns={'transformed_message_gpt3': f'transformed_message_{args.openai_model}'})
        assert gold_df[f'transformed_message_{args.openai_model}'].notnull().all()
        # rename commit_message to original_message
        gold_df = gold_df.rename(columns={'commit_message': 'original_message'})
        # rename transformed_message to commit_message
        gold_df = gold_df.rename(columns={f'transformed_message_{args.openai_model}': 'commit_message'})
        print(f'Found gold data for {repo_name} with shape {gold_df.shape} at {gold_data_path}')
        print(gold_df.info())

        # run BM25 on gold data first
        # print('Running BM25 on gold data...')
        # bm25_gold_output_path = os.path.join(eval_path, f'bm25_{args.openai_model}_gold_metrics.txt')
        # bm25_gold_eval = model_evaluator.evaluate_sampling(n=n, k=K, output_file_path=bm25_gold_output_path, aggregation_strategy=params['bm25_aggr_strategy'], gold_df=gold_df, overwrite_eval=args.overwrite_eval)
        # print("BM25 Gold Evaluation")
        # print(bm25_gold_eval)


        # get gold eval with reranking
        print('Running BERT on gold data...')
        bert_gold_output_path = os.path.join(eval_path, f'multi_bert_{args.openai_model}_gold.txt')
        bert_gold_eval = model_evaluator.evaluate_sampling(n=n, k=K, output_file_path=bert_gold_output_path, aggregation_strategy=params['aggregation_strategy'], rerankers=rerankers, gold_df=gold_df, overwrite_eval=args.overwrite_eval)

        print("BERT Gold Evaluation")
        print(bert_gold_eval)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run BM25 and/or BERT Reranker evaluation.')
    parser.add_argument('--index_path', type=str, help='Path to the index directory.', required=True)
    parser.add_argument('--repo_path', type=str, help='Path to the repository directory.', required=True)
    parser.add_argument('-k', '--k', type=int, default=1000, help='The number of top documents to retrieve (default: 1000)')
    parser.add_argument('-n', '--n', type=int, default=100, help='The number of commits to sample (default: 100)')
    # parser.add_argument('--no_bm25', action='store_true', help='Do not run BM25.')
    parser.add_argument('-m', '--model_path', type=str, help='Path to the pretrained model.')
    parser.add_argument('-o', '--overwrite_cache', action='store_true', help='Overwrite existing cache files.')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('-e', '--num_epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5, help='Learning rate (default: 5e-5)')
    # parser.add_argument('-l', '--load_model', action='store_true', help='Load a pretrained model.')
    parser.add_argument('--num_positives', type=int, default=10, help='Number of positive samples per query (default: 10)')
    parser.add_argument('--num_negatives', type=int, default=10, help='Number of negative samples per query (default: 10)')
    parser.add_argument('--train_depth', type=int, default=1000, help='Number of samples to train on (default: 1000)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader (default: 8)')
    parser.add_argument('--train_commits', type=int, default=1500, help='Number of commits to train on (default: 1500)')
    parser.add_argument('--psg_cnt', type=int, default=5, help='Number of passages to retrieve per query (default: 5)')
    parser.add_argument('--aggregation_strategy', type=str, default='sump', help='Aggregation strategy (default: sump)')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU.')
    parser.add_argument('--rerank_depth', type=int, default=250, help='Number of commits to rerank (default: 250)')
    parser.add_argument('--do_train', action='store_true', help='Train the model.')
    parser.add_argument('--do_eval', action='store_true', help='Evaluate the model.')
    parser.add_argument('--eval_gold', action='store_true', help='Evaluate the model on gold data.')
    parser.add_argument('--openai_model', choices=['gpt3', 'gpt4'], help='OpenAI model to use for transforming commit messages.')
    parser.add_argument('--overwrite_eval', action='store_true', help='Replace evaluation files if they already exist.')
    parser.add_argument('--sanity_check_triplets', action='store_true', help='Run sanity check on training data.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
    # parser.add_argument('--do_combined_train', action='store_true', help='Train on combined data from multiple repositories.')
    # parser.add_argument('--repo_paths', nargs='+', help='List of repository paths for combined training.', required='--do_combined_train' in sys.argv)
    parser.add_argument('--best_model_path', type=str, help='Path to the best model.')
    args = parser.parse_args()
    print(args)
    combined_df = get_combined_df(args.repo_path)
    main(args)