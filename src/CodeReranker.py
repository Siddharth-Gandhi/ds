import argparse
import gc
import json
import os
import sys
from typing import List

import git
import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import wandb
from BERTReranker import BERTReranker
from bm25_v2 import BM25Searcher
from eval import ModelEvaluator, SearchEvaluator
from models import CodeReranker
from splitter import *
from utils import (
    balance_labels,
    get_code_df,
    get_combined_df,
    get_recent_df,
    prepare_code_triplets,
    sanity_check_code,
    set_seed,
)

# set seed
set_seed(42)
gc.collect() # These commands help you when you face CUDA OOM error
torch.cuda.empty_cache()

def do_training(triplet_data, reranker, hf_output_dir, args):
    def tokenize_hf(example):
        return reranker.tokenizer(example['query'], example['passage'], truncation=True, padding='max_length', max_length=reranker.max_seq_length, return_tensors='pt', add_special_tokens=True)


    # triplet_data = triplet_data.sample(1000, random_state=42)
    print('Training the model...')
    print('Label distribution:')
    print(triplet_data['label'].value_counts())

    # merge columns file_path and passage into one column called passage
    # triplet_data['passage'] = triplet_data['file_path'] + ' ' + triplet_data['passage']


    triplet_data['passage'] = triplet_data['passage'] # ! no file path as that just leads to confusion


    # Step 1: convert triplet_data to HuggingFace Dataset
    # convert triplet_data to HuggingFace Dataset
    if args.train_mode == 'regression':
        # ! important for regression
        triplet_data['label'] = triplet_data['label'].astype(float)

    train_df, val_df = train_test_split(triplet_data, test_size=0.2, random_state=42, stratify=triplet_data['label'])
    train_hf_dataset = HFDataset.from_pandas(train_df, split='train') # type: ignore
    val_hf_dataset = HFDataset.from_pandas(val_df, split='validation') # type: ignore
    # Step 2: tokenize the data
    tokenized_train_dataset = train_hf_dataset.map(tokenize_hf, batched=True)
    tokenized_val_dataset = val_hf_dataset.map(tokenize_hf, batched=True)

    # Step 3: set format for pytorch
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(['query', 'passage', 'file_path'])
    tokenized_val_dataset = tokenized_val_dataset.remove_columns(['query', 'passage', 'file_path'])

    # Step 4: rename label column to labels
    tokenized_train_dataset = tokenized_train_dataset.rename_column('label', 'labels')
    tokenized_val_dataset = tokenized_val_dataset.rename_column('label', 'labels')

    # Step 5: set format to pytorch
    tokenized_train_dataset = tokenized_train_dataset.with_format('torch')
    tokenized_val_dataset = tokenized_val_dataset.with_format('torch')
    print('Training dataset features:')
    print(tokenized_train_dataset.features)

    # Step 6: set up training arguments
    train_args = TrainingArguments(
        output_dir=hf_output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='epoch',
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        save_total_limit=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=True,
        dataloader_num_workers=args.num_workers,
        report_to='wandb' if not args.debug else "none" # type: ignore
        )

    small_train_dataset = tokenized_train_dataset.shuffle(seed=42).select(range(100))
    small_val_dataset = tokenized_val_dataset.shuffle(seed=42).select(range(100))

    if args.debug:
        print('Running in debug mode, using small datasets')
        tokenized_train_dataset = small_train_dataset
        tokenized_val_dataset = small_val_dataset

    # Step 7: set up trainer
    trainer = Trainer(
        model = reranker.model,
        args = train_args,
        train_dataset = tokenized_train_dataset, # type: ignore
        eval_dataset = tokenized_val_dataset, # type: ignore
        # compute_metrics=compute_metrics,
    )

    # Step 8: train the model
    trainer.train()

    # Step 9: save the model
    best_model_path = os.path.join(hf_output_dir, 'best_model')
    trainer.save_model(best_model_path)
    print(f'Saved model to {best_model_path}')
    print('Training complete')


def main(args):
    # print torch devices available
    print('Available devices: ', torch.cuda.device_count())
    if torch.cuda.is_available():
        print('Current cuda device: ', torch.cuda.current_device())
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    # metrics = ['MAP', 'P@10', 'P@100', 'P@1000', 'MRR', 'Recall@100', 'Recall@1000']
    metrics = ['MAP', 'P@1', 'P@10', 'P@20', 'P@30', 'MRR', 'R@1', 'R@10', 'R@100', 'R@1000']
    data_path = args.data_path
    repo_name = data_path.split('/')[-1]
    github_repo_path = os.path.join('repos', repo_name) # ! important
    index_path = args.index_path
    K = args.k
    n = args.n
    combined_df = get_combined_df(data_path)
    # github_repo = git.Repo(github_repo_path) # type: ignore
    BM25_AGGR_STRAT = 'maxp'

    # create eval directory to store results
    # eval_path = os.path.join(data_path, 'eval', 'coderr')

    # load fid_to_path and path_to_fid json files to dicts
    with open(f"facebook_react_FID_to_paths.json") as f:
        fid_to_path = json.load(f)

    # make all fids ints
    fid_to_path = {int(k): v for k, v in fid_to_path.items()}

    with open(f"facebook_react_path_to_FID.json") as f:
        path_to_fid = json.load(f)

    save_name = f'code_reranker_{args.code_reranker_mode}'

    eval_path = os.path.join('out', repo_name, save_name, args.eval_folder)

    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    bm25_searcher = BM25Searcher(index_path, fid_to_path, path_to_fid)
    evaluator = SearchEvaluator(metrics)
    model_evaluator = ModelEvaluator(bm25_searcher, evaluator, combined_df, fid_to_path, path_to_fid)

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
        'psg_len': args.psg_len,
        'psg_stride': args.psg_stride,
        'output_length': args.output_length
    }

    tokenizer = AutoTokenizer.from_pretrained(params['model_name'])

    split_strategy = TokenizedLineSplitStrategy(tokenizer=tokenizer, psg_len=args.psg_len, psg_stride=args.psg_stride)

    code_reranker = CodeReranker(params, github_repo_path, fid_to_path, path_to_fid, args.code_reranker_mode, args.train_mode, split_strategy, filter_invalid=args.filter_invalid)

    hf_output_dir = os.path.join('models', repo_name, save_name, args.eval_folder)
    if not os.path.exists(hf_output_dir):
        os.makedirs(hf_output_dir)
    best_model_path = os.path.join(hf_output_dir, 'best_model')




    # training methods
    if args.do_train:

        # cache_path = os.path.join(data_path, 'cache', args.eval_folder)
        cache_path = os.path.join('cache', repo_name, save_name, args.eval_folder)
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        gold_dir = os.path.join('gold', repo_name)
        if not os.path.exists(gold_dir):
            raise ValueError(f'Gold directory {gold_dir} does not exist, please run openai_transform.py first')

        gold_train_file = os.path.join(gold_dir, f'v2_{repo_name}_{args.openai_model}_train.parquet')
        if not os.path.exists(gold_train_file):
            raise ValueError(f'Gold train file {gold_train_file} does not exist, please run openai_transform.py first')

        gold_df = pd.read_parquet(gold_train_file)
        # rename column commit_message to original_message and transformed_message_gpt4 to commit_message
        gold_df = gold_df.rename(columns={'commit_message': 'original_message', f'transformed_message_{args.openai_model}': 'commit_message'})
        recent_df = gold_df

        # else: # TODO uncomment to remove 4X train
        if not args.use_gpt_train:
            recent_df = get_recent_df(combined_df=combined_df, repo_name=repo_name, ignore_gold_in_training=args.ignore_gold_in_training)
            # Step 6: randomly sample params['train_commits'] commits for training or if less than that, use all commits
            if len(recent_df) < params['train_commits']:
                print(f'Number of commits in train_df: {len(recent_df)}')
                print(f'Using all commits for training')
                recent_df = recent_df.sample(len(recent_df))
            else:
                recent_df = recent_df.sample(params['train_commits'])

            #! merging gold df and rest from combined df
            # recent_df = get_recent_df(combined_df=combined_df, repo_name=repo_name, ignore_gold_in_training=args.ignore_gold_in_training, skip_midpoint_filter=True)
            # # remove commits from recent_df that are in gold_df
            # recent_df = recent_df[~recent_df['commit_id'].isin(gold_df['commit_id'])]
            # assert gold_df['commit_id'].unique().tolist() not in recent_df['commit_id'].unique().tolist(), 'Gold commits are present in recent_df'
            # # random sample 1500 commits
            # recent_df = recent_df.sample(params['train_commits'], random_state=42)
            # # add a column original_message to recent_df which is the same as commit_message
            # recent_df['original_message'] = recent_df['commit_message']
            # # merge gold_df and recent_df
            # recent_df = pd.concat([gold_df, recent_df])


        print(f'Number of unique commits: {len(recent_df["commit_id"].unique())}')


        print(f'Number of train commits: {len(recent_df)}')
        code_df_cache = args.code_df_cache or os.path.join(cache_path, 'code_df.parquet')
        code_df = get_code_df(recent_df, bm25_searcher, params['train_depth'], params['num_positives'], params['num_negatives'], combined_df, code_df_cache, args.overwrite_cache, debug=args.debug)

        processed_code_df = code_df
        if args.sanity_check:
            # (i.e. for a train query, one file does not have both label 0 and 1)
            print('Running sanity check on training data...')
            processed_code_df = sanity_check_code(code_df)
            print('Sanity check complete')

        print(f'Processed code dataframe shape after sanity check: {processed_code_df.shape}')
        print(processed_code_df.info())

        triplet_cache = args.triplet_cache_path or os.path.join(cache_path, 'diff_code_triplets.parquet')

        # break the file_content (huge) into manageable chunks for BERT based on commonality with diff
        triplets = prepare_code_triplets(processed_code_df, args, mode=args.triplet_mode, cache_file=triplet_cache ,overwrite=args.overwrite_cache)

        #### Sampling to keep number of triplets reasonable.
        print(f'Triplet dataframe shape (before sampling): {triplets.shape}')
        if 'diff' not in args.triplet_mode:
            triplet_size = min(100000, triplets.shape[0])
            triplets = triplets.sample(triplet_size, random_state=42)

        triplets = balance_labels(triplets)
        print(f'Triplet dataframe shape (after sampling): {triplets.shape}')

        print(triplets.info())

        do_training(triplets, code_reranker, hf_output_dir, args)


    # load the best model from args.best_model_path for do_eval and eval_gold
    if args.do_eval or args.eval_gold:
        if args.best_model_path is None:
            print(f'WARNING: No best model path provided, using {best_model_path}')
            # raise ValueError('Please provide the path to the best model')
        cur_best_model_path = args.best_model_path or best_model_path
        if not os.path.exists(cur_best_model_path):
            raise ValueError(f'Best model path {cur_best_model_path} does not exist, please train the model first')
        print(f'Loading model from {cur_best_model_path}...')
        if args.train_mode == 'classification':
            code_reranker.model = AutoModelForSequenceClassification.from_pretrained(cur_best_model_path, num_labels=2)
        elif args.train_mode == 'regression':
            code_reranker.model = AutoModelForSequenceClassification.from_pretrained(cur_best_model_path, num_labels=1, problem_type='regression')
        code_reranker.model.to(code_reranker.device)

        if args.bert_best_model is not None:
            print(f'Loading BERT model from {args.bert_best_model}...')
            bert_reranker = BERTReranker(bert_params, train_mode = 'classification')
            bert_reranker.model = AutoModelForSequenceClassification.from_pretrained(args.bert_best_model, num_labels=2, problem_type='classification')
            bert_reranker.model.to(bert_reranker.device)
            rerankers = [bert_reranker, code_reranker]
        else:
            rerankers = [code_reranker]

        # print rerankers with their rerank_depth and psg_cnt
        print('Rerankers:')
        for reranker in rerankers:
            print(f'{reranker.__class__.__name__} with rerank_depth: {reranker.rerank_depth}, psg_cnt: {reranker.psg_cnt}')


    if args.do_eval:

        bert_with_training_output_path = os.path.join(eval_path, 'commit')
        gold_dir = os.path.join('gold', repo_name)
        if not os.path.exists(gold_dir):
            raise ValueError(f'Gold directory {gold_dir} does not exist, please run openai_transform.py first')
        # check if gold data exists
        gold_data_path = os.path.join(gold_dir, f'v2_{repo_name}_{args.openai_model}_gold.parquet')
        if not os.path.exists(gold_data_path):
            raise ValueError(f'Gold data {gold_data_path} does not exist, please run openai_transform.py first')
        print(f'Model: {args.openai_model}')
        gold_df = pd.read_parquet(gold_data_path)
        bert_with_training_eval = model_evaluator.evaluate_sampling(n=n, k=K, output_folder_path=bert_with_training_output_path, aggregation_strategy=params['aggregation_strategy'], rerankers=rerankers, overwrite_eval=args.overwrite_eval, gold_df=gold_df)

        print("BERT Evaluation with training")
        print(bert_with_training_eval)

    if args.eval_gold:
        gold_dir = os.path.join('gold', repo_name)
        if not os.path.exists(gold_dir):
            raise ValueError(f'Gold directory {gold_dir} does not exist, please run openai_transform.py first')
        # check if gold data exists
        gold_data_path = os.path.join(gold_dir, f'v2_{repo_name}_{args.openai_model}_gold.parquet')
        if not os.path.exists(gold_data_path):
            raise ValueError(f'Gold data {gold_data_path} does not exist, please run openai_transform.py first')
        print(f'Model: {args.openai_model}')
        gold_df = pd.read_parquet(gold_data_path)
        # assert all transformed_message_gpt3 are not NaN
        assert gold_df[f'transformed_message_{args.openai_model}'].notnull().all()
        # rename commit_message to original_message
        gold_df = gold_df.rename(columns={'commit_message': 'original_message'})
        # rename transformed_message to commit_message
        gold_df = gold_df.rename(columns={f'transformed_message_{args.openai_model}': 'commit_message'})
        print(f'Found gold data for {repo_name} with shape {gold_df.shape} at {gold_data_path}')
        print(gold_df.info())

        # get gold eval with reranking
        print('Running BERT on gold data...')
        bert_gold_output_path = os.path.join(eval_path, 'gold')
        bert_gold_eval = model_evaluator.evaluate_sampling(n=n, k=K, output_folder_path=bert_gold_output_path, aggregation_strategy=params['aggregation_strategy'], rerankers=rerankers, gold_df=gold_df, overwrite_eval=args.overwrite_eval)

        print("BERT Gold Evaluation")
        print(bert_gold_eval)
        if not args.debug:
            wandb.log(bert_gold_eval)





if __name__ == '__main__':
    print('Running CodeReranker.py')
    parser = argparse.ArgumentParser(description='Train/Eval/Run CodeReranker model for files and patches')
    parser.add_argument('--index_path', type=str, help='Path to the index directory.', required=True)
    parser.add_argument('--data_path', type=str, help='Path to the data folder with .parquet files.', required=True)
    parser.add_argument('-k', '--k', type=int, default=1000, help='The number of top documents to retrieve (default: 1000)')
    parser.add_argument('-n', '--n', type=int, default=100, help='The number of commits to sample (default: 100)')
    parser.add_argument('--output_length', type=int, default=1000, help='The length of the output sequence (default: 1000)')
    parser.add_argument('-m', '--model_path', type=str, help='Path to the pretrained model.')
    parser.add_argument('-o', '--overwrite_cache', action='store_true', help='Overwrite existing cache files.')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('-e', '--num_epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5, help='Learning rate (default: 5e-5)')
    parser.add_argument('--run_name', type=str, help='Wandb run name.')
    parser.add_argument('--notes', type=str, help='Wandb run notes.', default='')
    parser.add_argument('--num_positives', type=int, default=10, help='Number of positive samples per query (default: 10)')
    parser.add_argument('--num_negatives', type=int, default=10, help='Number of negative samples per query (default: 10)')
    parser.add_argument('--train_depth', type=int, default=1000, help='Number of samples to train on (default: 1000)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader (default: 8)')
    parser.add_argument('--train_commits', type=int, default=1500, help='Number of commits to train on (default: 1500)')
    parser.add_argument('--psg_cnt', type=int, default=10, help='Number of passages to retrieve per query (default: 5)')
    parser.add_argument('--aggregation_strategy', type=str, default='sump', help='Aggregation strategy (default: sump)')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU.')
    parser.add_argument('--rerank_depth', type=int, default=250, help='Number of commits to rerank (default: 250)')
    parser.add_argument('--do_train', action='store_true', help='Train the model.')
    parser.add_argument('--do_eval', action='store_true', help='Evaluate the model.')
    parser.add_argument('--eval_gold', action='store_true', help='Evaluate the model on gold data.')
    parser.add_argument('--openai_model', choices=['gpt3', 'gpt4'], help='OpenAI model to use for transforming commit messages.')
    parser.add_argument('--overwrite_eval', action='store_true', help='Replace evaluation files if they already exist.')
    parser.add_argument('--sanity_check', action='store_true', help='Run sanity check on training data.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
    parser.add_argument('--filter_invalid', action='store_true', help='Filter invalid in CodeReranker.')
    # parser.add_argument('--do_combined_train', action='store_true', help='Train on combined data from multiple repositories.')
    # parser.add_argument('--repo_paths', nargs='+', help='List of repository paths for combined training.', required='--do_combined_train' in sys.argv)
    parser.add_argument('--best_model_path', type=str, help='Path to the best model.')
    parser.add_argument('--bert_best_model', type=str, help='Path to the best BERT model.')
    parser.add_argument('--psg_len', type=int, default=250, help='Length of each passage (default: 250)')
    parser.add_argument('--psg_stride', type=int, default=200, help='Stride of each passage (default: 250)')
    parser.add_argument('--ignore_gold_in_training', action='store_true', help='Ignore gold commits in training data.')
    parser.add_argument('--eval_folder', type=str, help='Folder name to store evaluation files.', required=True)
    parser.add_argument('--use_gpt_train', action='store_true', help='Use GPT data for training.')
    parser.add_argument('--triplet_mode', choices=['parse_functions', 'sliding_window', 'diff_content', 'diff_subsplit'], default='', help='Mode for preparing triplets (default: diff_code)')
    parser.add_argument('--code_df_cache', type=str, help='Path to the code dataframe cache file.')
    parser.add_argument('--use_previous_file', action='store_true', help='Use the previous file for training.')
    parser.add_argument('--code_reranker_mode', choices=['file', 'patch'], default='file', help='Mode for code reranker (default: file)')
    parser.add_argument('--triplet_cache_path', type=str, help='Path to the triplet cache file.')
    parser.add_argument('--train_mode', choices=['regression', 'classification'], default='regression', help='Mode for training (default: regression)')
    args = parser.parse_args()

    if not args.debug:
        run = wandb.init(project='ds-code_rerank', name=args.run_name, reinit=True, config=args, notes=args.notes) # type: ignore
        # metrics = ['MAP', 'P@1', 'P@10', 'P@20', 'P@30', 'MRR', 'Recall@1', 'Recall@10', 'Recall@100', 'Recall@1000']
        run.define_metric('MAP', summary='max') # type: ignore
        run.define_metric('P@1', summary='max') # type: ignore
        run.define_metric('P@10', summary='max') # type: ignore
        run.define_metric('P@20', summary='max') # type: ignore
        run.define_metric('P@30', summary='max') # type: ignore
        run.define_metric('MRR', summary='max') # type: ignore
        run.define_metric('R@1', summary='max') # type: ignore
        run.define_metric('R@10', summary='max') # type: ignore
        run.define_metric('R@100', summary='max') # type: ignore
        run.define_metric('R@1000', summary='max') # type: ignore
        wandb.save('src/*', policy='now')
        wandb.save('scripts/code_rerank.sh', policy='now')
    if args.debug:
        print('Running in debug mode')
    print(args)
    combined_df = get_combined_df(args.data_path)
    bert_params = {
        'model_name': args.model_path,
        'psg_cnt': 5,
        'aggregation_strategy': 'maxp',
        'batch_size': args.batch_size,
        'use_gpu': args.use_gpu,
        'rerank_depth': 1000,
        'num_epochs': args.num_epochs,
        'lr': args.learning_rate,
        'num_positives': args.num_positives,
        'num_negatives': args.num_negatives,
        'train_depth': args.train_depth,
        'num_workers': args.num_workers,
        'train_commits': args.train_commits,
        'bm25_aggr_strategy': 'maxp',
        'output_length': args.output_length
    }
    main(args)
