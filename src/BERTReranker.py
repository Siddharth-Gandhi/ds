import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

import wandb
from bm25_v2 import BM25Searcher
from eval import ModelEvaluator, SearchEvaluator
from models import BERTReranker
from utils import (
    balance_labels,
    get_combined_df,
    get_recent_df,
    prepare_triplet_data_from_df,
    sanity_check_bertreranker,
    set_seed,
)

# set seed
set_seed(42)
gc.collect() # These commands help you when you face CUDA OOM error
torch.cuda.empty_cache()

def do_training(triplet_data, bert_reranker, hf_output_dir, args):
    def tokenize_hf(example):
        return bert_reranker.tokenizer(example['query'], example['passage'], truncation=True, padding='max_length', max_length=bert_reranker.max_seq_length, return_tensors='pt', add_special_tokens=True)
    print('Training the model...')
    print('Initial Label distribution:')
    print(triplet_data['label'].value_counts())

    # make label distribution more balanced
    triplet_data = balance_labels(triplet_data)

    print('Balanced Label distribution:')
    print(triplet_data['label'].value_counts())

    if args.sanity_check:
        print('Running sanity check on training data...')
        triplet_data = sanity_check_bertreranker(triplet_data)

    # Step 7: convert triplet_data to HuggingFace Dataset
    # convert triplet_data to HuggingFace Dataset
    if args.train_mode == 'regression':
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
        learning_rate=args.learning_rate,
        save_strategy='epoch',
        num_train_epochs=args.num_epochs,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        save_total_limit=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # logging_steps=1000,
        lr_scheduler_type='constant',
        logging_strategy='epoch',
        fp16=True,
        dataloader_num_workers=args.num_workers,
        report_to='wandb' if not args.debug else "none", # type: ignore
        )

    small_train_dataset = tokenized_train_dataset.shuffle(seed=42).select(range(100))
    small_val_dataset = tokenized_val_dataset.shuffle(seed=42).select(range(100))

    if args.debug:
        print('Running in debug mode, using small datasets')
        tokenized_train_dataset = small_train_dataset
        tokenized_val_dataset = small_val_dataset

    # Step 11: set up trainer
    trainer = Trainer(
        model = bert_reranker.model,
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
    metrics = ['MAP', 'P@1', 'P@10', 'P@20', 'P@30', 'MRR', 'R@1', 'R@10', 'R@100', 'R@1000']
    data_path = args.data_path
    repo_name = data_path.split('/')[-1]
    index_path = args.index_path
    # TODO remove K and n everywhere
    K = args.k
    n = args.n
    combined_df = get_combined_df(data_path)
    # ! important
    BM25_AGGR_STRAT = 'sump'


    # create eval directory to store results
    eval_path = os.path.join('out', repo_name, 'bert_reranker', args.eval_folder)

    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    bm25_searcher = BM25Searcher(index_path)
    evaluator = SearchEvaluator(metrics)
    model_evaluator = ModelEvaluator(bm25_searcher, evaluator, combined_df)

    # Reranking with BERT
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


    bert_reranker = BERTReranker(params, train_mode=args.train_mode)

    # hf_output_dir = os.path.join(data_path, 'models', model_name)
    hf_output_dir = os.path.join('models', repo_name, 'bert_reranker', args.eval_folder)

    best_model_path = os.path.join(hf_output_dir, 'best_model')

    if not os.path.exists(hf_output_dir):
        os.makedirs(hf_output_dir)

    if args.do_train:
        if args.use_gpt_train:
            gold_dir = os.path.join('gold', repo_name)
            if not os.path.exists(gold_dir):
                raise ValueError(f'Gold directory {gold_dir} does not exist, please run openai_transform.py first')

            gold_train_file = os.path.join(gold_dir, f'v2_{repo_name}_{args.openai_model}_train.parquet')
            if not os.path.exists(gold_train_file):
                raise ValueError(f'Gold train file {gold_train_file} does not exist, please run openai_transform.py first')

            recent_df = pd.read_parquet(gold_train_file)
            # ! rename column commit_message to original_message and transformed_message_gpt4 to commit_message
            recent_df = recent_df.rename(columns={'commit_message': 'original_message', f'transformed_message_{args.openai_model}': 'commit_message'})
        else:
            recent_df = get_recent_df(combined_df=combined_df, repo_name=repo_name, ignore_gold_in_training=args.ignore_gold_in_training)
            # Step 6: randomly sample params['train_commits'] commits for training or if less than that, use all commits
            if len(recent_df) < params['train_commits']:
                print(f'Number of commits in train_df: {len(recent_df)}')
                print(f'Using all commits for training')
                recent_df = recent_df.sample(len(recent_df))
            else:
                recent_df = recent_df.sample(params['train_commits'])

        cache_folder = os.path.join('cache', repo_name, 'bert_reranker', args.eval_folder)
        if not os.path.exists(cache_folder) and not args.triplet_cache_path:
            os.makedirs(cache_folder)
        triplet_cache = args.triplet_cache_path or os.path.join(cache_folder, 'triplet_data_cache.pkl')

        print(f'Number of commits in train_df: {len(recent_df)}')

        # Step 7: Prepare triplet data

        triplet_data = prepare_triplet_data_from_df(recent_df, bm25_searcher, search_depth=params['train_depth'], num_positives=params['num_positives'], num_negatives=params['num_negatives'], cache_file=triplet_cache, overwrite=args.overwrite_cache)

        do_training(triplet_data, bert_reranker, hf_output_dir, args)
    elif args.do_combined_train:
            print("Performing combined training on multiple repositories...")
            print(f'Found {len(args.repo_paths)} repositories: {args.repo_paths}')
            combined_triplet_data = pd.DataFrame()
            for data_path in args.repo_paths:
                if args.use_gpt_train:
                    triplet_cache = os.path.join(data_path, 'cache', 'gpt_triplet_data_cache.pkl')
                else:
                    triplet_cache = os.path.join(data_path, 'cache', 'triplet_data_cache.pkl')

                triplet_cache = args.triplet_cache_path or triplet_cache
                if os.path.exists(triplet_cache):
                    repo_triplet_data = pd.read_pickle(triplet_cache)
                    combined_triplet_data = pd.concat([combined_triplet_data, repo_triplet_data], ignore_index=True)
                else:
                    print(f"Warning: Triplet cache not found for {data_path}, skipping this repository.")

            if combined_triplet_data.empty:
                raise ValueError("No triplet data found in the specified repositories.")

            print(f'Shape of combined triplet data: {combined_triplet_data.shape}')

            combined_hf_output_dir = os.path.join('data', 'combined_commit_train' if not args.use_gpt_train else 'combined_gpt_train', args.eval_folder)
            if not os.path.exists(combined_hf_output_dir):
                os.makedirs(combined_hf_output_dir)

            do_training(combined_triplet_data, bert_reranker, combined_hf_output_dir, args)


    # common settings for evaluation
    if args.do_eval or args.eval_gold:
        if not args.best_model_path:
            print(f'WARNING: No best model path provided, using default path {best_model_path}')
            # raise ValueError('Best model path not provided, please provide a path to the best model')
        cur_best_model_path = args.best_model_path or best_model_path
        if not os.path.exists(cur_best_model_path):
            raise ValueError(f'Best model path {cur_best_model_path} does not exist, please train the model first')
        print(f'Loading model from {cur_best_model_path}...')
        if args.train_mode == 'regression':
            print('Using regression model')
            bert_reranker.model = AutoModelForSequenceClassification.from_pretrained(cur_best_model_path, num_labels=1, problem_type='regression')
        elif args.train_mode == 'classification':
            print('Using classification model')
            bert_reranker.model = AutoModelForSequenceClassification.from_pretrained(cur_best_model_path, num_labels=2)
        bert_reranker.model.to(bert_reranker.device)
        rerankers = [bert_reranker]

        gold_dir = os.path.join('gold', repo_name)
        if not os.path.exists(gold_dir):
            raise ValueError(f'Gold directory {gold_dir} does not exist, please run openai_transform.py first')
        # check if gold data exists
        gold_data_path = os.path.join(gold_dir, f'v2_{repo_name}_{args.openai_model}_gold.parquet')
        if not os.path.exists(gold_data_path):
            raise ValueError(f'Gold data {gold_data_path} does not exist, please run openai_transform.py first')
        print(f'Model: {args.openai_model}')
        gold_df = pd.read_parquet(gold_data_path)
        print(f'Found gold data for {repo_name} with shape {gold_df.shape} at {gold_data_path}')
        print(gold_df.info())


    if args.do_eval:
        bert_with_training_output_path = os.path.join(eval_path, 'commit')
        # ! not renaming transformed message into query so works
        bert_with_training_eval = model_evaluator.evaluate_sampling(n=n, k=K, output_folder_path=bert_with_training_output_path, aggregation_strategy=params['aggregation_strategy'], rerankers=rerankers, overwrite_eval=args.overwrite_eval, gold_df=gold_df)
        print("BERT Commit Evaluation")
        print(bert_with_training_eval)
        # Assuming bert_with_training_eval and bert_gold_eval are your dicts

    if args.eval_gold:
        bert_gold_output_path = os.path.join(eval_path, 'gold')
        assert gold_df[f'transformed_message_{args.openai_model}'].notnull().all()
        #! rename commit_message to original_message
        gold_df = gold_df.rename(columns={'commit_message': 'original_message'})
        #!rename transformed_message to commit_message
        gold_df = gold_df.rename(columns={f'transformed_message_{args.openai_model}': 'commit_message'})

        # get gold eval with reranking
        print('Running BERT on gold data...')
        bert_gold_eval = model_evaluator.evaluate_sampling(n=n, k=K, output_folder_path=bert_gold_output_path, aggregation_strategy=params['aggregation_strategy'], rerankers=rerankers, gold_df=gold_df, overwrite_eval=args.overwrite_eval)

        print("BERT Gold Evaluation")
        print(bert_gold_eval)
        if not args.debug:
            wandb.log(bert_gold_eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run BM25 and/or BERT Reranker evaluation.')
    parser.add_argument('--index_path', type=str, help='Path to the index directory.', required=True)
    parser.add_argument('--data_path', type=str, help='Path to the repository directory.', required=True)
    parser.add_argument('-k', '--k', type=int, default=1000, help='The number of top documents to retrieve (default: 1000)')
    parser.add_argument('-n', '--n', type=int, default=100, help='The number of commits to sample (default: 100)')
    parser.add_argument('-m', '--model_path', type=str, help='Path to the pretrained model.')
    parser.add_argument('-o', '--overwrite_cache', action='store_true', help='Overwrite existing cache files.')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('-e', '--num_epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5, help='Learning rate (default: 5e-5)')
    parser.add_argument('--run_name', type=str, help='Name of the run for wandb.')
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
    parser.add_argument('--sanity_check', action='store_true', help='Run sanity check on training data.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
    parser.add_argument('--do_combined_train', action='store_true', help='Train on combined data from multiple repositories.')
    parser.add_argument('--repo_paths', nargs='+', help='List of repository paths for combined training.', required='--do_combined_train' in sys.argv)
    parser.add_argument('--best_model_path', type=str, help='Path to the best model.')
    parser.add_argument('--ignore_gold_in_training', action='store_true', help='Ignore gold commits in training data.')
    parser.add_argument('--use_gpt_train', action='store_true', help='Use GPT transformed training data.')
    parser.add_argument('--eval_folder', type=str, help='Folder to store evaluation results for a particular experiment.', required=True)
    parser.add_argument('--notes', type=str, help='Notes for the run.', default='')
    parser.add_argument('--train_mode', choices=['regression', 'classification'], default='regression', help='Training mode for the model (default: regression')
    parser.add_argument('--triplet_cache_path', type=str, help='Path to the triplet data cache file.')
    args = parser.parse_args()
    if not args.debug:
        run = wandb.init(project='ds-bert_rerank', name=args.run_name, reinit=True, config=args, notes=args.notes) # type: ignore
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

        # save the file scripts/bert_rerank.sh to wandb
        # wandb.save('bert_rerank.sh', policy='now')
        wandb.save('src/*', policy='now')
        wandb.save('scripts/bert_rerank.sh', policy='now')
    print(args)
    main(args)