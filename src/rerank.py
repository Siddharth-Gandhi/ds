import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

from bm25 import average_precision_score, mean_reciprocal_rank, precision_at_k, search
from utils import get_combined_df, reverse_tokenize, tokenize


def sample_query(df, seed=42):
    """
    Sample a query from the dataframe
    """
    sampled_commit = df.drop_duplicates(subset='commit_id').sample(1, random_state=seed).iloc[0]
    return {
        'commit_message': sampled_commit['commit_message'],
        'commit_id': sampled_commit['commit_id'],
        'commit_date': sampled_commit['commit_date'],
        'actual_files_modified': df[df['commit_id'] == sampled_commit['commit_id']]['file_path'].tolist()
    }


def print_search_results(query, results, showOnlyActualModified = False):
    """
    Print the search results
    """
    actual_modified_files = query['actual_files_modified']
    for i in range(len(results)):
    # print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}')
    # print with repo name and file name
        obj = json.loads(results[i].raw)
        if (
            showOnlyActualModified
            and obj["file_path"] in actual_modified_files
            or not showOnlyActualModified
        ):
            commit_date = int(obj["commit_date"])
            print(f'{i+1:2} {obj["file_path"]:4} {results[i].score:.5f} {commit_date}')
def evaluate(hits, actual_modified_files, k=1000):
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

def prepare_data_from_df(df, searcher, n_positive=10, n_negative=10):
    data = []

    for _, row in df.iterrows():
        commit_message = row['commit_message']
        actual_files_modified = row['actual_files_modified']
        search_results = search(searcher, commit_message, row['commit_date'], 1000)

        # Get positive and negative samples
        positive_samples = [res for res in search_results if res in actual_files_modified][:n_positive]
        negative_samples = [res for res in search_results if res not in actual_files_modified][:n_negative]

        for sample in positive_samples:
            sample_msg  = reverse_tokenize(json.loads(sample.raw)['contents'])
            data.append((commit_message, sample_msg, 1))

        for sample in negative_samples:
            sample_msg  = reverse_tokenize(json.loads(sample.raw)['contents'])
            data.append((commit_message, sample_msg, 0))

    return data

def bm25_baseline_evaluation(df_test, searcher, k=1000):
    # Store evaluation results
    results = []

    for _, row in df_test.iterrows():
        commit_message = row['commit_message']
        actual_files_modified = row['actual_files_modified']

        # Retrieve documents using BM25
        search_results = search(searcher, commit_message, row['commit_date'], k)

        # Evaluate search results
        evaluation = evaluate(search_results, actual_files_modified, k)

        results.append(evaluation)

    # Aggregate the evaluation metrics
    metrics = {
        'MAP': np.mean([res['MAP'] for res in results]),
        'P@10': np.mean([res['P@10'] for res in results]),
        'P@100': np.mean([res['P@100'] for res in results]),
        'P@1000': np.mean([res['P@1000'] for res in results]),
        'MRR': np.mean([res['MRR'] for res in results]),
        f'Recall@{k}': np.mean([res[f'Recall@{k}'] for res in results])
    }

    return metrics


def rerank_with_model(search_results, model, mlp_model, tokenizer, device, commit_message):
    """Rerank search results using the provided model."""
    reranked_results = []
    scores = []

    for hit in tqdm(search_results):
        # Convert query and doc to BERT input format
        bert_input = get_bert_input(tokenizer, commit_message, json.loads(hit.raw)['contents'])
        input_ids = bert_input["input_ids"].to(device)
        attention_mask = bert_input["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            cls_output = outputs[0][:, 0]
            score = mlp_model(cls_output)
            scores.append(score.item())

    # Sort hits by scores
    reranked_results = [hit for _, hit in sorted(zip(scores, search_results), reverse=True, key=lambda pair: pair[0])]

    return reranked_results

def codebert_mlp_baseline_evaluation(df_test, searcher, model, mlp_model, tokenizer, device, k=1000):
    # Store evaluation results
    results = []


    # for _, row in tqdm(df_test.iterrows()):
    # tqdm with total=len(df_test) gives a nice progress bar
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        commit_message = row['commit_message']
        actual_files_modified = row['actual_files_modified']

        # Retrieve documents using BM25
        search_results = search(searcher, commit_message, row['commit_date'], k)

        # Rerank search results using CodeBERT + MLP model
        reranked_results = rerank_with_model(search_results, model, mlp_model, tokenizer, device, commit_message)

        # Evaluate reranked results
        evaluation = evaluate(reranked_results, actual_files_modified, k)

        results.append(evaluation)

    # Aggregate the evaluation metrics
    metrics = {
        'MAP': np.mean([res['MAP'] for res in results]),
        'P@10': np.mean([res['P@10'] for res in results]),
        'P@100': np.mean([res['P@100'] for res in results]),
        'P@1000': np.mean([res['P@1000'] for res in results]),
        'MRR': np.mean([res['MRR'] for res in results]),
        f'Recall@{k}': np.mean([res[f'Recall@{k}'] for res in results])
    }

    return metrics


def get_bert_input(tokenizer, query_message, commit_message):
    """
    Convert the query and commit message into BERT input format.
    """
    return tokenizer(query_message + " [SEP] " + commit_message, return_tensors="pt", truncation=True, padding=True, max_length=512)

def batch_get_bert_input(tokenizer, query_messages, commit_messages):
    """
    Convert batches of queries and commit messages into BERT input format.
    """
    input_ids_list = []
    attention_mask_list = []

    # First tokenize without padding
    for query, commit in zip(query_messages, commit_messages):
        tokenized = tokenizer(query + " [SEP] " + commit,
                              return_tensors="pt",
                              truncation=True,
                              padding=False,
                              max_length=512)
        input_ids_list.append(tokenized["input_ids"])
        attention_mask_list.append(tokenized["attention_mask"])

    # Find the maximum length in this batch
    max_length = max([x.size(1) for x in input_ids_list])

    # Pad all sequences to this max_length
    for idx in range(len(input_ids_list)):
        if input_ids_list[idx].size(1) < max_length:
            difference = max_length - input_ids_list[idx].size(1)
            padding_ids = torch.full((input_ids_list[idx].size(0), difference), tokenizer.pad_token_id, dtype=torch.long)
            input_ids_list[idx] = torch.cat([input_ids_list[idx], padding_ids], dim=-1)

            padding_mask = torch.zeros(input_ids_list[idx].size(0), difference, dtype=torch.long)
            attention_mask_list[idx] = torch.cat([attention_mask_list[idx], padding_mask], dim=-1)

    # Concatenate tokenized results into tensor batches
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_mask = torch.cat(attention_mask_list, dim=0)

    return {"input_ids": input_ids, "attention_mask": attention_mask}

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    # print gpu info
    print(f'GPU available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU device: {torch.cuda.get_device_name(0)}')
        print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory}')
    else:
        print('No GPU available, return')
        sys.exit(0)
    print(os.getcwd())


    repo_path = '2_8/django_django'
    idx_path = f'{repo_path}/index_commit_tokenized'

    df = get_combined_df(repo_path)

    print(df.info())

    searcher = LuceneSearcher(idx_path)

    # Step 1: Filter necessary columns
    filtered_df = df[['commit_date', 'commit_message', 'commit_id', 'file_path']]

    # Step 2: Group by commit_id
    grouped_df = filtered_df.groupby(['commit_id', 'commit_date', 'commit_message'])['file_path'].apply(list).reset_index()
    grouped_df.rename(columns={'file_path': 'actual_files_modified'}, inplace=True)

    # Step 3: Determine midpoint and filter dataframe
    midpoint_date = np.median(grouped_df['commit_date'])
    recent_df = grouped_df[grouped_df['commit_date'] > midpoint_date]
    print(f'Number of commits after midpoint date: {len(recent_df)}')
    # sys.exit(0)

    recent_df = recent_df.head(1000)

    # Step 4: Split recent dataframe and prepare data
    df_train, df_temp = train_test_split(recent_df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

    print('Preparing data...')

    train_data = prepare_data_from_df(df_train, searcher)
    val_data = prepare_data_from_df(df_val, searcher)
    test_data = prepare_data_from_df(df_test, searcher)

    # print size of data
    print(f'Train data size: {len(train_data)}')
    print(f'Val data size: {len(val_data)}')
    print(f'Test data size: {len(test_data)}')

    print(f'Train data sample: {train_data[0]}')


    print('BM25 Baseline Evaluation...')
    # Get BM25 baseline evaluation metrics
    bm25_baseline_metrics = bm25_baseline_evaluation(df_test, searcher)

    print(bm25_baseline_metrics)


    # Rerate the search results using CodeBERT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.to(device)
    model.eval()

    # Assuming the output dimension of CodeBERT is 768 (as is the case for many BERT variants)
    INPUT_DIM = 768
    HIDDEN_DIM = 256  # you can adjust this value based on your needs
    OUTPUT_DIM = 1
    LEARNING_RATE = 0.001
    # HYPERPARAMETERS
    NUM_EPOCHS = 10
    BATCH_SIZE = 128

    mlp_model = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)

    # Get CodeBERT + MLP baseline evaluation metrics (before training)
    # print('CodeBERT + MLP metrics before training')
    # codebert_mlp_baseline_metrics = codebert_mlp_baseline_evaluation(df_test, searcher, model, mlp_model, tokenizer, device)

    # print(codebert_mlp_baseline_metrics)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)



    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        mlp_model.train()

        for queries, commit_msgs, labels in tqdm(train_loader):  # Batched data
            # Convert batch of queries and commit messages to BERT input format.
            # bert_input = batch_get_bert_input(queries, commit_msgs)
            bert_input = batch_get_bert_input(tokenizer, queries, commit_msgs)
            input_ids = bert_input["input_ids"].to(device)
            attention_mask = bert_input["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            cls_output = outputs[0][:,0,:]

            optimizer.zero_grad()
            predictions = mlp_model(cls_output).squeeze()  # Assuming output has shape (batch_size, 1)
            label_tensor = labels.float().to(device)
            loss = criterion(predictions, label_tensor)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation (optional)
        val_loss = 0.0
        mlp_model.eval()
        with torch.no_grad():
            for queries, commit_msgs, labels in tqdm(val_loader):  # Batched data
                # bert_input = batch_get_bert_input(queries, commit_msgs)
                bert_input = batch_get_bert_input(tokenizer, queries, commit_msgs)
                input_ids = bert_input["input_ids"].to(device)
                attention_mask = bert_input["attention_mask"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                cls_output = outputs[0][:,0,:]

                predictions = mlp_model(cls_output).squeeze()  # Assuming output has shape (batch_size, 1)
                label_tensor = labels.float().to(device)
                loss = criterion(predictions, label_tensor)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} => Training loss: {train_loss/len(train_loader)}, Validation loss: {val_loss/len(val_loader)}")

    # Get CodeBERT + MLP baseline evaluation metrics (after training)
    print('CodeBERT + MLP metrics after training')
    codebert_mlp_baseline_metrics = codebert_mlp_baseline_evaluation(df_test, searcher, model, mlp_model, tokenizer, device)
    print(codebert_mlp_baseline_metrics)