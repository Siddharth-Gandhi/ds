import argparse
from typing import List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from bm25_v2 import BM25Searcher
from eval import ModelEvaluator, SearchEvaluator
from utils import AggregatedSearchResult, get_combined_df


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

def prepare_data_from_df(df, searcher, depth=100, n_positive=10, n_negative=10):
    data = []
    print(f'Preparing data from dataframe of size: {len(df)}')

    for _, row in df.iterrows():
        commit_message = row['commit_message']
        actual_files_modified = row['actual_files_modified']
        # search_results = search(searcher, commit_message, row['commit_date'], 1000)

        # search_results = searcher.search(commit_message, row['commit_date'], 100)
        search_results = searcher.pipeline(commit_message, row['commit_date'], depth, 'sump')

        # flatten the contributing results for each aggregated result
        search_results = [result for agg_result in search_results for result in agg_result.contributing_results]

        # efficiently get the top n_positive and n_negative samples
        positive_samples = []
        negative_samples = []

        for result in search_results:
            if result.file_path in actual_files_modified and len(positive_samples) < n_positive:
                positive_samples.append(result.commit_msg)
            elif result.file_path not in actual_files_modified and len(negative_samples) < n_negative:
                negative_samples.append(result.commit_msg)

            if len(positive_samples) == n_positive and len(negative_samples) == n_negative:
                break

        # Get positive and negative samples
        # positive_samples = [res.commit_msg for res in search_results if res.file_path in actual_files_modified][:n_positive]
        # negative_samples = [res.commit_msg for res in search_results if res.file_path not in actual_files_modified][:n_negative]

        for sample_msg in positive_samples:
            # sample_msg  = reverse_tokenize(json.loads(sample.raw)['contents'])
            data.append((commit_message, sample_msg, 1))

        for sample_msg in negative_samples:
            # sample_msg  = reverse_tokenize(json.loads(sample.raw)['contents'])
            data.append((commit_message, sample_msg, 0))

    return data

class BERTReranker:
    # def __init__(self, model_name, psg_len, psg_cnt, psg_stride, agggreagtion_strategy, batch_size, use_gpu=True):
    def __init__(self, parameters):
        self.parameters = parameters
        self.model_name = parameters['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModel.from_pretrained(self.model_name, num_labels=1)
        self.model = AutoModel.from_pretrained(self.model_name)
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
        self.psg_cnt = parameters.get('psg_cnt', 1)
        # self.psg_stride = parameters.get('psg_stride', self.psg_len)
        self.aggregation_strategy = parameters['aggregation_strategy']
        self.batch_size = parameters['batch_size']
        # self.max_title_len = parameters.get('max_title_len', 0)
        # self.use_title = self.max_title_len > 0
        self.rerank_depth = parameters.get('rerank_depth', 100)
        # self.max_seq_length = parameters.get('max_seq_length', 512)
        self.max_seq_length = self.tokenizer.model_max_length

        print(f"Initialized BERT reranker with parameters: {parameters}")

        input_dim = parameters.get('INPUT_DIM', 768)  # Default BERT hidden size
        hidden_dim = parameters.get('HIDDEN_DIM', 512)  # Example hidden size
        output_dim = parameters.get('OUTPUT_DIM', 1)  # We want a single score as output
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(parameters.get('DROPOUT_RATE', 0.1)),  # Dropout for regularization
        #     nn.Linear(hidden_dim, output_dim)
        # ).to(self.device)

        self.mlp = MLP(input_dim, hidden_dim, output_dim).to(self.device)

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

        if len(query_passage_pairs) == 0:
            print('WARNING: No query passage pairs to rerank')
            return aggregated_results
        # query_passage_pairs = [(query, result.commit_msg) for aggregated_result in aggregated_results for result in aggregated_result.contributing_results]

        # print('Flattened query passage pairs')

        # tokenize the query passage pairs
        encoded_pairs = [self.tokenizer.encode_plus([query, passage], max_length=self.max_seq_length, truncation=True, padding='max_length', return_tensors='pt', add_special_tokens=True) for query, passage in query_passage_pairs]

        # print('Encoded query passage pairs')

        # create tensors for the input ids, attention masks
        input_ids = torch.cat([encoded_pair['input_ids'] for encoded_pair in encoded_pairs], dim=0) # type: ignore
        attention_masks = torch.cat([encoded_pair['attention_mask'] for encoded_pair in encoded_pairs], dim=0) # type: ignore

        # Create a dataloader for feeding the data to the model
        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        # print('Created dataloader')

        scores = self.get_scores(dataloader, self.model)

        score_index = 0
        for agg_result in aggregated_results:
            # Each aggregated result gets a slice of the scores equal to the number of contributing results it has
            end_index = score_index + len(agg_result.contributing_results)
            cur_passage_scores = scores[score_index:end_index]
            score_index = end_index

            # Aggregate the scores for the current aggregated result
            agg_score = self.aggregate_scores(cur_passage_scores)
            agg_result.score = agg_score  # Assign the aggregated score

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

                # # Forward pass, get logit predictions
                # outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask)
                # # logits = outputs.logits.data.cpu().numpy()
                # logits = outputs.logits.data.cpu().numpy().flatten()

                # # Move logits to CPU and convert to probabilities (optional)
                # # probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

                # # Collect the scores
                # scores.extend(logits)


                # Get the pooled output from BERT's [CLS] token
                pooled_output = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask).pooler_output

                # Pass the pooled output through the MLP to get the scores
                logits = self.mlp(pooled_output).squeeze(-1) # type: ignore

                # Collect the scores (detach them from the computation graph and move to CPU)
                scores.extend(logits.detach().cpu().numpy())

        return scores

    def train_mlp(self, train_dataloader, validation_dataloader):
        # Set BERT parameters to not require gradients
        for param in self.model.parameters():
            param.requires_grad = False

        # Set up the optimizer. Only parameters of the MLP will be updated.
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.parameters.get('LEARNING_RATE', 1e-4))

        # Set up the loss function
        criterion = nn.BCEWithLogitsLoss()  #

        # Set up training variables
        epochs = self.parameters.get('EPOCHS', 10)
        # Training loop

        print('Starting training loop')
        # for epoch in range(epochs):
        for epoch in tqdm(range(epochs)):
            self.model.eval()  # Make sure the BERT model is in evaluation mode
            self.mlp.train()  # MLP should be in training mode
            total_loss = 0

            for batch in train_dataloader:
                # b_input_ids, b_attention_mask, b_labels = batch
                queries, commits, b_labels = batch

                # tokenize the query passage pairs and create tensors for the input ids, attention masks, and token type ids
                encoded_pairs = [self.tokenizer.encode_plus([query, passage], max_length=self.max_seq_length, truncation=True, padding='max_length', return_tensors='pt', add_special_tokens=True) for query, passage in zip(queries, commits)]

                b_input_ids = torch.cat([encoded_pair['input_ids'] for encoded_pair in encoded_pairs], dim=0) # type: ignore
                b_attention_mask = torch.cat([encoded_pair['attention_mask'] for encoded_pair in encoded_pairs], dim=0) # type: ignore

                # b_input_ids = b_input_ids.to(self.device)
                # b_attention_mask = b_attention_mask.to(self.device)

                # tokenize the query passage pairs

                # b_labels = b_labels.to(self.device)
                b_input_ids = b_input_ids.to(self.device)
                b_attention_mask = b_attention_mask.to(self.device)
                b_labels = b_labels.float().to(self.device)

                # Forward pass
                with torch.no_grad():  # No need to calculate gradients for BERT
                    pooled_output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask).pooler_output
                logits = self.mlp(pooled_output).squeeze(-1) # type: ignore

                # Compute loss
                loss = criterion(logits, b_labels.float())
                total_loss += loss.item()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Calculate average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)

            # Validation step
            self.model.eval()
            self.mlp.eval()
            total_eval_loss = 0
            for batch in validation_dataloader:
                queries, commits, b_labels = batch

                # tokenize the query passage pairs and create tensors for the input ids, attention masks, and token type ids
                encoded_pairs = [self.tokenizer.encode_plus([query, passage], max_length=self.max_seq_length, truncation=True, padding='max_length', return_tensors='pt', add_special_tokens=True) for query, passage in zip(queries, commits)]

                b_input_ids = torch.cat([encoded_pair['input_ids'] for encoded_pair in encoded_pairs], dim=0) # type: ignore
                b_attention_mask = torch.cat([encoded_pair['attention_mask'] for encoded_pair in encoded_pairs], dim=0) # type: ignore

                # b_input_ids = b_input_ids.to(self.device)
                # b_attention_mask = b_attention_mask.to(self.device)

                # tokenize the query passage pairs

                # b_labels = b_labels.to(self.device)

                b_input_ids = b_input_ids.to(self.device)
                b_attention_mask = b_attention_mask.to(self.device)
                b_labels = b_labels.float().to(self.device)

                with torch.no_grad():
                    pooled_output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask).pooler_output
                    logits = self.mlp(pooled_output).squeeze(-1) # type: ignore

                # Compute loss
                loss = criterion(logits, b_labels.float())
                total_eval_loss += loss.item()

            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Average training loss: {avg_train_loss}")
            print(f"Validation Loss: {avg_val_loss}")

            # Here you can add early stopping based on validation loss

        print("Training complete!")

    def aggregate_scores(self, passage_scores):
        """
        Aggregate passage scores based on the specified strategy.
        """
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
        reranked_results = self.rerank(query, aggregated_results)
        return reranked_results

def main(args):
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
        'psg_cnt': 3,
        'psg_stride': 32,
        'aggregation_strategy': 'sump',
        'batch_size': 512,
        'use_gpu': True,
        'rerank_depth': 1000,
        'EPOCHS': 10
        # 'max_seq_length': 512,
    }


    bert_reranker = BERTReranker(params)
    rerankers = [bert_reranker]


    df = get_combined_df(repo_path)
    print(df.info())

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

    train_depth = 1000
    train_data = prepare_data_from_df(df_train, bm25_searcher, depth=train_depth)
    val_data = prepare_data_from_df(df_val, bm25_searcher, depth=train_depth)
    test_data = prepare_data_from_df(df_test, bm25_searcher, depth=train_depth)

    # get distribution of labels
    train_labels = [label for _, _, label in train_data]
    val_labels = [label for _, _, label in val_data]
    test_labels = [label for _, _, label in test_data]

    # print size of data
    print(f'Train data size: {len(train_data)}')
    print(f'Val data size: {len(val_data)}')
    print(f'Test data size: {len(test_data)}')

    print(f'Train data sample: {train_data[0]}')

    print(f'Train label distribution: {np.unique(train_labels, return_counts=True)}')
    print(f'Val label distribution: {np.unique(val_labels, return_counts=True)}')
    print(f'Test label distribution: {np.unique(test_labels, return_counts=True)}')

    # Step 5: train the MLP
    train_dataloader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=True)

    bert_reranker.train_mlp(train_dataloader, val_dataloader)

    bert_reranker_eval = model_evaluator.evaluate_sampling(n=n, k=K, output_file='bert_reranker_metrics.txt', aggregation_strategy='sump', rerankers=rerankers, repo_path=repo_path)

    print("BERT Reranker Evaluation")
    print(bert_reranker_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BM25 and BERT Reranker evaluation.')
    parser.add_argument('index_path', type=str, help='Path to the index directory.')
    parser.add_argument('repo_path', type=str, help='Path to the repository directory.')
    parser.add_argument('-k', '--k', type=int, default=1000, help='The number of top documents to retrieve (default: 1000)')
    parser.add_argument('-n', '--n', type=int, default=100, help='The number of commits to sample (default: 100)')
    args = parser.parse_args()
    print(args)
    main(args)