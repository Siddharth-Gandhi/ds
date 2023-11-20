import argparse
import os

# import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.init as init
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from bm25_v2 import BM25Searcher
from eval import ModelEvaluator, SearchEvaluator
from utils import (
    AggregatedSearchResult,
    TripletDataset,
    get_combined_df,
    prepare_triplet_data_from_df,
    set_seed,
)

# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_dim, output_dim)
#         # self.fc1 = nn.Linear(input_dim, hidden_dim)
#         # self.relu = nn.ReLU()
#         # self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self.sigmoid = nn.Sigmoid()
#         # Initialize weights
#     #     self.init_weights()

#     # def init_weights(self):
#     #     init.xavier_uniform_(self.fc1.weight)
#     #     init.xavier_uniform_(self.fc2.weight)


#         # init.zeros_(self.fc1.bias)
#         # init.zeros_(self.fc2.bias)

#     def forward(self, x):
#         x = self.fc1(x)
#         # x = self.relu(x)
#         # x = self.fc2(x)
#         x = self.sigmoid(x)
#         return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Adding an intermediate layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class BERTReranker:
    # def __init__(self, model_name, psg_len, psg_cnt, psg_stride, agggreagtion_strategy, batch_size, use_gpu=True):
    def __init__(self, parameters):
        self.parameters = parameters
        self.model_name = parameters['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModel.from_pretrained(self.model_name, num_labels=1)
        self.model = AutoModel.from_pretrained(self.model_name)
        # self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() and parameters['use_gpu'] else "cpu")
        self.model.to(self.device)

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
        # hidden_dim = parameters['HIDDEN_DIM']   # Example hidden size
        # output_dim = parameters['OUTPUT_DIM']  # We want a single score as output

        self.mlp = MLP(self.model.config.hidden_size, parameters['hidden_dim'], 1, parameters['dropout_prob']).to(self.device)

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
                (query, result.commit_message)
                for result in agg_result.contributing_results[: self.psg_cnt]
            )

        if not query_passage_pairs:
            print('WARNING: No query passage pairs to rerank')
            print(query, aggregated_results, self.psg_cnt)
            return aggregated_results

        # tokenize the query passage pairs
        encoded_pairs = [self.tokenizer.encode_plus([query, passage], max_length=self.max_seq_length, truncation=True, padding='max_length', return_tensors='pt', add_special_tokens=True) for query, passage in query_passage_pairs]

        # create tensors for the input ids, attention masks
        input_ids = torch.stack([encoded_pair['input_ids'].squeeze() for encoded_pair in encoded_pairs], dim=0) # type: ignore
        attention_masks = torch.stack([encoded_pair['attention_mask'].squeeze() for encoded_pair in encoded_pairs], dim=0) # type: ignore

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
        with torch.no_grad():
            for batch in dataloader:
                # Unpack the batch and move it to GPU
                b_input_ids, b_attention_mask = batch
                b_input_ids = b_input_ids.to(self.device)
                b_attention_mask = b_attention_mask.to(self.device)

                # Get the pooled output from BERT's [CLS] token
                # pooled_output = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask).pooler_output

                cls_output = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask).last_hidden_state[:, 0, :]

                # # Pass the pooled output through the MLP to get the scores
                # logits = self.mlp(pooled_output).squeeze(-1) # type: ignore
                logits = self.mlp(cls_output).squeeze(-1) # type: ignore

                # # Collect the scores (detach them from the computation graph and move to CPU)
                scores.extend(logits.detach().cpu().numpy())


                # outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask)
                # logits = outputs.logits
                # scores.extend(logits.detach().cpu().numpy().squeeze(-1))

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


def train_reranker(bertranker, train_dataloader, validation_dataloader, freeze_bert, save_dir):
    # Set BERT parameters to not require gradients
    save_dir = os.path.join(save_dir, 'models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # for param in bertranker.model.parameters():
    #     param.requires_grad = False if freeze_bert else True


    # if freeze_bert:
    #     optimizer = torch.optim.Adam(bertranker.mlp.parameters(), lr=bertranker.parameters['mlp_lr'], weight_decay=bertranker.parameters['weight_decay'])
    # else:
    #     optimizer = torch.optim.Adam([
    #         {'params': bertranker.model.parameters(), 'lr': bertranker.parameters['bert_lr'], 'weight_decay': bertranker.parameters['weight_decay']},
    #         {'params': bertranker.mlp.parameters(), 'lr': bertranker.parameters['mlp_lr'], 'weight_decay': bertranker.parameters['weight_decay']}
    #             ], lr=bertranker.parameters['mlp_lr'])

    optimizer = torch.optim.Adam(bertranker.model.parameters(), lr=bertranker.parameters['bert_lr'])

    # one optimizer for both BERT and MLP with same learning rate


    print(f'Optimizer: {optimizer}')

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    # Set up the loss function
    criterion = nn.BCEWithLogitsLoss()  #

    # Set up training variables
    num_epochs = bertranker.parameters['num_epochs']
    # print train and val dataloader sizes
    print(f'Train dataloader size: {len(train_dataloader)}')
    print(f'Val dataloader size: {len(validation_dataloader)}')
    # Training loop
    print('Starting training loop')

    if freeze_bert:
        print('BERT is frozen, training only MLP')
    else:
        print('BERT is unfrozen, training BERT and MLP')
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    # model_name = 'bert_reranker_frozen' if freeze_bert else 'bert_reranker'
    model_name = bertranker.parameters['model_name'].replace('/', '_') + '_frozen' if freeze_bert else bertranker.parameters['model_name'].replace('/', '_')
    model_name += '_frozen' if freeze_bert else ''
    print(f'Model name: {model_name}')
    # for epoch in range(epochs):
    for epoch in tqdm(range(num_epochs)):
        # self.model.eval()  # Make sure the BERT model is in evaluation mode
        # if freeze_bert:
        #     bertranker.model.eval()  # BERT finetuning should be in eval mode
        # else:
        #     bertranker.model.train()  # BERT finetuning should be in train mode

        bertranker.model.train()  # BERT finetuning should be in train mode
        bertranker.mlp.train()  # MLP should be in training mode
        total_loss = 0

        for batch in train_dataloader:
            # breakpoint()
            b_input_ids, b_attention_mask, b_labels = batch
            b_input_ids = b_input_ids.to(bertranker.device)
            b_attention_mask = b_attention_mask.to(bertranker.device)
            b_labels = b_labels.float().to(bertranker.device)

            # Forward pass
            if freeze_bert:
                with torch.no_grad():
                    # pooled_output = bertranker.model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask).pooler_output
                    cls_output = bertranker.model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask).last_hidden_state[:, 0, :]

            else:
                pooled_output = bertranker.model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask).pooler_output
            cls_output = bertranker.model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask).last_hidden_state[:, 0, :]

            logits = bertranker.mlp(cls_output).squeeze(-1) # type: ignore

            # outputs = bertranker.model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask)
            # logits = bertranker.mlp(outputs.logits).squeeze(-1) # type: ignore
            # logits = outputs.logits.squeeze(-1) # type: ignore
            # Compute loss
            loss = criterion(logits, b_labels)
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Validation step
        bertranker.model.eval()
        bertranker.mlp.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for batch in validation_dataloader:
                b_input_ids, b_attention_mask, b_labels = batch
                b_input_ids = b_input_ids.to(bertranker.device)
                b_attention_mask = b_attention_mask.to(bertranker.device)
                b_labels = b_labels.float().to(bertranker.device)

                pooled_output = bertranker.model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask).pooler_output
                cls_output = bertranker.model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask).last_hidden_state[:, 0, :]
                logits = bertranker.mlp(cls_output).squeeze(-1) # type: ignore

                # outputs = bertranker.model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask)
                # logits = outputs.logits.squeeze(-1) # type: ignore

                # Compute loss
                loss = criterion(logits, b_labels.float())
                total_eval_loss += loss.item()

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # scheduler.step(avg_val_loss)
        # Save losses
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Average training loss: {avg_train_loss}")
        print(f"Validation Loss: {avg_val_loss}")
        print(f'Best validation loss: {best_val_loss}')

        # save graph of losses
        plt.plot(train_losses, label='Training loss', color='blue', linestyle='dashed', linewidth=1, marker='o', markerfacecolor='blue', markersize=3)
        plt.plot(val_losses, label='Validation loss', color='red', linestyle='dashed', linewidth=1, marker='o', markerfacecolor='red', markersize=3)
        plt.legend(frameon=False)
        plt.savefig(os.path.join(save_dir, f'{model_name}_losses.png'))
        plt.close()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # model name with frozen or unfrozen bert
            save_path = os.path.join(save_dir, f'{model_name}_best_model.pth')
            mlp_save_path = os.path.join(save_dir, f'{model_name}_best_mlp.pth')
            torch.save(bertranker.model.state_dict(), save_path)
            torch.save(bertranker.mlp.state_dict(), mlp_save_path)

            print(f"Model saved with validation loss: {best_val_loss}")

            # evaluate on train set



        # Here you can add early stopping based on validation loss

    print("Training complete!")



def main(args):
    # magic seed
    set_seed(42)
    metrics = ['MAP', 'P@10', 'P@100', 'P@1000', 'MRR', 'Recall@100', 'Recall@1000']
    repo_path = args.repo_path
    index_path = args.index_path
    K = args.k
    n = args.n
    combined_df = get_combined_df(repo_path)
    BM25_AGGR_STRAT = 'sump'

    eval_path = os.path.join(repo_path, 'eval')
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    bm25_searcher = BM25Searcher(index_path)
    evaluator = SearchEvaluator(metrics)
    model_evaluator = ModelEvaluator(bm25_searcher, evaluator, combined_df)

    bm25_output_path = os.path.join(eval_path, f'bm25_baseline_N{n}_K{K}_metrics.txt')
    print(f'BM25 output path: {bm25_output_path}')

    bm25_baseline_eval = model_evaluator.evaluate_sampling(n=n, k=K, output_file_path=bm25_output_path, aggregation_strategy=BM25_AGGR_STRAT, repo_path=repo_path)

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
        'batch_size': 16,
        # 'batch_size': 512,
        # 'batch_size': 1,
        'use_gpu': True,
        'rerank_depth': 250,
        'num_epochs': 3,
        # 'mlp_lr': 1e-2,
        'mlp_lr': 1e-3,
        'bert_lr': 5e-5,
        'hidden_dim': 128,
        'num_positives': 20,
        'num_negatives': 20,
        'train_depth': 1000,
        'num_workers': 8,
        # 'weight_decay': 0.01,
        # 'dropout_prob': 0.5,
    }


    bert_reranker = BERTReranker(params)
    rerankers = [bert_reranker]

    save_model_name = params['model_name'].replace('/', '_')

    # print(combined_df.info())

    # Step 1: Filter necessary columns
    filtered_df = combined_df[['commit_date', 'commit_message', 'commit_id', 'file_path']]

    # Step 2: Group by commit_id
    grouped_df = filtered_df.groupby(['commit_id', 'commit_date', 'commit_message'])['file_path'].apply(list).reset_index()
    grouped_df.rename(columns={'file_path': 'actual_files_modified'}, inplace=True)

    # Step 3: Determine midpoint and filter dataframe
    midpoint_date = np.median(grouped_df['commit_date'])
    recent_df = grouped_df[grouped_df['commit_date'] > midpoint_date]
    print(f'Number of commits after midpoint date: {len(recent_df)}')
    # sys.exit(0)

    recent_df = recent_df.head(2000)

    # Step 4: Split recent dataframe and prepare data
    # TODO remove magic numbers
    df_train, df_temp = train_test_split(recent_df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

    print('Preparing data...')

    train_depth = params['train_depth']
    if not os.path.exists(os.path.join(repo_path, 'cache')):
        os.makedirs(os.path.join(repo_path, 'cache'))
    train_cache = os.path.join(repo_path, 'cache', 'train_data_cache.pkl')
    val_cache = os.path.join(repo_path, 'cache', 'val_data_cache.pkl')
    test_cache = os.path.join(repo_path, 'cache', 'test_data_cache.pkl')


    train_data = prepare_triplet_data_from_df(df_train, bm25_searcher, depth=train_depth, n_positive=params['num_positives'], n_negative=params['num_negatives'], cache_file=train_cache, overwrite=args.overwrite_cache)
    val_data = prepare_triplet_data_from_df(df_val, bm25_searcher, depth=train_depth, n_positive=params['num_positives'], n_negative=params['num_negatives'], cache_file=val_cache, overwrite=args.overwrite_cache)
    test_data = prepare_triplet_data_from_df(df_test, bm25_searcher, depth=train_depth, n_positive=params['num_positives'], n_negative=params['num_negatives'], cache_file=test_cache, overwrite=args.overwrite_cache)

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

    train_dataset = TripletDataset(train_data, bert_reranker.tokenizer, bert_reranker.max_seq_length)
    val_dataset = TripletDataset(val_data, bert_reranker.tokenizer, bert_reranker.max_seq_length)
    test_dataset = TripletDataset(test_data, bert_reranker.tokenizer, bert_reranker.max_seq_length)

    # Step 5: train the MLP
    train_dataloader = DataLoader(train_dataset, batch_size=bert_reranker.batch_size, shuffle=True, num_workers=params['num_workers'])
    val_dataloader = DataLoader(val_dataset, batch_size=bert_reranker.batch_size, shuffle=False, num_workers=params['num_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=bert_reranker.batch_size, shuffle=False, num_workers=params['num_workers'])

    # bert_reranker.train_mlp(train_dataloader, val_dataloader)
    train_reranker(bert_reranker, train_dataloader, val_dataloader, freeze_bert=args.freeze_bert, save_dir=repo_path)

    reranker_output_file = f"925_bert_reranker_{save_model_name}_N{args.n}_K{args.k}_non_frozen_metrics.txt" if not args.freeze_bert else f"bert_reranker_{save_model_name}_N{args.n}_K{args.k}_frozen_metrics.txt"

    # reranker_output_file = f"bert_reranker_{save_model_name}_N{args.n}_K{args.k}_without_mlp_metrics.txt"
    reranker_output_path = os.path.join(eval_path, reranker_output_file)

    bert_reranker_eval = model_evaluator.evaluate_sampling(n=n, k=K, output_file_path=reranker_output_path, aggregation_strategy='sump', rerankers=rerankers, repo_path=repo_path)

    print("BERT Reranker Evaluation")
    print(bert_reranker_eval)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BM25 and BERT Reranker evaluation.')
    parser.add_argument('--index_path', type=str, help='Path to the index directory.', required=True)
    parser.add_argument('--repo_path', type=str, help='Path to the repository directory.', required=True)
    parser.add_argument('-k', '--k', type=int, default=1000, help='The number of top documents to retrieve (default: 1000)')
    parser.add_argument('-n', '--n', type=int, default=100, help='The number of commits to sample (default: 100)')
    parser.add_argument('-o', '--overwrite_cache', action='store_true', help='Overwrite existing cache files.')
    parser.add_argument('--freeze_bert', action='store_true', help='Freeze BERT layers during training.')
    args = parser.parse_args()
    print(args)
    main(args)


