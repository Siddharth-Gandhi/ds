import glob
import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset

os.environ['TIKTOKEN_CACHE_DIR'] = ""
ENCODING = 'p50k_base'
enc = tiktoken.get_encoding(ENCODING)
assert enc.decode(enc.encode("hello world")) == "hello world"

def tokenize(text):
    return ' '.join(map(str, enc.encode(text, disallowed_special=())))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_combined_df(repo_dir):
    all_files = glob.glob(os.path.join(repo_dir, '*.parquet'))
    all_files.sort()
    all_dataframes = [pd.read_parquet(file) for file in all_files]
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df['commit_date'] = (combined_df['commit_date'].astype('int64') / 1e9).astype('int64')
    return combined_df


def count_commits(repo_dir):
    combined_df = get_combined_df(repo_dir)
    return combined_df.commit_id.nunique()


def reverse_tokenize(text):
    return enc.decode(list(map(int, text.split(' '))))


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




class TripletDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        query, passage, label = self.data[index]

        # tokenize the query passage pairs and create tensors for the input ids, attention masks, and token type ids
        encoded_pair = self.tokenizer.encode_plus([query, passage], max_length=self.max_seq_length, truncation=True, padding='max_length', return_tensors='pt', add_special_tokens=True)

        input_ids = encoded_pair['input_ids'].squeeze(0)
        attention_mask = encoded_pair['attention_mask'].squeeze(0)

        return input_ids, attention_mask, label



def prepare_triplet_data_from_df(df, searcher, depth, n_positive, n_negative, cache_file, overwrite=False):
    # Check if cache file exists
    if os.path.exists(cache_file) and not overwrite:
        print(f"Loading data from cache file: {cache_file}")
        with open(cache_file, 'rb') as file:
            return pickle.load(file)


    data = []
    print(f'Preparing data from dataframe of size: {len(df)} with depth: {depth}')

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
    # Write data to cache file
    with open(cache_file, 'wb') as file:
        pickle.dump(data, file)
        print(f"Saved data to cache file: {cache_file}")
    return data