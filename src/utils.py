import glob
import json
import math
import os
import pickle
import random
from turtle import pos

import numpy as np
import pandas as pd
import tiktoken
import torch
import tqdm
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
    def __init__(self, commit_id, file_path, score, commit_date, commit_message):
        self.commit_id = commit_id
        self.file_path = file_path
        self.score = score
        self.commit_date = commit_date
        self.commit_message = commit_message


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


class AggregatedCommitResult:
    def __init__(self, commit_id, aggregated_score, contributing_results):
        self.commit_id = commit_id
        self.score = aggregated_score
        self.contributing_results = contributing_results

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(commit_id={self.commit_id!r}, score={self.score}, " \
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

def prepare_triplet_data_from_df(df, searcher, search_depth, num_positives, num_negatives, cache_file, overwrite=False):
    # Check if cache file exists
    if cache_file and os.path.exists(cache_file) and not overwrite:
        print(f"Loading data from cache file: {cache_file}")
        with open(cache_file, 'rb') as file:
            return pickle.load(file)


    data = []
    print(f'Preparing data from dataframe of size: {len(df)} with search_depth: {search_depth}')
    # for _, row in df.iterrows():
    total_positives, total_negatives = 0, 0
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        cur_positives = 0
        cur_negatives = 0
        pos_commit_ids = set()
        neg_commit_ids = set()
        commit_message = row['commit_message']
        actual_files_modified = row['actual_files_modified']

        agg_search_results = searcher.pipeline(commit_message, row['commit_date'], search_depth, 'sump', aggregate_on='commit')

        # for each agg_result, find out how many files it has edited are in actual_files_modified and sort by score

        for agg_result in agg_search_results:
            agg_result_files = set([result.file_path for result in agg_result.contributing_results])
            intersection = agg_result_files.intersection(actual_files_modified)
            # TODO maybe try this for training
            # agg_result.score = len(intersection) / len(agg_result_files) # how focused the commit is
            agg_result.score = len(intersection) / len(agg_result_files) # how focused the commit is
            # agg_result.score = math.log(cur_score+1)
            # agg_result.score = len(intersection)

        agg_search_results.sort(key=lambda res: res.score, reverse=True)

        # go from top to bottom, first num_positives non-0 scores are positive samples and the next num_negatives are negative samples
        for agg_result in agg_search_results:
            cur_commit_msg = agg_result.contributing_results[0].commit_message
            if cur_positives < num_positives and agg_result.score > 0:
                # meaning there is at least one file in the agg_result that is in actual_files_modified
                # pos_commits.append(agg_result)
                data.append((commit_message, cur_commit_msg, 1))
                cur_positives += 1
                pos_commit_ids.add(agg_result.commit_id)
            elif cur_negatives < num_negatives:
                # neg_commits.append(agg_result)
                data.append((commit_message, cur_commit_msg, 0))
                cur_negatives += 1
                neg_commit_ids.add(agg_result.commit_id)
            if cur_positives == num_positives and cur_negatives == num_negatives:
                break

        assert len(pos_commit_ids.intersection(neg_commit_ids)) == 0, 'Positive and negative commit ids should not intersect'
        # print(f"Total positives: {cur_positives}, Total negatives: {cur_negatives}")
        total_positives += cur_positives
        total_negatives += cur_negatives

    # convert to pandas dataframe
    data = pd.DataFrame(data, columns=['query', 'passage', 'label'])

    # Write data to cache file
    if cache_file:
        with open(cache_file, 'wb') as file:
            pickle.dump(data, file)
            print(f"Saved data to cache file: {cache_file}")

    # print distribution of labels
    print(f"Total positives: {total_positives}, Total negatives: {total_negatives}")
    # print percentage of positives and negatives
    denom = total_positives + total_negatives
    print(f"Percentage of positives: {total_positives / denom}, Percentage of negatives: {total_negatives / denom}")
    return data

# OLD CODE
# def prepare_triplet_data_from_df(df, searcher, search_depth, num_positives, num_negatives, cache_file, overwrite=False):
#     # Check if cache file exists
#     if os.path.exists(cache_file) and not overwrite:
#         print(f"Loading data from cache file: {cache_file}")
#         with open(cache_file, 'rb') as file:
#             return pickle.load(file)


#     data = []
#     print(f'Preparing data from dataframe of size: {len(df)} with search_depth: {search_depth}')
#     total_positives = 0
#     total_negatives = 0
#     for _, row in df.iterrows():
#         commit_message = row['commit_message']
#         actual_files_modified = row['actual_files_modified']

#         search_results = searcher.pipeline(commit_message, row['commit_date'], search_depth, 'sump')

#         # flatten the contributing results for each aggregated result
#         search_results = [result for agg_result in search_results for result in agg_result.contributing_results]

#         # efficiently get the top num_positives and num_negatives samples
#         positive_samples = []
#         negative_samples = []

#         for result in search_results:
#             if result.file_path in actual_files_modified and len(positive_samples) < num_positives:
#                 positive_samples.append(result.commit_message)
#                 total_positives += 1
#             elif result.file_path not in actual_files_modified and len(negative_samples) < num_negatives:
#                 negative_samples.append(result.commit_message)
#                 total_negatives += 1

#             if len(positive_samples) == num_positives and len(negative_samples) == num_negatives:
#                 break


#         for sample_msg in positive_samples:
#             data.append((commit_message, sample_msg, 1))

#         for sample_msg in negative_samples:
#             data.append((commit_message, sample_msg, 0))
#     # Write data to cache file
#     with open(cache_file, 'wb') as file:
#         pickle.dump(data, file)
#         print(f"Saved data to cache file: {cache_file}")

#     # print distribution of labels
#     print(f"Total positives: {total_positives}, Total negatives: {total_negatives}")
#     # print percentage of positives and negatives
#     denom = total_positives + total_negatives
#     print(f"Percentage of positives: {total_positives / denom}, Percentage of negatives: {total_negatives / denom}")
#     return data