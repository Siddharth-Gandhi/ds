import glob

# import json
# import math
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import tiktoken
import torch
import tqdm
from torch.utils.data import Dataset

# from turtle import pos


os.environ['TIKTOKEN_CACHE_DIR'] = ""
ENCODING = 'p50k_base'
enc = tiktoken.get_encoding(ENCODING)
assert enc.decode(enc.encode("hello world")) == "hello world"

def print_random_commit_message(df):
    print(df['commit_message'].sample().values[0])

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


def sanity_check_triplets(data):
    """
    Perform a sanity check on the triplets data.

    Args:
        data: The input data containing triplets.

    Returns:
        The sanitized data after removing problematic rows.

    Examples:
        >>> data = pd.DataFrame({'query': ['apple', 'banana', 'apple'], 'passage': ['red fruit', 'yellow fruit', 'red fruit'], 'label': [0, 1, 0]})
        >>> sanity_check_triplets(data)
        Assertion failed at index 0: query      apple
        passage    red fruit
        label             0
        Name: 0, dtype: object
        Dropped row at index 0
        Total number of problems in sanity check of training data: 1
        # Output: DataFrame without the problematic row
    """
    problems = 0
    for i, row in tqdm.tqdm(data.iterrows(), total=len(data)):
        try:
            if row['label'] == 0:
                assert data[(data['query'] == row['query']) & (data['passage'] == row['passage'])]['label'].values[0] == 0
            else:
                assert data[(data['query'] == row['query']) & (data['passage'] == row['passage'])]['label'].values[0] == 1
        except AssertionError:
            print(f"Assertion failed at index {i}: {row}")
            # break  # Optional: break after the first failure, remove if you want to see all failures
            # remove the row with label 0

            if row['label'] == 0:
                problems += 1
                data.drop(i, inplace=True)
                print(f"Dropped row at index {i}")

    print(f"Total number of problems in sanity check of training data: {problems}")
    return data


def get_recent_df(combined_df, repo_name=None, ignore_gold_in_training=False):
    # Prepare the data for training
    print('Preparing training data...')
    # Step 1: Filter out only the columns we need
    filtered_df = combined_df[['commit_date', 'commit_message', 'commit_id', 'file_path', 'diff']]

    # Step 2: Group by commit_id
    grouped_df = filtered_df.groupby(['commit_id', 'commit_date', 'commit_message'])['file_path'].apply(list).reset_index()
    grouped_df.rename(columns={'file_path': 'actual_files_modified'}, inplace=True)

    # Step 3: Determine midpoint and filter dataframe
    midpoint_date = np.median(grouped_df['commit_date'])
    recent_df = grouped_df[grouped_df['commit_date'] > midpoint_date]
    print(f'Number of commits after midpoint date: {len(recent_df)}')

    # Step 4: Filter out commits with less than average length commit messages
    average_commit_len = recent_df['commit_message'].str.split().str.len().mean()
    # filter out commits with less than average length
    recent_df = recent_df[recent_df['commit_message'].str.split().str.len() > average_commit_len] # type: ignore
    print(f'Number of commits after filtering by commit message length: {len(recent_df)}')

    # Step 5: Filter out gold commits
    if repo_name:
        gold_dir = os.path.join('gold', repo_name)
        gold_commit_file = os.path.join(gold_dir, f'{repo_name}_gpt4_gold_commit_ids.txt')


        if not os.path.exists(gold_commit_file):
            if ignore_gold_in_training:
                print(f'Gold commit file {gold_commit_file} does not exist, but ignore_gold_in_training is set to True, so continuing...')
            else:
                print(f'Gold commit file {gold_commit_file} does not exist and ignore_gold_in_training is set to False, so exiting...')
                sys.exit(1)
            # print(f'Gold commit file {gold_commit_file} does not exist, skipping this step.')
        else:
            gold_commits = pd.read_csv(gold_commit_file, header=None, names=['commit_id']).commit_id.tolist()

            print(f'Found {len(gold_commits)} gold commits for {repo_name}')

            print('Removing gold commits from training data...')
            recent_df = recent_df[~recent_df['commit_id'].isin(gold_commits)]
            print(f'Number of commits after removing gold commits: {len(recent_df)}')
    return recent_df

# def prepare_code_triplets(diff_data, code_reranker, cache_file, overwrite=False):
#     # given diff_data, the passage column is way too long. We need to split it into passages of length psg_len with stride psg_stride
#     # then we can create triplets from that

#     # diff_data has columns: commit_id, file_path, query, passage, label

#     if cache_file and os.path.exists(cache_file) and not overwrite:
#         print(f"Loading data from cache file: {cache_file}")
#         # with open(cache_file, 'rb') as file:
#         #     return pickle.load(file)
#         return pd.read_parquet(cache_file)

#     def full_tokenize(s):
#         return code_reranker.tokenizer.encode_plus(s, max_length=None, truncation=False, return_tensors='pt', add_special_tokens=True, return_attention_mask=False, return_token_type_ids=False)['input_ids'].squeeze().tolist()

#     triplets = []

#     print('Preparing triplets from scratch')

#     for _, row in tqdm.tqdm(diff_data.iterrows(), total=len(diff_data)):
#         # get the input ids
#         # input_ids = file_content['input_ids'].squeeze()
#         # get the number of tokens in the file content
#         file_tokens = full_tokenize(row['passage'])
#         # query_tokens = full_tokenize(row['query'])
#         # path_tokens = full_tokenize(row['file_path'])
#         total_tokens = len(file_tokens)

#         cur_psg_cnt = 0
#         for cur_start in range(0, total_tokens, code_reranker.psg_stride):
#             cur_passage = []
#             # add query tokens and path tokens
#             # cur_passage.extend(query_tokens)
#             # cur_passage.extend(path_tokens)

#             # add the file tokens
#             cur_passage.extend(file_tokens[cur_start:cur_start+code_reranker.psg_len])

#             # now convert cur_passage into a string
#             cur_passage_decoded = code_reranker.tokenizer.decode(cur_passage)


#             # add the cur_passage to cur_result_passages
#             triplets.append((row['query'], row['file_path'], cur_passage_decoded, row['label']))

#             cur_psg_cnt += 1

#             if cur_psg_cnt == code_reranker.psg_cnt:
#                 break

#     # convert to pandas dataframe
#     triplets = pd.DataFrame(triplets, columns=['query', 'file_path', 'passage', 'label'])
#     # Write data to cache file
#     if cache_file:
#         # with open(cache_file, 'wb') as file:
#         #     pickle.dump(triplets, file)
#         #     print(f"Saved data to cache file: {cache_file}")
#         print(f"Saving data to cache file: {cache_file}")
#         triplets.to_parquet(cache_file)
#     return triplets

def prep_line(line):
        return line.rstrip().lstrip()

def parse_diff(diff):
    return [
        line[1:] if line.startswith('+') else line
        for line in diff.split('\n')
        if not (line.startswith('-') or len(line) == 0 or (line.startswith('@@') and line.count('@@') > 1))
        and len(prep_line(line)) > 2
    ]

def prepare_code_triplets(diff_data, code_reranker, cache_file, combined_df, overwrite=False):
    print(f'Preparing code triplets from scratch for {len(diff_data)} diffs with psg_len: {code_reranker.psg_len}, psg_stride: {code_reranker.psg_stride}, psg_cnt: {code_reranker.psg_cnt}')
    if cache_file and os.path.exists(cache_file) and not overwrite:
        print(f"Loading data from cache file: {cache_file}")
        # with open(cache_file, 'rb') as file:
        #     return pickle.load(file)
        return pd.read_parquet(cache_file)

    def full_tokenize(s):
        return code_reranker.tokenizer.encode_plus(s, max_length=None, truncation=False, return_tensors='pt', add_special_tokens=True, return_attention_mask=False, return_token_type_ids=False)['input_ids'].squeeze().tolist()



    def count_matching_lines(passage_lines, diff_lines):
        # Create a 2D array to store the lengths of the longest common subsequences
        dp = [[0] * (len(diff_lines) + 1) for _ in range(len(passage_lines) + 1)]

        # Fill the dp array
        for i in range(1, len(passage_lines) + 1):
            for j in range(1, len(diff_lines) + 1):
                if prep_line(passage_lines[i - 1]) == prep_line(diff_lines[j - 1]):
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[-1][-1]

    triplets = []

    for _, row in tqdm.tqdm(diff_data.iterrows(), total=len(diff_data)):
        file_tokens = full_tokenize(row['passage'])
        total_tokens = len(file_tokens)
        cur_diff = combined_df[(combined_df['commit_id'] == row['commit_id']) & (combined_df['file_path'] == row['file_path'])]['diff'].values[0]
        if pd.isna(cur_diff):
            # if diff is NA/NaN
            continue
        cur_diff_lines = parse_diff(cur_diff)
        cur_triplets = []
        for cur_start in range(0, total_tokens, code_reranker.psg_stride):
            cur_passage = []

            cur_passage.extend(file_tokens[cur_start:cur_start+code_reranker.psg_len])

            # now convert cur_passage into a string
            cur_passage_decoded = code_reranker.tokenizer.decode(cur_passage)

            cur_passage_lines = cur_passage_decoded.split('\n')

            # remove lines with less than 2 characters
            cur_passage_lines = [line for line in cur_passage_lines if len(prep_line(line)) > 2]

            # check if there are lines matching the diff lines
            # if there are, then we can add this directly to the triplets
            # common_lines = set(cur_passage_lines).intersection(set(cur_diff_lines))
            common_line_count = count_matching_lines(cur_passage_lines, cur_diff_lines)

            # add the cur_passage to cur_result_passages
            cur_triplets.append((common_line_count, (row['query'], row['file_path'], cur_passage_decoded, row['label'])))

        # sort the cur_triplets by the number of common lines
        cur_triplets.sort(key=lambda x: x[0], reverse=True)

        # now we want to filter cur_triplets to have all tuplets with x[0] > 3 to be in order and shuffle the rest

        # now add the top code_reranker.psg_cnt to triplets
        for triplet in cur_triplets[:code_reranker.psg_cnt]:
            # print(f"Found {triplet[0]} matching lines for diff in cur_passage at index")
            triplets.append(triplet[1])


    # convert to pandas dataframe
    triplets = pd.DataFrame(triplets, columns=['query', 'file_path', 'passage', 'label'])
    if cache_file:
        # with open(cache_file, 'wb') as file:
        #     pickle.dump(triplets, file)
        #     print(f"Saved data to cache file: {cache_file}")
        print(f"Saving data to cache file: {cache_file}")
        triplets.to_parquet(cache_file)
    return triplets