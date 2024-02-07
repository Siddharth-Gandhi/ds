import glob
import os
import pickle
import random
import sys
from re import search

import numpy as np
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from tree_sitter import Language, Parser

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
    for _, row in tqdm(df.iterrows(), total=len(df)):
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
    for i, row in tqdm(data.iterrows(), total=len(data)):
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


def get_code_df(recent_df, searcher, search_depth, num_positives, num_negatives, combined_df, cache_file, overwrite=False):

    # takes a bunch of train

    if cache_file and os.path.exists(cache_file) and not overwrite:
        print(f"Loading data from cache file: {cache_file}")
        return pd.read_parquet(cache_file)

    code_data = []
    print(f'Preparing code data from dataframe of size: {len(recent_df)} with search_depth: {search_depth}')
    total_positives, total_negatives = 0, 0
    for _, row in tqdm(recent_df.iterrows(), total=len(recent_df)):
        cur_positives = 0
        cur_negatives = 0
        train_original_message = row['original_message']
        train_commit_message = row['commit_message']
        actual_files_modified = row['actual_files_modified']
        train_commit_id = row['commit_id']

        agg_search_results = searcher.pipeline(train_commit_message, row['commit_date'], search_depth, 'sump', sort_contributing_result_by_date=True)

        for agg_result in agg_search_results:
            most_recent_search_result = agg_result.contributing_results[0] # get the most recent version at the time of train query
            search_result_file_path = most_recent_search_result.file_path
            search_result_commit_id = most_recent_search_result.commit_id
            search_result_current_file_content = combined_df[(combined_df['commit_id'] == search_result_commit_id) & (combined_df['file_path'] == search_result_file_path)]['cur_file_content'].values[0]
            # search_result_current_file_content = 'Empty for now, uncomment #317 in utils'
            search_result_diff = combined_df[(combined_df['commit_id'] == search_result_commit_id) & (combined_df['file_path'] == search_result_file_path)]['diff'].values[0]

            if search_result_file_path in actual_files_modified and cur_positives < num_positives:
                # this is a positive sample
                code_data.append((train_commit_id, train_commit_message, train_original_message, search_result_file_path, search_result_commit_id, search_result_current_file_content, search_result_diff, 1))
                cur_positives += 1
                total_positives += 1
            elif search_result_file_path not in actual_files_modified and cur_negatives < num_negatives:
                # this is a negative sample
                code_data.append((train_commit_id, train_commit_message, train_original_message, search_result_file_path, search_result_commit_id, search_result_current_file_content, search_result_diff, 0))
                cur_negatives += 1
                total_negatives += 1

            if cur_positives == num_positives and cur_negatives == num_negatives:
                break

        # if _ == 3:
        # break

    code_df = pd.DataFrame(code_data, columns=['train_commit_id', 'train_query', 'train_original_message', 'SR_file_path', 'SR_commit_id', 'SR_file_content', 'SR_diff' ,'label'])

    # print distribution of labels
    print(f"Total positives: {total_positives}, Total negatives: {total_negatives}")
    denom = total_positives + total_negatives
    print(f"Percentage of positives: {total_positives / denom}, Percentage of negatives: {total_negatives / denom}")


    if cache_file:
        print(f"Saving data to cache file: {cache_file}")
        code_df.to_parquet(cache_file)

    return code_df


def sanity_check_code(data):
    problems = 0
    for i, row in tqdm(data.iterrows(), total=len(data)):
        val = data[(data['train_query'] == row['train_query']) & (data['SR_commit_id'] == row['SR_commit_id']) & (data['SR_file_path'] == row['SR_file_path'])]['label'].values[0]
        try:
            if row['label'] == 0:
                assert val == 0
            else:
                assert val == 1
        except AssertionError:
            print(f"Assertion failed at index {i}: {row}")
            # break  # Optional: break after the first failure, remove if you want to see all failures
            # remove the row with label 0

            if row['label'] == 0:
                problems += 1
                # data.drop(i, inplace=True)
                data = data.drop(i)
                # print(f"Dropped row at index {i}")

    print(f"Total number of problems in sanity check of training data: {problems}")
    return data


def prepare_code_triplets(code_df, code_reranker, cache_file, combined_df, overwrite=False):
    print(f'Preparing code triplets from scratch for {len(code_df)} diffs with psg_len: {code_reranker.psg_len}, psg_stride: {code_reranker.psg_stride}, psg_cnt: {code_reranker.psg_cnt}')

    if cache_file and os.path.exists(cache_file) and not overwrite:
        print(f"Loading data from cache file: {cache_file}")
        return pd.read_parquet(cache_file)

    JS_LANGUAGE = Language('src/parser/my-languages.so', 'javascript')
    parser = Parser()
    parser.set_language(JS_LANGUAGE)



    def prep_line(line):
        return line.rstrip().lstrip()

    def parse_diff(diff):
        return [
            line[1:] if line.startswith('+') else line
            for line in diff.split('\n')
            if not (line.startswith('-') or len(line) == 0 or (line.startswith('@@') and line.count('@@') > 1))
            and len(prep_line(line)) > 2
        ]

    # def parse_diff2(diff):
    #     return [
    #         line[1:] if (line.startswith('+') or line.startswith('-')) else line
    #         for line in diff.split('\n')
    #         if not (len(line) == 0 or (line.startswith('@@') and line.count('@@') > 1))
    #     ]

    def full_tokenize(s):
        return code_reranker.tokenizer.encode_plus(s, max_length=None, truncation=False, return_tensors='pt', add_special_tokens=True, return_attention_mask=False, return_token_type_ids=False)['input_ids'].squeeze().tolist()

    def extract_function_texts(node, source_code):
        function_texts = []
        # Check if the node represents a function declaration
        if node.type == 'function_declaration':
            start_byte = node.start_byte
            end_byte = node.end_byte
            function_texts.append(source_code[start_byte:end_byte].decode('utf8'))
        # Check for variable declarations that might include function expressions or arrow functions
        elif node.type == 'variable_declaration':
            for child in node.children:
                if child.type == 'variable_declarator':
                    init_node = child.child_by_field_name('init')
                    if init_node and (init_node.type in ['function', 'arrow_function', 'function_expression']):
                        start_byte = node.start_byte
                        end_byte = node.end_byte
                        function_texts.append(source_code[start_byte:end_byte].decode('utf8'))
                        break  # Assuming one function per variable declaration for simplicity
        # Recursively process all child nodes
        else:
            for child in node.children:
                function_texts.extend(extract_function_texts(child, source_code))
        return function_texts

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

    for _, row in tqdm(code_df.iterrows(), total=len(code_df)):
        # file_tokens = full_tokenize(row['SR_file_content'])
        # total_tokens = len(file_tokens)
        # cur_diff = combined_df[(combined_df['commit_id'] == row['SR_commit_id']) & (combined_df['file_path'] == row['SR_file_path'])]['diff'].values[0]
        cur_diff = row['SR_diff']



        # Convert the source code to bytes for tree-sitter
        source_code_bytes = bytes(row['SR_file_content'], "utf8")

        # Parse the code
        tree = parser.parse(source_code_bytes)

        # Extract function texts
        root_node = tree.root_node
        function_texts = extract_function_texts(root_node, source_code_bytes)

        # Print or return the list of full function texts

        if pd.isna(cur_diff):
            # if diff is NA/NaN, then skip this row
            # possible when commit removes or renames this file or maybe god decided to remove the diff
            continue

        # cur_diff_lines = parse_diff2(cur_diff)
        cur_diff_lines = parse_diff(cur_diff)

        # diff_tokens = full_tokenize(''.join(cur_diff_lines))
        # total_tokens = len(diff_tokens)



        cur_triplets = []

        for func in function_texts:

            cur_func_lines = func.split('\n')

            # remove lines with less than 2 characters
            cur_func_lines = [line for line in cur_func_lines if len(prep_line(line)) > 2]
            # common_lines = set(cur_func_lines).intersection(set(cur_diff_lines))
            common_line_count = count_matching_lines(cur_func_lines, cur_diff_lines)
            cur_triplets.append((common_line_count, (row['train_query'], row['SR_file_path'], func, row['label'])))


        # for cur_start in range(0, total_tokens, code_reranker.psg_stride):
        #     cur_passage = []

        #     cur_passage.extend(diff_tokens[cur_start:cur_start+code_reranker.psg_len])

        #     # now convert cur_passage into a string
        #     cur_passage_decoded = code_reranker.tokenizer.decode(cur_passage)

        #     # cur_passage_lines = cur_passage_decoded.split('\n')

        #     # remove lines with less than 2 characters
        #     # cur_passage_lines = [line for line in cur_passage_lines if len(prep_line(line)) > 2]

        #     # check if there are lines matching the diff lines
        #     # if there are, then we can add this directly to the triplets
        #     # common_lines = set(cur_passage_lines).intersection(set(cur_diff_lines))
        #     # common_line_count = count_matching_lines(cur_passage_lines, cur_diff_lines)

        #     # add the cur_passage to cur_result_passages
        #     # cur_triplets.append((common_line_count, (row['train_query'], row['SR_file_path'], cur_passage_decoded, row['label'])))
        #     triplets.append((row['train_query'], row['SR_file_path'], cur_passage_decoded, row['label']))

        # # sort the cur_triplets by the number of common lines
        cur_triplets.sort(key=lambda x: x[0], reverse=True)

        # # now we want to filter cur_triplets to have all tuplets with x[0] > 3 to be in order and shuffle the rest

        # # now add the top code_reranker.psg_cnt to triplets
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


