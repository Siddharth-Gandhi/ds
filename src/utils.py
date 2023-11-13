import glob
import json
import os

import pandas as pd
import tiktoken

os.environ['TIKTOKEN_CACHE_DIR'] = ""
ENCODING = 'p50k_base'
enc = tiktoken.get_encoding(ENCODING)
assert enc.decode(enc.encode("hello world")) == "hello world"

def tokenize(text):
    return ' '.join(map(str, enc.encode(text, disallowed_special=())))


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
