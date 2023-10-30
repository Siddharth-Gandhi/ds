import glob
import json
import os

import pandas as pd
import tiktoken

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

# def reverse_tokenize(text):
#     text = json.loads(text)
#     # print(list(text['contents'].split(' ')))
#     text['contents'] = enc.decode([int(i) for i in text['contents'].split(' ')])
#     # return string
#     return json.dumps(text, indent=2)

def reverse_tokenize(text):
    return enc.decode([int(i) for i in text.split(' ')])