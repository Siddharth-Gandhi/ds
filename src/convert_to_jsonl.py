import argparse
import configparser
import datetime
import glob
import json
import logging
import logging.config
import os
import time

import numpy as np
import pandas as pd
import tiktoken

from utils import count_commits, get_combined_df, tokenize


def set_log_filename(config_file, log_filename):
    """
    Modify the log filename in the logging configuration.

    Args:
    - config_file (str): Path to the logging configuration file.
    - log_filename (str): New log filename.
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    # Change the filename for the fileHandler
    config.set('handler_fileHandler', 'args', f"('logs/{log_filename}', 'w')")

    # Save the updated configuration back to the file
    with open(config_file, 'w') as configfile:
        config.write(configfile)

set_log_filename('logging.conf', 'convert_to_jsonl.log')
logging.config.fileConfig(fname="logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# Configuration for tiktoken encoding
# ENCODING = 'p50k_base'
# enc = tiktoken.get_encoding(ENCODING)
# assert enc.decode(enc.encode("hello world")) == "hello world"


# def tokenize(text):
#     return ' '.join(map(str, enc.encode(text, disallowed_special=())))


# def get_combined_df(repo_dir):
#     all_files = glob.glob(os.path.join(repo_dir, '*.parquet'))
#     all_files.sort()
#     all_dataframes = [pd.read_parquet(file) for file in all_files]
#     combined_df = pd.concat(all_dataframes, ignore_index=True)
#     combined_df['commit_date'] = (combined_df['commit_date'].astype('int64') / 1e9).astype('int64')
#     return combined_df


# def count_commits(repo_dir):
#     combined_df = get_combined_df(repo_dir)
#     return combined_df.commit_id.nunique()


def convert_repo_to_jsonl(repo_dir, output_file, content_option='commit', use_tokenizer=False):
    combined_df = get_combined_df(repo_dir)
    combined_df['commit_message'] = combined_df['commit_message'].fillna('')
    combined_df['cur_file_content'] = combined_df['cur_file_content'].fillna('')

    logging.info('Combined Memory Usage: %0.2f MB for %d rows', combined_df.memory_usage(deep=True).sum() / 1024 ** 2, len(combined_df))

    with open(output_file, 'x', encoding='utf-8') as f:
        for index, row in combined_df.iterrows():
            if content_option == "commit":
                content = row['commit_message']
            elif content_option == "code":
                content = row['cur_file_content']
            elif content_option == "both":
                content = row['commit_message'] + '\n' + row['cur_file_content']

            if use_tokenizer:
                content = tokenize(content)

            doc = {
                'id': row['commit_id'],
                'contents': content,
                'repo_name': row['repo_name'],
                'file_path': row['file_path'],
                'commit_date': row['commit_date'],
            }
            f.write(json.dumps(doc) + '\n')
    logging.info('Wrote %d rows to %s', len(combined_df), output_file)


def main(args):
    # jsonl_dir_name = 'jsonl_tk'
    jsonl_dir_name = f'jsonl_{args.content_option}'
    if args.use_tokenizer:
        jsonl_dir_name += '_tokenized'
    data_dir = args.data_path

    for repo_name in [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]:
        start_time = time.perf_counter()
        logger.info('Processing %s', repo_name)
        repo_dir = os.path.join(data_dir, repo_name)
        os.makedirs(os.path.join(repo_dir, jsonl_dir_name), exist_ok=True)
        output_name = f'{repo_name}_commit_only_tk.jsonl'
        output_jsonl_file = os.path.join(repo_dir, jsonl_dir_name, output_name)

        if os.path.exists(output_jsonl_file):
            os.remove(output_jsonl_file)

        convert_repo_to_jsonl(repo_dir, output_jsonl_file, args.content_option, args.use_tokenizer)
        end_time = time.perf_counter()
        logger.info('Finished processing %s in %0.2f seconds', repo_name, end_time - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert repositories to JSONL format.")
    parser.add_argument("data_path", help="Path to the 'data' directory containing repositories.")
    parser.add_argument("--use_tokenizer", action="store_true", help="Use tokenizer on content.")
    parser.add_argument("--content_option", choices=["commit", "code", "both"], default="commit", help="Choose content: 'commit', 'code', or 'both'.")

    args = parser.parse_args()
    main(args)