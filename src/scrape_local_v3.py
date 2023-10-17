import argparse
import cProfile
import json
import os
import pstats

# import subprocess
# import time
import traceback

import git
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.getcwd()

with open(
    os.path.join(BASE_DIR, "misc/code_extensions.json"), "r", encoding="utf-8"
) as f:
    code_extensions = set(json.load(f))


def get_all_commits(repo):
    return [commit.hexsha for commit in repo.iter_commits()]


# def get_files_changed_in_commit(commit):
#     files_changed = []
#     is_merge_request = len(commit.parents) > 1

#     # If the commit has a parent, compare it with its first parent
#     if commit.parents:
#         diff_index = commit.diff(commit.parents[0])
#         for diff in diff_index:
#             files_changed.append(diff.b_path)

#     return files_changed, is_merge_request


# V2
# def get_files_changed_in_commit(commit):
#     files_changed = []
#     is_merge_request = len(commit.parents) > 1

#     # Check if it's the initial commit
#     if not commit.parents:
#         return [item.path for item in commit.tree], is_merge_request

#     # For other commits, compare with the first parent to get the diff
#     diff_index = commit.diff(commit.parents[0])
#     for diff in diff_index:
#         files_changed.append(diff.b_path)

#     return files_changed, is_merge_request

def get_all_files_from_tree(tree):
    files = []
    for item in tree:
        if item.type == 'blob':
            files.append(item.path)
        elif item.type == 'tree':
            files.extend(get_all_files_from_tree(item))
    return files

def get_files_changed_in_commit(commit):
    is_merge_request = len(commit.parents) > 1

    # Check if it's the initial commit
    if not commit.parents:
        return get_all_files_from_tree(commit.tree), is_merge_request

    # For other commits, compare with the first parent to get the diff
    diff_index = commit.diff(commit.parents[0])
    files_changed = [diff.b_path for diff in diff_index]

    return files_changed, is_merge_request

def get_file_content_at_commit(commit, file_path, parent=False):
    target_commit = commit.parents[0] if parent else commit
    return target_commit.tree[file_path].data_stream.read().decode()

def determine_status(previous_content, new_content):
    if not previous_content and new_content:
        return "added"
    elif previous_content and not new_content:
        return "deleted"
    elif previous_content and new_content:
        return "modified"
    else:
        return "unknown"


def set_dtype(df):
    dtypes = {
        "owner": "string",
        "repo_name": "string",
        "commit_date": "datetime64[ns]",
        "commit_id": "string",
        "commit_message": "string",
        "file_path": "string",
        "previous_commit_id": "string",
        "previous_file_content": "string",
        "cur_file_content": "string",
        "diff": "string",
        "status": "category",
        "is_merge_request": "bool",
        "file_extension": "category",
    }
    for column, dtype in dtypes.items():
        if column == "commit_date":
            df[column] = pd.to_datetime(df[column], format="ISO8601", utc=True)
            continue
        df[column] = df[column].astype(dtype)
    return df


def save_to_parquet(data, owner, repo_name, batch_num, dir_path):
    df = pd.DataFrame(data)
    df = set_dtype(df)
    parquet_file = os.path.join(
        dir_path, f"{owner}_{repo_name}_commit_data_{batch_num}.parquet"
    )
    df.to_parquet(parquet_file, index=False)
    print(f"Saved {len(df)} rows to {parquet_file}")

def extract_diff_from_at_symbol(diff_string):
    idx = diff_string.find("@@")
    if idx != -1:
        return diff_string[idx:]
    return ""


def scrape_repository(repo_path, CHUNK_SIZE):
    owner, repo_name = repo_path.lower().split("/")
    local_path = os.path.join(BASE_DIR, f"repos/{owner}_{repo_name}")
    repo = git.Repo(local_path)

    repo.git.checkout("b440ef8c96bb5175b44c275a7b4ed450061e9bae")
    print(f"Checking out {repo_path} at b440ef8c96bb5175b44c275a7b4ed450061e9bae")
    dir_path = os.path.join(BASE_DIR, f"data/{owner}_{repo_name}")
    os.makedirs(dir_path, exist_ok=True)

    all_commits = get_all_commits(repo)
    print('b0a11594aec50892a02cd8d129eee2dfe93a8bb8' in all_commits)
    data_batch = []
    batch_num = 0

    for idx, commit_sha in tqdm(enumerate(all_commits), total=len(all_commits)):
        if commit_sha == 'b0a11594aec50892a02cd8d129eee2dfe93a8bb8':
            print('HELLO WORKSDLKN KLSMDFKLSMKLSFMDKLFMSDFLKM')
        commit = repo.commit(commit_sha)
        files_changed, is_merge_request = get_files_changed_in_commit(commit)
        for file_path in files_changed:
            file_extension = file_path.split(".")[-1]
            if f".{file_extension}" not in code_extensions:
                continue
            try:
                previous_content = get_file_content_at_commit(
                    commit, file_path, parent=True
                )
                prev_commit = commit.parents[0].hexsha
            except Exception:
                previous_content = None
                prev_commit = None

            try:
                new_content = get_file_content_at_commit(
                    commit, file_path, parent=False
                )
            except Exception:
                new_content = None

            diff = None
            if previous_content and new_content:
                diff = repo.git.diff(commit.parents[0].hexsha, commit.hexsha, "--", file_path)
                diff = extract_diff_from_at_symbol(diff)

            status = determine_status(previous_content, new_content)

            data_batch.append(
                {
                    "owner": owner,
                    "repo_name": repo_name,
                    "commit_date": commit.committed_datetime,
                    "commit_id": commit.hexsha,
                    "commit_message": commit.message,
                    "file_path": file_path,
                    "previous_commit_id": prev_commit,
                    "previous_file_content": previous_content,
                    "cur_file_content": new_content,
                    "diff": diff,
                    "status": status,
                    "is_merge_request": is_merge_request,
                    "file_extension": file_extension,
                }
            )

        if (idx + 1) % CHUNK_SIZE == 0:
            save_to_parquet(data_batch, owner, repo_name, batch_num, dir_path)
            data_batch = []
            batch_num += 1

    if data_batch:
        save_to_parquet(data_batch, owner, repo_name, batch_num, dir_path)


def main(args):
    if not os.path.exists("../data"):
        os.makedirs("../data")

    with open(os.path.join(BASE_DIR, args.file_path), "r", encoding="utf-8") as f:
        repos = f.read().splitlines()[args.start_index : args.end_index + 1]

    for repo in repos:
        owner, repo_name = repo.lower().split("/")
        local_path = os.path.join(BASE_DIR, f"repos/{owner}_{repo_name}")

        if not os.path.exists(local_path) or not os.listdir(local_path):
            print(f"{repo} does not exist/is empty in the repos folder. Please clone")
            continue

        try:
            scrape_repository(repo, args.chunk_size)
        except Exception as e:
            print(f"Failed to scrape {repo}")
            print(e)
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path",
        type=str,
        help="File path for the .txt file containing the list of repos",
    )
    parser.add_argument(
        "start_index",
        type=int,
        default=0,
        help="Start index for the repos to be processed",
    )
    parser.add_argument(
        "end_index",
        type=int,
        default=0,
        help="End index for the repos to be processed",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=1000, help="Number of rows per parquet file"
    )

    args = parser.parse_args()

    # print working directory & all arguments
    print("Working directory: ", os.getcwd())
    print("All arguments: ", args)
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    main(args)
