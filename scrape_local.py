import argparse
import asyncio
import json
import os
import subprocess
import time
import traceback
from asyncio.subprocess import PIPE, STDOUT

import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
code_extensions = set(json.load(open(os.path.join(BASE_DIR, "code_extensions.json"), "r")))
# CHUNK_SIZE = 1000


def clone_repository(repo_path):
    """
    Clone a repository if it doesn't exist locally.
    """
    owner, repo_name = repo_path.lower().split("/")
    # local_path = f"repos/{owner}_{repo_name}"
    local_path = os.path.join(BASE_DIR, f"repos/{owner}_{repo_name}")
    print(f"Cloning {repo_path} to {local_path}...")
    if not os.path.exists(local_path):
        os.makedirs(local_path)
        # subprocess.run(["git", "clone", f"https://github.com/{repo_path}.git", local_path])
        run_command(f"git clone https://github.com/{repo_path}.git {local_path}")
    return local_path


def run_command(command: str, cwd=None, timeout=None):
    result = subprocess.run(
        [command], timeout=timeout, capture_output=True, text=True, shell=True, cwd=cwd
    )
    return "" if result.returncode != 0 else result.stdout


def get_all_commits(local_path=None):
    cmd = "git log --pretty=format:'%H'"
    return run_command(cmd, cwd=local_path).splitlines()


# V3
def get_files_changed_in_commit(commit_sha, local_path=None):
    num_parents = len(run_command(f"git log --pretty=%P -n 1 {commit_sha}", cwd=local_path).split())
    is_merge_request = False
    if num_parents <= 1:
        return (
            run_command(
                f"git show {commit_sha} --pretty='' --name-only", cwd=local_path
            ).splitlines(),
            is_merge_request,
        )
    # It's a merge commit
    is_merge_request = True
    parents = run_command(f"git log --pretty=%P -n 1 {commit_sha}", cwd=local_path).split()
    # Find the common ancestor of the two parents
    base = run_command(f"git merge-base {parents[0]} {parents[1]}", cwd=local_path).strip()
    # Get changes introduced by the merged branch
    return (
        run_command(f"git diff --name-only {base} {parents[1]}", cwd=local_path).splitlines(),
        is_merge_request,
    )


def get_commit_message(commit_sha, local_path=None):
    cmd = f"git show -s --format=%B {commit_sha}"
    return run_command(cmd, cwd=local_path)


def get_file_content_at_commit(commit_sha, file_path, parent=False, local_path=None):
    # If parent=True, get the content from the parent commit (before the change)
    cmd = f"git show {commit_sha}{'^' if parent else ''}:{file_path}"
    return run_command(cmd, cwd=local_path)


def get_diff(cur_sha, prev_sha, file_path, local_path=None):
    cmd = f"git diff {prev_sha}:{file_path} {cur_sha}:{file_path} | awk '/@@/ {{flag=1}} flag'"
    # cmd = f"git diff {prev_sha}:{file_path} {cur_sha}:{file_path} | awk '/@@/ {{flag=1}} flag'"
    return run_command(cmd, cwd=local_path)


def get_previous_commit(commit_sha, local_path=None):
    cmd = f"git rev-parse {commit_sha}^"
    return run_command(cmd, cwd=local_path)


def get_date(commit_sha, local_paths=None):
    cmd = f"git show -s --format=%cI {commit_sha}"
    return run_command(cmd, cwd=local_paths)


def process_commit(commit, local_path, owner, repo_name):
    files_changed, is_merge_request = get_files_changed_in_commit(commit, local_path)
    commit_message = get_commit_message(commit, local_path)
    commit_date = get_date(commit, local_path)
    data_list = []
    for file in files_changed:
        file_extension = file.split(".")[-1]
        if f".{file_extension}" not in code_extensions:
            continue
        try:
            previous_content = get_file_content_at_commit(
                commit, file, parent=True, local_path=local_path
            )
            prev_commit = get_previous_commit(commit, local_path=local_path)
        except Exception:
            previous_content = None
            prev_commit = None
        try:
            new_content = get_file_content_at_commit(
                commit, file, parent=False, local_path=local_path
            )
        except Exception:
            new_content = None
        diff = None
        if previous_content and new_content:
            diff = get_diff(commit, f"{commit}^", file, local_path=local_path)
        status = determine_status(previous_content, new_content)

        data_list.append(
            {
                "owner": owner,
                "repo_name": repo_name,
                "commit_date": commit_date,
                "commit_id": commit,
                "commit_message": commit_message,
                "file_path": file,
                "previous_commit_id": prev_commit,
                "previous_file_content": previous_content,
                "cur_file_content": new_content,
                "diff": diff,
                "status": status,
                "is_merge_request": is_merge_request,
                "file_extension": file_extension,
            }
        )
    return data_list


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
    df["owner"] = df["owner"].astype("string")
    df["repo_name"] = df["repo_name"].astype("string")
    df["commit_date"] = pd.to_datetime(df["commit_date"], format="ISO8601", utc=True)
    df["commit_id"] = df["commit_id"].astype("string")
    df["commit_message"] = df["commit_message"].astype("string")
    df["file_path"] = df["file_path"].astype("string")
    df["previous_commit_id"] = df["previous_commit_id"].astype("string")
    df["previous_file_content"] = df["previous_file_content"].astype("string")
    df["cur_file_content"] = df["cur_file_content"].astype("string")
    df["diff"] = df["diff"].astype("string")
    df["status"] = df["status"].astype("category")
    df["is_merge_request"] = df["is_merge_request"].astype("bool")
    df["file_extension"] = df["file_extension"].astype("category")
    return df


def save_to_parquet(data, owner, repo_name, batch_num, dir_path):
    """
    Save a list of data to a Parquet file.
    """
    df = pd.DataFrame(data)
    df = set_dtype(df)
    # convert empty strings to None -> BAD IDEA, pyserini can't handle None
    # df = df.replace(r"^\s*$", None, regex=True)
    parquet_file = os.path.join(dir_path, f"{owner}_{repo_name}_commit_data_{batch_num}.parquet")
    df.to_parquet(parquet_file, index=False)
    print(f"Saved {len(df)} rows to {parquet_file}")


def scrape_repository(repo_path):
    owner, repo_name = repo_path.split("/")
    local_path = f"repos/{owner}_{repo_name}"

    # Create a directory for each repository
    dir_path = os.path.join(BASE_DIR, f"data/{owner}_{repo_name}")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create a stats directory for each repository
    stats_dir_path = os.path.join(BASE_DIR, f"data/{owner}_{repo_name}/stats")
    if not os.path.exists(stats_dir_path):
        os.makedirs(stats_dir_path)

    # ... [rest of your code in scrape_repository function]
    # Start from the batch specified
    start_commit_idx = args.resume_batch * CHUNK_SIZE
    if start_commit_idx:
        print(f"Resuming from batch {args.resume_batch} (commit index: {start_commit_idx})")
    all_commits = get_all_commits(local_path)
    print(f"Found {len(all_commits)} commits in {repo_path}")
    data_batch = []
    batch_num = args.resume_batch if start_commit_idx else 0
    # for idx, commit in tqdm(enumerate(all_commits), total=len(all_commits)):
    for idx, commit in tqdm(
        enumerate(all_commits[start_commit_idx:]), initial=start_commit_idx, total=len(all_commits)
    ):
        # print(f"Processing commit {idx} out of {len(all_commits)}")
        # print based on size of all_commits
        # if it's less than 1000, print every 100
        # if it's more than 1000, print every 1000
        if (
            len(all_commits) < 1000
            and idx
            and idx % 100 == 0
            or len(all_commits) >= 1000
            and idx
            and idx % 1000 == 0
        ):
            print(f"{idx} commits processed out of {len(all_commits)}")
        if idx and idx % CHUNK_SIZE == 0:
            save_to_parquet(data_batch, owner, repo_name, batch_num, dir_path)
            data_batch = []
            batch_num += 1
            print(f"{idx} commits processed and saved to Parquet out of {len(all_commits)}.")

            # dump the stats file too at the currennt state in case of failure in owner_repo/stats

            stats_file = os.path.join(stats_dir_path, f"{owner}_{repo_name}_stats_{batch_num}.prof")
            stats = pstats.Stats(pr)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.dump_stats(stats_file)

        data_batch.extend(process_commit(commit, local_path, owner, repo_name))

    # Save remaining data to Parquet
    if data_batch:
        save_to_parquet(data_batch, owner, repo_name, batch_num, dir_path)


def main():
    if not os.path.exists("data"):
        os.makedirs("data")

    with open(os.path.join(BASE_DIR, args.file_path), "r") as f:
        repos = f.read().splitlines()[args.start_index : args.end_index + 1]

    print(f"Found {len(repos)} repos to scrape")
    print(
        f"Scraping repos {args.start_index} to {args.end_index} (inclusive)"
        f" from {args.file_path}"
    )
    if args.resume_batch:
        print(
            f"Resuming from batch {args.resume_batch} (commit index: {args.resume_batch * CHUNK_SIZE})"
        )
    print(f"Processing {CHUNK_SIZE} commits in one batch")

    for repo in repos:
        # check if repo exists
        owner, repo_name = repo.lower().split("/")
        local_path = f"repos/{owner}_{repo_name}"
        if not os.path.exists(local_path) or not os.listdir(local_path):
            print(f"{repo} does not exist/is empty in the repos folder. Please clone")
            continue
        try:
            start_time = time.perf_counter()
            scrape_repository(repo)
            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) / 60
            print(f"Scraped {repo} in {elapsed_time:.2f} minutes")
        except Exception as e:
            print(f"Failed to scrape {repo}")
            # print stacktrace
            print(e)
            traceback.print_exc()


if __name__ == "__main__":
    # print(args)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", type=str, help="File path for the .txt file containing the list of repos"
    )
    parser.add_argument("--start_index", type=int, help="Start index for repos")
    parser.add_argument("--end_index", type=int, help="End index for repos")
    parser.add_argument(
        "--resume_batch", type=int, default=0, help="Batch number from where to resume (0 indexed)."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of commits to process in one batch. Default: 1000",
    )
    args = parser.parse_args()
    CHUNK_SIZE = args.chunk_size
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    stats.dump_stats("karpathy_scrape_local.prof")
