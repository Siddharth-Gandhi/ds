import argparse
import cProfile
import json
import os
import pstats
import subprocess
import time
import traceback

import pandas as pd
from tqdm import tqdm

BASE_DIR = os.getcwd()
code_extensions = set(
    json.load(
        open(
            os.path.join(BASE_DIR, "misc/code_extensions.json"),
            "r",
            encoding="utf-8",
        )
    )
)


def run_command(command: str, cwd=None, timeout=None):
    result = subprocess.run(
        [command],
        timeout=timeout,
        capture_output=True,
        text=True,
        shell=True,
        cwd=cwd,
    )
    return result.stdout


def get_all_commits(local_path=None):
    cmd = "git log --pretty=format:'%H'"
    return run_command(cmd, cwd=local_path).splitlines()


# V3
def get_files_changed_in_commit(commit_sha, local_path=None):
    num_parents = len(
        run_command(f"git log --pretty=%P -n 1 {commit_sha}", cwd=local_path).split()
    )
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
    parents = run_command(
        f"git log --pretty=%P -n 1 {commit_sha}", cwd=local_path
    ).split()
    # Find the common ancestor of the two parents
    base = run_command(
        f"git merge-base {parents[0]} {parents[1]}", cwd=local_path
    ).strip()
    # Get changes introduced by the merged branch
    return (
        run_command(
            f"git diff --name-only {base} {parents[1]}", cwd=local_path
        ).splitlines(),
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
        except FileNotFoundError:
            previous_content = None
            prev_commit = None
        try:
            new_content = get_file_content_at_commit(
                commit, file, parent=False, local_path=local_path
            )
        except FileNotFoundError:
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
    # df = df.replace(r"^\s*$", None, regex=True) # convert empty strings to None -> BAD IDEA, pyserini can't handle None
    parquet_file = os.path.join(
        dir_path, f"{owner}_{repo_name}_commit_data_{batch_num}.parquet"
    )
    df.to_parquet(parquet_file, index=False)
    print(f"Saved {len(df)} rows to {parquet_file}")


def scrape_repository(repo_path):
    owner, repo_name = repo_path.lower().split("/")
    local_path = f"repos/{owner}_{repo_name}"
    print(f"Base dir: {BASE_DIR}")

    # Create a directory for each repository
    dir_path = os.path.join(BASE_DIR, f"data/{owner}_{repo_name}")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print(f"Storing data in {dir_path}...")
    # Create a stats directory for each repository
    stats_dir_path = os.path.join(BASE_DIR, f"profiling/{owner}_{repo_name}/")
    if not os.path.exists(stats_dir_path):
        os.makedirs(stats_dir_path)
    print(f"Storing stats in {stats_dir_path}...")
    # Start from the batch specified

    start_commit_idx = args.resume_batch * CHUNK_SIZE
    if start_commit_idx:
        print(
            f"Resuming from batch {args.resume_batch} (commit index: {start_commit_idx})"
        )

    all_commits = get_all_commits(local_path)
    print(f"Found {len(all_commits)} commits in {repo_path}")

    data_batch = []
    batch_num = args.resume_batch if start_commit_idx else 0
    update_freq = max(1, len(all_commits) // 100)

    pr = cProfile.Profile()
    pr.enable()

    with tqdm(
        total=len(all_commits),
        initial=start_commit_idx,
        miniters=update_freq,
    ) as pbar:
        for idx, commit in enumerate(
            all_commits[start_commit_idx:], start=start_commit_idx
        ):
            # print(f"Processing commit {idx} out of {len(all_commits)}")
            if idx % update_freq == 0:
                pbar.set_postfix_str(
                    f"Processing commit {idx} out of {len(all_commits)}"
                )
                pbar.update(update_freq)
            if (
                len(all_commits) < 1000
                and idx
                and idx % 100 == 0
                or len(all_commits) >= 1000
                and idx
                and idx % 1000 == 0
            ):
                print(f"{idx} commits processed out of {len(all_commits)}")
            if idx and idx % CHUNK_SIZE == 0 and data_batch:
                pr.disable()
                save_to_parquet(data_batch, owner, repo_name, batch_num, dir_path)
                data_batch = []
                print(
                    f"{idx} commits processed and saved to Parquet out of {len(all_commits)}."
                )
                cur_stats_file = os.path.join(
                    stats_dir_path, f"{owner}_{repo_name}_stats_{batch_num}.prof"
                )
                cur_stats = pstats.Stats(pr)
                cur_stats.sort_stats(pstats.SortKey.TIME)
                cur_stats.dump_stats(cur_stats_file)
                print(
                    f"Dumped stats to {cur_stats_file} at commit {idx} out of {len(all_commits)} for batch {batch_num} out of {len(all_commits) // CHUNK_SIZE} for {repo_path}"
                )
                batch_num += 1
                # Uncomment to profile each batch else it will profile cumulatively till the current batch
                # pr = cProfile.Profile()
                pr.enable()

            data_batch.extend(process_commit(commit, local_path, owner, repo_name))

    # Save remaining data to Parquet
    if data_batch:
        save_to_parquet(data_batch, owner, repo_name, batch_num, dir_path)


def main():
    print(BASE_DIR)
    if not os.path.exists("../data"):
        os.makedirs("../data")

    with open(os.path.join(BASE_DIR, args.file_path), "r", encoding="utf-8") as f:
        repos = f.read().splitlines()[args.start_index : args.end_index + 1]

    print(f"Found {len(repos)} repos to scrape")
    print(repos)
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
        print(local_path)
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
        "file_path",
        type=str,
        help="File path for the .txt file containing the list of repos",
    )
    parser.add_argument("start_index", type=int, help="Start index for repos")
    parser.add_argument("end_index", type=int, help="End index for repos")
    parser.add_argument(
        "--resume_batch",
        type=int,
        default=0,
        help="Batch number from where to resume (0 indexed).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of commits to process in one batch. Default: 1000",
    )
    args = parser.parse_args()
    CHUNK_SIZE = args.chunk_size

    # print working directory & all arguments
    print("Working directory: ", os.getcwd())
    print("All arguments: ", args)
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    main()
