import argparse
import json
import time
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
code_extensions = set(json.load(open(os.path.join(BASE_DIR, "code_extensions.json"), "r")))


def clone_repository(repo_path):
    """
    Clone a repository if it doesn't exist locally.
    """
    owner, repo_name = repo_path.lower().split("/")
    # local_path = f"repos/{owner}_{repo_name}"
    local_path = os.path.join(BASE_DIR, f"repos/{owner}_{repo_name}")
    print(f'Cloning {repo_path} to {local_path}...')
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


# V1
# def get_files_changed_in_commit(commit_sha):
#     cmd = f"git show {commit_sha} --pretty='' --name-only"
#     return run_command(cmd).splitlines()


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
    cmd = f"git show -s --format=%ci {commit_sha}"
    return run_command(cmd, cwd=local_paths)


def scrape_repository(repo_path):
    """
    Scrape all commits and their details from a local repository.
    """
    owner, repo_name = repo_path.lower().split("/")
    local_path = f"repos/{owner}_{repo_name}"
    print(f'Local path of {repo_path}: {local_path}')
    if not os.path.exists(local_path) or not os.listdir(local_path):
        # print(f"Cloning {repo_path}...")
        # clone_repository(repo_path)
        # print(f"Finished cloning {repo_path} to {local_path}")
        print(f'{repo_path} does not exist/is empty in the repos folder. Please clone')
        return []
    # os.chdir(local_path)
        # run_command("git checkout main", cwd=local_path)
    # # TODO - maybe checkout HEAD instead?
    # try:
    #     run_command("git checkout main", cwd=local_path)
    # except Exception:
    #     run_command("git checkout master", cwd=local_path)
    # Ensure the repository is cloned locally
    # local_path = clone_repository(repo_path)
    print(f'Processing commits for {repo_path}')
    all_commits = get_all_commits(local_path)
    print(f"Found {len(all_commits)} commits in {repo_path}")

    data = []

    for commit in all_commits:
        if len(data) % 1000 == 0:
            print(f'{len(data)} commits finished')
        files_changed, is_merge_request = get_files_changed_in_commit(commit, local_path)
        commit_message = get_commit_message(commit, local_path)
        commit_date = get_date(commit, local_path)
        for file in files_changed:
            # Skip files that are not code
            file_extension = file.split(".")[-1]
            if f".{file_extension}" not in code_extensions:
                continue
            # TODO bad code, might need to fix
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
            if previous_content is not None and new_content is not None:
                diff = get_diff(commit, f"{commit}^", file, local_path=local_path)

            status = None
            if previous_content is None and new_content is not None:
                status = "added"
            elif previous_content is not None and new_content is None:
                status = "deleted"
            elif previous_content is not None and new_content is not None:
                status = "modified"
            else:
                status = "unknown"

            data.append(
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
                }
            )


    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path", type=str, help="File path for the .txt file containing the list of repos"
    )
    parser.add_argument("start_index", type=int, help="Start index for repos")
    parser.add_argument("end_index", type=int, help="End index for repos")
    args = parser.parse_args()

    # repos = ["facebook/react"]
    # repos = [
    #     "karpathy/nanoGPT",
    #     "karpathy/llama2.c",
    #     "siddharth-gandhi/refpred",
    #     "ggerganov/llama.cpp",
    # ]
    # repos = ["ggerganov/llama.cpp"]
    if not os.path.exists("data"):
        os.makedirs("data")
    # repo_file_name = "test_repos.txt"
    # repo_file_name = "top_repos.txt"
    repo_file_name = args.file_path
    with open(os.path.join(BASE_DIR, repo_file_name), "r") as f:
        repos = f.read().splitlines()[args.start_index : args.end_index + 1]
    CWD = os.getcwd()
    print(f"Scraping {len(repos)} repositories...")
    print(f"Current working directory: {CWD}")
    for repo in repos:
        owner, repo_name = repo.lower().split("/")
        start_time = time.perf_counter()
        done = False
        try:
            data = scrape_repository(repo)
            done = True
        except Exception as e:
            print(f"Failed to scrape {repo}")
            print(e)
            continue
        end_time = time.perf_counter()
        # in minutes
        elapsed_time = (end_time - start_time) / 60
        if done:
            print(f"Scraped {repo} in {elapsed_time:.2f} minutes storing {len(data)} commits")
        # reset path to cur_dir after each repo
        # os.chdir(CWD)
        with open(os.path.join(BASE_DIR, f"data/{owner}_{repo_name}_commit_data.json"), "w") as f:
            json.dump(data, f)