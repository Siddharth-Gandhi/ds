import json
import os
import subprocess

code_extensions = set(json.load(open("code_extensions.json", "r")))


def clone_repository(repo_path):
    """
    Clone a repository if it doesn't exist locally.
    """
    owner, repo_name = repo_path.split("/")
    local_path = f"repos/{owner}_{repo_name}"
    # print current working directory
    print(os.getcwd())
    if not os.path.exists(local_path):
        os.makedirs(local_path)
        # subprocess.run(["git", "clone", f"https://github.com/{repo_path}.git", local_path])
        run_command(f"git clone https://github.com/{repo_path}.git {local_path}")
    return local_path


def run_command(command: str, cwd=None, timeout=60):
    result = subprocess.run(
        [command], timeout=timeout, capture_output=True, text=True, shell=True, cwd=cwd
    )
    if result.returncode != 0:
        # print(f"Command '{command}' failed with error:")
        # print(result.stderr)
        # raise Exception(f"Command '{command}' failed with error: {result.stderr}")
        return "-1"  # or raise an exception
    return result.stdout


def get_all_commits():
    cmd = "git log --pretty=format:'%H'"
    return run_command(cmd).splitlines()


def get_files_changed_in_commit(commit_sha):
    cmd = f"git show {commit_sha} --pretty='' --name-only"
    return run_command(cmd).splitlines()


def get_commit_message(commit_sha):
    cmd = f"git show -s --format=%B {commit_sha}"
    return run_command(cmd)


def get_file_content_at_commit(commit_sha, file_path, parent=False):
    # If parent=True, get the content from the parent commit (before the change)
    cmd = f"git show {commit_sha}{'^' if parent else ''}:{file_path}"
    return run_command(cmd)


def get_diff(cur_sha, prev_sha, file_path):
    cmd = f"git diff {cur_sha}:{file_path} {prev_sha}:{file_path} | awk '/@@/ {{flag=1}} flag'"
    return run_command(cmd)


def get_previous_commit(commit_sha):
    cmd = f"git rev-parse {commit_sha}^"
    return run_command(cmd)


def scrape_repository(repo_path):
    """
    Scrape all commits and their details from a local repository.
    """
    owner, repo_name = repo_path.split("/")
    local_path = f"repos/{owner}_{repo_name}"
    if not os.path.exists(local_path):
        clone_repository(repo_path)
    os.chdir(local_path)
    # TODO lmao
    try:
        run_command("git checkout main")
    except:
        run_command("git checkout master")
    # Ensure the repository is cloned locally
    local_path = clone_repository(repo_path)
    all_commits = get_all_commits()
    print(f"Found {len(all_commits)} commits in {repo_path}")

    data = []

    for commit in all_commits:
        files_changed = get_files_changed_in_commit(commit)
        commit_message = get_commit_message(commit)
        for file in files_changed:
            # Skip files that are not code
            file_extension = file.split(".")[-1]
            if f".{file_extension}" not in code_extensions:
                continue
            # TODO bad code, need to fix
            try:
                previous_content = get_file_content_at_commit(commit, file, parent=True)
                prev_commit = get_previous_commit(commit)
            except:
                previous_content = None
                prev_commit = None
            try:
                new_content = get_file_content_at_commit(commit, file, parent=False)
            except:
                new_content = None
            diff = None
            if previous_content is not None and new_content is not None:
                diff = get_diff(commit, f"{commit}^", file)

            data.append(
                {
                    "commit_id": commit,
                    "commit_message": commit_message,
                    "file_path": file,
                    "previous_commit": prev_commit,
                    "previous_content": previous_content,
                    "new_content": new_content,
                    "diff": diff,
                }
            )

    return data


if __name__ == "__main__":
    CWD = os.getcwd()
    # repos = ["siddharth-gandhi/refpred"]
    repos = ["karpathy/llama2.c"]
    # commit_data = extract_commits_info("repos/makemore")
    if not os.path.exists("data_local"):
        os.makedirs("data_local")
    for repo in repos:
        owner, repo_name = repo.split("/")
        data = scrape_repository(repo)
        # reset path to cur_dir after each repo
        os.chdir(CWD)
        with open(f"data_local/{owner}_{repo_name}_commit_data_local.json", "w") as f:
            json.dump(data, f)
    # Save or process the data as required
    # with open("commit_data_local.json", "w") as f:
    #     json.dump(commit_data, f)
