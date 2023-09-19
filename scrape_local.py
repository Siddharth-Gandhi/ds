import subprocess
import os
import json

def run_git_command(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout

def get_all_commits():
    cmd = "git log --pretty=format:'%H'"
    return run_git_command(cmd).splitlines()

def get_files_changed_in_commit(commit_sha):
    cmd = f"git show {commit_sha} --pretty='' --name-only"
    return run_git_command(cmd).splitlines()

def get_file_content_at_commit(commit_sha, file_path, parent=False):
    # If parent=True, get the content from the parent commit (before the change)
    cmd = f"git show {commit_sha}{'^' if parent else ''}:{file_path}"
    return run_git_command(cmd)

def extract_commits_info(repo_path):
    os.chdir(repo_path)
    all_commits = get_all_commits()

    data = []

    for commit in all_commits:
        files_changed = get_files_changed_in_commit(commit)
        for file in files_changed:
            previous_content = get_file_content_at_commit(commit, file, parent=True)
            new_content = get_file_content_at_commit(commit, file, parent=False)

            data.append({
                "commit_id": commit,
                "file_path": file,
                "previous_content": previous_content,
                "new_content": new_content,
            })

    return data

if __name__ == "__main__":
    original_path = os.getcwd()
    commit_data = extract_commits_info('repos/makemore')
    # Save or process the data as required
    os.chdir(original_path)
    with open("commit_data_local.json", "w") as f:
        json.dump(commit_data, f)