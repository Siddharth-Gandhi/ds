import os
import subprocess
import sys


def clone_repository(repo_path, destination_dir):
    owner, repo_name = repo_path.lower().split("/")
    local_path = os.path.join(destination_dir, f"{owner}_{repo_name}")

    if not os.path.exists(local_path) or not os.listdir(local_path):
        print(f"Cloning {repo_path} to {local_path}...")
        success = run_command(f"git clone https://github.com/{repo_path}.git {local_path}")
        if not success:
            with open("failed_clone.txt", "a") as f:
                f.write(f"{repo_path}\n")
        else:
            print(f"Finished cloning {repo_path} at {local_path}")
    else:
        print(f"{repo_path} already exists at {local_path}")


def run_command(command: str, cwd=None, capture_output=True):
    try:
        result = subprocess.run([command], capture_output=capture_output, text=True, shell=True, cwd=cwd)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
        return False


def main(file_name, start_index, end_index, destination_dir="repos"):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    with open(file_name, "r") as f:
        repos = [line.strip() for line in f.readlines()][start_index : end_index + 1]

    for repo_path in repos:
        clone_repository(repo_path, destination_dir)
    return


if __name__ == "__main__":
    file_path = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    main(file_path, start, end)