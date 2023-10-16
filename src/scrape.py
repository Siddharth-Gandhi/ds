# RATE LIMIT : 5000 requests per hour
import json
import os
import time

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

import requests

TOKEN = os.getenv("GITHUB_PERSONAL_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "User-Agent": "GitHub Commit Scraper (https://www.ssgandhi.com/posts/github_scrape)",
    "Accept": "application/vnd.github.json",
}

BASE_URL = "https://api.github.com/repos"
request_count = 0
# modified_commit_count = 0
total_commit_count = 0
stored_commit_count = 0

code_extensions = set(json.load(open("../misc/code_extensions.json", "r")))


# def get_commits(repo_path):
#     """
#     Get all commits from a specific repository.
#     """
#     global request_count, total_commit_count
#     url = f"{BASE_URL}/{repo_path}/commits"
#     response = requests.get(url, headers=HEADERS)
#     if response.status_code != 200:
#         print(response.json())
#         raise Exception(f"Error getting file content: {response.status_code}")
#     request_count += 1
#     total_commit_count += len(response.json())
#     return response.json()


def get_commits(repo_path):
    """
    Get all commits from a specific repository.
    """
    global request_count, total_commit_count

    commits = []
    page = 1
    while True:
        url = f"{BASE_URL}/{repo_path}/commits?per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)

        if response.status_code != 200:
            print(response.json())
            raise Exception(f"Error getting file content: {response.status_code}")

        request_count += 1
        current_commits = response.json()
        total_commit_count += len(current_commits)
        commits.extend(current_commits)

        # Check if there are more pages of commits
        if "next" in response.links:
            page += 1
        else:
            break

    return commits


def get_commit_details(repo_path, commit_sha):
    """
    Get details of a specific commit.
    """
    global request_count
    url = f"{BASE_URL}/{repo_path}/commits/{commit_sha}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(response.json())
        raise Exception(f"Error getting file content: {response.status_code}")
    request_count += 1
    return response.json()


# def extract_commit_info(repo_path, commit):
#     """
#     Extract desired information from a commit.
#     """
#     commit_details = get_commit_details(repo_path, commit["sha"])
#     data = []
#
#     for file in commit_details["files"]:
#         if file['status'] == 'modified':
#             data.append(
#                 {
#                     "commit_id": commit["sha"],
#                     "commit_message": commit["commit"]["message"],
#                     "relative_path": file["filename"],
#                     "previous_code_file": file[
#                         "patch"
#                     ],  # This will give the diff, if you want the raw content, you'd need another API call.
#                     "new_code_file": file["patch"],  # Same as above.
#                 }
#             )
#
#     return data


def get_file_content_at_commit(repo_path, file_path, commit_sha):
    """
    Get the content of a file at a specific commit.
    """
    global request_count
    url = f"{BASE_URL}/{repo_path}/contents/{file_path}?ref={commit_sha}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(response.json())
        raise Exception(f"Error getting file content: {response.status_code}")
    request_count += 1
    content_data = response.json()

    # The content is base64 encoded in the response
    import base64

    decoded_content = base64.b64decode(content_data["content"]).decode("utf-8")
    return decoded_content


def extract_commit_info(repo_path, commit):
    # sourcery skip: raise-specific-error
    """
    Extract desired information from a commit.
    """
    global stored_commit_count
    commit_details = get_commit_details(repo_path, commit["sha"])
    data = []

    for file in commit_details["files"]:
        file_name = file["filename"]
        # extract the extension
        file_extension = file_name.split(".")[-1]
        if f".{file_extension}" not in code_extensions:
            continue
        stored_commit_count += 1
        # 4 cases: added, removed, modified, renamed
        if file["status"] == "modified":
            patch = file.get("patch", None)
            previous_commit_sha = commit["parents"][0]["sha"]
            # Get previous and new file contents
            previous_file_content = get_file_content_at_commit(
                repo_path, file["filename"], previous_commit_sha
            )
            new_file_content = get_file_content_at_commit(
                repo_path, file["filename"], commit["sha"]
            )

            data.append(
                {
                    "commit_id": commit["sha"],
                    "commit_message": commit["commit"]["message"],
                    "relative_path": file["filename"],
                    "previous_code_file": previous_file_content,
                    "previous_id": commit["parents"][0]["sha"],
                    "new_code_file": new_file_content,
                    "diff": patch,
                    "status": "modified",
                }
            )
        elif file["status"] == "added":
            new_file_content = get_file_content_at_commit(
                repo_path, file["filename"], commit["sha"]
            )
            data.append(
                {
                    "commit_id": commit["sha"],
                    "commit_message": commit["commit"]["message"],
                    "relative_path": file["filename"],
                    "previous_code_file": None,
                    "new_code_file": new_file_content,
                    "diff": None,
                    "status": "added",
                }
            )
        elif file["status"] == "removed":
            previous_commit_sha = commit["parents"][0]["sha"]
            previous_file_content = get_file_content_at_commit(
                repo_path, file["filename"], previous_commit_sha
            )
            data.append(
                {
                    "commit_id": commit["sha"],
                    "commit_message": commit["commit"]["message"],
                    "relative_path": file["filename"],
                    "previous_code_file": previous_file_content,
                    "previous_id": commit["parents"][0]["sha"],
                    "new_code_file": None,
                    "diff": None,
                    "status": "removed",
                }
            )
        elif file["status"] == "renamed":
            previous_commit_sha = commit["parents"][0]["sha"]
            previous_file_content = get_file_content_at_commit(
                repo_path, file["previous_filename"], previous_commit_sha
            )
            new_file_content = get_file_content_at_commit(
                repo_path, file["filename"], commit["sha"]
            )
            data.append(
                {
                    "commit_id": commit["sha"],
                    "commit_message": commit["commit"]["message"],
                    "relative_path": file["filename"],
                    "previous_code_file": previous_file_content,
                    "previous_id": commit["parents"][0]["sha"],
                    "previous_file_name": file["previous_filename"],
                    "new_code_file": new_file_content,
                    "diff": None,
                    "status": "renamed",
                }
            )
        else:
            raise Exception(f"Unknown status: {file['status']}")

    return data


def scrape_repository(repo_path):
    """
    Scrape all commits and their details from a repository.
    """
    owner, repo_name = repo_path.split("/")

    # Ensure the 'data' directory exists
    if not os.path.exists("../data"):
        os.makedirs("../data")

    commits = get_commits(repo_path)
    data = []

    prev_time = time.time()
    # for commit in tqdm(commits):
    with tqdm(commits, unit="commit") as pbar:
        for commit in pbar:
            prev_requests = request_count
            data.extend(extract_commit_info(repo_path, commit))
            elapsed_time = time.time() - prev_time
            avg_rate = (request_count - prev_requests) / elapsed_time
            pbar.set_description(f"Repo: {repo_path}")
            pbar.set_postfix(avg_req_per_sec=f"{avg_rate:.2f}")
            pbar.set_postfix(total_req=request_count)
            prev_time = time.time()
    # with open("commit_data.json", "w") as f:
    #     json.dump(data, f)
    with open(f"data/{owner}_{repo_name}_commit_data.json", "w") as f:
        json.dump(data, f)


# sourcery skip: hoist-statement-from-loop
if __name__ == "__main__":
    # repo_path = "karpathy/micrograd"  # In the format: <owner>/<repository_name>
    # delete stats file
    # if os.path.exists("request_stats.txt"):
    #     os.remove("request_stats.txt")
    repos = ["ggerganov/whisper.cpp"]
    # repos = ["karpathy/nanoGPT"]
    # repos = ["karpathy/micrograd", "karpathy/makemore"]
    # repos = ["karpathy/llama2.c"]
    # scrape_repository(repo_path)
    for repo_path in repos:
        request_count = 0
        total_commit_count = 0
        # modified_commit_count = 0
        # add a timer

        start_time = time.perf_counter()
        scrape_repository(repo_path)
        end_time = time.perf_counter()
        stats_message = f"Repo: {repo_path} - Total Commits: {total_commit_count}, Stored Commits: {stored_commit_count}, Total Number of requests: {request_count} - Time Taken: {end_time - start_time:.2f}s"

        print(stats_message)
        with open("request_stats.txt", "a+") as stats_file:
            stats_file.write(stats_message + "\n")
    # print(
    #     f"Total Number of requests for {total_commit_count} total & {modified_commit_count} modified commits is {request_count}"
    # )
