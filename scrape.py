import json
import os

from dotenv import load_dotenv

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
modified_commit_count = 0
commit_count = 0

code_extensions = set(json.load(open("code_extensions.json", "r")))


def get_commits(repo_path):
    """
    Get all commits from a specific repository.
    """
    global request_count, commit_count
    url = f"{BASE_URL}/{repo_path}/commits"
    response = requests.get(url, headers=HEADERS)
    request_count += 1
    commit_count += len(response.json())
    return response.json()


def get_commit_details(repo_path, commit_sha):
    """
    Get details of a specific commit.
    """
    global request_count
    url = f"{BASE_URL}/{repo_path}/commits/{commit_sha}"
    response = requests.get(url, headers=HEADERS)
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
    request_count += 1
    content_data = response.json()

    # The content is base64 encoded in the response
    import base64

    decoded_content = base64.b64decode(content_data["content"]).decode("utf-8")
    return decoded_content


def extract_commit_info(repo_path, commit):
    """
    Extract desired information from a commit.
    """
    global modified_commit_count
    commit_details = get_commit_details(repo_path, commit["sha"])
    data = []

    for file in commit_details["files"]:
        file_name = file["filename"]
        # extract the extension
        file_extension = file_name.split(".")[-1]
        if f".{file_extension}" not in code_extensions:
            continue
        modified_commit_count += 1
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
    commits = get_commits(repo_path)
    data = []

    for commit in commits:
        data.extend(extract_commit_info(repo_path, commit))
    with open("commit_data.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    repo_path = "karpathy/micrograd"  # In the format: <owner>/<repository_name>
    scrape_repository(repo_path)
    print(
        f"Total Number of requests for {commit_count} total & {modified_commit_count} modified commits is {request_count}"
    )
