import json
import os
import re

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Set up GitHub API details
TOKEN = os.getenv("GITHUB_PERSONAL_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "User-Agent": "GitHub Commit Scraper (https://www.ssgandhi.com/posts/github_scrape)",
    "Accept": "application/vnd.github.v3+json",
}


def fetch_repo_details(repo_name):
    # Fetch basic repo details
    url = f"https://api.github.com/repos/{repo_name}"
    response = requests.get(url, headers=HEADERS)
    repo_data = response.json()

    # Extract relevant details
    repo_info = {
        "repo_name": repo_data["name"],
        "owner": repo_data["owner"]["login"],
        "number_of_stars": repo_data["stargazers_count"],
        "open_issue_count": repo_data["open_issues_count"],
        "language": repo_data["language"],
    }

    # Fetch commit count (this involves pagination due to GitHub API limitations)
    commits_url = f"https://api.github.com/repos/{repo_name}/commits"
    response = requests.get(commits_url, headers=HEADERS, params={"per_page": 1})
    commit_count = 0
    if "Link" in response.headers:
        # Use regex to extract last page number from Link header
        last_page_match = re.search(
            r'page=(\d+)>; rel="last"', response.headers["Link"]
        )
        if last_page_match:
            commit_count = int(last_page_match.group(1))
        else:
            # If "last" link is not found, it means there's only one page of commits
            commit_count = len(response.json())
    else:
        commit_count = len(response.json())
    repo_info["number_of_commits"] = commit_count

    # Fetch pull requests count
    prs_url = f"https://api.github.com/repos/{repo_name}/pulls"
    response = requests.get(
        prs_url, headers=HEADERS, params={"per_page": 1, "state": "all"}
    )
    pr_count = 0
    if "Link" in response.headers:
        # Use regex to extract last page number from Link header for PRs
        last_page_match = re.search(
            r'page=(\d+)>; rel="last"', response.headers["Link"]
        )
        if last_page_match:
            pr_count = int(last_page_match.group(1))
        else:
            pr_count = len(response.json())
    else:
        pr_count = len(response.json())
    repo_info["num_pull_requests"] = pr_count

    return repo_info


def main():
    with open("../misc/top_repos.txt", "r") as file:
        repo_list = file.read().splitlines()

    repo_details_list = []
    for repo in tqdm(repo_list, desc="Fetching repo details"):
        repo_details_list.append(fetch_repo_details(repo))

    with open("../misc/repo_info.json", "w") as json_file:
        json.dump(repo_details_list, json_file, indent=4)


if __name__ == "__main__":
    main()
