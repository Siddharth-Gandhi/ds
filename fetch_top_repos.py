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
    "User-Agent": "GitHub Starred Repos Scraper",
    "Accept": "application/vnd.github.v3+json",
}


# def fetch_most_starred_repos(page_number):
#     url = f"https://api.github.com/search/repositories?q=stars:>1&sort=stars&order=desc&page={page_number}&per_page=100"
#     response = requests.get(url, headers=HEADERS)
#     if response.status_code != 200:
#         raise Exception(f"Failed to fetch data: {response.text}")
#     return response.json()["items"]


def fetch_most_starred_repos(page_number, star_range):
    url = f"https://api.github.com/search/repositories?q=stars:{star_range}&sort=stars&order=desc&page={page_number}&per_page=100"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(
            f"Failed to fetch data for stars:{star_range} on page {page_number}: {response.text}"
        )
        return []
    return response.json()["items"]


def is_valid_language(repo):
    with open("programming_languages.json", "r") as file:
        valid_languages = json.load(file)
    return repo["language"] and repo["language"].lower() in valid_languages


def fetch_additional_details(repo_full_name):
    commits_url = f"https://api.github.com/repos/{repo_full_name}/commits"
    prs_url = f"https://api.github.com/repos/{repo_full_name}/pulls"

    commit_response = requests.get(commits_url, headers=HEADERS, params={"per_page": 1})
    pr_response = requests.get(
        prs_url, headers=HEADERS, params={"per_page": 1, "state": "all"}
    )

    commit_count, pr_count = 0, 0

    # Extract commit count
    if "Link" in commit_response.headers:
        last_page_match = re.search(
            r'page=(\d+)>; rel="last"', commit_response.headers["Link"]
        )
        commit_count = (
            int(last_page_match.group(1))
            if last_page_match
            else len(commit_response.json())
        )
    else:
        commit_count = len(commit_response.json())

    # Extract PR count
    if "Link" in pr_response.headers:
        last_page_match = re.search(
            r'page=(\d+)>; rel="last"', pr_response.headers["Link"]
        )
        pr_count = (
            int(last_page_match.group(1))
            if last_page_match
            else len(pr_response.json())
        )
    else:
        pr_count = len(pr_response.json())

    return commit_count, pr_count


def main():
    TOTAL_REPOS = 500
    repo_details_list = []
    star_ranges = [">10000", "5000..10000", "1000..4999"]  # Example ranges
    for star_range in star_ranges:
        for page in tqdm(
            range(1, TOTAL_REPOS // 10 + 1),
            desc=f"Fetching repo details for stars:{star_range}",
        ):
            repos = fetch_most_starred_repos(page, star_range)
            for repo in repos:
                if is_valid_language(repo):
                    commit_count, pr_count = fetch_additional_details(repo["full_name"])
                    repo_info = {
                        "repo_name": repo["name"],
                        "owner": repo["owner"]["login"],
                        "number_of_stars": repo["stargazers_count"],
                        "open_issue_count": repo["open_issues_count"],
                        "language": repo["language"],
                        "number_of_commits": commit_count,
                        "num_pull_requests": pr_count,
                    }
                    repo_details_list.append(repo_info)

    # Sorting the list of repositories by stars in descending order
    repo_details_list.sort(key=lambda x: x["number_of_stars"], reverse=True)

    with open(f"repo_info_{TOTAL_REPOS}.json", "w") as json_file:
        json.dump(repo_details_list, json_file, indent=4)


if __name__ == "__main__":
    main()
