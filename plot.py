import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    # Text data provided
    data = """
    Repo: karpathy/llama2.c - Total Commits: 462, Stored Commits: 401, Total Number of requests: 1196 - Time Taken: 354.02s
    Repo: karpathy/micrograd - Total Commits: 24, Stored Commits: 14, Total Number of requests: 48 - Time Taken: 12.79s
    Repo: karpathy/makemore - Total Commits: 21, Stored Commits: 31, Total Number of requests: 55 - Time Taken: 13.04s
    Repo: siddharth-gandhi/refpred - Total Commits: 23, Stored Commits: 71, Total Number of requests: 140 - Time Taken: 33.02s
    Repo: karpathy/nanoGPT - Total Commits: 195, Stored Commits: 189, Total Number of requests: 560 - Time Taken: 136.30s
    Repo: ggerganov/whisper.cpp - Total Commits: 722, Stored Commits: 1125, Total Number of requests: 2785 - Time Taken: 823.13s
    """

    # Regular expression to extract the required data
    pattern = r"Repo: (.+?) - Total Commits: (\d+), .+? Total Number of requests: (\d+) - Time Taken: (\d+\.\d+)s"

    # Extract the data using regex
    matches = re.findall(pattern, data)

    # Create lists to store the extracted data
    repos = [match[0] for match in matches]
    total_commits = [int(match[1]) for match in matches]
    total_requests = [int(match[2]) for match in matches]
    time_taken = [float(match[3]) for match in matches]

    # Setting up the plot
    plt.figure(figsize=(12, 8))
    plt.plot(total_commits, total_requests)
    plt.scatter(total_commits, total_requests, s=100, c="blue", marker="o")

    # Annotating each point with its repo name and time taken
    for i, repo in enumerate(repos):
        plt.annotate(
            f"{repo} ({time_taken[i]:.2f}s)", (total_commits[i], total_requests[i]), fontsize=9
        )

    # Setting plot labels and title
    plt.xlabel("Total Commits")
    plt.ylabel("Total Requests")
    plt.title("Total Requests vs Total Commits for Different Repos")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("commit_v_request_plot.png")

    # plt.show()
