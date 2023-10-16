import json


def get_commit_ids_from_file(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return set(item["commit_id"] for item in data)


def main(file1, file2):
    commits_file1 = get_commit_ids_from_file(file1)
    commits_file2 = get_commit_ids_from_file(file2)

    in_file1_not_in_file2 = commits_file1 - commits_file2
    in_file2_not_in_file1 = commits_file2 - commits_file1

    if in_file1_not_in_file2:
        print(f"Commits in {file1} but not in {file2}:")
        for commit in in_file1_not_in_file2:
            print(commit)

    if in_file2_not_in_file1:
        print(f"\nCommits in {file2} but not in {file1}:")
        for commit in in_file2_not_in_file1:
            print(commit)

    with open(file1, "r") as f:
        data = json.load(f)
        for item in data:
            if item["commit_id"] in in_file1_not_in_file2:
                print(item["commit_id"])
                print(item["commit_message"])
                # print(item["file_path"])
                # print(item["previous_commit"])
                # print(True if item["diff"] else False)

    with open(file2, "r") as f:
        data = json.load(f)
        for item in data:
            if item["commit_id"] in in_file2_not_in_file1:
                print(item["commit_id"])
                print(item["commit_message"])


if __name__ == "__main__":
    # Replace 'file1.json' and 'file2.json' with your filenames
    main(
        "data/karpathy_llama2.c_commit_data.json",
        "data_local/karpathy_llama2.c_commit_data_local.json",
    )
