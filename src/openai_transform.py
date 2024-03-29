import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from utils import get_combined_df

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# SYSTEM_MESSAGE="""
# You are a developer who is tasked with a very specific goal. You will be given commit messages from a Github repo, and your task is to identify the core problem(s) that this particular commit is trying to solve. With that information, you need to write a short description of problem itself in the style of a Github issue before this change was commited. In essence you are trying to reconstruct what potential bugs led to certain commits happening.  Do not mention any information about the solution in the commit (as that would be cheating). Just what the potential issue or problem could have been that led to this commit being needed.

# Do not mention the names or contact information of any people involved in the commits (like authors or reviewers). Do not start responding with any description, just start the issue.

# For example:
# I will give you this commit message:
# '[Fizz] Fix for failing id overwrites for postpone (#27684)
# When we postpone during a render we inject a new segment synchronously
# which we postpone. That gets assigned an ID so we can refer to it
# immediately in the postponed state.

# When we do that, the parent segment may complete later even though it's
# also synchronous. If that ends up not having any content in it, it'll
# inline into the child and that will override the child's segment id
# which is not correct since it was already assigned one.

# To fix this, we simply opt-out of the optimization in that case which is
# unfortunate because we'll generate many more unnecessary empty segments.
# So we should come up with a new strategy for segment id assignment but
# this fixes the bug.

# Co-authored-by: Josh Story <story@hey.com>'


# And I want you to respond:
# 'When postponing during a render, injecting a new segment synchronously that gets assigned an ID can lead to issues. If the parent segment, also synchronous, completes later without any content, it inlines into the child segment, incorrectly overriding the child's segment ID which was already assigned.'
# """

SYSTEM_MESSAGE="""
You are a professional software developer who is given a very specific task. You will be given commit messages solving one or more problems from any big open-source Github repository, and your task is to identify those core problem(s) that this particular commit is trying to solve. With that information, you need to write a short description of problem itself in the style of a Github issue which would have existed *before* this change was committed. In essence you are trying to reconstruct what potential bugs led to certain commits happening.  Do not mention any information about the solution in the commit (as that would be cheating). Just what the potential issue or problem could have been that led to this commit being needed.

Do not mention the names or contact information of any people involved in the commits (like authors or reviewers). Do not start responding with any description or titles, just start the issue. Limit your response to at most 2 lines. Just think of yourself as a developer who encountered a bug in a hurry and has to write 2 lines which captures as much information about the problem as possible. And remember do NOT leak any details of the solution in the issue message.

For example, given the commit message below:
'[Fizz] Fix for failing id overwrites for postpone (#27684)
When we postpone during a render we inject a new segment synchronously which we postpone. That gets assigned an ID so we can refer to it immediately in the postponed state.

When we do that, the parent segment may complete later even though it's also synchronous. If that ends up not having any content in it, it'll inline into the child and that will override the child's segment id which is not correct since it was already assigned one.

To fix this, we simply opt-out of the optimization in that case which is unfortunate because we'll generate many more unnecessary empty segments. So we should come up with a new strategy for segment id assignment but this fixes the bug.

Co-authored-by: Josh Story <story@hey.com>'

I want you to respond similar to:
'Synchronous render with postponed segments results in incorrect segment ID overrides, causing empty segments to be generated unnecessarily.'
"""

def filter_df(combined_df):
    # Step 1: Filter out only the columns we need
    filtered_df = combined_df[['commit_date', 'commit_message', 'commit_id', 'file_path', 'diff']]

    # Step 2: Group by commit_id
    grouped_df = filtered_df.groupby(['commit_id', 'commit_date', 'commit_message'])['file_path'].apply(list).reset_index()
    grouped_df.rename(columns={'file_path': 'actual_files_modified'}, inplace=True)

    # Step 3: Determine midpoint and filter dataframe
    midpoint_date = np.median(grouped_df['commit_date'])
    recent_df = grouped_df[grouped_df['commit_date'] > midpoint_date]
    print(f'Number of commits after midpoint date: {len(recent_df)}')

    # Step 4: Filter out commits with less than average length commit messages
    average_commit_len = recent_df['commit_message'].str.split().str.len().mean()
    # filter out commits with less than average length
    recent_df = recent_df[recent_df['commit_message'].str.split().str.len() > average_commit_len]
    print(f'Number of commits after filtering by commit message length: {len(recent_df)}')

    # Step 5: Remove outliers based on commit message length and number of files modified
    q1_commit_len = recent_df['commit_message'].str.len().quantile(0.25)
    q3_commit_len = recent_df['commit_message'].str.len().quantile(0.75)
    iqr_commit_len = q3_commit_len - q1_commit_len

    q1_files = recent_df['actual_files_modified'].apply(len).quantile(0.25)
    q3_files = recent_df['actual_files_modified'].apply(len).quantile(0.75)
    iqr_files = q3_files - q1_files

    filter_condition = (
        (recent_df['commit_message'].str.len() <= (q3_commit_len + 1.5 * iqr_commit_len)) &
        (recent_df['actual_files_modified'].apply(len) <= (q3_files + 1.5 * iqr_files))
    )
    filtered_df = recent_df[filter_condition]

    print(f'Number of commits after filtering by commit message length and number of files modified: {len(filtered_df)}')

    return filtered_df


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def transform_message(commit_message, model):
    user_message = f"Now do the same thing for this commit message: {commit_message}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content

def process_repository(repo_path, model_name, save_model_name, train_sample_size, test_sample_size, VERSION, save_dir):
    print(f'Processing {repo_path}')
    repo_name = repo_path.split('/')[-1]
    gold_dir = f'gold/{repo_name}'
    # train_dir = f'{gold_dir}/train'
    # create gold_dir if it does not exist
    if not os.path.exists(gold_dir):
        os.makedirs(gold_dir)

    gold_commit_ids_file = f'{gold_dir}/{repo_name}_gpt4_gold_commit_ids.txt'
    combined_df = get_combined_df(repo_path)
    filtered_df = filter_df(combined_df)
    test_commit_ids = []
    create_gold = False
    # check if gold_commit_ids_file exists
    if os.path.exists(gold_commit_ids_file):
        print(f'Found gold commit ids file for {repo_path}')
        with open(gold_commit_ids_file, 'r') as f:
            for line in f:
                test_commit_ids.append(line.strip())
    else:
        create_gold = True
        print(f'No gold commit ids file found for {repo_path}')
        # this repo does not have test commit ids, so we will randomly sample 100 commits
        test_commit_ids = filtered_df['commit_id'].sample(n=test_sample_size, random_state=42, replace=False).tolist()

        # write test_commit_ids to a file
        with open(gold_commit_ids_file, 'w') as f:
            for commit_id in test_commit_ids:
                f.write(f'{commit_id}\n')

    # filter out test_commit_ids from filtered_df to get train_df
    train_df = filtered_df[~filtered_df['commit_id'].isin(test_commit_ids)]
    print(f'Number of commits in train_df: {len(train_df)}')

    # assert none of commits between train_df and test_df overlap
    assert not set(train_df['commit_id'].tolist()).intersection(
        set(test_commit_ids)
    )

    # sample train_sample_size commits from train_df with seed 42 without replacement
    train_df = train_df.sample(n=train_sample_size, random_state=42, replace=False)

    output_dir = os.path.join(save_dir, repo_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # write train_commit_ids to a file
    train_commit_ids_file = os.path.join(output_dir, f'{VERSION}_{repo_name}_{save_model_name}_train_commit_ids.txt')
    if os.path.exists(train_commit_ids_file):
        print(f'Found train commit ids file for {repo_path}, skipping')
        return

    with open(train_commit_ids_file, 'w') as f:
        for commit_id in train_df['commit_id'].tolist():
            f.write(f'{commit_id}\n')



    def process_df(final_df, is_train):
        print(f'Number of unique commit_ids: {len(final_df)}')
        # add a new column to final_df
        final_df[f'transformed_message_{save_model_name}'] = np.nan
        for index, row in tqdm(final_df.iterrows(), total=final_df.shape[0]):
            # check if transformed_message_gpt3 is NaN or not
            if not pd.isna(row[f'transformed_message_{save_model_name}']):
                continue
            transformed_message = transform_message(row['commit_message'], model_name)
            final_df.at[index, f'transformed_message_{save_model_name}'] = transformed_message
        if not is_train:
            csv_save_path = os.path.join(output_dir, f'{VERSION}_{repo_name}_{save_model_name}_gold.csv')
        else:
            csv_save_path = os.path.join(output_dir, f'{VERSION}_{repo_name}_{save_model_name}_train.csv')
        final_df.to_csv(csv_save_path, index=False)
        print(f'Saved {csv_save_path}')

        if not is_train:
            parquet_save_path = os.path.join(output_dir, f'{VERSION}_{repo_name}_{save_model_name}_gold.parquet')
        else:
            parquet_save_path = os.path.join(output_dir, f'{VERSION}_{repo_name}_{save_model_name}_train.parquet')
        final_df.to_parquet(parquet_save_path, index=False)
        print(f'Saved {parquet_save_path}')

        # save a list of commit_ids to a file {repo_name}_{save_model_name}_gold_commit_ids.txt
        # all_commit_ids = final_df['commit_id'].tolist()
        # commit_ids_save_path = os.path.join(output_dir, f'{VERSION}_{repo_name}_{save_model_name}_gold_commit_ids.txt')
        #
        # with open(commit_ids_save_path, 'w') as f:
        #     for commit_id in all_commit_ids:
        #         f.write(f'{commit_id}\n')

        print(f'Finished processing {repo_path}')


    if create_gold:
        test_df = filtered_df[filtered_df['commit_id'].isin(test_commit_ids)]
        print(f'Number of commits in test_df: {len(test_df)}')
        process_df(test_df, is_train=False)

    process_df(train_df, is_train=True)

def main():
    save_dir = 'gold/'
    # model_name = "gpt-3.5-turbo"
    model_name = "gpt-4"
    save_model_name = 'gpt4' if model_name == 'gpt-4' else 'gpt3.5'
    VERSION = 'v2'
    train_sample_size = 500
    test_sample_size = 100

    # REPO_LIST = []
    print(os.getcwd())
    # REPO_LIST = ['2_7/facebook_react']
    # REPO_LIST = ['2_7/apache_spark', '2_7/apache_kafka', '2_8/angular_angular', '2_8/django_django',
    REPO_LIST = ['2_7/julialang_julia', '2_7/ruby_ruby','2_8/pytorch_pytorch', '2_9/redis_redis', '2_9/huggingface_transformers']
    # Specify the model name as a variable
    print(f'Total number of repos: {len(REPO_LIST)}')
    print(f'Using model: {model_name} with train_sample_size: {train_sample_size} and test_sample_size: {test_sample_size}')

    for repo_path in REPO_LIST:
        process_repository(repo_path, model_name, save_model_name, train_sample_size, test_sample_size, VERSION, save_dir)

if __name__ == "__main__":
    main()
