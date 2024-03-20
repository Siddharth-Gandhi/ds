import json
import os
from datetime import date
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from models import CodeReranker
from utils import AggregatedSearchResult, get_avg_metrics


# Function to generate .teIn file
def generate_teIn_file(commit_results: Dict[str, List[AggregatedSearchResult]], file_path: str):
    # os.makedirs(folder_path, exist_ok=True)
    # file_path = os.path.join(folder_path, "output.teIn")

    with open(file_path, "w") as file:
        for commit_id, results in commit_results.items():
            for result in results:
                file.write(f'{commit_id}\t{result.file_path}\t{result.score}\t{[{"score": sr.score, "file_path": sr.file_path, "commit_id": sr.commit_id, "commit_date": sr.commit_date} for sr in result.contributing_results]}\n')

# Function to generate .qry file
def generate_qry_file(gold_df, file_path: str, use_gpt_train: bool = True):
    # Ensure folder exists
    # os.makedirs(folder_path, exist_ok=True)
    # file_path = os.path.join(folder_path, "output.qry")
    with open(file_path, "w") as file:
        for index, row in gold_df.iterrows():
            cid, date, orig, files, query = row
            for f in files:
                file.write(f'{cid}\t{date}\t{f}\n')

            # write in order cid, date, files, query, orig
            # file.write(f'{cid}\t{[f"{f}" for f in files]}\n')
            # file.write(f"{row['commit_id']}\t{msg}\n")

def evaluate_results(teIn_path, qry_path, output_path, evaluator):
    # Read actual files modified from .qry file
    actual_files = {}
    total_list = []
    with open(qry_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            cid = parts[0]
            files = parts[2]
            # files_str = parts[2].replace("'", '"')
            # try:
            #     files = json.loads(files_str)
            # except json.JSONDecodeError as e:
            #     print(f"Error decoding JSON for commit {cid}: {e}")
            #     continue  # Skip this line or handle error as needed
            if cid not in actual_files:
                actual_files[cid] = []
            actual_files[cid].append(files)

    # Read predicted files and scores from .teIn file
    predicted_files = {}
    with open(teIn_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            cid = parts[0]
            file_path = parts[1]
            # score = float(parts[2]) # Score is not used directly in evaluation, but might be useful for other purposes

            if cid not in predicted_files:
                predicted_files[cid] = []
            predicted_files[cid].append(file_path)

    # Evaluate results and write to .teOut file
    # os.makedirs(output_folder, exist_ok=True)
    # output_path = os.path.join(output_folder, "output.teOut")

    with open(output_path, 'w') as f:
        for cid, files_predicted in predicted_files.items():
            actual_files_modified = actual_files.get(cid, [])
            evaluation_results = evaluator.evaluate(files_predicted, actual_files_modified)
            total_list.append(evaluation_results)

            f.write(f"{cid}\t{json.dumps(evaluation_results)}\n")

        # Print average metrics
        f.write(f"Average metrics: {json.dumps(get_avg_metrics(total_list))}\n")


class SearchEvaluator:
    def __init__(self, metrics):
        self.metrics = metrics

    @staticmethod
    def precision_at_k(relevant, k):
        return sum(relevant[:k]) / k

    @staticmethod
    def mean_reciprocal_rank(relevant):
        for idx, value in enumerate(relevant):
            if value == 1:
                return 1 / (idx + 1)
        return 0

    @staticmethod
    def calculate_average_precision(relevant):
        pred_rel = [1] * len(relevant)
        relevant_documents_count = 0
        cumulative_precision = 0.0

        # We iterate through the predicted relevance scores
        for i in range(len(pred_rel)):
            # Check if the prediction at this rank is correct (i.e., if it is a relevant document)
            if pred_rel[i] == 1 and relevant[i] == 1:
                relevant_documents_count += 1
                precision_at_i = relevant_documents_count / (i + 1)
                cumulative_precision += precision_at_i

        # The average precision is the cumulative precision divided by the number of relevant documents
        average_precision = cumulative_precision / sum(relevant) if sum(relevant) > 0 else 0
        return average_precision

    # @staticmethod
    # def calculate_recall(relevant, total_modified_files, k):
    #   # Does not work for commit based approach as it can have multiple mentions of the same file across commits leading to a higher than 1 recall
    #     print(total_modified_files)
    #     print(relevant)
    #     return sum(relevant[:k]) / total_modified_files

    @staticmethod
    def calculate_recall(retrieved_files, actual_modified_files, relevant, k):
        # this complicated mess is required as compared to the above much simpler code to support both commit-based and file-based approaches
        # in file-based approach, this is equivalent to the above code
        # in code-based approach, duplicates could be present in retrieved_files, which is why we need to filter them out (the above code would not work in this case)

        return len({file for idx, file in enumerate(retrieved_files[:k])
                        if relevant[idx] == 1
                    }) / len(actual_modified_files) if len(actual_modified_files) > 0 else 0


    def evaluate(self, search_results, actual_modified_files):
        # check if search results is a list of strings (i.e. file paths) instead of a list of SearchResult objects
        if isinstance(search_results[0], str):
            retrieved_files = search_results
        else:
            retrieved_files = [result.file_path for result in search_results]

        relevant = [1 if file in actual_modified_files else 0 for file in retrieved_files]

        evaluations = {}
        for metric in self.metrics:
            if metric == 'MAP':
                evaluations[metric] = self.calculate_average_precision(relevant)
            elif metric == 'MRR':
                evaluations[metric] = self.mean_reciprocal_rank(relevant)
            elif metric.startswith('P@'):
                k = int(metric.split('@')[1])
                evaluations[metric] = self.precision_at_k(relevant, k)
            elif metric.startswith('R@'):
                k = int(metric.split('@')[1])
                evaluations[metric] = self.calculate_recall(retrieved_files, actual_modified_files, relevant, k)

        return {k: round(v, 4) for k, v in evaluations.items()}



class ModelEvaluator:
    def __init__(self, model, eval_model, combined_df, seed=42):
        self.model = model
        self.eval_model = eval_model
        self.combined_df = combined_df
        self.seed = seed

    def sample_commits(self, n):
        if self.combined_df.commit_id.nunique() < n:
            raise ValueError(f'Not enough commits to sample. Required: {n}, available: {self.combined_df.commit_id.nunique()}')

        midpoint_date = np.median(self.combined_df['commit_date'])
        recent_df = self.combined_df[self.combined_df['commit_date'] > midpoint_date]

        return recent_df.drop_duplicates(subset='commit_id').sample(n=n, replace=False, random_state=self.seed)

    def evaluate_df(self, df, k, aggregation_strategy, rerankers, output_folder_path):
        results = []
        if output_folder_path:
            qry_file_path = os.path.join(output_folder_path, "output.qry")
            teIn_file_path = os.path.join(output_folder_path, "output.teIn")
            teOut_file_path = os.path.join(output_folder_path, "output.teOut")
            generate_qry_file(df, qry_file_path)

        query_res_dict = {}

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            cur_query = row['commit_message']
            search_results = self.model.pipeline(cur_query, row['commit_date'], ranking_depth=k, aggregation_method=aggregation_strategy)
            for reranker in rerankers:
                if isinstance(reranker, CodeReranker):
                    search_results = reranker.rerank_pipeline(cur_query, search_results, row['commit_id'])
                else:
                    search_results = reranker.rerank_pipeline(cur_query, search_results)

            query_res_dict[row['commit_id']] = search_results

            if 'actual_modified_files' in df.columns:
                actual_modified_files = row['actual_modified_files']
            else:
                actual_modified_files = self.combined_df[self.combined_df['commit_id'] == row['commit_id']]['file_path'].tolist()
            evaluation = self.eval_model.evaluate(search_results, actual_modified_files)
            results.append(evaluation)

        if output_folder_path:
            generate_teIn_file(query_res_dict, teIn_file_path)
            evaluate_results(teIn_file_path, qry_file_path, teOut_file_path, self.eval_model)

        return results

    def evaluate_sampling(self, n=100, k=1000, output_folder_path=None, overwrite_eval=False, aggregation_strategy=None, rerankers=None, gold_df=None, output_path=None): #, repo_path=None):
        # if repo_path is None:
        #     print("Repo path not provided, using current working directory")
            # repo_path = os.getcwd()
        if rerankers is None:
            rerankers = []


        if output_folder_path is None:
            print("WARNING: Output file path not provided, not writing results to file")
            # output_file_path = os.path.join(repo_path, f'{self.model.__class__.__name__}_results.txt')

        # output_file_path = os.path.join(repo_path, output_file)
        model_name = self.model.__class__.__name__

        if not overwrite_eval and output_folder_path:
            if os.path.exists(output_folder_path):
                print(f'Output file {output_folder_path} already exists - not writing to file, set overwrite_eval flag to True for that...')
                output_folder_path=None
            else:
                os.makedirs(output_folder_path, exist_ok=True)

        if gold_df is None:
            sampled_commits = self.sample_commits(n)
            results = self.evaluate_df(sampled_commits, k, aggregation_strategy, rerankers, output_folder_path)
        else:
            print(f'Found gold_df, evaluating on {len(gold_df)} commits')
            print(gold_df.info())
            results = self.evaluate_df(gold_df, k, aggregation_strategy, rerankers, output_folder_path)

        avg_scores = {metric: round(np.mean([result[metric] for result in results]), 4) for metric in results[0]}

        if output_folder_path:
            # with open(output_file_path, "w") as file:
            #     file.write(f"Model Name: {model_name}\n")
            #     # write name of each reranker
            #     if len(rerankers) > 0:
            #         file.write("Rerankers:\n")
            #         for reranker in rerankers:
            #             reranker_model_name = reranker.model.config.name_or_path
            #             # replace / with _
            #             reranker_model_name = reranker_model_name.replace('/', '_')
            #             file.write(f"{reranker.__class__.__name__} ({reranker_model_name}) @ {reranker.rerank_depth}\n")


            #     file.write(f"Sample Size: {n}\n")
            #     file.write("Evaluation Metrics:\n")
            #     for key, value in avg_scores.items():
            #         file.write(f"{key}: {value}\n")

            print(f'Evaluation results written to {output_folder_path}')

        return avg_scores