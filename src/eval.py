import os

import numpy as np
from tqdm import tqdm


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
            elif metric.startswith('Recall@'):
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

    def evaluate_df(self, df, k=1000, aggregation_strategy=None, rerankers=None):
        results = []
        if rerankers is None:
            rerankers = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            cur_query = row['commit_message']
            search_results = self.model.pipeline(cur_query, row['commit_date'], ranking_depth=k, aggregation_method=aggregation_strategy)
            for reranker in rerankers:
                search_results = reranker.rerank_pipeline(cur_query, search_results)
            evaluation = self.eval_model.evaluate(search_results,
                                                   self.combined_df[self.combined_df['commit_id'] == row['commit_id']]['file_path'].tolist())
            results.append(evaluation)
        return results

    def evaluate_sampling(self, n=100, k=1000, output_file_path=None, skip_existing=False, aggregation_strategy=None, rerankers=None, repo_path=None):
        if repo_path is None:
            print("Repo path not provided, using current working directory")
            repo_path = os.getcwd()


        if output_file_path is None:
            print("WARNING: Output file path not provided, using default")
            output_file_path = os.path.join(repo_path, f'{self.model.__class__.__name__}_results.txt')

        # output_file_path = os.path.join(repo_path, output_file)
        model_name = self.model.__class__.__name__

        if skip_existing and os.path.exists(output_file_path):
            print(f'Output file {output_file_path} already exists, skipping...')
            return

        sampled_commits = self.sample_commits(n)
        results = self.evaluate_df(sampled_commits, k, aggregation_strategy, rerankers)

        avg_scores = {metric: round(np.mean([result[metric] for result in results]), 4) for metric in results[0]}

        with open(output_file_path, "w") as file:
            file.write(f"Model Name: {model_name}\n")
            file.write(f"Sample Size: {n}\n")
            file.write("Evaluation Metrics:\n")
            for key, value in avg_scores.items():
                file.write(f"{key}: {value}\n")

        print(f'Evaluation results written to {output_file_path}')

        return avg_scores