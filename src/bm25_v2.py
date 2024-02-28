import json
import os
from collections import defaultdict

import numpy as np
from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher

from utils import (
    AggregatedCommitResult,
    AggregatedSearchResult,
    SearchResult,
    reverse_tokenize,
    tokenize,
)


class BM25Searcher:
    def __init__(self, index_path):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index at {index_path} does not exist!")
        self.searcher = LuceneSearcher(index_path)
        print(f"Loaded index at {index_path}")
        print(f'Index Stats: {IndexReader(index_path).stats()}')
        # self.ranking_depth = ranking_depth

    def search(self, query, query_date, ranking_depth):
        # TODO maybe change this to mean returning reranking_depths total results instead of being pruned by the query date
        tokenized_query = tokenize(query)
        try:
            hits = self.searcher.search(tokenized_query, ranking_depth)
        except Exception as e:
            print(e)
            print(f"Error searching for query of length {len(tokenized_query)}")
            return []
        unix_date = query_date
        filtered_hits = [
            SearchResult(hit.docid, json.loads(hit.raw)['file_path'], hit.score, int(json.loads(hit.raw)["commit_date"]), reverse_tokenize(json.loads(hit.raw)['contents']))
            for hit in hits if int(json.loads(hit.raw)["commit_date"]) < unix_date
        ]
        return filtered_hits

    def search_full(self, query, query_date, ranking_depth):
        filtered_hits = []
        step_size = ranking_depth  # Initial search window
        total_hits_retrieved = 0

        while len(filtered_hits) < ranking_depth and step_size > 0:
            current_hits = self.searcher.search(tokenize(query), total_hits_retrieved + step_size)
            if not current_hits:
                break  # No more results to retrieve

            # Filter hits by query date
            for hit in current_hits:
                loaded_obj = json.loads(hit.raw)
                if int(loaded_obj["commit_date"]) < query_date:
                    filtered_hits.append(
                        SearchResult(hit.docid, loaded_obj['file_path'], hit.score,
                                     int(loaded_obj["commit_date"]),
                                     reverse_tokenize(loaded_obj['contents']))
                    )
                if len(filtered_hits) == ranking_depth:
                    break  # We have enough results

            total_hits_retrieved += step_size
            step_size = ranking_depth - len(filtered_hits)  # Decrease step size to only get as many as needed

        return filtered_hits[:ranking_depth]  # Return up to ranking_depth results

    def aggregate_file_scores(self, search_results, aggregation_method='sump', sort_contributing_result_by_date=False):
        file_to_results = defaultdict(list)
        for result in search_results:
            file_to_results[result.file_path].append(result)

        aggregated_results = []
        for file_path, results in file_to_results.items():
            # aggregated_score = sum(result.score for result in results)
            if aggregation_method == 'sump':
                aggregated_score = sum(result.score for result in results)
            elif aggregation_method == 'maxp':
                aggregated_score = max(result.score for result in results)
            # elif aggregation_method == 'firstp':
            #     aggregated_score = results[0].score
            elif aggregation_method == 'avgp':
                aggregated_score = np.mean([result.score for result in results])
            else:
                raise ValueError(f"Unknown aggregation method {aggregation_method}")

            aggregated_results.append(AggregatedSearchResult(file_path, aggregated_score, results))


        aggregated_results.sort(key=lambda result: result.score, reverse=True)
        if sort_contributing_result_by_date: # by default, it is sorted by score
            # for each aggregated result, sort the contributing results by date
            for result in aggregated_results:
                result.contributing_results.sort(key=lambda result: result.commit_date, reverse=True)
        return aggregated_results

    def aggregate_commit_scores(self, search_results, aggregation_method='sump'):
        commit_to_results = defaultdict(list)
        for result in search_results:
            commit_to_results[result.commit_id].append(result)

        aggregated_results = []
        for commit_id, results in commit_to_results.items():
            # aggregated_score = sum(result.score for result in results)
            if aggregation_method == 'sump':
                aggregated_score = sum(result.score for result in results)
            elif aggregation_method == 'maxp':
                aggregated_score = max(result.score for result in results)
            # elif aggregation_method == 'firstp':
            #     aggregated_score = results[0].score
            elif aggregation_method == 'avgp':
                aggregated_score = np.mean([result.score for result in results])
            else:
                raise ValueError(f"Unknown aggregation method {aggregation_method}")

            aggregated_results.append(AggregatedCommitResult(commit_id, aggregated_score, results))

        # if sort_by == 'date':
        #     aggregated_results.sort(key=lambda result: result.commit_date, reverse=True)
        # else:
        aggregated_results.sort(key=lambda result: result.score, reverse=True)
        return aggregated_results

    def pipeline(self, query, query_date, ranking_depth, aggregation_method, aggregate_on='file', sort_contributing_result_by_date=False):
        search_results = self.search(query, query_date, ranking_depth)
        if aggregation_method is not None:
            if aggregate_on == 'file':
                aggregated_results = self.aggregate_file_scores(search_results, aggregation_method, sort_contributing_result_by_date)
            elif aggregate_on == 'commit':
                aggregated_results = self.aggregate_commit_scores(search_results, aggregation_method)
            else:
                raise ValueError(f"Unknown aggregation type {aggregate_on}")
            # aggregated_results = self.aggregate_file_scores(search_results, aggregation_method)
            return aggregated_results
        return search_results