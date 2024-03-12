class Reranker:
    def __init__(self, parameters):
        self.parameters = parameters

        self.device = torch.device("cuda" if torch.cuda.is_available() and parameters['use_gpu'] else "cpu")
        print(f'Using device: {self.device}')
        # print GPU info
        if torch.cuda.is_available() and parameters['use_gpu']:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f'GPU Device Count: {torch.cuda.device_count()}')
            print(f"GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        self.aggregation_strategy = parameters['aggregation_strategy'] # how to aggregate the scores of the psg_cnt contributing_results
        self.batch_size = parameters['batch_size'] # batch size for reranking efficiently
        self.rerank_depth = parameters['rerank_depth']

    def rerank(self, query, aggregated_results: List[AggregatedSearchResult]):
        raise NotImplementedError

    def aggregate_scores(self, passage_scores):
        """
        Aggregate passage scores based on the specified strategy.
        """
        if len(passage_scores) == 0:
            return 0.0

        if self.aggregation_strategy == 'firstp':
            return passage_scores[0]
        if self.aggregation_strategy == 'maxp':
            return max(passage_scores)
        if self.aggregation_strategy == 'avgp':
            return sum(passage_scores) / len(passage_scores)
        if self.aggregation_strategy == 'sump':
            return sum(passage_scores)
        # else:
        raise ValueError(f"Invalid score aggregation method: {self.aggregation_strategy}")

    def get_scores(self, dataloader, model):
        scores = []
        with torch.no_grad():
            for batch in dataloader:
                # Unpack the batch and move it to GPU
                b_input_ids, b_attention_mask = batch
                b_input_ids = b_input_ids.to(self.device)
                b_attention_mask = b_attention_mask.to(self.device)

                # Get scores from the model
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask)
                scores.extend(outputs.logits.detach().cpu().numpy().squeeze(-1))
        return scores

class CodeReranker_R(Reranker):
    def __init__(self, parameters, combined_df):
        super().__init__(parameters)

        # specific to CodeReranker type
        self.combined_df = combined_df

        self.model_name = parameters['model_name']
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1, problem_type='regression')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.max_seq_length = self.tokenizer.model_max_length # max sequence length for the model


        self.psg_len = parameters['psg_len']
        self.psg_cnt = parameters['psg_cnt'] # how many contributing_results to use per file for reranking
        self.psg_stride = parameters.get('psg_stride', self.psg_len)
        print(f"Initialized Patch Code Reranker with parameters: {parameters}")


    def rerank(self, query, aggregated_results: List[AggregatedSearchResult], train_commit_id):
        """
        Rerank the BM25 aggregated search results using BERT model scores.

        query: The issue query string.
        aggregated_results: A list of AggregatedSearchResult objects from BM25 search.
        """
        self.model.eval()

        query_passage_pairs, per_result_contribution = self.split_into_query_passage_pairs(query, aggregated_results, train_commit_id)

        if not query_passage_pairs:
            print('WARNING: No query passage pairs to rerank, returning original results from previous stage')
            print(query, aggregated_results, self.psg_cnt)
            return aggregated_results

        # tokenize the query passage pairs
        encoded_pairs = [self.tokenizer.encode_plus([query, passage], max_length=self.max_seq_length, truncation=True, padding='max_length', return_tensors='pt', add_special_tokens=True) for query, passage in query_passage_pairs]

        # create tensors for the input ids, attention masks
        input_ids = torch.stack([encoded_pair['input_ids'].squeeze() for encoded_pair in encoded_pairs], dim=0) # type: ignore
        attention_masks = torch.stack([encoded_pair['attention_mask'].squeeze() for encoded_pair in encoded_pairs], dim=0) # type: ignore

        # Create a dataloader for feeding the data to the model
        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False) # shuffle=False very important for reconstructing the results back into the original order

        scores = self.get_scores(dataloader, self.model)

        score_index = 0

        # Now assign the scores to the aggregated results by mapping the scores to the contributing results
        for i, agg_result in enumerate(aggregated_results):
            # Each aggregated result gets a slice of the scores equal to the number of contributing results it has which should be min(psg_cnt, len(contributing_results))
            assert score_index < len(scores), f'score_index {score_index} is greater than or equal to scores length {len(scores)}'
            end_index = score_index + per_result_contribution[i] # only use psg_cnt contributing_results
            cur_passage_scores = scores[score_index:end_index]
            score_index = end_index

            # Aggregate the scores for the current aggregated result
            agg_score = self.aggregate_scores(cur_passage_scores)
            agg_result.score = agg_score  # Assign the aggregated score

        assert score_index == len(scores), f'score_index {score_index} does not equal scores length {len(scores)}, indices probably not working correctly'

        # Sort by the new aggregated score
        aggregated_results.sort(key=lambda res: res.score, reverse=True)

        return aggregated_results


    def split_into_query_passage_pairs(self, query, aggregated_results, train_commit_id):
        # Flatten the list of results into a list of (query, passage) pairs but only keep max psg_cnt passages per file
        def full_tokenize(s):
            return self.tokenizer.encode_plus(s, max_length=None, truncation=False, return_tensors='pt', add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)['input_ids'].squeeze().tolist()

        query_passage_pairs = []
        per_result_contribution = []

        if self.combined_df is not None:
            combined_df = self.combined_df

        for agg_result in aggregated_results:
            file_path = agg_result.file_path

            # get most recent version of the file
            file_content = get_file_at_commit_from_git(file_path, train_commit_id)

            # file_content = combined_df[(combined_df['commit_id'] == commit_id) & (combined_df['file_path'] == file_path)]['previous_file_content'].values[0]

            # now need to split this file content into psg_cnt passages
            # first tokenize the file content

            # warning these asserts are useless since we are using NaNs
            # assert file_content is not None, f'file_content is None for commit_id: {commit_id}, file_path: {file_path}'
            # assert file_path is not None, f'file_path is None for commit_id: {commit_id}'
            assert query is not None, 'query is None'

            # query_tokens = full_tokenize(query)
            path_tokens = full_tokenize(file_path)

            if pd.isna(file_content):
                # if file_content is NaN, then we can just set file_content to empty string
                print(f'WARNING: file_content is NaN for commit_id: {train}, file_path: {file_path}, setting file_content to empty string')
                file_content = ''

            file_tokens = full_tokenize(file_content)


            # now split the file content into psg_cnt passages
            cur_result_passages = []
            # get the input ids
            # input_ids = file_content['input_ids'].squeeze()
            # get the number of tokens in the file content
            total_tokens = len(file_tokens)

            for cur_start in range(0, total_tokens, self.psg_stride):
                cur_passage = []
                # add query tokens and path tokens
                # cur_passage.extend(query_tokens)
                cur_passage.extend(path_tokens)

                # add the file tokens
                cur_passage.extend(file_tokens[cur_start:cur_start+self.psg_len])

                # now convert cur_passage into a string
                cur_passage_decoded = self.tokenizer.decode(cur_passage)

                # add the cur_passage to cur_result_passages
                cur_result_passages.append(cur_passage_decoded)

                # if len(cur_result_passages) == self.psg_cnt:
                #     break

            # now add the query, passage pairs to query_passage_pairs
            per_result_contribution.append(len(cur_result_passages))
            query_passage_pairs.extend((query, passage) for passage in cur_result_passages)
        return query_passage_pairs, per_result_contribution

    def rerank_pipeline(self, query, aggregated_results, train_commit_id):
        if len(aggregated_results) == 0:
            return aggregated_results
        top_results = aggregated_results[:self.rerank_depth]
        bottom_results = aggregated_results[self.rerank_depth:]
        reranked_results = self.rerank(query, top_results, train_commit_id)
        min_top_score = reranked_results[-1].score
        # now adjust the scores of bottom_results
        for i, result in enumerate(bottom_results):
            result.score = min_top_score - i - 1
        # combine the results
        reranked_results.extend(bottom_results)
        assert(len(reranked_results) == len(aggregated_results))
        return reranked_results




# Patch code reranker
class PatchResult:
    def __init__(self, file_path, passage, score):
        self.file_path = file_path
        self.score = score
        self.passage = passage

    def __repr__(self):
        class_name = self.__class__.__name__
        return f'{class_name}(file_path={self.file_path}, passage={self.passage}, score={self.score})'

class PatchCodeReranker(Reranker):
    def __init__(self, parameters, split_strategy: SplitStrategy):
        super().__init__(parameters)

        # specific to CodeReranker type

        self.model_name = parameters['model_name']
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1, problem_type='regression')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.max_seq_length = self.tokenizer.model_max_length # max sequence length for the model


        self.psg_len = parameters['psg_len']
        self.psg_cnt = parameters['psg_cnt'] # how many contributing_results to use per file for reranking
        self.psg_stride = parameters.get('psg_stride', self.psg_len)
        print(f"Initialized Patch Code Reranker with parameters: {parameters}")

        # Passage/patch splitting
        self.split_strategy = split_strategy
        self.passage_splitter = PassageSplitter(split_strategy)


    def rerank(self, query, aggregated_results: List[AggregatedSearchResult], train_commit_id):
        """
        Rerank a aggregated search result list by splitting into patches and getting model scores for each patch.

        query: The issue query string.
        aggregated_results: A list of AggregatedSearchResult objects from BM25 search.

        Returns:
        - List[PatchResult]         [IMP: DOES NOT return file_list]
        """
        self.model.eval()
        query_passage_pairs = self.split_into_query_passage_pairs(query, aggregated_results, train_commit_id)

        if not query_passage_pairs:
            print('WARNING: No query passage pairs to rerank, returning []')
            return []

        # tokenize the query passage pairs
        encoded_pairs = [self.tokenizer.encode_plus([query, obj.passage], max_length=self.max_seq_length, truncation=True, padding='max_length', return_tensors='pt', add_special_tokens=True) for query, file_path, obj in query_passage_pairs]

        # create tensors for the input ids, attention masks
        input_ids = torch.stack([encoded_pair['input_ids'].squeeze() for encoded_pair in encoded_pairs], dim=0) # type: ignore
        attention_masks = torch.stack([encoded_pair['attention_mask'].squeeze() for encoded_pair in encoded_pairs], dim=0) # type: ignore

        # Create a dataloader for feeding the data to the model
        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False) # shuffle=False very important for reconstructing the results back into the original order

        scores = self.get_scores(dataloader, self.model)

        # convert the scores to PatchResult objects
        patch_results = [PatchResult(file_path, obj, score) for (query, file_path, obj), score in zip(query_passage_pairs, scores)]

        # sort patch_results by the scores
        sorted_patch_results = sorted(patch_results, key=lambda res: res.score, reverse=True)

        return sorted_patch_results

    def split_into_query_passage_pairs(self, query, aggregated_results, train_commit_id):
        query_passage_pairs = []
        for agg_result in aggregated_results:
            # get any file result
            most_recent_search_result = agg_result.contributing_results[0] # doesn't matter which version we take, we only care about file_path

            # get the file_path
            file_path = most_recent_search_result.file_path

            file_content = get_file_at_commit_from_git(file_path, train_commit_id)
            if not file_content:
                # useless file
                continue

            # warning these asserts are useless since we are using NaNs
            assert file_content is not None, f'file_content is None for commit_id: {train_commit_id}, file_path: {file_path}'
            assert file_path is not None, f'file_path is None for commit_id: {train_commit_id}'
            assert query is not None, 'query is None'

            if pd.isna(file_content):
                # if file_content is NaN, then we can just set file_content to empty string
                print(f'WARNING: file_content is NaN for commit_id: {train_commit_id}, file_path: {file_path}, setting file_content to empty string')
                file_content = ''

            cur_result_passages = self.passage_splitter.split_passages(file_content)

            query_passage_pairs.extend((query, file_path, obj) for obj in cur_result_passages)

        return query_passage_pairs

    def rerank_pipeline(self, query, aggregated_results, train_commit_id):
        if len(aggregated_results) == 0:
            return aggregated_results
        top_results = aggregated_results[:self.rerank_depth]
        # bottom_results = aggregated_results[self.rerank_depth:]
        reranked_results = self.rerank(query, top_results, train_commit_id)
        # min_top_score = reranked_results[-1].score
        # now adjust the scores of bottom_results
        # for i, result in enumerate(bottom_results):
            # result.score = min_top_score - i - 1
        # combine the results
        # reranked_results.extend(bottom_results)
        # assert(len(reranked_results) == len(aggregated_results))
        return reranked_results