import random
import re
from abc import ABC, abstractmethod


class SplitStrategy(ABC):
    @abstractmethod
    def split(self, file_content):
        pass

class LineSpanPassage:
        def __init__(self, si, ei, passage):
            self.start_line = si
            self.end_line = ei
            self.passage = passage

        def __repr__(self):
            return f'LineSpanPassage(start_line={self.start_line}, end_line={self.end_line}, passage=\n{self.passage})'

class DiffSplitStrategy(SplitStrategy):
    def __init__(self, context_lines=5):
        super().__init__()
        self.context_lines = context_lines

    @classmethod
    def extract_modified_lines(cls, diff):
        """
        Extracts line numbers of modified lines from a diff string.
        Args:
        - diff (str): The diff string in Linux diff format.
        Returns:
        - List[int]: A list of line numbers that were modified in the previous file state.
        """
        modified_lines = []

        # Regular expression to find all instances of line number indicators in the diff
        line_indicator_regex = re.compile(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@')

        for match in line_indicator_regex.finditer(diff): # type: ignore
            start_line = int(match.group(1))
            line_count = int(match.group(2))

            # Adding all affected lines by this change to the list
            # for i in range(line_count):
                # modified_lines.append(start_line + i)

            modified_lines.append((start_line, line_count))

        return modified_lines

    def split(self, file_content, diff, *args, **kwargs):
        modified_lines = self.extract_modified_lines(diff)
        lines = file_content.split('\n')
        passages = []
        for start_line, line_count in modified_lines:
            start_index = max(0, start_line - 1 - self.context_lines)
            end_index = min(len(lines), start_line - 1 + line_count + self.context_lines)
            section = lines[start_index:end_index]
            lsp = LineSpanPassage(start_index, end_index, '\n'.join(section))
            passages.append(lsp)
        return passages

class TokenizedPassage:
    def __init__(self, st, et, passage):
        self.start_token = st
        self.end_token = et
        self.passage = passage


    def __repr__(self):
        return f'TokenizedPassage(start_token_index={self.start_token}, end_token_index={self.end_token}, passage=\n{self.passage})'

class TokenizedSplitStrategy(SplitStrategy):

    def __init__(self, tokenizer, psg_len, psg_stride):
        super().__init__()
        self.tokenizer = tokenizer
        self.psg_len = psg_len
        self.psg_stride = psg_stride

    # def full_tokenize(self, s):
    #         return self.tokenizer.encode_plus(s, max_length=None, truncation=False, return_tensors='pt', add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)['input_ids'].squeeze().tolist()

    def full_tokenize(self, s):
        tokens = self.tokenizer.encode_plus(s, max_length=None, truncation=False, return_tensors='pt', add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)['input_ids'].squeeze().tolist()
        # Ensure tokens is always a list
        if not isinstance(tokens, list):
            tokens = [tokens]
        return tokens

    def split(self, file_content, *args, **kwargs):
        # Tokenize the entire file content
        file_tokens = self.full_tokenize(file_content)
        total_tokens = len(file_tokens)
        res = []
        for cur_start in range(0, total_tokens, self.psg_stride):
            # get tokens for current passage
            cur_psg = self.tokenizer.decode(file_tokens[cur_start:cur_start+self.psg_len])
            res.append(TokenizedPassage(cur_start, min(total_tokens-1, cur_start+self.psg_len-1), cur_psg))

        return res

class TokenizedLineSplitStrategy(SplitStrategy):
    def __init__(self, tokenizer, psg_len, psg_stride):
        super().__init__()
        self.tokenizer = tokenizer
        self.psg_len = psg_len
        self.psg_stride = psg_stride

    def full_tokenize(self, s):
        tokens = self.tokenizer.encode_plus(s, max_length=None, truncation=False, return_tensors='pt', add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)['input_ids'].squeeze().tolist()
        # Ensure tokens is always a list
        if not isinstance(tokens, list):
            tokens = [tokens]
        return tokens

    def split(self, file_content, *args, **kwargs):
        lines = file_content.split('\n')
        passages = []
        cur_lines = []
        cur_tokens = 0
        start_line = 0

        for i, line in enumerate(lines):
            # Tokenize the current line to count tokens
            line_tokens = len(self.full_tokenize(line))

            # Check if adding this line would exceed the token limit for a passage
            if cur_tokens + line_tokens > self.psg_len and cur_lines:
                # Add the current passage without exceeding the token limit
                passages.append(LineSpanPassage(start_line, i, '\n'.join(cur_lines)))
                # Reset for the next passage
                cur_lines = [line] if i >= start_line else []
                cur_tokens = line_tokens if i >= start_line else 0
                start_line = i + 1
            else:
                # Add line to the current passage
                cur_lines.append(line)
                cur_tokens += line_tokens

        # Add the last passage if there are any remaining lines
        if cur_lines:
            passages.append(LineSpanPassage(start_line, len(lines), '\n'.join(cur_lines)))

        return passages


class PassageSplitter:
    def __init__(self, strategy: SplitStrategy):
        self.strategy = strategy

    def split_passages(self, file_content, *args, **kwargs):
        return self.strategy.split(file_content=file_content, *args, **kwargs)