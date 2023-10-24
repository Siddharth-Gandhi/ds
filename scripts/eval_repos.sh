#!/bin/bash

# Function to get the dynamic directory name
get_dir_name() {
    local prefix=$1  # either 'jsonl' or 'index'
    local content_option=$2
    local use_tokenizer=$3

    local dir_name="${prefix}_${content_option}"

    if [ "$use_tokenizer" = true ]; then
        dir_name+="_tokenized"
    fi

    echo "$dir_name"
}

# Input arguments
DATA_PATH=$1
CONTENT_OPTION=$2  # commit, code, or both
USE_TOKENIZER=$3   # true or false
NUM_SAMPLES=${4:-100}  # Number of samples (default to 100 if not provided)
TOP_K=${5:-1000}  # Number of samples (default to 1000 if not provided)

SCRIPT_FILE="src/bm25.py"

# Check if the given data path directory exists
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Directory $DATA_PATH does not exist."
    exit 1
fi

# Loop over each repo in the data path
for REPO_PATH in "$DATA_PATH"/*; do
    if [ -d "$REPO_PATH" ]; then
        echo "Processing $REPO_PATH"
        REPO_NAME=$(basename "$REPO_PATH")

        # Construct directory names
        INDEX_DIR_NAME=$(get_dir_name "index" "$CONTENT_OPTION" "$USE_TOKENIZER")
        INDEX_DIR="$REPO_PATH/$INDEX_DIR_NAME"

        if [ ! -d "$INDEX_DIR" ]; then
            echo "Warning: JSONL directory $INDEX_DIR does not exist. Skipping $REPO_NAME."
            continue
        fi

        python "$SCRIPT_FILE" "$REPO_PATH" "$INDEX_DIR" --n "$NUM_SAMPLES" --k "$TOP_K"
    fi
done