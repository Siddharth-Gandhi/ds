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
NUM_THREADS=${4:-4}  # Number of threads (default to 4 if not provided)

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
        JSONL_DIR_NAME=$(get_dir_name "jsonl" "$CONTENT_OPTION" "$USE_TOKENIZER")
        INDEX_DIR_NAME=$(get_dir_name "index" "$CONTENT_OPTION" "$USE_TOKENIZER")

        JSONL_DIR="$REPO_PATH/$JSONL_DIR_NAME"
        INDEX_DIR="$REPO_PATH/$INDEX_DIR_NAME"

        # Check if the JSONL directory exists
        if [ ! -d "$JSONL_DIR" ]; then
            echo "Warning: JSONL directory $JSONL_DIR does not exist. Skipping $REPO_NAME."
            continue
        fi

        # Remove all files in the index directory
        rm -rf "$INDEX_DIR"

        # Create the directory if it doesn't exist
        mkdir -p "$INDEX_DIR"

        # Build the index with the specified number of threads
        python -m pyserini.index.lucene -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
         -threads "$NUM_THREADS" -input "$JSONL_DIR" -index "$INDEX_DIR" -storePositions -storeDocvectors -storeRaw -impact -pretokenized

        # Log the repo being processed
        echo "Processed $REPO_NAME"
    fi
done