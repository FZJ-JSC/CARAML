#!/usr/bin/env bash

# Important directories

# The main directory you want to work in.
if [ "xROOT_DIR" = x ]; then
    echo "ROOT_DIR must be set to rootdir path of this folder to execute the benchmark"
    exit 1
fi

echo "$ROOT_DIR is set as ROOT_DIR"

# Input data
VOCAB_FILE="$ROOT_DIR"/aux/tokenizers/gpt2-vocab.json
MERGE_FILE="$ROOT_DIR"/aux/tokenizers/gpt2-merges.txt

# Path to a singular, preprocessed dataset.
LLM_DATA_PATH="$ROOT_DIR"/llm_data/oscar_text_document

# Output data
# The main directory you want to store output in.
ROOT_OUTPUT_DIR="$ROOT_DIR"/output

export PYTHONPATH="$ROOT_DIR/Megatron-LM":$PYTHONPATH

# Check whether variables were set.
[ "x$ROOT_DIR" = x ] \
    && echo 'Please set `ROOT_DIR` in `llm_variables.bash.' && return 1
[ "x$ROOT_OUTPUT_DIR" = x ] \
    && echo 'Please set `ROOT_OUTPUT_DIR` in `llm_variables.bash.' && return 1
[ "x$VOCAB_FILE" = x ] \
    && echo 'Please set `VOCAB_FILE` in `llm_variables.bash.' && return 1
[ "x$MERGE_FILE" = x ] \
    && echo 'Please set `MERGE_FILE` in `llm_variables.bash.' && return 1
[ "x$LLM_DATA_PATH" = x ] \
    && echo 'Please set `LLM_DATA_PATH` in `llm_variables.bash.' && return 1

:
