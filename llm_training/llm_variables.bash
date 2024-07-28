#!/usr/bin/env bash

# Important directories

# The main directory you want to work in.
if [ "xBENCH_DIR" = x ]; then
    echo "BENCH_DIR must be set to `llm_training` folder to execute the benchmark"
    exit 1
fi

echo "$BENCH_DIR is set as BENCH_DIR"

# Input data
VOCAB_FILE="$BENCH_DIR"/aux/tokenizers/gpt2-vocab.json
MERGE_FILE="$BENCH_DIR"/aux/tokenizers/gpt2-merges.txt

# Path to a singular, preprocessed dataset.
LLM_DATA_PATH="$BENCH_DIR"/llm_data/oscar_text_document

# Output data
# The main directory you want to store output in.
BENCH_OUTPUT_DIR="$BENCH_DIR"/output

export PYTHONPATH="$BENCH_DIR/Megatron-LM":$PYTHONPATH

# Check whether variables were set.
[ "x$BENCH_DIR" = x ] \
    && echo 'Please set `BENCH_DIR` in `llm_variables.bash.' && return 1
[ "x$BENCH_OUTPUT_DIR" = x ] \
    && echo 'Please set `ROOT_OUTPUT_DIR` in `llm_variables.bash.' && return 1
[ "x$VOCAB_FILE" = x ] \
    && echo 'Please set `VOCAB_FILE` in `llm_variables.bash.' && return 1
[ "x$MERGE_FILE" = x ] \
    && echo 'Please set `MERGE_FILE` in `llm_variables.bash.' && return 1
[ "x$LLM_DATA_PATH" = x ] \
    && echo 'Please set `LLM_DATA_PATH` in `llm_variables.bash.' && return 1

:
