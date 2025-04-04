#!/bin/bash
TIMEOUT=$(command -v timeout || command -v gtimeout)
if [ -z "$TIMEOUT" ]; then
  echo "Error: timeout command not found. Please install coreutils or gtimeout."
  exit 1
fi

# Function to process a list of token IDs for a specific split
process_token_ids() {
  local model=$1
  local split=$2
  local threshold=$3
  shift 3
  local token_ids=("$@")

  for token_id in "${token_ids[@]}"; do
    echo "Processing ${split}:${token_id}"
    $TIMEOUT 180m ./experiments/circuits/extract.sh $model $split $token_id $threshold
  done
}

# Empty defaults
TRAIN_TOKEN_IDS=()
VAL_TOKEN_IDS=()

# DIRNAME="toy-v0"
# TRAIN_TOKEN_IDS=(7010 300553 393485 512822 780099 872699)
# VAL_TOKEN_IDS=(15 1039 15524 69324 75324 85424)

# DIRNAME="comparisons-random"
# DIRNAME="comparisons-zero"
# DIRNAME="comparisons-cluster"
# VAL_TOKEN_IDS=(1282 7554 8834 9218) # 3 char sequences
# VAL_TOKEN_IDS=(6159) # 16 char sequence

DIRNAME="toy-local"
TRAIN_TOKEN_IDS=()
# VAL_TOKEN_IDS=(15 1039 2063 3087 4111 5135 6159 7183)
# VAL_TOKEN_IDS=(102431 103455 104479 105503 106527 107551 108575 109599 110623 111647 112671 113695)

# Process training data
process_token_ids "${DIRNAME}" "train" 0.15 "${TRAIN_TOKEN_IDS[@]}"

# Process validation data
process_token_ids "${DIRNAME}" "val" 0.25 "${VAL_TOKEN_IDS[@]}"