#!/bin/bash

BASE_DIR="/gws/ssde/j25a/ai4er/users/pn341/climate-rl"
ALGOS=("ddpg" "dpg" "td3" "reinforce" "trpo" "ppo" "sac" "avg" "tqc")

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
NC="\033[0m"    # No Colour

# Print header
printf "%-50s" "Run Directory"
for algo in "${ALGOS[@]}"; do
    printf "%15s" "$algo"
done
echo

# Loop over top-level run directories
for top_dir in "$BASE_DIR"/records/*/; do
    run_name=$(basename "$top_dir")
    printf "%-50s" "$run_name"
    for algo in "${ALGOS[@]}"; do
        if [[ "$run_name" == *infx10* ]]; then
            if [[ "$run_name" == *rce* ]]; then
                step_file="step_500.pth"
            elif [[ "$run_name" == *scbc* ]]; then
                step_file="step_200.pth"
            fi
        else
            if [[ "$run_name" == *rce* ]]; then
                step_file="step_10000.pth"
            elif [[ "$run_name" == *scbc* ]]; then
                step_file="step_60000.pth"
            fi
        fi
        step_file_count=$(find "$top_dir" -mindepth 1 -maxdepth 1 -type d -name "*_${algo}_*" -exec find {} -name "$step_file" \; | wc -l)
        if [[ "$step_file_count" -eq 1 || ( "$step_file_count" -eq 9 && "$run_name" == *x9* ) || ( "$step_file_count" -eq 10 && "$run_name" == *infx10* ) ]]; then
            printf "${GREEN}%15s${NC}" "$step_file_count"
        else
            printf "${RED}%15d${NC}" "$step_file_count"
        fi
    done
    echo
done
