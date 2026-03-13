#!/bin/sh

BASE_DIR="/gws/ssde/j25a/ai4er/users/pn341/climate-rl"

# Check if a command-line argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <tag_name>"
    exit 1
fi

# Set the tag name from the first argument
TAG_NAME=$1

# Remove the specified tag from each directory
rm -rf "$BASE_DIR/runs/$TAG_NAME"*
rm -rf "$BASE_DIR/videos/$TAG_NAME"*
rm -rf "$BASE_DIR/slurm"/*.err
rm -rf "$BASE_DIR/slurm"/*.out
rm -rf "$BASE_DIR/param_tune/tmp"/*.tmp
