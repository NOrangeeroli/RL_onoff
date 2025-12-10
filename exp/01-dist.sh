#!/bin/bash
# Script to run 01-dist.py with automatic backend detection
# Uses accelerate launch for HuggingFace backend with num_process > 1
# Otherwise uses regular python

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default config path
CONFIG_PATH="${1:-$SCRIPT_DIR/experiment_config.yaml}"

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Parse YAML to check backend type and num_process
# Using Python to parse YAML (more reliable than bash)
BACKEND_INFO=$(python3 <<EOF
import yaml
import sys

try:
    with open("$CONFIG_PATH", 'r') as f:
        config = yaml.safe_load(f)
    
    # Get 01_dist section
    dist_config = config.get('01_dist', {})
    backend_config = dist_config.get('backend', {})
    
    backend_type = backend_config.get('backend_type', 'huggingface')
    num_process = backend_config.get('backend_specific', {}).get('num_process', 1)
    
    print(f"{backend_type}|{num_process}")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
EOF
)

if [ $? -ne 0 ]; then
    echo "Error: Failed to parse config file"
    exit 1
fi

# Split the result
BACKEND_TYPE=$(echo "$BACKEND_INFO" | cut -d'|' -f1)
NUM_PROCESS=$(echo "$BACKEND_INFO" | cut -d'|' -f2)

echo "Detected backend: $BACKEND_TYPE"
echo "Detected num_process: $NUM_PROCESS"

# Determine how to run
if [ "$BACKEND_TYPE" = "huggingface" ] && [ "$NUM_PROCESS" -gt 1 ]; then
    echo "Using accelerate launch (HuggingFace backend with num_process=$NUM_PROCESS)"
    # Pass through all arguments except the first one (config path) if it was provided
    if [ "$1" = "$CONFIG_PATH" ] && [ "$1" != "$SCRIPT_DIR/experiment_config.yaml" ]; then
        # Config path was provided as first argument, shift it
        shift
    fi
    accelerate launch --num_processes="$NUM_PROCESS" "$SCRIPT_DIR/01-dist.py" --config "$CONFIG_PATH" "$@"
else
    echo "Using regular python"
    # Pass through all arguments
    if [ "$1" = "$CONFIG_PATH" ] && [ "$1" != "$SCRIPT_DIR/experiment_config.yaml" ]; then
        # Config path was provided as first argument, shift it
        shift
    fi
    python3 "$SCRIPT_DIR/01-dist.py" --config "$CONFIG_PATH" "$@"
fi

