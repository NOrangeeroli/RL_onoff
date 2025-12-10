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

# Parse YAML to check backend type and tp_size
# Using Python to parse YAML (more reliable than bash)
BACKEND_INFO=$(python3 <<EOF
import yaml
import sys
import subprocess

try:
    with open("$CONFIG_PATH", 'r') as f:
        config = yaml.safe_load(f)
    
    # Get 01_dist section
    dist_config = config.get('01_dist', {})
    backend_config = dist_config.get('backend', {})
    backend_specific = backend_config.get('backend_specific', {})
    
    backend_type = backend_config.get('backend_type', 'huggingface')
    tp_size = backend_specific.get('tp_size') or backend_specific.get('num_process')  # Backward compatibility
    
    # Calculate number of processes if tp_size is set
    num_processes = None
    if backend_type == "huggingface" and tp_size is not None and tp_size > 1:
        # Get number of GPUs
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                num_gpus = len([line for line in result.stdout.strip().split('\n') if line.strip()])
                if num_gpus > 0 and num_gpus % tp_size == 0:
                    num_processes = num_gpus // tp_size
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
    
    print(f"{backend_type}|{tp_size or 1}|{num_processes or 1}")
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
TP_SIZE=$(echo "$BACKEND_INFO" | cut -d'|' -f2)
NUM_PROCESSES=$(echo "$BACKEND_INFO" | cut -d'|' -f3)

echo "Detected backend: $BACKEND_TYPE"
echo "Detected tp_size: $TP_SIZE"
echo "Calculated num_processes: $NUM_PROCESSES"

# Determine how to run
if [ "$BACKEND_TYPE" = "huggingface" ] && [ "$NUM_PROCESSES" -gt 1 ]; then
    echo "Using accelerate launch (HuggingFace backend with tp_size=$TP_SIZE, num_processes=$NUM_PROCESSES)"
    # Pass through all arguments except the first one (config path) if it was provided
    if [ "$1" = "$CONFIG_PATH" ] && [ "$1" != "$SCRIPT_DIR/experiment_config.yaml" ]; then
        # Config path was provided as first argument, shift it
        shift
    fi
    accelerate launch --num_processes="$NUM_PROCESSES" "$SCRIPT_DIR/01-dist.py" --config "$CONFIG_PATH" "$@"
else
    echo "Using regular python"
    # Pass through all arguments
    if [ "$1" = "$CONFIG_PATH" ] && [ "$1" != "$SCRIPT_DIR/experiment_config.yaml" ]; then
        # Config path was provided as first argument, shift it
        shift
    fi
    python3 "$SCRIPT_DIR/01-dist.py" --config "$CONFIG_PATH" "$@"
fi

