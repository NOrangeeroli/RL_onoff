#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: $0 <pytest_command_and_args>"
    exit 1
fi

CMD="$*"
OUTPUT_FILE="test_output.txt"

{
    echo "Test Execution Report"
    echo "===================="
    echo "Command: $CMD"
    echo "Date: $(date)"
    echo "Working Directory: $(pwd)"
    echo ""
    echo "Output:"
    echo "-------"
    eval "$CMD"
} > "$OUTPUT_FILE" 2>&1

echo "Output saved to: $OUTPUT_FILE"
cat "$OUTPUT_FILE"