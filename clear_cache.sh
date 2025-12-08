#!/bin/bash
# Script to clear Python cache files (__pycache__ directories and .pyc files)

echo "Clearing Python cache files..."

# Find and remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null

# Find and remove all .pyc files
find . -type f -name "*.pyc" -delete 2>/dev/null

# Find and remove all .pyo files (optimized bytecode)
find . -type f -name "*.pyo" -delete 2>/dev/null

# Find and remove all .pyd files (Python extension modules on Windows)
find . -type f -name "*.pyd" -delete 2>/dev/null

# Remove .DS_Store files (macOS)
find . -type f -name ".DS_Store" -delete 2>/dev/null

# Remove pytest cache
find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null

# Remove .mypy_cache
find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null

# Remove .ruff_cache
find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null

echo "Cache cleared successfully!"

