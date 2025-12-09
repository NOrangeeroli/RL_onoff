# Experimental Scripts

This directory contains numbered experimental scripts for various research tasks.

## Structure

Each script follows the naming convention: `NN-description.py` where:
- `NN` is a two-digit number (00, 01, 02, ...)
- `description` is a short description of what the script does

Each script has its own output directory: `exp/NN-description/output/`

## Scripts

### 00-sample.py

Samples responses from models given task configurations and datasets.

**Usage:**
```bash
python exp/00-sample.py \
    --task-config rl_onoff/tasks/configs/math_default.json \
    --dataset data/math/test.parquet \
    --backend-type huggingface \
    --model-name gpt2 \
    --max-new-tokens 256 \
    --temperature 1.0 \
    --num-samples 1
```

**Output:**
- `exp/00-sample/output/responses.json` - All responses with metadata
- `exp/00-sample/output/summary.json` - Summary statistics

**Features:**
- Loads task configuration (template, format, reward)
- Formats questions using task's chat template
- Samples responses from specified model
- Extracts answers from responses (if references available)
- Saves results in structured JSON format

