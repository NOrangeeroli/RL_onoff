# Experimental Scripts

This directory contains numbered experimental scripts for various research tasks. All scripts share a unified configuration file (`experiment_config.yaml`) where each script has its own configuration section.

## Structure

Each script follows the naming convention: `NN-description.py` where:
- `NN` is a two-digit number (00, 01, 02, ...)
- `description` is a short description of what the script does

Each script has its own output directory: `exp/NN-description/output/`

## Configuration

All scripts use a unified configuration file: `exp/experiment_config.yaml`

Each script has its own section in the config file:
- `00_sample`: Configuration for `00-sample.py`
- `01_dist`: Configuration for `01-dist.py`
- `02_visualize`: Configuration for `02-visualize.py`

See `experiment_config.yaml` for detailed configuration options.

## Scripts

### 00-sample.py

Samples responses from models given task configurations and datasets.

**Purpose:**
- Loads a dataset (configurable via `experiment_config.yaml`)
- Uses a backend to generate responses (configurable via `experiment_config.yaml`)
- Uses sampling config for generation parameters (configurable via `experiment_config.yaml`)
- Uses a task to format questions, generate responses, and evaluate rewards
- Saves results to output directory

**Usage:**
```bash
# Using default config (experiment_config.yaml in same directory)
python exp/00-sample.py

# Using custom config file
python exp/00-sample.py --config path/to/config.yaml
```

**Configuration (in `experiment_config.yaml` under `00_sample`):**
- `task`: Task configuration (template_type, reward_type, format_type) OR `task_config`: path to task config file
- `dataset`: Dataset configuration (name, split, num_examples)
- `backend`: Backend configuration (backend_type, model_name, backend_specific)
- `sampling`: Sampling configuration (max_length, temperature, top_k, top_p, do_sample, num_samples, batch_size, seed, stop_strings) OR `sampling_config`: path to sampling config file
- `output`: Output directory path

**Output:**
- `exp/00-sample/output/results.json` - All questions, responses, and rewards
- `exp/00-sample/output/statistics.json` - Overall statistics (accuracy, sampling length, etc.)

**Features:**
- Supports all datasets available in `rl_onoff.utils.dataset` (aime2025, amc23, gsm8k_level1, math)
- Flexible task configuration (can use config file or direct configuration)
- Flexible sampling configuration (can use config file or direct configuration)
- Progress bars for long-running operations
- Automatic batch processing

**Example Output Structure:**
```json
{
  "results": [
    {
      "question": "...",
      "reference": "...",
      "samples": ["...", "..."],
      "rewards": [1.0, 0.0],
      "extracted_answers": ["...", "..."]
    }
  ],
  "statistics": {
    "accuracy": 0.75,
    "avg_sampling_length": 150.5,
    "num_questions": 10,
    "num_samples_per_question": 8
  }
}
```

---

### 01-dist.py

Extracts token-wise distributions from sampled responses.

**Purpose:**
- Loads results from `00-sample.py`
- For each prompt/sample pair, extracts token-wise distributions (logits or probabilities)
- Splits the sample string according to tokenization
- Calculates per-token entropy
- Saves results in a compact format (NPZ for distributions, JSON for metadata)

**Usage:**
```bash
# Using default config (experiment_config.yaml in same directory)
python exp/01-dist.py

# Using custom config file
python exp/01-dist.py --config path/to/config.yaml
```

**Configuration (in `experiment_config.yaml` under `01_dist`):**
- `input`: Input configuration
  - `results_file`: Path to results.json from `00-sample.py`
- `backend`: Backend configuration (must match the backend used in `00-sample.py`)
- `distribution`: Distribution extraction configuration
  - `use_logits`: If true, extract logits; if false, extract probabilities
  - `temperature`: Temperature for probability normalization (if use_logits=true)
- `output`: Output directory path

**Output:**
- `exp/01-dist/output/distributions.json` - Metadata and structure (without large arrays)
- `exp/01-dist/output/distributions.npz` - Compressed NumPy archive with all distribution arrays

**Features:**
- Efficient storage: Large distributions stored in NPZ format, metadata in JSON
- Per-token entropy calculation
- Token string extraction (cleaned of special characters like 'Ġ')
- Progress bars for long-running operations
- Automatic batch processing

**Loading Results:**
```python
from rl_onoff.distributions.extractor import load_distributions

# Load distributions (lazy loading - NPZ only loaded when needed)
data = load_distributions("exp/01-dist/output/distributions.json")

# Access metadata
print(data["metadata"])
print(data["prompts"])
print(data["samples"])

# Access distributions (loads from NPZ)
for prompt_idx, prompt_data in enumerate(data["results"]):
    for sample_idx, sample_data in enumerate(prompt_data["samples"]):
        distributions = sample_data["distributions"]  # (num_tokens, vocab_size)
        token_strings = sample_data["token_strings"]  # List[str]
        entropies = sample_data["entropies"]  # List[float]
```

**Example Output Structure:**
```json
{
  "metadata": {
    "backend": "...",
    "distribution_type": "probabilities",
    "temperature": 1.0,
    "num_prompts": 10,
    "distributions_file": "distributions.npz"
  },
  "prompts": ["...", "..."],
  "results": [
    {
      "samples": [
        {
          "text": "...",
          "token_strings": ["...", "..."],
          "distributions": "distributions.npz:results[0].samples[0].distributions",
          "entropies": [2.5, 1.8, ...]
        }
      ]
    }
  ]
}
```

---

### 02-visualize.py

Interactive Streamlit app for visualizing token distributions and entropy.

**Purpose:**
- Loads results from `01-dist.py`
- Interactive web interface for exploring token distributions
- Visualizes token entropy with color coding
- Shows probability distributions for individual tokens

**Usage:**
```bash
# Run locally
streamlit run exp/02-visualize.py

# Run on server with port forwarding (recommended)
streamlit run exp/02-visualize.py --server.port 8501
```

**Configuration (in `experiment_config.yaml` under `02_visualize`):**
- `input`: Input configuration
  - `results_dir`: Path to `01-dist` output directory

**Features:**
- Interactive prompt/response selection
- Color-coded tokens by entropy (darker = higher entropy)
- Clickable tokens to view probability distributions
- Bar plots sorted by probability magnitude
- Lazy loading for large datasets (only loads data when needed)
- Caching for improved performance

**Running on a Server:**

The Streamlit app can be run on a server in several ways:

**Option 1: SSH Port Forwarding (Recommended for Security)**

This is the most secure method - the app runs on localhost on the server, and you access it through an SSH tunnel.

On the server:
```bash
streamlit run exp/02-visualize.py --server.port 8501
```

On your local machine:
```bash
# Direct connection
ssh -L 8501:localhost:8501 user@server-address

# With jump host (if you have SSH config set up)
ssh -L 8501:localhost:8501 mpi-txiao
```

Then open `http://localhost:8501` in your browser.

**Option 2: Bind to All Interfaces**

Make the app accessible via the server's IP address.

On the server:
```bash
streamlit run exp/02-visualize.py --server.address 0.0.0.0 --server.port 8501
```

Then access it at `http://server-ip:8501` from any machine on the network.

**⚠️ Security Warning**: This makes the app accessible to anyone on the network. Only use this on trusted networks or behind a firewall.

**Option 3: Using Streamlit Config File**

Create a `.streamlit/config.toml` file in the project root:

```toml
[server]
address = "0.0.0.0"
port = 8501
```

Then just run:
```bash
streamlit run exp/02-visualize.py
```

**Option 4: Background Process (nohup)**

To run in the background on a server:

```bash
nohup streamlit run exp/02-visualize.py --server.address 0.0.0.0 --server.port 8501 > streamlit.log 2>&1 &
```

Check the process:
```bash
ps aux | grep streamlit
```

Stop it:
```bash
pkill -f streamlit
```

**Troubleshooting:**
- **Port already in use**: Change the port with `--server.port 8502`
- **Can't access from remote**: Make sure firewall allows the port, or use SSH port forwarding
- **App stops when SSH disconnects**: Use `nohup` or `screen`/`tmux` to keep it running
- **Large data loading slowly**: The app uses lazy loading and caching - first load may be slow, subsequent interactions should be faster

**Note on Data Transmission:**
The app uses lazy loading - large distribution arrays (NPZ files) are only loaded when you interact with specific prompts/responses. This means:
- Initial page load is fast (only loads metadata)
- Data is loaded on-demand as you select prompts/responses
- If running on a server, only the data you view is transmitted to your browser

---

## Workflow

The scripts are designed to work together in a pipeline:

1. **00-sample.py**: Generate responses for a dataset
   - Input: Dataset, task config, backend config, sampling config
   - Output: `exp/00-sample/output/results.json`, `statistics.json`

2. **01-dist.py**: Extract token distributions from responses
   - Input: Results from `00-sample.py`, backend config
   - Output: `exp/01-dist/output/distributions.json`, `distributions.npz`

3. **02-visualize.py**: Visualize distributions interactively
   - Input: Results from `01-dist.py`
   - Output: Interactive web interface

## Common Configuration Patterns

### Using a Task Config File

```yaml
00_sample:
  task_config: "rl_onoff/tasks/configs/math_default.yaml"
  # ... rest of config
```

### Configuring Task Directly

```yaml
00_sample:
  task:
    template_type: "chatml"
    reward_type: "math_verify"
    format_type: "boxed"
  # ... rest of config
```

### Using a Sampling Config File

```yaml
00_sample:
  sampling_config: "rl_onoff/sampling/configs/default.yaml"
  # ... rest of config
```

### Configuring Sampling Directly

```yaml
00_sample:
  sampling:
    max_length: 2048
    temperature: 1.0
    num_samples: 8
    batch_size: 48
  # ... rest of config
```

### Multi-GPU Hybrid Parallelism

```yaml
00_sample:
  backend:
    backend_type: "huggingface"
    model_name: "Qwen/Qwen3-8B"
    backend_specific:
      num_process: 2  # 2 model replicas (data parallelism)
      # Each replica uses device_map="auto" for tensor parallelism
```

## Dependencies

- **00-sample.py**: `rl_onoff.backends`, `rl_onoff.sampling`, `rl_onoff.tasks`, `rl_onoff.utils.dataset`
- **01-dist.py**: `rl_onoff.backends`, `rl_onoff.distributions`
- **02-visualize.py**: `streamlit`, `rl_onoff.distributions`

## Notes

- All scripts automatically add the project root to `sys.path`, so they can be run from anywhere
- Output directories are created automatically if they don't exist
- Progress bars are shown for long-running operations
- All scripts use the same unified configuration file for consistency
- Backend configurations should match between `00-sample.py` and `01-dist.py` for consistent tokenization
