# Environment Files

This directory contains Conda environment files and corresponding pip requirements files for different LLM backends.

## Files

### Environment Files (Conda)

- **`environment_huggingface.yml`** - For HuggingFace backend (flexible PyTorch version)
- **`environment_vllm.yml`** - For vLLM backend (requires torch==2.5.1)
- **`environment_sglang.yml`** - For SGLang backend (requires compiled kernels)
- **`environment.yml`** - Legacy/backward compatibility file

### Requirements Files (Pip)

- **`requirements_huggingface.txt`** - Pip packages for HuggingFace backend
- **`requirements_vllm.txt`** - Pip packages for vLLM backend
- **`requirements_sglang.txt`** - Pip packages for SGLang backend

## Usage

### Creating a Conda Environment

1. Create the environment from the YAML file:
   ```bash
   conda env create -f envs/environment_huggingface.yml
   ```

2. Activate the environment:
   ```bash
   conda activate rl_onoff_hf
   ```

3. Install pip packages separately (faster than installing during conda env create):
   ```bash
   pip install -r envs/requirements_huggingface.txt
   ```

### Available Environments

- **HuggingFace**: `conda env create -f envs/environment_huggingface.yml` → `conda activate rl_onoff_hf`
- **vLLM**: `conda env create -f envs/environment_vllm.yml` → `conda activate rl_onoff_vllm`
- **SGLang**: `conda env create -f envs/environment_sglang.yml` → `conda activate rl_onoff_sglang`

## Notes

- Pip packages are separated from Conda environment files to speed up environment creation
- After creating the environment, always install the corresponding requirements file
- Each backend has specific version requirements (especially PyTorch and CUDA), so use the correct environment file

