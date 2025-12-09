#!/usr/bin/env python3
"""Experimental script 02: Visualize token distributions and entropy.

Simplified version for debugging - just shows first prompt-response pair.
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import streamlit as st

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_distributions(
    json_file: Union[str, Path],
    distributions_file: Optional[Union[str, Path]] = None
) -> Dict:
    """Load distributions from JSON metadata and NPZ distributions file."""
    json_path = Path(json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    st.write("📂 Loading JSON metadata...")
    # Load JSON metadata
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    st.write(f"✅ JSON loaded. Found {len(data.get('results', []))} examples.")
    
    # Determine distributions file path
    if distributions_file is None:
        dist_filename = data.get("distributions_file", "distributions.npz")
        distributions_file = json_path.parent / dist_filename
    else:
        distributions_file = Path(distributions_file)
    
    if not distributions_file.exists():
        st.warning(f"⚠️ NPZ file not found: {distributions_file}")
        return data
    
    st.write(f"📦 Loading NPZ file: {distributions_file}")
    st.write(f"📊 File size: {distributions_file.stat().st_size / 1e9:.2f} GB")
    
    # Load distributions from NPZ
    distributions_data = np.load(distributions_file)
    st.write(f"✅ NPZ loaded. Found {len(distributions_data.files)} distributions.")
    
    # Reconstruct full structure by adding distributions to each sample
    results = data.get("results", [])
    st.write(f"🔄 Processing {len(results)} examples...")
    
    processed_count = 0
    for example_result in results:
        for sample in example_result.get("samples", []):
            if "error" in sample:
                continue
            
            dist_index = sample.get("distribution_index")
            if dist_index is not None:
                dist_key = f"dist_{dist_index}"
                if dist_key in distributions_data:
                    sample["distributions"] = distributions_data[dist_key].tolist()
                    processed_count += 1
                    if processed_count % 10 == 0:
                        st.write(f"  Processed {processed_count} distributions...")
    
    st.write(f"✅ Done! Processed {processed_count} distributions.")
    return data


def main():
    """Main Streamlit app - simplified for debugging."""
    st.set_page_config(page_title="Token Distribution Visualizer (Debug)", layout="wide")
    
    st.title("Token Distribution Visualizer (Debug Mode)")
    st.markdown("**Simplified version** - Shows first prompt-response pair only")
    
    # Get results directory
    default_results_dir = project_root / "exp" / "01-dist" / "output"
    results_dir_str = st.sidebar.text_input(
        "Results Directory",
        value=str(default_results_dir),
        help="Path to 01-dist/output directory"
    )
    results_dir = Path(results_dir_str)
    
    if not results_dir.exists():
        st.error(f"❌ Results directory not found: {results_dir}")
        st.stop()
    
    # Load data with progress indicators
    try:
        json_file = results_dir / "distributions.json"
        if not json_file.exists():
            st.error(f"❌ JSON file not found: {json_file}")
            st.stop()
        
        st.write("=" * 60)
        st.write("## Loading Data")
        st.write("=" * 60)
        
        data = load_distributions(str(json_file))
        
        st.write("=" * 60)
        st.write("## Data Loaded Successfully!")
        st.write("=" * 60)
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        st.stop()
    
    results = data.get("results", [])
    if not results:
        st.warning("⚠️ No results found in data file")
        st.stop()
    
    st.write(f"📋 Total examples in file: {len(results)}")
    st.write("---")
    
    # Get first example
    first_example = results[0]
    st.write("## First Example")
    
    # Get first valid sample
    samples = first_example.get("samples", [])
    valid_samples = [s for s in samples if "error" not in s]
    
    if not valid_samples:
        st.warning("⚠️ No valid samples found in first example")
        st.stop()
    
    first_sample = valid_samples[0]
    
    # Display prompt
    st.write("### Prompt")
    prompt = first_example.get("prompt", "")
    st.text_area("", prompt, height=150, disabled=True, key="prompt_display")
    
    # Display response
    st.write("### Response")
    response = first_sample.get("response", "")
    st.text_area("", response, height=200, disabled=True, key="response_display")
    
    # Display token information
    token_strings = first_sample.get("token_strings", [])
    per_token_entropy = first_sample.get("per_token_entropy", [])
    token_ids = first_sample.get("token_ids", [])
    
    if token_strings and per_token_entropy:
        st.write("### Token Information")
        st.write(f"**Number of tokens:** {len(token_strings)}")
        
        # Show first 10 tokens
        st.write("**First 10 tokens:**")
        for i, (token, entropy, token_id) in enumerate(zip(
            token_strings[:10], 
            per_token_entropy[:10], 
            token_ids[:10]
        )):
            st.write(f"  Token {i}: `{token}` | ID: {token_id} | Entropy: {entropy:.4f}")
        
        # Entropy stats
        if per_token_entropy:
            st.write("**Entropy Statistics:**")
            st.write(f"  Min: {min(per_token_entropy):.4f}")
            st.write(f"  Max: {max(per_token_entropy):.4f}")
            st.write(f"  Mean: {np.mean(per_token_entropy):.4f}")
            st.write(f"  Std: {np.std(per_token_entropy):.4f}")
    else:
        st.warning("⚠️ No token information available")
    
    # Check if distributions are loaded
    distributions = first_sample.get("distributions")
    if distributions:
        st.write("### Distribution Information")
        if isinstance(distributions, list):
            distributions = np.array(distributions)
        st.write(f"**Distribution shape:** {distributions.shape}")
        st.write(f"  - Number of tokens: {distributions.shape[0]}")
        st.write(f"  - Vocabulary size: {distributions.shape[1]}")
        
        # Show distribution for first token
        if len(distributions) > 0:
            st.write("**First token distribution (top 10):**")
            first_token_dist = distributions[0]
            top_indices = np.argsort(first_token_dist)[::-1][:10]
            for idx in top_indices:
                st.write(f"  Token ID {idx}: {first_token_dist[idx]:.6f}")
    else:
        st.warning("⚠️ Distribution data not loaded")
    
    # Raw data view
    with st.expander("🔍 View Raw Sample Data"):
        st.json({
            "example_id": first_example.get("example_id"),
            "sample_id": first_sample.get("sample_id"),
            "has_distributions": "distributions" in first_sample,
            "num_tokens": len(token_strings) if token_strings else 0,
            "has_entropy": len(per_token_entropy) > 0
        })


if __name__ == "__main__":
    main()
