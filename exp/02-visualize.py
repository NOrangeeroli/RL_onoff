#!/usr/bin/env python3
"""Experimental script 02: Visualize token distributions and entropy.

Interactive Streamlit app for visualizing token-wise distributions.
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import streamlit as st

# Try to import transformers for tokenizer
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import load_distributions from 01-dist.py
# We need to import it from the module path since it's in the exp directory
import importlib.util
exp_01_dist_path = Path(__file__).parent / "01-dist.py"
spec = importlib.util.spec_from_file_location("exp_01_dist", exp_01_dist_path)
exp_01_dist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp_01_dist)
_load_distributions_raw = exp_01_dist.load_distributions

# Create a cached wrapper for load_distributions
@st.cache_data
def load_distributions_cached(json_file: Union[str, Path], distributions_file: Optional[Union[str, Path]] = None) -> Dict:
    """Cached wrapper for load_distributions to avoid reloading on every rerun."""
    return _load_distributions_raw(json_file, distributions_file)


@st.cache_resource
def load_tokenizer(model_name: str):
    """Load tokenizer for a given model name."""
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        st.warning(f"Could not load tokenizer for {model_name}: {e}")
        return None


@st.cache_data
def load_single_distribution(distributions_file: Path, dist_index: int, vocab_size: Optional[int] = None, top_k: Optional[int] = None) -> Optional[Dict]:
    """Load a single distribution from NPZ file on-demand.
    
    Returns top-k data directly (indices and values) without reconstructing full distribution.
    
    Args:
        distributions_file: Path to the NPZ file
        dist_index: Index of the distribution to load
        vocab_size: Vocabulary size (from metadata, if available)
        top_k: Number of top entries stored (from metadata, if available)
        
    Returns:
        Dictionary with keys:
        - 'indices': (num_tokens, top_k) array of token IDs
        - 'values': (num_tokens, top_k) array of probabilities/logits
        - 'vocab_size': vocabulary size
        - 'top_k': actual top-k value
        or None if not found
    """
    try:
        distributions_data = np.load(distributions_file)
        
        # Load top-k format
        indices_key = f"dist_{dist_index}_indices"
        values_key = f"dist_{dist_index}_values"
        vocab_size_key = f"dist_{dist_index}_vocab_size"
        top_k_key = f"dist_{dist_index}_top_k"
        
        if indices_key in distributions_data and values_key in distributions_data:
            # Load top-k data
            top_k_indices = distributions_data[indices_key]  # (num_tokens, top_k)
            top_k_values = distributions_data[values_key]   # (num_tokens, top_k)
            
            # Get vocab_size and top_k from NPZ if available, otherwise use provided values
            if vocab_size_key in distributions_data:
                vocab_size = int(distributions_data[vocab_size_key][0])
            if top_k_key in distributions_data:
                top_k = int(distributions_data[top_k_key][0])
            
            return {
                "indices": top_k_indices,
                "values": top_k_values,
                "vocab_size": vocab_size,
                "top_k": top_k
            }
            
    except Exception as e:
        st.error(f"Error loading distribution {dist_index}: {e}")
    return None


def get_entropy_color(entropy: float, min_entropy: float, max_entropy: float) -> str:
    """Get color for a token based on its entropy.
    
    Higher entropy = darker color (more uncertainty)
    
    Args:
        entropy: Token entropy value
        min_entropy: Minimum entropy in the response
        max_entropy: Maximum entropy in the response
        
    Returns:
        Hex color string
    """
    if max_entropy == min_entropy:
        # All tokens have same entropy
        return "#808080"  # Medium gray
    
    # Normalize entropy to [0, 1]
    normalized = (entropy - min_entropy) / (max_entropy - min_entropy)
    
    # Use grayscale: higher entropy = darker (lower RGB values)
    # Scale from light gray (240) to black (0)
    gray_value = int(240 - (normalized * 240))
    gray_value = max(0, min(255, gray_value))
    
    return f"#{gray_value:02x}{gray_value:02x}{gray_value:02x}"


def format_token_html(token: str, entropy: float, min_entropy: float, max_entropy: float) -> str:
    """Format a token as HTML with entropy-based coloring.
    
    Args:
        token: Token string
        entropy: Token entropy
        min_entropy: Minimum entropy in response
        max_entropy: Maximum entropy in response
        
    Returns:
        HTML string for the token
    """
    color = get_entropy_color(entropy, min_entropy, max_entropy)
    # Escape HTML special characters in token
    token_escaped = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    return f'<span style="background-color: {color}; color: white; padding: 2px 4px; margin: 1px; border-radius: 3px; display: inline-block;" title="Entropy: {entropy:.3f}">{token_escaped}</span>'


def get_top_tokens_from_topk(indices: np.ndarray, values: np.ndarray, requested_k: int) -> List[Tuple[int, float]]:
    """Get top K tokens from already-stored top-k data.
    
    Args:
        indices: Array of token IDs (top_k,)
        values: Array of probabilities/logits (top_k,)
        requested_k: Number of top tokens to return (must be <= len(indices))
        
    Returns:
        List of (token_id, probability) tuples, sorted by probability descending
    """
    k = min(requested_k, len(indices))
    # The data is already sorted in descending order by value, so just take first k
    return [(int(indices[i]), float(values[i])) for i in range(k)]


def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="Token Distribution Visualizer", layout="wide")
    
    st.title("Token Distribution Visualizer")
    st.markdown("Visualize token-wise distributions and entropy from 01-dist.py results")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Get results directory (can load from config if needed)
    default_results_dir = project_root / "exp" / "01-dist" / "output"
    
    # Try to load from config file
    config_path = Path(__file__).parent / "experiment_config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                all_configs = yaml.safe_load(f)
            viz_config = all_configs.get("02_visualize", {})
            input_config = viz_config.get("input", {})
            if "results_dir" in input_config:
                default_results_dir = project_root / input_config["results_dir"]
        except Exception:
            pass  # Use default if config loading fails
    
    results_dir_str = st.sidebar.text_input(
        "Results Directory",
        value=str(default_results_dir),
        help="Path to 01-dist/output directory"
    )
    results_dir = Path(results_dir_str)
    
    # Load data
    if not results_dir.exists():
        st.error(f"Results directory not found: {results_dir}")
        st.stop()
    
    try:
        json_file = results_dir / "distributions.json"
        if not json_file.exists():
            st.error(f"JSON file not found: {json_file}")
            st.stop()
        
        # Load data with caching (only loads once, cached across reruns)
        data = load_distributions_cached(str(json_file))
        # Store distributions file path for on-demand loading (needed for load_single_distribution)
        # The imported function doesn't set this, so we add it here
        dist_filename = data.get("distributions_file", "distributions.npz")
        distributions_file_path = json_file.parent / dist_filename
        data["_distributions_file"] = str(distributions_file_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        st.stop()
    
    results = data.get("results", [])
    if not results:
        st.warning("No results found in data file")
        st.stop()
    
    # Get distributions file path
    distributions_file = Path(data.get("_distributions_file", results_dir / "distributions.npz"))
    
    # Example selection
    st.sidebar.header("Select Example")
    example_options = [f"Example {r['example_id']}" for r in results]
    selected_example_idx = st.sidebar.selectbox(
        "Choose an example",
        range(len(example_options)),
        format_func=lambda i: example_options[i],
        key="example_selector"
    )
    
    selected_example = results[selected_example_idx]
    
    # Sample selection
    samples = selected_example.get("samples", [])
    if not samples:
        st.warning("No samples found for this example")
        st.stop()
    
    # Filter out samples with errors
    valid_samples = [s for s in samples if "error" not in s]
    if not valid_samples:
        st.warning("No valid samples found for this example")
        st.stop()
    
    st.sidebar.header("Select Sample")
    sample_options = [f"Sample {s['sample_id']}" for s in valid_samples]
    selected_sample_idx = st.sidebar.selectbox(
        "Choose a sample",
        range(len(sample_options)),
        format_func=lambda i: sample_options[i],
        key="sample_selector"
    )
    
    selected_sample = valid_samples[selected_sample_idx]

    # Load distribution for selected sample on-demand, cached in session_state
    dist_index = selected_sample.get("distribution_index")
    dist_cache_key = f"dist_{dist_index}" if dist_index is not None else None
    if dist_index is not None and distributions_file.exists():
        if dist_cache_key not in st.session_state:
            # Get vocab_size and top_k from sample metadata
            distribution_shape = selected_sample.get("distribution_shape", [])
            vocab_size = distribution_shape[1] if len(distribution_shape) >= 2 else None
            top_k = selected_sample.get("top_k")

            topk_data = load_single_distribution(
                distributions_file,
                dist_index,
                vocab_size=vocab_size,
                top_k=top_k,
            )
            if topk_data is not None:
                # Cache in session_state for reuse across reruns
                st.session_state[dist_cache_key] = topk_data
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Prompt")
        prompt = selected_example.get("prompt", "")
        st.text_area("Prompt", prompt, height=200, disabled=True, label_visibility="collapsed")
        
        st.subheader("Response")
        response = selected_sample.get("response", "")
        token_strings = selected_sample.get("token_strings", [])
        per_token_entropy = selected_sample.get("per_token_entropy", [])
        
        # Debug: Check if entropy data exists and matches token count
        if token_strings and per_token_entropy:
            if len(token_strings) != len(per_token_entropy):
                st.warning(f"⚠️ Mismatch: {len(token_strings)} tokens but {len(per_token_entropy)} entropy values")
        
        if not token_strings or not per_token_entropy:
            st.text_area("Response", response, height=300, disabled=True, label_visibility="collapsed")
        else:
            # Calculate min/max entropy for coloring
            if per_token_entropy:
                min_entropy = min(per_token_entropy)
                max_entropy = max(per_token_entropy)
            else:
                min_entropy = max_entropy = 0.0
            
            # Create HTML with colored tokens
            token_html_parts = []
            for i, (token, entropy) in enumerate(zip(token_strings, per_token_entropy)):
                token_html = format_token_html(token, entropy, min_entropy, max_entropy)
                token_html_parts.append(token_html)
            
            token_html = " ".join(token_html_parts)
            
            # Display with HTML
            st.markdown(f'<div style="line-height: 2; word-wrap: break-word; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">{token_html}</div>', unsafe_allow_html=True)
            
            # Entropy legend
            st.caption(f"Entropy range: {min_entropy:.3f} (light gray) to {max_entropy:.3f} (dark gray/black)")
    
    with col2:
        st.subheader("Token Distribution")
        
        # Token selection
        if token_strings:
            # Create token options with entropy info
            token_options = []
            for i, token in enumerate(token_strings):
                entropy_val = per_token_entropy[i] if i < len(per_token_entropy) else 0.0
                token_display = token[:40] + "..." if len(token) > 40 else token
                token_options.append(f"Token {i}: {token_display} (entropy: {entropy_val:.3f})")
            
            selected_token_idx = st.selectbox(
                "Select a token to view its distribution",
                range(len(token_strings)),
                format_func=lambda i: token_options[i],
                key="token_selector"
            )
            
            # Get distribution for selected token (top-k data)
            topk_data = (
                st.session_state.get(dist_cache_key) if dist_cache_key is not None else None
            )
            if topk_data is not None and isinstance(topk_data, dict):
                indices = topk_data["indices"]  # (num_tokens, top_k)
                values = topk_data["values"]    # (num_tokens, top_k)
                stored_top_k = topk_data.get("top_k", len(indices[0]) if len(indices) > 0 else 0)
                
                if selected_token_idx < len(indices):
                    token_indices = indices[selected_token_idx]  # (top_k,)
                    token_values = values[selected_token_idx]    # (top_k,)
                    
                    # Show available top-k tokens
                    max_k = min(stored_top_k, len(token_indices))
                    requested_k = st.slider(
                        f"Number of top tokens to show (max: {max_k})", 
                        10, 
                        max_k, 
                        min(50, max_k)
                    )
                    top_tokens = get_top_tokens_from_topk(token_indices, token_values, requested_k)
                    
                    # Create bar chart (will be created later with token labels)
                    # Ensure sorted by probability (descending)
                    top_tokens = sorted(top_tokens, key=lambda x: x[1], reverse=True)
                    probs = [t[1] for t in top_tokens]
                    
                    # Show details with bounds checking
                    if selected_token_idx < len(token_strings):
                        st.write(f"**Selected token:** `{token_strings[selected_token_idx]}`")
                    else:
                        st.write(f"**Selected token:** N/A (index out of range)")
                    
                    token_ids_list = selected_sample.get('token_ids', [])
                    if selected_token_idx < len(token_ids_list):
                        st.write(f"**Token ID:** {token_ids_list[selected_token_idx]}")
                    else:
                        st.write(f"**Token ID:** N/A")
                    
                    # Get entropy with proper bounds checking
                    if per_token_entropy and selected_token_idx < len(per_token_entropy):
                        st.write(f"**Entropy:** {per_token_entropy[selected_token_idx]:.4f}")
                    elif per_token_entropy:
                        st.write(f"**Entropy:** N/A (index {selected_token_idx} out of range, len={len(per_token_entropy)})")
                    else:
                        st.write(f"**Entropy:** N/A (no entropy data available)")
                    
                    # Get tokenizer to decode token IDs
                    statistics = data.get("statistics", {})
                    backend_info = statistics.get("backend", {})
                    model_name = backend_info.get("model_name")
                    
                    # Create bar chart with token strings if possible
                    token_labels = []
                    if model_name and TRANSFORMERS_AVAILABLE:
                        tokenizer = load_tokenizer(model_name)
                        if tokenizer:
                            for token_id, prob in top_tokens:
                                try:
                                    token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                                    # Clean up the token string
                                    if token_str.startswith(" ") and len(token_str) > 1:
                                        token_str = token_str[1:]
                                    token_labels.append(f"{token_str[:20]}... (ID:{token_id})" if len(token_str) > 20 else f"{token_str} (ID:{token_id})")
                                except:
                                    token_labels.append(f"Token {token_id}")
                        else:
                            token_labels = [f"Token {tid}" for tid, _ in top_tokens]
                    else:
                        token_labels = [f"Token {tid}" for tid, _ in top_tokens]
                    
                    # Create DataFrame for better chart with labels
                    # Ensure data is sorted by probability (descending) for display
                    try:
                        import pandas as pd
                        # Create DataFrame and sort by probability descending
                        chart_df = pd.DataFrame({
                            "Token": token_labels,
                            "Probability": probs
                        })
                        # Sort by probability descending to ensure bars are ordered
                        chart_df = chart_df.sort_values("Probability", ascending=False).reset_index(drop=True)
                        
                        # Try to use plotly for better control over ordering
                        try:
                            import plotly.express as px
                            fig = px.bar(
                                chart_df,
                                x="Token",
                                y="Probability",
                                title=f"Token Distribution (Token {selected_token_idx})",
                                labels={"Token": "Token", "Probability": "Probability"}
                            )
                            # Sort bars by probability (descending)
                            fig.update_layout(
                                xaxis={'categoryorder': 'total descending'},
                                xaxis_tickangle=-45
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except ImportError:
                            # Fallback to streamlit bar_chart
                            # Create a dict with sorted data to preserve order
                            chart_dict = {row["Token"]: row["Probability"] for _, row in chart_df.iterrows()}
                            st.bar_chart(chart_dict)
                    except ImportError:
                        # Fallback if pandas not available - create dict sorted by value
                        sorted_data = sorted(zip(token_labels, probs), key=lambda x: x[1], reverse=True)
                        chart_data = {label: prob for label, prob in sorted_data}
                        st.bar_chart(chart_data)
                        st.caption("Token labels: " + ", ".join([l for l, _ in sorted_data[:5]]) + "...")
                    
                    # Show top tokens table
                    with st.expander("Top tokens (expand to see details)"):
                        if model_name and TRANSFORMERS_AVAILABLE:
                            tokenizer = load_tokenizer(model_name)
                            if tokenizer:
                                for token_id, prob in top_tokens:
                                    try:
                                        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                                        st.write(f"Token ID {token_id} (`{token_str}`): {prob:.6f}")
                                    except:
                                        st.write(f"Token ID {token_id}: {prob:.6f}")
                            else:
                                for token_id, prob in top_tokens:
                                    st.write(f"Token ID {token_id}: {prob:.6f}")
                        else:
                            for token_id, prob in top_tokens:
                                st.write(f"Token ID {token_id}: {prob:.6f}")
            else:
                st.warning("Distribution data not available for this sample.")
        else:
            st.info("No token information available")
    
    # Statistics
    st.sidebar.header("Statistics")
    st.sidebar.write(f"**Total examples:** {len(results)}")
    st.sidebar.write(f"**Samples in this example:** {len(valid_samples)}")
    if per_token_entropy and len(per_token_entropy) > 0:
        # Check if all entropy values are 0
        if all(e == 0.0 for e in per_token_entropy):
            st.sidebar.warning("⚠️ All entropy values are 0.0 - this may indicate an issue with entropy calculation.")
        st.sidebar.write(f"**Average entropy:** {np.mean(per_token_entropy):.3f}")
        st.sidebar.write(f"**Min entropy:** {min(per_token_entropy):.3f}")
        st.sidebar.write(f"**Max entropy:** {max(per_token_entropy):.3f}")
    elif not per_token_entropy:
        st.sidebar.warning("⚠️ No entropy data available")


if __name__ == "__main__":
    main()
