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


def load_distributions(
    json_file: Union[str, Path],
    distributions_file: Optional[Union[str, Path]] = None
) -> Dict:
    """Load distributions from JSON metadata (skip NPZ loading for performance)."""
    json_path = Path(json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # Load JSON metadata
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determine distributions file path
    if distributions_file is None:
        dist_filename = data.get("distributions_file", "distributions.npz")
        distributions_file = json_path.parent / dist_filename
    else:
        distributions_file = Path(distributions_file)
    
    # Store distributions file path in data for later use
    data["_distributions_file"] = str(distributions_file)
    
    return data


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
def load_single_distribution(distributions_file: Path, dist_index: int) -> Optional[np.ndarray]:
    """Load a single distribution from NPZ file on-demand.
    
    Args:
        distributions_file: Path to the NPZ file
        dist_index: Index of the distribution to load
        
    Returns:
        Distribution array or None if not found
    """
    try:
        distributions_data = np.load(distributions_file)
        dist_key = f"dist_{dist_index}"
        if dist_key in distributions_data:
            return distributions_data[dist_key]
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


def get_top_tokens(distribution: np.ndarray, top_k: int = 20) -> List[Tuple[int, float]]:
    """Get top K tokens by probability from a distribution.
    
    Args:
        distribution: Probability distribution array (vocab_size,)
        top_k: Number of top tokens to return
        
    Returns:
        List of (token_id, probability) tuples, sorted by probability descending
    """
    top_indices = np.argsort(distribution)[::-1][:top_k]
    return [(int(idx), float(distribution[idx])) for idx in top_indices]


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
        with st.spinner("Loading metadata..."):
            json_file = results_dir / "distributions.json"
            if not json_file.exists():
                st.error(f"JSON file not found: {json_file}")
                st.stop()
            
            data = load_distributions(str(json_file))
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
    
    # Load distribution for selected sample on-demand
    dist_index = selected_sample.get("distribution_index")
    if dist_index is not None and distributions_file.exists():
        distribution = load_single_distribution(distributions_file, dist_index)
        if distribution is not None:
            # Convert to list and store in sample
            selected_sample["distributions"] = distribution.tolist()
    
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
            
            # Get distribution for selected token
            distributions = selected_sample.get("distributions")
            if distributions:
                # Convert to numpy array if it's a list
                if isinstance(distributions, list):
                    distributions = np.array(distributions)
                
                if selected_token_idx < len(distributions):
                    token_dist = distributions[selected_token_idx]
                    
                    # Get all tokens or top K
                    show_all = st.checkbox("Show all tokens (may be slow for large vocabularies)", value=False)
                    if show_all:
                        # Get all tokens sorted by probability
                        all_indices = np.argsort(token_dist)[::-1]  # Sort descending
                        top_tokens = [(int(idx), float(token_dist[idx])) for idx in all_indices]
                    else:
                        top_k = st.slider("Number of top tokens to show", 10, 1000, 50)
                        top_tokens = get_top_tokens(token_dist, top_k=top_k)
                    
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
                st.warning("Distribution data not available. Loading...")
                # Try to load if we have the index
                if dist_index is not None and distributions_file.exists():
                    with st.spinner("Loading distribution..."):
                        distribution = load_single_distribution(distributions_file, dist_index)
                        if distribution is not None:
                            selected_sample["distributions"] = distribution.tolist()
                            st.rerun()
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
