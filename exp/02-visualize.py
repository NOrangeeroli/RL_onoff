#!/usr/bin/env python3
"""Experimental script 02: Visualize token distributions and entropy.

This Streamlit app allows interactive visualization of:
- Token-wise distributions from 01-dist.py results
- Entropy-based token coloring
- Clickable tokens to view distribution barplots
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import streamlit as st

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Define load_distributions function (same as in 01-dist.py)
def load_distributions(
    json_file: Union[str, Path],
    distributions_file: Optional[Union[str, Path]] = None
) -> Dict:
    """Load distributions from JSON metadata and NPZ distributions file."""
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
    
    if not distributions_file.exists():
        raise FileNotFoundError(f"Distributions file not found: {distributions_file}")
    
    # Load distributions from NPZ
    distributions_data = np.load(distributions_file)
    
    # Reconstruct full structure by adding distributions to each sample
    results = data.get("results", [])
    for example_result in results:
        for sample in example_result.get("samples", []):
            if "error" in sample:
                continue
            
            dist_index = sample.get("distribution_index")
            if dist_index is not None:
                dist_key = f"dist_{dist_index}"
                if dist_key in distributions_data:
                    sample["distributions"] = distributions_data[dist_key].tolist()
    
    return data


def load_data(results_dir: Path) -> Dict:
    """Load distribution data from 01-dist output directory.
    
    Args:
        results_dir: Path to 01-dist/output directory
        
    Returns:
        Dictionary with loaded data
    """
    json_file = results_dir / "distributions.json"
    if not json_file.exists():
        raise FileNotFoundError(f"Results file not found: {json_file}")
    
    data = load_distributions(str(json_file))
    return data


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
        return "#000000"  # Black
    
    # Normalize entropy to [0, 1]
    normalized = (entropy - min_entropy) / (max_entropy - min_entropy)
    
    # Use grayscale: higher entropy = darker (lower RGB values)
    # Scale from light gray (200) to black (0)
    gray_value = int(200 - (normalized * 200))
    gray_value = max(0, min(255, gray_value))
    
    return f"#{gray_value:02x}{gray_value:02x}{gray_value:02x}"


def format_token_html(token: str, entropy: float, min_entropy: float, max_entropy: float, token_idx: int, selected: bool = False) -> str:
    """Format a token as HTML with entropy-based coloring.
    
    Args:
        token: Token string
        entropy: Token entropy
        min_entropy: Minimum entropy in response
        max_entropy: Maximum entropy in response
        token_idx: Token index
        selected: Whether this token is currently selected
        
    Returns:
        HTML string for the token
    """
    color = get_entropy_color(entropy, min_entropy, max_entropy)
    # Escape HTML special characters in token
    token_escaped = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Add border if selected
    border_style = "border: 2px solid red;" if selected else ""
    return f'<span style="background-color: {color}; color: white; padding: 2px 4px; margin: 1px; border-radius: 3px; display: inline-block; {border_style}" title="Token {token_idx}, Entropy: {entropy:.3f}">{token_escaped}</span>'


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
    
    # Get results directory
    default_results_dir = project_root / "exp" / "01-dist" / "output"
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
        with st.spinner("Loading data..."):
            data = load_data(results_dir)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    results = data.get("results", [])
    if not results:
        st.warning("No results found in data file")
        st.stop()
    
    # Example selection
    st.sidebar.header("Select Example")
    example_options = [f"Example {r['example_id']}" for r in results]
    selected_example_idx = st.sidebar.selectbox(
        "Choose an example",
        range(len(example_options)),
        format_func=lambda i: example_options[i]
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
        format_func=lambda i: sample_options[i]
    )
    
    selected_sample = valid_samples[selected_sample_idx]
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Prompt")
        st.text(selected_example.get("prompt", ""))
        
        st.subheader("Response")
        response = selected_sample.get("response", "")
        token_strings = selected_sample.get("token_strings", [])
        per_token_entropy = selected_sample.get("per_token_entropy", [])
        
        if not token_strings or not per_token_entropy:
            st.text(response)
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
                token_html = format_token_html(token, entropy, min_entropy, max_entropy, i)
                token_html_parts.append(token_html)
            
            token_html = " ".join(token_html_parts)
            
            # Display with HTML
            st.markdown(f'<div style="line-height: 2; word-wrap: break-word;">{token_html}</div>', unsafe_allow_html=True)
            
            # Entropy legend
            st.caption(f"Entropy range: {min_entropy:.3f} (light) to {max_entropy:.3f} (dark)")
    
    with col2:
        st.subheader("Token Distribution")
        
        # Token selection (via dropdown)
        if token_strings:
            # Create token options with entropy info
            token_options = []
            for i, token in enumerate(token_strings):
                entropy_val = per_token_entropy[i] if i < len(per_token_entropy) else 0.0
                token_display = token[:30] + "..." if len(token) > 30 else token
                token_options.append(f"Token {i}: {token_display} (entropy: {entropy_val:.3f})")
            
            selected_token_idx = st.selectbox(
                "Select a token to view its distribution",
                range(len(token_strings)),
                format_func=lambda i: token_options[i],
                key="token_selector"
            )
            
            # Store in session state for highlighting
            st.session_state['selected_token_idx'] = selected_token_idx
            
            # Get distribution for selected token
            distributions = selected_sample.get("distributions")
            if distributions:
                # Convert to numpy array if it's a list
                if isinstance(distributions, list):
                    distributions = np.array(distributions)
                
                if selected_token_idx < len(distributions):
                    token_dist = distributions[selected_token_idx]
                    
                    # Get top tokens
                    top_k = st.slider("Number of top tokens to show", 10, 50, 20)
                    top_tokens = get_top_tokens(token_dist, top_k=top_k)
                    
                    # Create bar chart
                    token_ids = [t[0] for t in top_tokens]
                    probs = [t[1] for t in top_tokens]
                    
                    # Create labels (try to decode token IDs if possible)
                    # For now, just use token IDs
                    labels = [f"Token {tid}" for tid in token_ids]
                    
                    # Display chart
                    chart_data = {"Probability": probs}
                    st.bar_chart(chart_data)
                    
                    # Show details
                    st.write(f"**Selected token:** {token_strings[selected_token_idx]}")
                    st.write(f"**Token ID:** {selected_sample.get('token_ids', [])[selected_token_idx] if selected_token_idx < len(selected_sample.get('token_ids', [])) else 'N/A'}")
                    st.write(f"**Entropy:** {per_token_entropy[selected_token_idx]:.4f}")
                    
                    # Show top tokens table
                    with st.expander("Top tokens (expand to see details)"):
                        for token_id, prob in top_tokens:
                            st.write(f"Token ID {token_id}: {prob:.6f}")
            else:
                st.warning("Distribution data not available. Make sure distributions are loaded.")
        else:
            st.info("No token information available")
    
    # Statistics
    st.sidebar.header("Statistics")
    st.sidebar.write(f"**Total examples:** {len(results)}")
    st.sidebar.write(f"**Samples in this example:** {len(valid_samples)}")
    if per_token_entropy:
        st.sidebar.write(f"**Average entropy:** {np.mean(per_token_entropy):.3f}")
        st.sidebar.write(f"**Min entropy:** {min(per_token_entropy):.3f}")
        st.sidebar.write(f"**Max entropy:** {max(per_token_entropy):.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Token Distribution Visualizer")
    parser.add_argument(
        "--server-address",
        type=str,
        default="localhost",
        help="Address to bind the server to (default: localhost). Use '0.0.0.0' to make it accessible from network"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the server on (default: 8501)"
    )
    
    args = parser.parse_args()
    
    # Note: Streamlit doesn't accept command-line args directly for server config
    # These would need to be set via streamlit run command or config file
    # For now, we'll just run main()
    # To use custom server address/port, run:
    # streamlit run exp/02-visualize.py --server.address 0.0.0.0 --server.port 8501
    main()

