"""
Embedding-Based Motion Similarity Analysis Framework - Per-Module Analysis with Validation Set Evaluation

This script analyzes motion data similarity for each animation type separately,
using the embedding-based similarity network instead of the raw motion CNN.
It compares embedding-based and raw feature-based approaches including geodesic distance
calculations for quaternion data. It processes each animation module individually,
providing comprehensive analysis of how well each method correlates with human perception.

The framework includes:
1. Loading and processing motion data for each animation type
2. Creating train/validation splits similar to run_motion_triplet_training.py
3. Generating embeddings using a trained embedding-based neural network
4. Extracting variable-length raw features directly from pickle files
5. Calculating L2 distances for embeddings and geodesic distances for raw features
6. Analyzing the relationship between distance metrics and human comparison data
7. Comparing the different approaches for each animation type separately
8. Specifically evaluating on the validation (out-of-sample) subset

Key differences from original script:
- Uses EmbeddingAwareSimilarityNetwork instead of SimilarityNetwork
- Loads embedding model checkpoint instead of raw motion checkpoint
- Processes pre-computed autoencoder embeddings as input
"""

import os
import sys
import numpy as np
import pandas as pd
import random
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from dtaidistance import dtw_ndim
# TensorFlow is optional — only used for the GPU availability check helper.
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
import torch
import pickle
import time
from itertools import combinations
from pathlib import Path

# Add necessary paths
# Resolve project root from this file's location so the script works
# whether invoked from pipelines/ or from the project root.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'networks'))
# Import required modules
from networks.similarity_network import EmbeddingRefiningSimilarityNetwork
from networks.similarity_data_loader import SimilarityDataLoader
from networks.triplet_mining import TripletMining
from bvh_visualizing.datasetLoad import BVHDataset
from bvh_visualizing.bvhvisualize import DualBVHAnimator
from bvh_visualizing.bvh import BVH
from Config import Config
import src.organize_synthetic_data as osd

from src.organize_synthetic_data import load_similarity_data_from_embeddings, EmbeddingSimilarityDataLoader


def create_train_val_split(similarity_dicts, val_ratio=0.4):
    """
    Create deterministic training and validation indices for each animation type.

    Args:
        similarity_dicts: List of dictionaries containing class exemplars for each animation
        val_ratio: Ratio of classes to use for validation

    Returns:
        train_indices: List of sets containing training class indices for each animation
        val_indices: List of sets containing validation class indices for each animation
    """

    train_indices = []
    val_indices = []

    for anim_dict in similarity_dicts:
        # Get keys except neutral and sort them for deterministic order
        keys = sorted(k for k in anim_dict.keys() if k != (0, 0, 0, 0))

        # Determine validation set size
        val_size = max(1, int(len(keys) * val_ratio))

        # Deterministic split based on sorted order
        val_keys = set(keys[:val_size])
        train_keys = set(keys[val_size:])

        # Add neutral exemplar to both sets if present
        if (0, 0, 0, 0) in anim_dict:
            train_keys.add((0, 0, 0, 0))
            val_keys.add((0, 0, 0, 0))

        train_indices.append(train_keys)
        val_indices.append(val_keys)

    return train_indices, val_indices


def get_inverse_direct_comparison_value(key1, key2, triplet_modules):
    """
    Get the inverse of direct comparison value (1 - value) for a pair of keys.

    Args:
        key1: First key (action_type, effort_tuple)
        key2: Second key (action_type, effort_tuple)
        triplet_modules: Triplet module object or list of triplet module objects

    Returns:
        Inverse comparison value or None if not available
    """
    if not isinstance(triplet_modules, list):
        triplet_modules = [triplet_modules]

    action1, effort1 = key1
    action2, effort2 = key2

    # Skip neutral exemplars
    if effort1 == (0, 0, 0, 0) or effort2 == (0, 0, 0, 0):
        return None

    # Only process pairs from the same animation type
    if action1 != action2:
        print(f"Skipping different animation types: {action1} and {action2}")
        return None

    # Find the appropriate module for this animation
    for module in triplet_modules:
        if module.anim_name == action1:
            # Check if alpha_dataframes exists
            if not hasattr(module, 'alpha_dataframes') or module.alpha_dataframes is None:
                print(f"No alpha_dataframes found for {action1}")
                return None

            # Look for the pair in the efforts_tuples column
            df = module.df_comparisons

            pair_match = df[df['efforts_tuples'].apply(lambda x:
                                                       set(x) == set([effort1, effort2]) if isinstance(x,
                                                                                                       list) else False)]

            # Target the specific row we want (selected0=0, selected1=2)
            target_row = pair_match[(pair_match['selected0'] == 0) & (pair_match['selected1'] == 2)]

            if not target_row.empty:
                # Check if we have a count_normalized value
                if 'count_normalized' in target_row.columns and not pd.isna(target_row['count_normalized'].iloc[0]):
                    # Return 1 minus the comparison value
                    return 1 - target_row['count_normalized'].iloc[0]

            print(f"No direct comparison value found for {key1} and {key2}")
            return None

    return None


def calculate_real_variable_length_dtw(dict_raw_features):
    """
    Calculate DTW distances between all raw motion features with variable lengths.

    Args:
        dict_raw_features: Dictionary mapping (action_type, effort_tuple) to raw motion features

    Returns:
        List of tuples (distance, key1, key2) sorted by distance in ascending order
    """
    print("Calculating DTW distances with variable-length raw features...")
    distances = []
    keys = list(dict_raw_features.keys())
    total_pairs = len(keys) * (len(keys) - 1) // 2

    print(f"Processing {total_pairs} pairs across {len(keys)} features")

    # First, verify we have variable-length data
    lengths = {key: data.shape[0] for key, data in dict_raw_features.items()}
    unique_lengths = sorted(set(lengths.values()))
    print(f"DTW processing sequences with {len(unique_lengths)} unique lengths: {unique_lengths}")

    # Optional: Track progress
    progress_interval = max(1, total_pairs // 10)  # Show progress 10 times
    pair_count = 0
    start_time = time.time()

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key1 = keys[i]
            key2 = keys[j]

            # Get motion data
            motion1 = dict_raw_features[key1]
            motion2 = dict_raw_features[key2]

            # Reshape for DTW if needed
            if len(motion1.shape) > 2:
                motion1_dtw = motion1.reshape(motion1.shape[0], -1)
                motion2_dtw = motion2.reshape(motion2.shape[0], -1)
            else:
                motion1_dtw = motion1
                motion2_dtw = motion2

            # Ensure data is in double precision format
            motion1_dtw = motion1_dtw.astype(np.double)
            motion2_dtw = motion2_dtw.astype(np.double)

            try:
                # Calculate DTW distance
                dtw_distance = dtw_ndim.distance(motion1_dtw, motion2_dtw)
                distances.append((dtw_distance, key1, key2))
            except Exception as e:
                print(f"Error calculating DTW for {key1} and {key2}: {e}")
                print(f"Shapes: {motion1.shape}, {motion2.shape}")
                continue

            # Update progress
            pair_count += 1
            if pair_count % progress_interval == 0:
                elapsed = time.time() - start_time
                percent = (pair_count / total_pairs) * 100
                eta = (elapsed / pair_count) * (total_pairs - pair_count) if pair_count > 0 else 0
                print(
                    f"Progress: {pair_count}/{total_pairs} pairs ({percent:.1f}%) - Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")

    # Sort by distance (ascending)
    distances.sort()
    print(f"Computed {len(distances)} DTW distances")

    return distances


def compare_distance_inverse_comparison_value_relationships(embedding_distances, embedding_alphas,
                                                            dtw_distances, dtw_alphas, method="geodesic"):
    """
    Compare the relationship between distances and inverse comparison values for embedding and DTW methods.

    Args:
        embedding_distances: List of distance values from embedding method
        embedding_alphas: List of inverse comparison values corresponding to embedding_distances
        dtw_distances: List of distance values from raw feature method
        dtw_alphas: List of inverse comparison values corresponding to dtw_distances
        method: String indicating which comparison is being performed

    Returns:
        Dictionary containing comparison results with correlation metrics and summary
    """
    # Filter out None values
    emb_pairs = [(d, a) for d, a in zip(embedding_distances, embedding_alphas) if a is not None]
    dtw_pairs = [(d, a) for d, a in zip(dtw_distances, dtw_alphas) if a is not None]

    if not emb_pairs or not dtw_pairs:
        return {"error": "No valid distance-alpha pairs found"}

    # Unzip the filtered pairs
    embedding_distances, embedding_alphas = zip(*emb_pairs)
    dtw_distances, dtw_alphas = zip(*dtw_pairs)

    # Create dataframes for easier analysis
    df_embedding = pd.DataFrame({
        'distance': embedding_distances,
        'alpha': embedding_alphas,
        'method': f'Embedding_L2'
    })

    df_dtw = pd.DataFrame({
        'distance': dtw_distances,
        'alpha': dtw_alphas,
        'method': f'Raw_Features_{method}'
    })

    # Combine dataframes
    df_combined = pd.concat([df_embedding, df_dtw])

    results = {}
    output = []

    # 1. CORRELATION ANALYSIS
    # Calculate Pearson correlation (linear relationship)
    emb_pearson = stats.pearsonr(embedding_distances, embedding_alphas)
    dtw_pearson = stats.pearsonr(dtw_distances, dtw_alphas)

    # Calculate Spearman rank correlation (monotonic relationship)
    emb_spearman = stats.spearmanr(embedding_distances, embedding_alphas)
    dtw_spearman = stats.spearmanr(dtw_distances, dtw_alphas)

    output.append("1. CORRELATION ANALYSIS")
    output.append(f"Embedding L2 distances - Pearson: r={emb_pearson[0]:.4f}, p={emb_pearson[1]:.4f}")
    output.append(f"Raw feature {method} distances - Pearson: r={dtw_pearson[0]:.4f}, p={dtw_pearson[1]:.4f}")
    output.append(f"Embedding L2 distances - Spearman: r={emb_spearman[0]:.4f}, p={emb_spearman[1]:.4f}")
    output.append(f"Raw feature {method} distances - Spearman: r={dtw_spearman[0]:.4f}, p={dtw_spearman[1]:.4f}")
    output.append("")

    results['correlation'] = {
        'embedding_pearson': {'r': emb_pearson[0], 'p': emb_pearson[1]},
        'dtw_pearson': {'r': dtw_pearson[0], 'p': dtw_pearson[1]},
        'embedding_spearman': {'r': emb_spearman[0], 'p': emb_spearman[1]},
        'dtw_spearman': {'r': dtw_spearman[0], 'p': dtw_spearman[1]}
    }

    # 2. LINEAR REGRESSION ANALYSIS
    X_emb = np.array(embedding_distances).reshape(-1, 1)
    y_emb = np.array(embedding_alphas)
    X_dtw = np.array(dtw_distances).reshape(-1, 1)
    y_dtw = np.array(dtw_alphas)

    model_emb = LinearRegression().fit(X_emb, y_emb)
    model_dtw = LinearRegression().fit(X_dtw, y_dtw)

    y_pred_emb = model_emb.predict(X_emb)
    y_pred_dtw = model_dtw.predict(X_dtw)

    r2_emb = r2_score(y_emb, y_pred_emb)
    r2_dtw = r2_score(y_dtw, y_pred_dtw)

    output.append("2. LINEAR REGRESSION ANALYSIS")
    output.append(
        f"Embedding L2 - slope: {model_emb.coef_[0]:.4f}, intercept: {model_emb.intercept_:.4f}, R²: {r2_emb:.4f}")
    output.append(
        f"Raw Feature {method} - slope: {model_dtw.coef_[0]:.4f}, intercept: {model_dtw.intercept_:.4f}, R²: {r2_dtw:.4f}")
    output.append("")

    # Evaluation to determine which method is better
    better_pearson = "Embedding L2" if abs(emb_pearson[0]) > abs(dtw_pearson[0]) else f"Raw Feature {method}"
    better_spearman = "Embedding L2" if abs(emb_spearman[0]) > abs(dtw_spearman[0]) else f"Raw Feature {method}"
    better_r2 = "Embedding L2" if r2_emb > r2_dtw else f"Raw Feature {method}"

    # Determine which method has a stronger relationship
    def determine_relationship_strength():
        emb_strength = 0
        dtw_strength = 0

        if abs(emb_pearson[0]) > abs(dtw_pearson[0]):
            emb_strength += 1
        else:
            dtw_strength += 1

        if abs(emb_spearman[0]) > abs(dtw_spearman[0]):
            emb_strength += 1
        else:
            dtw_strength += 1

        if r2_emb > r2_dtw:
            emb_strength += 1
        else:
            dtw_strength += 1

        if emb_pearson[0] > 0:
            emb_strength += 0.5
        if dtw_pearson[0] > 0:
            dtw_strength += 0.5

        if emb_strength > dtw_strength:
            return f"Embedding_L2"
        elif dtw_strength > emb_strength:
            return f"Raw_Features_{method}"
        else:
            return "Both methods are equally strong"

    stronger_method = determine_relationship_strength()

    output.append("3. SUMMARY AND CONCLUSION")
    output.append(f"Linear correlation (Pearson): {better_pearson} is better")
    output.append(f"Rank correlation (Spearman): {better_spearman} is better")
    output.append(f"Linear relationship (R²): {better_r2} is better")
    output.append(
        f"Overall, the {stronger_method} method shows a stronger relationship between distances and inverse comparison values")

    results['summary'] = {
        'better_pearson': better_pearson,
        'better_spearman': better_spearman,
        'better_r2': better_r2,
        'stronger_method': stronger_method
    }

    results['output_text'] = '\n'.join(output)

    return results


def collect_distance_inverse_comparison_value_pairs(distance_tuples, triplet_modules,
                                                    get_inverse_direct_comparison_value):
    """
    Collect distance and inverse comparison values from distance tuples.

    Args:
        distance_tuples: List of tuples (distance, key1, key2)
        triplet_modules: Triplet module object or list of triplet module objects
        get_inverse_direct_comparison_value: Function to get inverse comparison value for a pair of keys

    Returns:
        tuple: (distances, inverse_comparison_values)
    """
    distances = []
    alphas = []

    for distance, key1, key2 in distance_tuples:
        alpha_value = get_inverse_direct_comparison_value(key1, key2, triplet_modules)
        distances.append(distance)
        alphas.append(alpha_value)

    return distances, alphas


def create_triplet_module(anim_name, bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
                          squared_left_right_euc_dist, squared_class_neut_euc_dist, config, valid_indices=None):
    """
    Create a single triplet module for the specified animation.

    Args:
        anim_name: Name of the animation (e.g., "walking", "pointing", "picking")
        bool_drop_neutral_exemplar: Whether to drop neutral exemplar
        bool_fixed_neutral_embedding: Whether to fix neutral embedding
        squared_left_right_euc_dist: Whether to square left-right Euclidean distance
        squared_class_neut_euc_dist: Whether to square class-neutral Euclidean distance
        config: Configuration object
        valid_indices: Set of valid indices for filtering (train or validation subset)

    Returns:
        TripletMining object for the specified animation
    """
    triplet_module = TripletMining(
        bool_drop_neutral_exemplar,
        bool_fixed_neutral_embedding,
        squared_left_right_euc_dist,
        squared_class_neut_euc_dist,
        anim_name,
        config,
        valid_indices=valid_indices
    )
    return triplet_module


def load_embedding_model(checkpoint_path, architecture_variant, config, data_loader, triplet_module):
    """
    Load a trained embedding-based similarity network model from a PyTorch checkpoint file.

    Args:
        checkpoint_path: Path to the saved model weights (.pt file)
        architecture_variant: Architecture variant number
        config: Configuration object containing model parameters
        data_loader: Data loader object for model initialization
        triplet_module: Triplet module for model initialization

    Returns:
        Loaded and initialized embedding network model ready for inference
    """
    # Load checkpoint to inspect metadata and get correct input dimensions
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print(f"Loaded embedding checkpoint: {checkpoint_path}")

        # Check embedding-specific metadata
        if 'use_embeddings' in checkpoint:
            print(f"Confirmed embedding model: {checkpoint['use_embeddings']}")
        if 'input_type' in checkpoint:
            print(f"Input type: {checkpoint['input_type']}")
        if 'embedding_dim' in checkpoint:
            print(f"Input embedding dimension: {checkpoint['embedding_dim']}")
        if 'output_dim' in checkpoint:
            print(f"Output embedding dimension: {checkpoint['output_dim']}")

        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Model trained for {epoch} epochs")

        # Extract the correct input dimension from the saved model
        # Look at the first layer weights to determine input size
        if 'model_state_dict' in checkpoint:
            first_layer_key = None
            for key in checkpoint['model_state_dict'].keys():
                if 'fc1.weight' in key or 'linear1.weight' in key or key.endswith('.weight'):
                    first_layer_key = key
                    break

            if first_layer_key:
                input_dim = checkpoint['model_state_dict'][first_layer_key].shape[1]
                print(f"Detected input dimension from model weights: {input_dim}")
            else:
                # Fallback to checkpoint metadata
                input_dim = checkpoint.get('embedding_dim', (512,))
                if isinstance(input_dim, tuple):
                    input_dim = input_dim[0]
                print(f"Using input dimension from checkpoint metadata: {input_dim}")
        else:
            raise ValueError("No model_state_dict found in checkpoint")

    except Exception as e:
        print(f"Error examining checkpoint: {e}")
        raise

    print(f"Creating network directly with input dimension: {input_dim}")

    # Import the network class directly
    from networks.similarity_network import EmbeddingSimilarityNetworkV0

    # Create the network directly with the correct input dimension
    # Bypass the EmbeddingAwareSimilarityNetwork's automatic dimension detection
    output_dim = config.embedding_refinement_model_output_size
    network = EmbeddingSimilarityNetworkV0(input_dim, output_dim)

    # Move to appropriate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = network.to(device)

    print(f"Created network with input_dim={input_dim}, output_dim={output_dim}")
    print("Network architecture:")
    print(network)

    # Verify the network has the correct input dimension
    first_layer = None
    for name, module in network.named_modules():
        if isinstance(module, torch.nn.Linear):
            first_layer = module
            break

    if first_layer:
        print(f"Network first layer input features: {first_layer.in_features}")
        if first_layer.in_features != input_dim:
            raise ValueError(f"Network input dimension {first_layer.in_features} doesn't match expected {input_dim}")

    # Load the state dict into the properly initialized model
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        network.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded embedding model weights")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        # Try with strict=False
        try:
            network.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded embedding model weights with strict=False (some weights may be missing)")
        except Exception as last_e:
            print(f"Critical error loading embedding model: {last_e}")
            raise

    # Set to evaluation mode
    network.eval()

    return network


def create_embedding_data_loader(similarity_dict, config, valid_indices=None):
    """
    Create a data loader specifically for autoencoder embeddings using EmbeddingSimilarityDataLoader.

    Args:
        similarity_dict: Dictionary of similarity data
        config: Configuration object
        valid_indices: Valid indices for filtering

    Returns:
        EmbeddingSimilarityDataLoader configured for embedding input
    """
    # Import the embedding data loader
    from src.organize_synthetic_data import EmbeddingSimilarityDataLoader

    # Convert TensorFlow tensors to NumPy arrays if needed
    processed_similarity_dict = {}
    for class_tuple, embeddings in similarity_dict.items():
        processed_embeddings = []
        for embedding in embeddings:
            # Handle different embedding types
            if hasattr(embedding, 'numpy'):  # TensorFlow tensor
                processed_embeddings.append(embedding.numpy())
            elif isinstance(embedding, np.ndarray):  # Already NumPy
                processed_embeddings.append(embedding)
            elif hasattr(embedding, 'detach'):  # PyTorch tensor
                processed_embeddings.append(embedding.detach().cpu().numpy())
            else:  # Unknown type, try to convert
                try:
                    processed_embeddings.append(np.array(embedding))
                except:
                    print(f"Warning: Could not convert embedding of type {type(embedding)}")
                    processed_embeddings.append(embedding)
        processed_similarity_dict[class_tuple] = processed_embeddings

    print(f"Processed {len(processed_similarity_dict)} classes, converting tensors to NumPy arrays")

    # Create the data loader with embedding configuration
    data_loader = EmbeddingSimilarityDataLoader(
        [processed_similarity_dict],
        config,
        shuffle=False,  # No shuffle for inference
        valid_indices=[valid_indices] if valid_indices is not None else None
    )

    print(f"Created EmbeddingSimilarityDataLoader:")
    print(f"  Classes: {data_loader.num_classes}")
    print(f"  Embedding dimension: {data_loader.exemplar_dim}")
    print(f"  Batch size: {data_loader.batch_size}")

    return data_loader


def generate_embeddings_from_dataloader(model, data_loader, similarity_dict, anim_name):
    """
    Generate embeddings using a PyTorch embedding model and DataLoader.

    Args:
        model: Loaded PyTorch embedding model
        data_loader: DataLoader object containing autoencoder embeddings
        similarity_dict: Dictionary of similarity data for this animation
        anim_name: Name of the animation being processed

    Returns:
        Dictionary mapping (action_type, effort_tuple) to refined embedding vectors
    """
    print("Generating refined embeddings from autoencoder embeddings...")

    # Get the mapping between batch indices and keys
    embedding_keys = []
    for class_tuple, _value in similarity_dict.items():
        new_key = (anim_name, class_tuple)
        embedding_keys.append(new_key)

    # Process the data through the embedding model
    model.eval()
    embeddings = {}

    with torch.no_grad():
        for batch_idx, (batch_features, _) in enumerate(data_loader):
            # Handle different input shapes for embeddings
            print(f"Initial batch_features.shape: {batch_features.shape}")

            # Handle various embedding input formats
            if len(batch_features.shape) == 3:
                if batch_features.shape[2] == 1:
                    batch_features = batch_features.squeeze(-1)  # Remove last dim if it's 1
                else:
                    # Flatten if it's (batch, seq_len, embedding_dim)
                    batch_features = batch_features.reshape(batch_features.shape[0], -1)

            # If it's still 3D or higher, flatten to 2D
            if len(batch_features.shape) > 2:
                batch_features = batch_features.reshape(batch_features.shape[0], -1)

            batch_features = batch_features.to(torch.float32)
            print(f"Processed embedding input batch_features.shape: {batch_features.shape}")

            batch_embeddings = model(batch_features).cpu().numpy()
            print(f"Embedding output batch_embeddings.shape: {batch_embeddings.shape}")

            # Map embeddings to keys
            batch_start_idx = batch_idx * batch_features.shape[0]
            for i, embedding in enumerate(batch_embeddings):
                key_idx = batch_start_idx + i
                if key_idx < len(embedding_keys):
                    embeddings[embedding_keys[key_idx]] = embedding

    print(f"Generated {len(embeddings)} refined embeddings")
    return embeddings


# def load_similarity_data_from_embeddings(anim_name, config, valid_indices=None, embedding_dir=None):
#     """
#     Load similarity data that contains actual autoencoder embeddings instead of raw motion data.
#
#     Args:
#         anim_name: Animation name (e.g., "walking")
#         config: Configuration object
#         valid_indices: Valid indices for filtering
#         embedding_dir: Directory containing embedding files (if different from config)
#
#     Returns:
#         Dictionary mapping effort tuples to autoencoder embeddings
#     """
#     # Try to find embedding-based similarity files
#     if embedding_dir is None:
#         embedding_dir = config.similarity_exemplars_dir
#
#     # Look for embedding similarity files with various naming patterns
#     embedding_file_patterns = [
#         f"{anim_name}_embedding_similarity_exemplars.pickle",
#         f"{anim_name}_embeddings_similarity_exemplars.pickle",
#         f"{anim_name}_autoencoder_similarity_exemplars.pickle",
#         f"embedding_{anim_name}_similarity_exemplars.pickle"
#     ]
#
#     embedding_file_path = None
#     for pattern in embedding_file_patterns:
#         test_path = os.path.join(embedding_dir, pattern)
#         if os.path.exists(test_path):
#             embedding_file_path = test_path
#             break
#
#     if embedding_file_path is None:
#         print(f"WARNING: No embedding similarity file found for {anim_name}")
#         print(f"Searched for patterns: {embedding_file_patterns}")
#         print(f"In directory: {embedding_dir}")
#         print("Will attempt to use raw motion data (this may cause dimension mismatches)")
#
#         # Fallback to original motion data loading
#         return load_original_similarity_data(anim_name, config, valid_indices)
#
#     print(f"Loading embedding similarity data from: {embedding_file_path}")
#
#     with open(embedding_file_path, "rb") as f:
#         embedding_similarity_dict = pickle.load(f)
#
#     print(f"Loaded {len(embedding_similarity_dict)} embedding classes")
#
#     # Filter by valid_indices if provided
#     if valid_indices is not None:
#         filtered_dict = {k: v for k, v in embedding_similarity_dict.items() if k in valid_indices}
#         print(f"Filtered to {len(filtered_dict)} classes using valid indices")
#         embedding_similarity_dict = filtered_dict
#
#     # Verify we have embeddings, not raw motion data
#     if embedding_similarity_dict:
#         sample_key = next(iter(embedding_similarity_dict.keys()))
#         sample_data = embedding_similarity_dict[sample_key][0]  # First exemplar
#
#         if hasattr(sample_data, 'shape'):
#             print(f"Sample embedding shape: {sample_data.shape}")
#             # Check if this looks like an embedding (1D) vs raw motion (2D)
#             if len(sample_data.shape) == 1:
#                 print("✅ Confirmed: Data appears to be 1D embeddings")
#             elif len(sample_data.shape) == 2 and sample_data.shape[0] > sample_data.shape[1]:
#                 print("⚠️  Warning: Data appears to be 2D motion data, not embeddings")
#             else:
#                 print(f"⚠️  Uncertain: Data shape {sample_data.shape} - please verify")
#
#     return embedding_similarity_dict


def load_original_similarity_data(anim_name, config, valid_indices=None):
    """
    Fallback to load original similarity data (raw motion).

    Args:
        anim_name: Animation name
        config: Configuration object
        valid_indices: Valid indices for filtering

    Returns:
        Dictionary of similarity data
    """
    import src.organize_synthetic_data as osd

    print("Loading original similarity data (raw motion)...")

    # Load using the original method
    anim_similarity_dict_partition = osd.load_similarity_data(True, anim_name, config)
    original_dict = anim_similarity_dict_partition["train"]
    original_dict.update(anim_similarity_dict_partition["test"])

    # Filter by valid_indices if provided
    if valid_indices is not None:
        filtered_dict = {k: v for k, v in original_dict.items() if k in valid_indices}
        print(f"Filtered to {len(filtered_dict)} classes using valid indices")
        return filtered_dict

    return original_dict


def get_raw_features_without_dataloader(anim_name, config, valid_indices=None, balance_class_frame_counts=True):
    """
    Directly load and extract raw features from the pickle file, completely bypassing
    the dataloader to preserve variable-length sequences.

    Args:
        anim_name: Name of the animation to load
        config: Configuration object with paths
        valid_indices: Set of valid indices (effort tuples) to include
        balance_class_frame_counts: Whether to balance class frame counts

    Returns:
        Dictionary mapping (action_type, effort_tuple) to raw feature vectors
    """
    # Directly open the pickle file
    file_path = config.similarity_exemplars_dir + anim_name + "_" + config.similarity_dict_file_name
    print(f"Loading data directly from pickle file: {file_path}")

    with open(file_path, "rb") as f:
        raw_dict = pickle.load(f)
        if balance_class_frame_counts:
            list_balanced_dict = osd.balance_single_exemplar_similarity_classes_by_frame_count(
                [raw_dict], 137)
            raw_dict = list_balanced_dict[0]

    # Print original dictionary information
    print(f"Loaded raw dictionary with {len(raw_dict)} keys")

    # Filter by valid_indices if provided
    if valid_indices is not None:
        original_keys = list(raw_dict.keys())
        filtered_dict = {k: v for k, v in raw_dict.items() if k in valid_indices}
        print(f"Filtered from {len(original_keys)} to {len(filtered_dict)} keys using validation indices")
        raw_dict = filtered_dict

    sample_keys = list(raw_dict.keys())[:2]

    print("Sample entries:")
    for key in sample_keys:
        exemplars = raw_dict[key]
        if exemplars:
            print(f"  Key {key}: Shape {exemplars[0].shape}")

    # Extract raw features without any processing
    raw_features = {}
    for class_tuple, exemplars in raw_dict.items():
        if not exemplars:
            continue

        # Get the raw tensor without conversion
        raw_tensor = exemplars[0]

        # Only convert to numpy for compatibility
        if _TF_AVAILABLE and isinstance(raw_tensor, tf.Tensor):
            numpy_tensor = raw_tensor.numpy().copy()
        elif isinstance(raw_tensor, np.ndarray):
            numpy_tensor = raw_tensor.copy()
        else:
            raise Exception(f"Unsupported type: {type(raw_tensor)}")
        key = (anim_name, class_tuple)
        raw_features[key] = numpy_tensor

    # Print sequence lengths
    lengths = [(key, tensor.shape[0]) for key, tensor in raw_features.items()]
    unique_lengths = sorted(set(length for _, length in lengths))

    print(f"Number of extracted features: {len(raw_features)}")
    print(f"Unique sequence lengths: {unique_lengths}")

    return raw_features


def calculate_pairwise_distances(embeddings):
    """
    Calculate pairwise Euclidean distances between all embeddings,
    with normalization to range [0,1].

    Args:
        embeddings: Dictionary mapping (action_type, effort_tuple) to embedding vectors

    Returns:
        List of tuples (distance, key1, key2) sorted by distance in ascending order
    """
    print("Calculating pairwise L2 distances for refined embeddings...")
    raw_distances = []
    embedding_keys = list(embeddings.keys())
    total_pairs = len(embedding_keys) * (len(embedding_keys) - 1) // 2

    print(f"Processing {total_pairs} pairs across {len(embedding_keys)} embeddings")

    # Optional: Track progress
    progress_interval = max(1, total_pairs // 20)  # Show progress 20 times
    pair_count = 0
    start_time = time.time()

    # First compute all raw distances
    for i in range(len(embedding_keys)):
        for j in range(i + 1, len(embedding_keys)):
            key1 = embedding_keys[i]
            key2 = embedding_keys[j]

            # Calculate Euclidean distance
            embedding1 = embeddings[key1]
            embedding2 = embeddings[key2]
            distance = np.linalg.norm(embedding1 - embedding2)

            # Store raw distance with keys
            raw_distances.append((distance, key1, key2))

            # Update progress
            pair_count += 1
            if pair_count % progress_interval == 0:
                elapsed = time.time() - start_time
                percent = (pair_count / total_pairs) * 100
                # print(f"Progress: {pair_count}/{total_pairs} pairs ({percent:.1f}%) - Elapsed: {elapsed:.1f}s")

    # Find the maximum distance for normalization
    if raw_distances:
        max_distance = max(dist for dist, _, _ in raw_distances)
    else:
        max_distance = 1.0  # Default if no distances

    # Normalize distances to [0,1] range
    normalized_distances = []
    if max_distance > 0:
        for dist, key1, key2 in raw_distances:
            normalized_dist = dist / max_distance
            normalized_distances.append((normalized_dist, key1, key2))
    else:
        # If all distances are zero (unlikely but possible)
        normalized_distances = [(0.0, k1, k2) for _, k1, k2 in raw_distances]

    # Sort by normalized distance (ascending)
    normalized_distances.sort()
    print(f"Calculated {len(normalized_distances)} normalized L2 distances (range [0,1])")

    return normalized_distances


def compute_geodesic_distances(dict_raw_features):
    # First, preprocess and normalize all quaternions in the dictionary
    normalized_features = {}
    print("Preprocessing: Normalizing all quaternions...")

    num_frames = 137  # Standard frame count
    num_joints = 28  # Standard joint count

    # Move this OUTSIDE the main loop - check a few samples first
    for key in list(dict_raw_features.keys())[:3]:  # Check first 3 samples
        sample = dict_raw_features[key]
        if isinstance(sample, list) and len(sample) > 0:
            sample = sample[0]
        if _TF_AVAILABLE and isinstance(sample, tf.Tensor):
            sample = sample.numpy()

        reshaped = sample.reshape(num_frames, num_joints, 4)
        norms = np.sqrt(np.sum(reshaped ** 2, axis=-1))

    # Now process all quaternions
    for key, sample in dict_raw_features.items():
        # Handle list case
        if isinstance(sample, list) and len(sample) > 0:
            sample = sample[0]

        # Convert to numpy if needed
        if _TF_AVAILABLE and isinstance(sample, tf.Tensor):
            sample = sample.numpy()
        elif not isinstance(sample, np.ndarray):
            raise Exception(f"Sample is not a tf.Tensor or np.ndarray but: {type(sample)}")

        # Reshape to (137, 28, 4)
        reshaped = sample.reshape(num_frames, num_joints, 4)

        # Compute norms and normalize
        norms = np.sqrt(np.sum(reshaped ** 2, axis=-1, keepdims=True))
        normalized = reshaped / (norms + 1e-10)  # Add epsilon to avoid division by zero

        # Store the normalized version
        normalized_features[key] = normalized

    print(f"Normalized quaternions for {len(normalized_features)} samples")

    # Now compute distances using the normalized quaternions
    keys = list(normalized_features.keys())
    num_samples = len(keys)

    distances = []
    total_pairs = num_samples * (num_samples - 1) // 2
    progress_interval = max(1, total_pairs // 10)
    pair_count = 0
    start_time = time.time()

    print(f"Computing geodesic distances for {total_pairs} pairs using normalized quaternions...")

    for i, j in combinations(range(num_samples), 2):
        key1 = keys[i]
        key2 = keys[j]

        # Get normalized quaternion data
        q1 = normalized_features[key1]
        q2 = normalized_features[key2]

        # Compute geodesic distances for all frames and joints
        # Note: No need to normalize again since they're already normalized
        distances_per_joint = 2 * np.arccos(np.clip(np.abs(np.sum(q1 * q2, axis=-1)), -1.0, 1.0))
        mean_distance = np.mean(distances_per_joint)

        # Store the result
        distances.append((mean_distance, key1, key2))

        # Update progress
        pair_count += 1
        if pair_count % progress_interval == 0:
            elapsed = time.time() - start_time
            percent = (pair_count / total_pairs) * 100
            eta = (elapsed / pair_count) * (total_pairs - pair_count) if pair_count > 0 else 0
            print(
                f"Progress: {pair_count}/{total_pairs} pairs ({percent:.1f}%) - Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")

    # Sort by distance (ascending)
    distances.sort()
    print(f"Computed {len(distances)} geodesic distances")

    return distances


def main_with_refinement():
    """
    Main execution function that analyzes each animation type separately using embedding-based networks
    and then performs an overall analysis across all animations.
    Additionally supports evaluating only on the validation (out-of-sample) subset.
    """

    # Initialize configuration
    config = Config()

    # Load the learned neutral representations
    neutral_path = Path(config.checkpoint_root_dir) / "learned_neutrals.pkl"

    if neutral_path.exists():
        with open(neutral_path, 'rb') as f:
            neutral_data = pickle.load(f)
        learned_neutrals = neutral_data['neutral_representations']
        print("Loaded learned neutral representations:")
        for anim_name, neutral in learned_neutrals.items():
            print(f"  {anim_name}: shape {neutral.shape}")
    else:
        print("WARNING: No learned_neutrals.pkl found!")
        learned_neutrals = None

    # Set up paths and model parameters for EMBEDDING MODEL
    architecture_variant = 0
    # MOTION_CHECKPOINT_FILE env var lets the experiment runner choose which
    # checkpoint to evaluate without editing this file.
    _default_chk = f"{architecture_variant}_final_embedding_model.pt"
    _chk_file = os.environ.get("MOTION_CHECKPOINT_FILE", _default_chk)
    checkpoint_path = os.path.join(config.checkpoint_root_dir, _chk_file)

    print(f"Using model checkpoint: {checkpoint_path}")

    bool_drop_neutral_exemplar = True
    bool_fixed_neutral_embedding = True
    squared_left_right_euc_dist = False
    squared_class_neut_euc_dist = False

    # Animation types to process
    animations = ["walking", "pointing", "picking"]
    # animations = ["walking"]

    # Create containers for aggregated results
    all_embeddings = {}
    all_raw_features = {}
    all_triplet_modules = []

    # Flag to control whether to evaluate only validation set
    evaluate_only_validation = True  # Set to True to evaluate only the validation set

    # Process each animation individually
    for anim_name in animations:
        print(f"\n{'=' * 70}")
        print(f"PROCESSING ANIMATION: {anim_name.upper()} (EMBEDDING-BASED)")
        print(f"{'=' * 70}")
        print(f"This analysis will evaluate which distance metric better correlates with human perception")
        print(f"for the {anim_name} animation type using embedding-based similarity networks.\n")

        # Get similarity data for train/val split
        anim_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, anim_name, config)
        original_anim_similarity_dict = anim_similarity_dict_partition["train"]
        original_anim_similarity_dict.update(anim_similarity_dict_partition["test"])

        # Create a train/validation split (same as in run_motion_triplet_training.py)
        single_anim_dict_list = [original_anim_similarity_dict]
        balanced_single_anim_dict_list = osd.balance_single_exemplar_similarity_classes_by_frame_count(
            single_anim_dict_list, 137)

        train_indices, val_indices = create_train_val_split(balanced_single_anim_dict_list)

        # Print out information about the split
        print(f"Created train/validation split:")
        print(f"  Training set: {len(train_indices[0])} classes")
        print(f"  Validation set: {len(val_indices[0])} classes")

        # Determine which subset to evaluate
        valid_indices = val_indices[0] if evaluate_only_validation else None
        subset_name = "VALIDATION SUBSET" if evaluate_only_validation else "ALL DATA"

        print(f"\nEvaluating on: {subset_name}")
        if evaluate_only_validation:
            print(f"Validation classes: {len(valid_indices)}")
            print(f"Sample classes: {list(valid_indices)[:3]}")

        # PART 1: Create triplet module for this animation
        print("\n1. CREATING TRIPLET MODULE")
        print("-" * 50)
        triplet_module = create_triplet_module(
            anim_name,
            bool_drop_neutral_exemplar,
            bool_fixed_neutral_embedding,
            squared_left_right_euc_dist,
            squared_class_neut_euc_dist,
            config,
            valid_indices=valid_indices
        )
        if learned_neutrals and anim_name in learned_neutrals:
            neutral = torch.tensor(learned_neutrals[anim_name], dtype=torch.float32)
            triplet_module.neutral_embedding = neutral
            triplet_module.bool_fixed_neutral_embedding = True
        all_triplet_modules.append(triplet_module)  # Store for overall analysis

        # PART 2: Load embedding data and generate refined embeddings
        print("\n2. LOADING EMBEDDING DATA AND GENERATING REFINED EMBEDDINGS")
        print("-" * 50)

        # Load embedding-based similarity data instead of raw motion data
        print("Loading pre-trained embeddings data...")
        try:
            # Try to load embedding-based similarity data first
            embedding_similarity_dict = load_similarity_data_from_embeddings(
                bool_drop=True, anim_name=anim_name, config=config, embedding_dir="../datasets/lma_perform_walking_ae_paired", combination_method="rots_only", force_regenerate=True)["train"]

            # Filter by valid_indices if evaluating only validation set
            if evaluate_only_validation:
                filtered_dict = {k: v for k, v in embedding_similarity_dict.items() if k in valid_indices}
                print(f"Filtered embedding dictionary from {len(embedding_similarity_dict)} to {len(filtered_dict)} classes for validation")
                working_dict = filtered_dict
            else:
                working_dict = embedding_similarity_dict

            balanced_anim_similarity_dict = working_dict  # No need to balance embeddings

        except Exception as e:
            print(f"Error loading embedding similarity data: {e}")
            print("Falling back to original motion data loading...")

            # Fallback to original method
            anim_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, anim_name, config)
            original_anim_similarity_dict = anim_similarity_dict_partition["train"]
            original_anim_similarity_dict.update(anim_similarity_dict_partition["test"])

            # Filter by valid_indices if evaluating only validation set
            if evaluate_only_validation:
                filtered_dict = {k: v for k, v in original_anim_similarity_dict.items() if k in valid_indices}
                print(f"Filtered original dictionary from {len(original_anim_similarity_dict)} to {len(filtered_dict)} classes for validation")
                working_dict = filtered_dict
            else:
                working_dict = original_anim_similarity_dict

            # Balance the working dictionary
            balanced_anim_similarity_dict = osd.balance_single_exemplar_similarity_classes_by_frame_count(
                [working_dict], 137)[0]

        # Create dataloader for the balanced dictionary using EmbeddingSimilarityDataLoader
        balanced_data_loader = create_embedding_data_loader(
            balanced_anim_similarity_dict,
            config,
            valid_indices
        )

        # Load embedding model
        model = load_embedding_model(checkpoint_path, architecture_variant, config, balanced_data_loader, triplet_module)

        # Generate refined embeddings from autoencoder embeddings
        embeddings = generate_embeddings_from_dataloader(model, balanced_data_loader, balanced_anim_similarity_dict,
                                                         anim_name)

        print(f"EXTRACTED {len(embeddings)} REFINED EMBEDDINGS")

        # PART 3: Extract raw features
        raw_features = get_raw_features_without_dataloader(anim_name, config, valid_indices=valid_indices)
        print(f"\n3. EXTRACTED {len(raw_features)} RAW FEATURES")
        print("-" * 50)

        # Store embeddings and raw features for overall analysis
        for key, value in embeddings.items():
            all_embeddings[key] = value

        for key, value in raw_features.items():
            all_raw_features[key] = value

        # PART 4: Calculate distances
        print("\n4. CALCULATING DISTANCES")
        print("-" * 50)

        # Calculate L2 distances for refined embeddings
        embedding_distances = calculate_pairwise_distances(embeddings)

        # Calculate geodesic distances for raw features
        geodesic_distances = compute_geodesic_distances(raw_features)

        # PART 5: Analyze relationships with human perception for embedding method and geodesic distance method
        print("\n5. ANALYZING RELATIONSHIPS WITH HUMAN PERCEPTION")
        print("-" * 50)

        # Collect distances and corresponding inverse comparison values
        print("\nCollecting distance and inverse comparison value pairs...")
        embedding_dist, embedding_inverse_comparison_values = collect_distance_inverse_comparison_value_pairs(
            embedding_distances, triplet_module, get_inverse_direct_comparison_value
        )

        geo_dist, geo_inverse_comparison_values = collect_distance_inverse_comparison_value_pairs(
            geodesic_distances, triplet_module, get_inverse_direct_comparison_value
        )

        # Count valid pairs (with human ratings)
        valid_embedding_pairs = sum(1 for v in embedding_inverse_comparison_values if v is not None)
        valid_geo_pairs = sum(1 for v in geo_inverse_comparison_values if v is not None)

        print(f"Found {valid_embedding_pairs} embedding pairs with human ratings")
        print(f"Found {valid_geo_pairs} geodesic pairs with human ratings")

        # Compare relationship between distances and human perception
        comparison_results = compare_distance_inverse_comparison_value_relationships(
            embedding_dist, embedding_inverse_comparison_values,
            geo_dist, geo_inverse_comparison_values,
            method="geodesic"
        )

        # Print final analysis results
        print("\n" + "=" * 70)

        print(f"{anim_name.upper()} ({subset_name}): PRETRAINED AND REFINED EMBEDDINGS VS GEODESIC DTW DISTANCE")
        print("=" * 70 + "\n")
        print(comparison_results['output_text'])

        # PART 6: Analyze relationships with human perception embedding method and DTW distance method
        print("\n6. ANALYZING RELATIONSHIPS WITH HUMAN PERCEPTION (DTW DISTANCE)")
        print("-" * 50)
        # Get raw features without dataloader
        raw_features = get_raw_features_without_dataloader(anim_name, config, valid_indices=valid_indices,
                                                           balance_class_frame_counts=False)
        # Calculate DTW distances
        dtw_distances = calculate_real_variable_length_dtw(raw_features)
        # Collect distances and corresponding inverse comparison values
        print("\nCollecting DTW distance and inverse comparison value pairs...")
        dtw_dist, dtw_inverse_comparison_values = collect_distance_inverse_comparison_value_pairs(
            dtw_distances, triplet_module, get_inverse_direct_comparison_value
        )
        # Count valid pairs (with human ratings)
        valid_dtw_pairs = sum(1 for v in dtw_inverse_comparison_values if v is not None)
        print(f"Found {valid_dtw_pairs} DTW pairs with human ratings")
        # Compare relationship between distances and human perception
        comparison_results_dtw = compare_distance_inverse_comparison_value_relationships(
            embedding_dist, embedding_inverse_comparison_values,
            dtw_dist, dtw_inverse_comparison_values,
            method="dtw"
        )
        # Print final analysis results
        print("\n" + "=" * 70)
        print(f"{anim_name.upper()} ({subset_name}): PRETRAINED AND REFINED EMBEDDINGS VS RAW FEATURE DTW DISTANCE")
        print("=" * 70 + "\n")
        print(comparison_results_dtw['output_text'])

    # OVERALL ANALYSIS ACROSS ALL ANIMATIONS
    print(f"\n{'=' * 70}")
    subset_label = "VALIDATION SUBSET" if evaluate_only_validation else "ALL DATA"
    print(f"OVERALL ANALYSIS, EMBEDDING METHOD VERSUS GEODESIC DISTANCE, ACROSS ALL ANIMATIONS ({subset_label})")
    print(f"{'=' * 70}")
    print(f"This analysis combines the results from individual animation analyses")
    print(f"(No cross-animation pairs are included since human comparisons only exist within animations)\n")

    # Combine the valid pairs from each individual analysis instead of recomputing
    combined_emb_dist = []
    combined_emb_alphas = []
    combined_geo_dist = []
    combined_geo_alphas = []

    # For each animation, collect the per-animation results
    for anim_name in animations:
        # Get similarity data for train/val split
        anim_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, anim_name, config)
        original_anim_similarity_dict = anim_similarity_dict_partition["train"]
        original_anim_similarity_dict.update(anim_similarity_dict_partition["test"])

        # Create a train/validation split
        single_anim_dict_list = [original_anim_similarity_dict]
        balanced_single_anim_dict_list = osd.balance_single_exemplar_similarity_classes_by_frame_count(
            single_anim_dict_list, 137)

        train_indices, val_indices = create_train_val_split(balanced_single_anim_dict_list)

        # Determine which subset to evaluate
        valid_indices = val_indices[0] if evaluate_only_validation else None

        # Create triplet module with appropriate validation filtering
        triplet_module = create_triplet_module(
            anim_name,
            bool_drop_neutral_exemplar,
            bool_fixed_neutral_embedding,
            squared_left_right_euc_dist,
            squared_class_neut_euc_dist,
            config,
            valid_indices=valid_indices
        )

        # Extract embeddings and raw features for this animation only
        anim_embeddings = {k: v for k, v in all_embeddings.items() if k[0] == anim_name}
        anim_raw_features = {k: v for k, v in all_raw_features.items() if k[0] == anim_name}

        # Calculate distances within this animation
        anim_embedding_distances = calculate_pairwise_distances(anim_embeddings)
        anim_geodesic_distances = compute_geodesic_distances(anim_raw_features)

        # Collect distances and corresponding inverse comparison values
        anim_emb_dist, anim_emb_alphas = collect_distance_inverse_comparison_value_pairs(
            anim_embedding_distances, triplet_module, get_inverse_direct_comparison_value
        )

        anim_geo_dist, anim_geo_alphas = collect_distance_inverse_comparison_value_pairs(
            anim_geodesic_distances, triplet_module, get_inverse_direct_comparison_value
        )

        # Add the valid pairs (filtering out None values)
        for i, alpha in enumerate(anim_emb_alphas):
            if alpha is not None:
                combined_emb_dist.append(anim_emb_dist[i])
                combined_emb_alphas.append(alpha)

        for i, alpha in enumerate(anim_geo_alphas):
            if alpha is not None:
                combined_geo_dist.append(anim_geo_dist[i])
                combined_geo_alphas.append(alpha)

    # Count valid pairs (with human ratings)
    print(f"Combined {len(combined_emb_alphas)} embedding pairs with human ratings")
    print(f"Combined {len(combined_geo_alphas)} geodesic pairs with human ratings")

    # Debug: Print sample pairs from the combined data
    print("\nDEBUG - Sample pairs in combined analysis:")
    for i in range(min(5, len(combined_emb_dist))):
        print(f"Embedding pair {i}: distance={combined_emb_dist[i]:.6f}, alpha={combined_emb_alphas[i]:.6f}")

    for i in range(min(5, len(combined_geo_dist))):
        print(f"Geodesic pair {i}: distance={combined_geo_dist[i]:.6f}, alpha={combined_geo_alphas[i]:.6f}")

    # Compare relationship between distances and human perception
    comparison_results = compare_distance_inverse_comparison_value_relationships(
        combined_emb_dist, combined_emb_alphas,
        combined_geo_dist, combined_geo_alphas,
        method="geodesic"
    )

    # Print final overall analysis results
    print("\n" + "=" * 70)
    print(f"FINAL OVERALL ANALYSIS: EMBEDDING L2 VS RAW FEATURE GEODESIC DISTANCE ({subset_label})")
    print("=" * 70 + "\n")
    print(comparison_results['output_text'])

    # Print concise conclusion
    print("\n" + "=" * 70)
    print(
        f"OVERALL CONCLUSION ({subset_label}): {comparison_results['summary']['stronger_method']} shows a stronger relationship with human perception across all animations")
    print("=" * 70)

    print(f"\n{'=' * 70}")
    print("EMBEDDING-BASED ANALYSIS COMPLETE")
    print(f"Model used: {checkpoint_path}")
    print(f"Evaluation subset: {subset_label}")
    print(f"Animations analyzed: {', '.join(animations)}")
    print(f"{'=' * 70}")


def generate_embeddings_without_refinement(similarity_dict, anim_name):
    """
    Extract pretrained autoencoder embeddings directly without refinement.

    Args:
        similarity_dict: Dictionary of similarity data containing pretrained embeddings
        anim_name: Name of the animation being processed

    Returns:
        Dictionary mapping (action_type, effort_tuple) to pretrained embedding vectors
    """
    print("Extracting pretrained autoencoder embeddings (without refinement)...")

    embeddings = {}

    for class_tuple, exemplars in similarity_dict.items():
        if not exemplars:
            continue

        # Get the first exemplar's embedding
        embedding = exemplars[0]

        # Convert to numpy if needed
        if hasattr(embedding, 'numpy'):  # TensorFlow tensor
            embedding = embedding.numpy()
        elif isinstance(embedding, np.ndarray):  # Already NumPy
            embedding = embedding
        elif hasattr(embedding, 'detach'):  # PyTorch tensor
            embedding = embedding.detach().cpu().numpy()
        else:
            try:
                embedding = np.array(embedding)
            except:
                print(f"Warning: Could not convert embedding of type {type(embedding)}")
                continue

        # Flatten if needed (in case it's 2D with shape (512, 1) or similar)
        if len(embedding.shape) > 1:
            embedding = embedding.flatten()

        key = (anim_name, class_tuple)
        embeddings[key] = embedding

    print(f"Extracted {len(embeddings)} pretrained embeddings")

    # Print sample embedding info
    if embeddings:
        sample_key = next(iter(embeddings.keys()))
        sample_embedding = embeddings[sample_key]
        print(f"Sample embedding shape: {sample_embedding.shape}")
        print(f"Embedding dimension: {sample_embedding.shape[0]}")

    return embeddings


def main_without_refinement():
    """
    Main execution function using pretrained embeddings without refinement.
    """

    # Initialize configuration
    config = Config()

    # Set up paths and parameters
    bool_drop_neutral_exemplar = True
    bool_fixed_neutral_embedding = True
    squared_left_right_euc_dist = False
    squared_class_neut_euc_dist = False

    # Animation types to process
    animations = ["walking", "pointing", "picking"]

    # Create containers for aggregated results
    all_embeddings = {}
    all_raw_features = {}
    all_triplet_modules = []

    # Flag to control whether to evaluate only validation set
    evaluate_only_validation = True

    # Process each animation individually
    for anim_name in animations:
        print(f"\n{'=' * 70}")
        print(f"PROCESSING ANIMATION: {anim_name.upper()} (PRETRAINED EMBEDDINGS WITHOUT REFINEMENT)")
        print(f"{'=' * 70}")
        print(f"This analysis uses pretrained autoencoder embeddings directly")
        print(f"without passing them through the embedding refinement network.\n")

        # Get similarity data for train/val split
        anim_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, anim_name, config)
        original_anim_similarity_dict = anim_similarity_dict_partition["train"]
        original_anim_similarity_dict.update(anim_similarity_dict_partition["test"])

        # Create a train/validation split
        single_anim_dict_list = [original_anim_similarity_dict]
        balanced_single_anim_dict_list = osd.balance_single_exemplar_similarity_classes_by_frame_count(
            single_anim_dict_list, 137)

        train_indices, val_indices = create_train_val_split(balanced_single_anim_dict_list)

        # Print split information
        print(f"Created train/validation split:")
        print(f"  Training set: {len(train_indices[0])} classes")
        print(f"  Validation set: {len(val_indices[0])} classes")

        # Determine which subset to evaluate
        valid_indices = val_indices[0] if evaluate_only_validation else None
        subset_name = "VALIDATION SUBSET" if evaluate_only_validation else "ALL DATA"

        print(f"\nEvaluating on: {subset_name}")
        if evaluate_only_validation:
            print(f"Validation classes: {len(valid_indices)}")

        # PART 1: Create triplet module for this animation
        print("\n1. CREATING TRIPLET MODULE")
        print("-" * 50)
        triplet_module = create_triplet_module(
            anim_name,
            bool_drop_neutral_exemplar,
            bool_fixed_neutral_embedding,
            squared_left_right_euc_dist,
            squared_class_neut_euc_dist,
            config,
            valid_indices=valid_indices
        )
        all_triplet_modules.append(triplet_module)

        # PART 2: Load pretrained embeddings (WITHOUT refinement)
        print("\n2. LOADING PRETRAINED EMBEDDINGS (NO REFINEMENT)")
        print("-" * 50)

        try:
            if anim_name == "picking":
                # Load embedding-based similarity data
                embedding_similarity_dict = load_similarity_data_from_embeddings(
                    bool_drop=True,
                    anim_name=anim_name,
                    config=config,
                    embedding_dir="../datasets/lma_perform_picking_ae_paired",
                    combination_method="rots_only",
                    force_regenerate=True
                )["train"]

            elif anim_name == "pointing":
                embedding_similarity_dict = load_similarity_data_from_embeddings(
                    bool_drop=True,
                    anim_name=anim_name,
                    config=config,
                    embedding_dir="../datasets/lma_perform_pointing_ae_paired",
                    combination_method="rots_only",
                    force_regenerate=True
                )["train"]

            elif anim_name == "walking":
                embedding_similarity_dict = load_similarity_data_from_embeddings(
                    bool_drop=True,
                    anim_name=anim_name,
                    config=config,
                    embedding_dir="../datasets/lma_perform_walking_ae_paired",
                    combination_method="rots_only",
                    force_regenerate=True
                )["train"]

            else:
                raise Exception(f"Unknown animation name: {anim_name}")

            # Filter by valid_indices if evaluating only validation set
            if evaluate_only_validation:
                filtered_dict = {k: v for k, v in embedding_similarity_dict.items() if k in valid_indices}
                print(f"Filtered from {len(embedding_similarity_dict)} to {len(filtered_dict)} classes for validation")
                working_dict = filtered_dict
            else:
                working_dict = embedding_similarity_dict

            # Extract pretrained embeddings directly (skip refinement)
            embeddings = generate_embeddings_without_refinement(working_dict, anim_name)

        except Exception as e:
            print(f"Error loading embedding similarity data: {e}")
            print("Cannot proceed without pretrained embeddings")
            continue

        print(f"EXTRACTED {len(embeddings)} PRETRAINED EMBEDDINGS (UNREFINED)")

        # PART 3: Extract raw features
        raw_features = get_raw_features_without_dataloader(anim_name, config, valid_indices=valid_indices)
        print(f"\n3. EXTRACTED {len(raw_features)} RAW FEATURES")
        print("-" * 50)

        # Store for overall analysis
        for key, value in embeddings.items():
            all_embeddings[key] = value

        for key, value in raw_features.items():
            all_raw_features[key] = value

        # PART 4: Calculate distances
        print("\n4. CALCULATING DISTANCES")
        print("-" * 50)

        # Calculate L2 distances for pretrained embeddings
        embedding_distances = calculate_pairwise_distances(embeddings)

        # Calculate geodesic distances for raw features
        geodesic_distances = compute_geodesic_distances(raw_features)

        # PART 5: Analyze relationships with human perception
        print("\n5. ANALYZING RELATIONSHIPS WITH HUMAN PERCEPTION")
        print("-" * 50)

        # Collect distances and corresponding inverse comparison values
        print("\nCollecting distance and inverse comparison value pairs...")
        embedding_dist, embedding_inverse_comparison_values = collect_distance_inverse_comparison_value_pairs(
            embedding_distances, triplet_module, get_inverse_direct_comparison_value
        )

        geo_dist, geo_inverse_comparison_values = collect_distance_inverse_comparison_value_pairs(
            geodesic_distances, triplet_module, get_inverse_direct_comparison_value
        )

        # Count valid pairs
        valid_embedding_pairs = sum(1 for v in embedding_inverse_comparison_values if v is not None)
        valid_geo_pairs = sum(1 for v in geo_inverse_comparison_values if v is not None)

        print(f"Found {valid_embedding_pairs} pretrained embedding pairs with human ratings")
        print(f"Found {valid_geo_pairs} geodesic pairs with human ratings")

        # Compare relationships
        comparison_results = compare_distance_inverse_comparison_value_relationships(
            embedding_dist, embedding_inverse_comparison_values,
            geo_dist, geo_inverse_comparison_values,
            method="geodesic"
        )

        # Print results
        print("\n" + "=" * 70)
        print(f"{anim_name.upper()} ({subset_name}): PRETRAINED EMBEDDINGS VS RAW FEATURE GEODESIC")
        print("=" * 70 + "\n")
        print(comparison_results['output_text'])

        # PART 6: DTW analysis
        print("\n6. ANALYZING WITH DTW DISTANCE")
        print("-" * 50)

        raw_features_dtw = get_raw_features_without_dataloader(
            anim_name, config, valid_indices=valid_indices, balance_class_frame_counts=False
        )

        dtw_distances = calculate_real_variable_length_dtw(raw_features_dtw)

        dtw_dist, dtw_inverse_comparison_values = collect_distance_inverse_comparison_value_pairs(
            dtw_distances, triplet_module, get_inverse_direct_comparison_value
        )

        valid_dtw_pairs = sum(1 for v in dtw_inverse_comparison_values if v is not None)
        print(f"Found {valid_dtw_pairs} DTW pairs with human ratings")

        comparison_results_dtw = compare_distance_inverse_comparison_value_relationships(
            embedding_dist, embedding_inverse_comparison_values,
            dtw_dist, dtw_inverse_comparison_values,
            method="dtw"
        )

        print("\n" + "=" * 70)
        print(f"{anim_name.upper()} ({subset_name}): PRETRAINED EMBEDDINGS VS RAW FEATURE DTW")
        print("=" * 70 + "\n")
        print(comparison_results_dtw['output_text'])

    # OVERALL ANALYSIS
    print(f"\n{'=' * 70}")
    subset_label = "VALIDATION SUBSET" if evaluate_only_validation else "ALL DATA"
    print(f"OVERALL ANALYSIS: PRETRAINED EMBEDDINGS VS GEODESIC ({subset_label})")
    print(f"{'=' * 70}\n")

    # Combine results from all animations
    combined_emb_dist = []
    combined_emb_alphas = []
    combined_geo_dist = []
    combined_geo_alphas = []

    for anim_name in animations:
        # Recreate validation split for this animation
        anim_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, anim_name, config)
        original_anim_similarity_dict = anim_similarity_dict_partition["train"]
        original_anim_similarity_dict.update(anim_similarity_dict_partition["test"])

        single_anim_dict_list = [original_anim_similarity_dict]
        balanced_single_anim_dict_list = osd.balance_single_exemplar_similarity_classes_by_frame_count(
            single_anim_dict_list, 137)

        train_indices, val_indices = create_train_val_split(balanced_single_anim_dict_list)
        valid_indices = val_indices[0] if evaluate_only_validation else None

        # Create triplet module
        triplet_module = create_triplet_module(
            anim_name,
            bool_drop_neutral_exemplar,
            bool_fixed_neutral_embedding,
            squared_left_right_euc_dist,
            squared_class_neut_euc_dist,
            config,
            valid_indices=valid_indices
        )

        # Extract embeddings and raw features for this animation
        anim_embeddings = {k: v for k, v in all_embeddings.items() if k[0] == anim_name}
        anim_raw_features = {k: v for k, v in all_raw_features.items() if k[0] == anim_name}

        # Calculate distances
        anim_embedding_distances = calculate_pairwise_distances(anim_embeddings)
        anim_geodesic_distances = compute_geodesic_distances(anim_raw_features)

        # Collect pairs
        anim_emb_dist, anim_emb_alphas = collect_distance_inverse_comparison_value_pairs(
            anim_embedding_distances, triplet_module, get_inverse_direct_comparison_value
        )

        anim_geo_dist, anim_geo_alphas = collect_distance_inverse_comparison_value_pairs(
            anim_geodesic_distances, triplet_module, get_inverse_direct_comparison_value
        )

        # Add valid pairs
        for i, alpha in enumerate(anim_emb_alphas):
            if alpha is not None:
                combined_emb_dist.append(anim_emb_dist[i])
                combined_emb_alphas.append(alpha)

        for i, alpha in enumerate(anim_geo_alphas):
            if alpha is not None:
                combined_geo_dist.append(anim_geo_dist[i])
                combined_geo_alphas.append(alpha)

    print(f"Combined {len(combined_emb_alphas)} pretrained embedding pairs with human ratings")
    print(f"Combined {len(combined_geo_alphas)} geodesic pairs with human ratings")

    # Final comparison
    comparison_results = compare_distance_inverse_comparison_value_relationships(
        combined_emb_dist, combined_emb_alphas,
        combined_geo_dist, combined_geo_alphas,
        method="geodesic"
    )

    print("\n" + "=" * 70)
    print(f"FINAL: PRETRAINED EMBEDDINGS VS RAW GEODESIC ({subset_label})")
    print("=" * 70 + "\n")
    print(comparison_results['output_text'])

    print("\n" + "=" * 70)
    print(f"CONCLUSION: {comparison_results['summary']['stronger_method']}")
    print("=" * 70)

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE (PRETRAINED EMBEDDINGS WITHOUT REFINEMENT)")
    print(f"Evaluation subset: {subset_label}")
    print(f"Animations analyzed: {', '.join(animations)}")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    # main_with_refinement()
    main_without_refinement()
