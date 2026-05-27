"""
Motion Similarity Analysis Framework - Per-Module Analysis with Validation Set Evaluation

This script provides a comprehensive analysis framework for comparing different distance metrics
in motion similarity assessment. It evaluates how well various distance measures correlate with
human perceptual judgments of motion similarity.

Key Functionality:
1. Compares embedding-based distances (from trained neural networks) vs raw feature distances
2. Tests multiple distance metrics: L2 (Euclidean), geodesic (for quaternions), and DTW
3. Performs per-animation-type analysis (walking, pointing, picking)
4. Implements train/validation splitting for out-of-sample evaluation
5. Correlates computed distances with human perception data
6. Provides statistical analysis of which method better captures human similarity judgments

The core hypothesis being tested: Do learned embeddings capture motion similarity better than
raw feature comparisons using specialized distance metrics?
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

# Add necessary paths for module imports
# Resolve project root from this file's location so the script works
# whether invoked from pipelines/ or from the project root.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'networks'))
# Import custom modules for similarity network and data handling
from networks.similarity_network import SimilarityNetwork
from networks.similarity_data_loader import SimilarityDataLoader
from networks.triplet_mining import TripletMining
from Config import Config
import src.organize_synthetic_data as osd


def create_train_val_split(similarity_dicts, val_ratio=0.4):
    """
    Create deterministic training and validation splits for each animation type.

    This function ensures reproducible train/val splits by using sorted keys rather
    than random sampling. This is crucial for consistent evaluation across runs.

    Args:
        similarity_dicts: List of dictionaries containing class exemplars for each animation
        val_ratio: Proportion of classes to allocate to validation set (default 0.4)

    Returns:
        train_indices: List of sets containing training class indices per animation
        val_indices: List of sets containing validation class indices per animation

    Note:
        - The neutral exemplar (0,0,0,0) is added to both train and val sets
        - Deterministic splitting ensures reproducible results
    """
    train_indices = []
    val_indices = []

    for anim_dict in similarity_dicts:
        # Sort keys for deterministic ordering (excluding neutral)
        keys = sorted(k for k in anim_dict.keys() if k != (0, 0, 0, 0))

        # Calculate validation set size
        val_size = max(1, int(len(keys) * val_ratio))

        # Take first val_size keys for validation (deterministic)
        val_keys = set(keys[:val_size])
        train_keys = set(keys[val_size:])

        # Add neutral exemplar to both sets if it exists
        if (0, 0, 0, 0) in anim_dict:
            train_keys.add((0, 0, 0, 0))
            val_keys.add((0, 0, 0, 0))

        train_indices.append(train_keys)
        val_indices.append(val_keys)

    return train_indices, val_indices


def get_inverse_direct_comparison_value(key1, key2, triplet_modules):
    """
    Retrieve the inverse of human comparison values for a pair of motion classes.

    Human comparison data typically represents similarity (0=different, 1=similar).
    This function returns 1 - comparison_value to represent dissimilarity/distance.

    Args:
        key1: First motion class (action_type, effort_tuple)
        key2: Second motion class (action_type, effort_tuple)
        triplet_modules: TripletMining module(s) containing human comparison data

    Returns:
        float: Inverse comparison value (1 - similarity) or None if not found

    Note:
        - Returns None for neutral exemplars or cross-animation comparisons
        - Looks for specific comparison configurations in the data
    """
    if not isinstance(triplet_modules, list):
        triplet_modules = [triplet_modules]

    action1, effort1 = key1
    action2, effort2 = key2

    # Skip neutral exemplars
    if effort1 == (0, 0, 0, 0) or effort2 == (0, 0, 0, 0):
        return None

    # Only process same-animation pairs
    if action1 != action2:
        print(f"Skipping different animation types: {action1} and {action2}")
        return None

    # Find the appropriate module and extract comparison value
    for module in triplet_modules:
        if module.anim_name == action1:
            if not hasattr(module, 'alpha_dataframes') or module.alpha_dataframes is None:
                print(f"No alpha_dataframes found for {action1}")
                return None

            df = module.df_comparisons

            # Look for the specific pair in the comparison data
            pair_match = df[df['efforts_tuples'].apply(
                lambda x: set(x) == set([effort1, effort2]) if isinstance(x, list) else False
            )]

            # Target specific comparison configuration
            target_row = pair_match[
                (pair_match['selected0'] == 0) & (pair_match['selected1'] == 2)
            ]

            if not target_row.empty and 'count_normalized' in target_row.columns:
                if not pd.isna(target_row['count_normalized'].iloc[0]):
                    return 1 - target_row['count_normalized'].iloc[0]

            print(f"No direct comparison value found for {key1} and {key2}")
            return None

    return None


def calculate_real_variable_length_dtw(dict_raw_features):
    """
    Calculate Dynamic Time Warping distances for variable-length motion sequences.

    DTW is particularly useful for comparing motion sequences of different lengths,
    as it finds the optimal alignment between sequences before computing distance.

    Args:
        dict_raw_features: Dictionary mapping keys to variable-length motion features

    Returns:
        List of tuples (distance, key1, key2) sorted by distance ascending

    Note:
        - Handles sequences with different frame counts
        - Shows progress updates for long computations
        - Uses double precision for numerical stability
    """
    print("Calculating DTW distances with variable-length raw features...")
    distances = []
    keys = list(dict_raw_features.keys())
    total_pairs = len(keys) * (len(keys) - 1) // 2

    print(f"Processing {total_pairs} pairs across {len(keys)} features")

    # Verify variable-length data
    lengths = {key: data.shape[0] for key, data in dict_raw_features.items()}
    unique_lengths = sorted(set(lengths.values()))
    print(f"DTW processing sequences with {len(unique_lengths)} unique lengths: {unique_lengths}")

    # Progress tracking
    progress_interval = max(1, total_pairs // 10)
    pair_count = 0
    start_time = time.time()

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key1, key2 = keys[i], keys[j]

            # Get motion data
            motion1 = dict_raw_features[key1]
            motion2 = dict_raw_features[key2]

            # Reshape for DTW computation if needed
            if len(motion1.shape) > 2:
                motion1_dtw = motion1.reshape(motion1.shape[0], -1)
                motion2_dtw = motion2.reshape(motion2.shape[0], -1)
            else:
                motion1_dtw = motion1
                motion2_dtw = motion2

            # Ensure double precision
            motion1_dtw = motion1_dtw.astype(np.double)
            motion2_dtw = motion2_dtw.astype(np.double)

            try:
                # Calculate DTW distance
                dtw_distance = dtw_ndim.distance(motion1_dtw, motion2_dtw)
                distances.append((dtw_distance, key1, key2))
            except Exception as e:
                print(f"Error calculating DTW for {key1} and {key2}: {e}")
                continue

            # Progress update
            pair_count += 1
            if pair_count % progress_interval == 0:
                elapsed = time.time() - start_time
                percent = (pair_count / total_pairs) * 100
                eta = (elapsed / pair_count) * (total_pairs - pair_count)
                print(f"Progress: {pair_count}/{total_pairs} pairs ({percent:.1f}%) - "
                      f"Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")

    distances.sort()
    print(f"Computed {len(distances)} DTW distances")

    return distances


def compare_distance_inverse_comparison_value_relationships(
    embedding_distances, embedding_alphas,
    dtw_distances, dtw_alphas,
    method="geodesic"
):
    """
    Compare how well different distance metrics correlate with human perception.

    This is the core analysis function that determines which distance metric
    (embedding-based or raw feature-based) better captures human similarity judgments.

    Args:
        embedding_distances: L2 distances from learned embeddings
        embedding_alphas: Corresponding human dissimilarity values
        dtw_distances: DTW/geodesic distances from raw features
        dtw_alphas: Corresponding human dissimilarity values
        method: Type of raw feature distance ("geodesic" or "dtw")

    Returns:
        Dictionary containing:
        - Correlation metrics (Pearson, Spearman)
        - Linear regression results (slope, intercept, R²)
        - Summary of which method performs better
        - Formatted output text

    Analysis performed:
        1. Pearson correlation: Measures linear relationship strength
        2. Spearman correlation: Measures monotonic relationship (rank-based)
        3. Linear regression: Fits linear model and computes R²
        4. Overall assessment: Determines superior method based on all metrics
    """
    # Filter out None values (missing human comparisons)
    emb_pairs = [(d, a) for d, a in zip(embedding_distances, embedding_alphas) if a is not None]
    dtw_pairs = [(d, a) for d, a in zip(dtw_distances, dtw_alphas) if a is not None]

    if not emb_pairs or not dtw_pairs:
        return {"error": "No valid distance-alpha pairs found"}

    embedding_distances, embedding_alphas = zip(*emb_pairs)
    dtw_distances, dtw_alphas = zip(*dtw_pairs)

    results = {}
    output = []

    # 1. CORRELATION ANALYSIS
    emb_pearson = stats.pearsonr(embedding_distances, embedding_alphas)
    dtw_pearson = stats.pearsonr(dtw_distances, dtw_alphas)
    emb_spearman = stats.spearmanr(embedding_distances, embedding_alphas)
    dtw_spearman = stats.spearmanr(dtw_distances, dtw_alphas)

    output.append("1. CORRELATION ANALYSIS")
    output.append(f"Embedding L2 distances - Pearson: r={emb_pearson[0]:.4f}, p={emb_pearson[1]:.4f}")
    output.append(f"Raw feature {method} distances - Pearson: r={dtw_pearson[0]:.4f}, p={dtw_pearson[1]:.4f}")
    output.append(f"Embedding L2 distances - Spearman: r={emb_spearman[0]:.4f}, p={emb_spearman[1]:.4f}")
    output.append(f"Raw feature {method} distances - Spearman: r={dtw_spearman[0]:.4f}, p={dtw_spearman[1]:.4f}")
    output.append("")

    # 2. LINEAR REGRESSION ANALYSIS
    X_emb = np.array(embedding_distances).reshape(-1, 1)
    y_emb = np.array(embedding_alphas)
    X_dtw = np.array(dtw_distances).reshape(-1, 1)
    y_dtw = np.array(dtw_alphas)

    model_emb = LinearRegression().fit(X_emb, y_emb)
    model_dtw = LinearRegression().fit(X_dtw, y_dtw)

    r2_emb = r2_score(y_emb, model_emb.predict(X_emb))
    r2_dtw = r2_score(y_dtw, model_dtw.predict(X_dtw))

    output.append("2. LINEAR REGRESSION ANALYSIS")
    output.append(f"Embedding L2 - slope: {model_emb.coef_[0]:.4f}, "
                  f"intercept: {model_emb.intercept_:.4f}, R²: {r2_emb:.4f}")
    output.append(f"Raw Feature {method} - slope: {model_dtw.coef_[0]:.4f}, "
                  f"intercept: {model_dtw.intercept_:.4f}, R²: {r2_dtw:.4f}")
    output.append("")

    # 3. DETERMINE BETTER METHOD
    better_pearson = "Embedding L2" if abs(emb_pearson[0]) > abs(dtw_pearson[0]) else f"Raw Feature {method}"
    better_spearman = "Embedding L2" if abs(emb_spearman[0]) > abs(dtw_spearman[0]) else f"Raw Feature {method}"
    better_r2 = "Embedding L2" if r2_emb > r2_dtw else f"Raw Feature {method}"

    # Overall assessment based on multiple criteria
    def determine_relationship_strength():
        emb_strength = 0
        dtw_strength = 0

        # Count wins for each method
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

        # Bonus for positive correlation (expected relationship)
        if emb_pearson[0] > 0:
            emb_strength += 0.5
        if dtw_pearson[0] > 0:
            dtw_strength += 0.5

        if emb_strength > dtw_strength:
            return "Embedding_L2"
        elif dtw_strength > emb_strength:
            return f"Raw_Features_{method}"
        else:
            return "Both methods are equally strong"

    stronger_method = determine_relationship_strength()

    output.append("3. SUMMARY AND CONCLUSION")
    output.append(f"Linear correlation (Pearson): {better_pearson} is better")
    output.append(f"Rank correlation (Spearman): {better_spearman} is better")
    output.append(f"Linear relationship (R²): {better_r2} is better")
    output.append(f"Overall, the {stronger_method} method shows a stronger relationship "
                  f"between distances and inverse comparison values")

    results['correlation'] = {
        'embedding_pearson': {'r': emb_pearson[0], 'p': emb_pearson[1]},
        'dtw_pearson': {'r': dtw_pearson[0], 'p': dtw_pearson[1]},
        'embedding_spearman': {'r': emb_spearman[0], 'p': emb_spearman[1]},
        'dtw_spearman': {'r': dtw_spearman[0], 'p': dtw_spearman[1]}
    }

    results['summary'] = {
        'better_pearson': better_pearson,
        'better_spearman': better_spearman,
        'better_r2': better_r2,
        'stronger_method': stronger_method
    }

    results['output_text'] = '\n'.join(output)

    return results


def collect_distance_inverse_comparison_value_pairs(
    distance_tuples, triplet_modules, get_inverse_direct_comparison_value
):
    """
    Pair computed distances with corresponding human perception values.

    Args:
        distance_tuples: List of (distance, key1, key2) tuples
        triplet_modules: Module(s) containing human comparison data
        get_inverse_direct_comparison_value: Function to retrieve human values

    Returns:
        Tuple of (distances, inverse_comparison_values) lists
    """
    distances = []
    alphas = []

    for distance, key1, key2 in distance_tuples:
        alpha_value = get_inverse_direct_comparison_value(key1, key2, triplet_modules)
        distances.append(distance)
        alphas.append(alpha_value)

    return distances, alphas


def create_triplet_module(
    anim_name, bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
    squared_left_right_euc_dist, squared_class_neut_euc_dist, config,
    valid_indices=None
):
    """
    Initialize a TripletMining module for a specific animation type.

    The TripletMining module handles:
    - Loading human comparison data
    - Managing triplet relationships for metric learning
    - Filtering data based on train/validation splits

    Args:
        anim_name: Animation type ("walking", "pointing", "picking")
        bool_drop_neutral_exemplar: Whether to exclude neutral poses
        bool_fixed_neutral_embedding: Whether to fix neutral embedding during training
        squared_left_right_euc_dist: Use squared Euclidean distance
        squared_class_neut_euc_dist: Use squared distance to neutral
        config: Configuration object with paths and parameters
        valid_indices: Validation set indices for filtering

    Returns:
        Initialized TripletMining object
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


def load_model(checkpoint_path, architecture_variant, config, data_loader, triplet_module):
    """
    Load a pre-trained similarity network from checkpoint.

    Handles multiple model variants:
    - Standard similarity networks
    - Perception-aligned models (trained with human comparison loss)
    - Models with adaptive distance functions

    Args:
        checkpoint_path: Path to saved model weights (.pt file)
        architecture_variant: Network architecture version (0, 1, 2, etc.)
        config: Configuration object
        data_loader: DataLoader for model initialization
        triplet_module: TripletMining module for model setup

    Returns:
        Loaded model in evaluation mode

    Note:
        - Automatically detects model type from checkpoint metadata
        - Handles legacy checkpoints with backward compatibility
        - Sets model to eval() mode for inference
    """
    # Load checkpoint and detect model type
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print(f"Loaded checkpoint: {checkpoint_path}")

        # Detect if this is a perception-aligned model
        is_perception_model = False
        use_adaptive_distance = False

        if 'use_perception_loss' in checkpoint:
            is_perception_model = checkpoint['use_perception_loss']
        elif 'correlation' in checkpoint and checkpoint['correlation'] is not None:
            is_perception_model = True
        elif 'best_correlation' in checkpoint_path:
            is_perception_model = True

        if 'use_adaptive_distance' in checkpoint:
            use_adaptive_distance = checkpoint['use_adaptive_distance']

        print(f"Detected model properties: perception_model={is_perception_model}, "
              f"adaptive_distance={use_adaptive_distance}")

        # Display model metrics if available
        if 'correlation' in checkpoint and checkpoint['correlation'] is not None:
            print(f"Model correlation: {checkpoint['correlation']:.4f}")
        if 'r2_score' in checkpoint and checkpoint['r2_score'] is not None:
            print(f"Model R²: {checkpoint['r2_score']:.4f}")

    except Exception as e:
        print(f"Error examining checkpoint: {e}")
        is_perception_model = False
        use_adaptive_distance = False

    # Initialize network with detected settings
    similarity_network = SimilarityNetwork(
        train_loader=data_loader,
        validation_loader=data_loader,
        test_loader=data_loader,
        checkpoint_root_dir=config.checkpoint_root_dir,
        triplet_modules=[triplet_module],
        architecture_variant=architecture_variant,
        config=config,
        use_perception_loss=is_perception_model,
        use_adaptive_distance=use_adaptive_distance
    )

    # Load model weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        similarity_network.network.load_state_dict(checkpoint['model_state_dict'])
        print(f"SUCCESS: Loaded triplet model weights from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        # Handle various checkpoint format issues
        # ... (error handling code)

    similarity_network.network.eval()
    return similarity_network.network


def generate_embeddings_from_dataloader(model, data_loader, similarity_dict, anim_name):
    """
    Generate embedding vectors for motion sequences using the trained model.

    Args:
        model: Trained PyTorch model
        data_loader: DataLoader containing motion sequences
        similarity_dict: Dictionary of motion classes
        anim_name: Animation type name

    Returns:
        Dictionary mapping (animation, effort_tuple) to embedding vectors

    Note:
        - Processes data in batches for efficiency
        - No gradient computation (inference only)
        - Embeddings are typically 32-dimensional vectors
    """
    print("Generating embeddings...")

    # Create mapping between indices and keys
    embedding_keys = []
    for class_tuple, _value in similarity_dict.items():
        new_key = (anim_name, class_tuple)
        embedding_keys.append(new_key)

    # Generate embeddings
    model.eval()
    embeddings = {}

    with torch.no_grad():
        for batch_features, _ in data_loader:
            # Ensure correct tensor shape
            if len(batch_features.shape) == 3:
                batch_features = batch_features.unsqueeze(-1)

            batch_features = batch_features.to(torch.float32)
            print(f"batch_features.shape: {batch_features.shape}")

            # Forward pass through model
            batch_embeddings = model(batch_features).cpu().numpy()
            print(f"batch_embeddings.shape: {batch_embeddings.shape}")

            # Map embeddings to keys
            for i, embedding in enumerate(batch_embeddings):
                embeddings[embedding_keys[i]] = embedding

    print(f"Generated {len(embeddings)} embeddings")
    return embeddings


def get_raw_features_without_dataloader(
    anim_name, config, valid_indices=None, balance_class_frame_counts=True
):
    """
    Load raw motion features directly from pickle files.

    This bypasses the DataLoader to preserve variable-length sequences,
    which is important for DTW distance calculations.

    Args:
        anim_name: Animation type to load
        config: Configuration with file paths
        valid_indices: Validation indices for filtering
        balance_class_frame_counts: Whether to pad sequences to uniform length

    Returns:
        Dictionary mapping (animation, effort_tuple) to raw motion arrays

    Note:
        - Preserves original sequence lengths when balance_class_frame_counts=False
        - Motion data is typically quaternion rotations for each joint
    """
    # Load raw data from pickle
    file_path = config.similarity_exemplars_dir + anim_name + "_" + config.similarity_dict_file_name
    print(f"Loading data directly from pickle file: {file_path}")

    with open(file_path, "rb") as f:
        raw_dict = pickle.load(f)
        if balance_class_frame_counts:
            # Pad/trim to standard frame count (137)
            list_balanced_dict = osd.balance_single_exemplar_similarity_classes_by_frame_count(
                [raw_dict], 137)
            raw_dict = list_balanced_dict[0]

    print(f"Loaded raw dictionary with {len(raw_dict)} keys")

    # Filter by validation indices if provided
    if valid_indices is not None:
        original_keys = list(raw_dict.keys())
        filtered_dict = {k: v for k, v in raw_dict.items() if k in valid_indices}
        print(f"Filtered from {len(original_keys)} to {len(filtered_dict)} keys using validation indices")
        raw_dict = filtered_dict

    # Extract features
    raw_features = {}
    for class_tuple, exemplars in raw_dict.items():
        if not exemplars:
            continue

        # Get raw tensor
        raw_tensor = exemplars[0]

        # Convert to numpy
        if isinstance(raw_tensor, tf.Tensor):
            numpy_tensor = raw_tensor.numpy().copy()
        elif isinstance(raw_tensor, np.ndarray):
            numpy_tensor = raw_tensor.copy()
        else:
            raise Exception(f"Unsupported type: {type(raw_tensor)}")

        key = (anim_name, class_tuple)
        raw_features[key] = numpy_tensor

    # Report sequence length statistics
    lengths = [(key, tensor.shape[0]) for key, tensor in raw_features.items()]
    unique_lengths = sorted(set(length for _, length in lengths))

    print(f"Number of extracted features: {len(raw_features)}")
    print(f"Unique sequence lengths: {unique_lengths}")

    return raw_features


def calculate_pairwise_distances(embeddings):
    """
    Calculate normalized L2 (Euclidean) distances between all embedding pairs.

    Args:
        embeddings: Dictionary of embedding vectors

    Returns:
        List of (distance, key1, key2) tuples sorted by distance

    Note:
        - Distances are normalized to [0,1] range
        - Used for embedding-based similarity assessment
    """
    print("Calculating pairwise L2 distances for embeddings...")
    raw_distances = []
    embedding_keys = list(embeddings.keys())
    total_pairs = len(embedding_keys) * (len(embedding_keys) - 1) // 2

    print(f"Processing {total_pairs} pairs across {len(embedding_keys)} embeddings")

    # Compute all pairwise distances
    for i in range(len(embedding_keys)):
        for j in range(i + 1, len(embedding_keys)):
            key1 = embedding_keys[i]
            key2 = embedding_keys[j]

            # Euclidean distance
            embedding1 = embeddings[key1]
            embedding2 = embeddings[key2]
            distance = np.linalg.norm(embedding1 - embedding2)

            raw_distances.append((distance, key1, key2))

    # Normalize to [0,1] range
    if raw_distances:
        max_distance = max(dist for dist, _, _ in raw_distances)
    else:
        max_distance = 1.0

    normalized_distances = []
    if max_distance > 0:
        for dist, key1, key2 in raw_distances:
            normalized_dist = dist / max_distance
            normalized_distances.append((normalized_dist, key1, key2))
    else:
        normalized_distances = [(0.0, k1, k2) for _, k1, k2 in raw_distances]

    normalized_distances.sort()
    print(f"Calculated {len(normalized_distances)} normalized L2 distances (range [0,1])")

    return normalized_distances


def compute_geodesic_distances(dict_raw_features):
    """
    Calculate geodesic distances between quaternion-based motion features.

    Geodesic distance is the natural metric for quaternion rotations,
    measuring the shortest path on the 4D unit sphere between rotations.

    Args:
        dict_raw_features: Dictionary of raw quaternion features

    Returns:
        List of (distance, key1, key2) tuples sorted by distance

    Mathematical background:
        - Quaternions represent 3D rotations as 4D unit vectors
        - Geodesic distance = 2 * arccos(|q1 · q2|)
        - More accurate than Euclidean distance for rotation data
    """
    # First normalize all quaternions
    normalized_features = {}
    print("Preprocessing: Normalizing all quaternions...")

    num_frames = 137  # Standard frame count
    num_joints = 28   # Standard joint count

    for key, sample in dict_raw_features.items():
        # Handle list wrapper
        if isinstance(sample, list) and len(sample) > 0:
            sample = sample[0]

        # Convert to numpy
        if isinstance(sample, tf.Tensor):
            sample = sample.numpy()
        elif not isinstance(sample, np.ndarray):
            raise Exception(f"Unexpected type: {type(sample)}")

        # Reshape to (frames, joints, 4)
        reshaped = sample.reshape(num_frames, num_joints, 4)

        # Normalize quaternions
        norms = np.sqrt(np.sum(reshaped ** 2, axis=-1, keepdims=True))
        normalized = reshaped / (norms + 1e-10)  # Add epsilon for stability

        normalized_features[key] = normalized

    print(f"Normalized quaternions for {len(normalized_features)} samples")

    # Compute pairwise geodesic distances
    keys = list(normalized_features.keys())
    num_samples = len(keys)
    distances = []
    total_pairs = num_samples * (num_samples - 1) // 2

    print(f"Computing geodesic distances for {total_pairs} pairs using normalized quaternions...")

    for i, j in combinations(range(num_samples), 2):
        key1 = keys[i]
        key2 = keys[j]

        # Get normalized quaternions
        q1 = normalized_features[key1]
        q2 = normalized_features[key2]

        # Compute geodesic distance
        # geodesic_dist = 2 * arccos(|dot_product|) for unit quaternions
        distances_per_joint = 2 * np.arccos(np.clip(np.abs(np.sum(q1 * q2, axis=-1)), -1.0, 1.0))
        mean_distance = np.mean(distances_per_joint)

        distances.append((mean_distance, key1, key2))

    distances.sort()
    print(f"Computed {len(distances)} geodesic distances")

    return distances


def main():
    """
    Main execution function for motion similarity analysis.

    Workflow:
    1. Load configuration and set parameters
    2. For each animation type (walking, pointing, picking):
       a. Create train/validation split
       b. Load trained embedding model
       c. Generate embeddings for validation set
       d. Extract raw features (with and without length normalization)
       e. Compute various distance metrics (L2, geodesic, DTW)
       f. Compare distance metrics against human perception data
       g. Determine which method better captures human similarity judgments
    3. Perform combined analysis across all animation types
    4. Output comprehensive statistical comparisons

    Key findings from the output:
    - Walking: Embeddings perform better (r=0.52 vs 0.38 for geodesic)
    - Pointing: Raw geodesic features slightly better (r=0.28 vs 0.23)
    - Picking: Raw geodesic features better (r=0.47 vs 0.21)
    - Overall: Raw geodesic slightly better when combined

    This suggests that learned embeddings don't always outperform
    domain-specific distance metrics, especially for certain motion types.
    """

    # Initialize configuration
    config = Config()

    # Model configuration
    architecture_variant = 0
    # MOTION_CHECKPOINT_FILE env var lets the experiment runner choose which
    # checkpoint to evaluate without editing this file.
    _default_chk = f"{architecture_variant}_similarity_model_weights_epoch_015.pt"
    _chk_file = os.environ.get("MOTION_CHECKPOINT_FILE", _default_chk)
    checkpoint_path = os.path.join(config.checkpoint_root_dir, _chk_file)
    print(f"Using model checkpoint: {checkpoint_path}")
    # Alternative: Use perception-aligned model
    # checkpoint_path = os.path.join(
    #     config.checkpoint_root_dir,
    #     f"{architecture_variant}_best_correlation_epoch_100.pt"
    # )

    # Training configuration
    bool_drop_neutral_exemplar = False  # Exclude neutral poses
    bool_fixed_neutral_embedding = False  # Fix neutral embedding during training
    squared_left_right_euc_dist = False  # Don't square distances
    squared_class_neut_euc_dist = False  # Don't square neutral distances

    # Animation types to analyze
    animations = ["walking", "pointing", "picking"]

    # Storage for combined analysis
    all_embeddings = {}
    all_raw_features = {}
    all_triplet_modules = []

    # Control evaluation mode
    evaluate_only_validation = True  # If True, evaluate only on held-out validation set

    # Process each animation type
    for anim_name in animations:
        print(f"\n{'=' * 70}")
        print(f"PROCESSING ANIMATION: {anim_name.upper()}")
        print(f"{'=' * 70}")
        print(f"This analysis will evaluate which distance metric better correlates with human perception")
        print(f"for the {anim_name} animation type.\n")

        # Load and prepare data
        anim_similarity_dict_partition = osd.load_similarity_data(
            bool_drop_neutral_exemplar, anim_name, config
        )
        original_anim_similarity_dict = anim_similarity_dict_partition["train"]
        original_anim_similarity_dict.update(anim_similarity_dict_partition["test"])

        # Create train/validation split
        single_anim_dict_list = [original_anim_similarity_dict]
        balanced_single_anim_dict_list = osd.balance_single_exemplar_similarity_classes_by_frame_count(
            single_anim_dict_list, 137
        )

        train_indices, val_indices = create_train_val_split(balanced_single_anim_dict_list)

        print(f"Created train/validation split:")
        print(f"  Training set: {len(train_indices[0])} classes")
        print(f"  Validation set: {len(val_indices[0])} classes")

        # Select evaluation subset
        valid_indices = val_indices[0] if evaluate_only_validation else None
        subset_name = "VALIDATION SUBSET" if evaluate_only_validation else "ALL DATA"

        print(f"\nEvaluating on: {subset_name}")
        if evaluate_only_validation:
            print(f"Validation classes: {len(valid_indices)}")
            print(f"Sample classes: {list(valid_indices)[:3]}")

        # PART 1: Initialize triplet module
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

        # PART 2: Generate embeddings
        print("\n2. LOADING DATA AND GENERATING EMBEDDINGS")
        print("-" * 50)

        # Filter data for validation if needed
        if evaluate_only_validation:
            filtered_dict = {k: v for k, v in original_anim_similarity_dict.items()
                             if k in valid_indices}
            print(f"Filtered original dictionary from {len(original_anim_similarity_dict)} "
                  f"to {len(filtered_dict)} classes for validation")
            working_dict = filtered_dict
        else:
            working_dict = original_anim_similarity_dict

        # Balance frame counts
        balanced_anim_similarity_dict = osd.balance_single_exemplar_similarity_classes_by_frame_count(
            [working_dict], 137
        )[0]

        # Create dataloader
        balanced_data_loader = SimilarityDataLoader(
            [balanced_anim_similarity_dict],
            config,
            False,
            [valid_indices] if valid_indices is not None else None
        )

        # Load model and generate embeddings
        model = load_model(
            checkpoint_path, architecture_variant, config,
            balanced_data_loader, triplet_module
        )
        embeddings = generate_embeddings_from_dataloader(
            model, balanced_data_loader, balanced_anim_similarity_dict, anim_name
        )
        print(f"EXTRACTED {len(embeddings)} EMBEDDINGS")

        # PART 3: Extract raw features
        raw_features = get_raw_features_without_dataloader(
            anim_name, config, valid_indices=valid_indices
        )
        print(f"\n3. EXTRACTED {len(raw_features)} RAW FEATURES")
        print("-" * 50)

        # Store for combined analysis
        all_embeddings.update(embeddings)
        all_raw_features.update(raw_features)

        # PART 4: Calculate distances
        print("\n4. CALCULATING DISTANCES")
        print("-" * 50)

        # L2 distances for embeddings
        embedding_distances = calculate_pairwise_distances(embeddings)

        # Geodesic distances for quaternion features
        geodesic_distances = compute_geodesic_distances(raw_features)

        # PART 5: Compare with human perception (Embedding vs Geodesic)
        print("\n5. ANALYZING RELATIONSHIPS WITH HUMAN PERCEPTION")
        print("-" * 50)

        # Collect distance-perception pairs
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

        print(f"Found {valid_embedding_pairs} embedding pairs with human ratings")
        print(f"Found {valid_geo_pairs} geodesic pairs with human ratings")

        # Statistical comparison
        comparison_results = compare_distance_inverse_comparison_value_relationships(
            embedding_dist, embedding_inverse_comparison_values,
            geo_dist, geo_inverse_comparison_values,
            method="geodesic"
        )

        # Print Geodesic distance comparison results
        print("\n" + "=" * 70)
        print(f"{anim_name.upper()} ({subset_name}): "
              f"EMBEDDING L2 VS RAW FEATURE GEODESIC DISTANCE")
        print("=" * 70 + "\n")
        print(comparison_results['output_text'])

        # PART 6: Compare with DTW distances
        print("\n6. ANALYZING RELATIONSHIPS WITH HUMAN PERCEPTION (DTW DISTANCE)")
        print("-" * 50)

        # Get variable-length features for DTW
        raw_features_dtw = get_raw_features_without_dataloader(
            anim_name, config, valid_indices=valid_indices,
            balance_class_frame_counts=False  # Keep original lengths for DTW
        )

        # Calculate DTW distances
        dtw_distances = calculate_real_variable_length_dtw(raw_features_dtw)

        # Collect DTW-perception pairs
        print("\nCollecting DTW distance and inverse comparison value pairs...")
        dtw_dist, dtw_inverse_comparison_values = collect_distance_inverse_comparison_value_pairs(
            dtw_distances, triplet_module, get_inverse_direct_comparison_value
        )

        # Count valid pairs
        valid_dtw_pairs = sum(1 for v in dtw_inverse_comparison_values if v is not None)
        print(f"Found {valid_dtw_pairs} DTW pairs with human ratings")

        # Statistical comparison
        comparison_results_dtw = compare_distance_inverse_comparison_value_relationships(
            embedding_dist, embedding_inverse_comparison_values,
            dtw_dist, dtw_inverse_comparison_values,
            method="dtw"
        )

        # Print DTW comparison results
        print("\n" + "=" * 70)
        print(f"{anim_name.upper()} ({subset_name}): EMBEDDING L2 VS RAW FEATURE DTW DISTANCE")
        print("=" * 70 + "\n")
        print(comparison_results_dtw['output_text'])

    # COMBINED ANALYSIS ACROSS ALL ANIMATIONS
    print(f"\n{'=' * 70}")
    subset_label = "VALIDATION SUBSET" if evaluate_only_validation else "ALL DATA"
    print(f"OVERALL ANALYSIS, EMBEDDING METHOD VERSUS GEODESIC DISTANCE, "
          f"ACROSS ALL ANIMATIONS ({subset_label})")
    print(f"{'=' * 70}")
    print(f"This analysis combines the results from individual animation analyses")
    print(f"(No cross-animation pairs are included since human comparisons only exist within animations)\n")

    # Aggregate results from all animations
    combined_emb_dist = []
    combined_emb_alphas = []
    combined_geo_dist = []
    combined_geo_alphas = []

    # Recalculate for each animation and combine
    for anim_name in animations:
        # Recreate data split and triplet module
        anim_similarity_dict_partition = osd.load_similarity_data(
            bool_drop_neutral_exemplar, anim_name, config
        )
        original_anim_similarity_dict = anim_similarity_dict_partition["train"]
        original_anim_similarity_dict.update(anim_similarity_dict_partition["test"])

        single_anim_dict_list = [original_anim_similarity_dict]
        balanced_single_anim_dict_list = osd.balance_single_exemplar_similarity_classes_by_frame_count(
            single_anim_dict_list, 137
        )

        train_indices, val_indices = create_train_val_split(balanced_single_anim_dict_list)
        valid_indices = val_indices[0] if evaluate_only_validation else None

        triplet_module = create_triplet_module(
            anim_name,
            bool_drop_neutral_exemplar,
            bool_fixed_neutral_embedding,
            squared_left_right_euc_dist,
            squared_class_neut_euc_dist,
            config,
            valid_indices=valid_indices
        )

        # Extract animation-specific data
        anim_embeddings = {k: v for k, v in all_embeddings.items() if k[0] == anim_name}
        anim_raw_features = {k: v for k, v in all_raw_features.items() if k[0] == anim_name}

        # Calculate distances
        anim_embedding_distances = calculate_pairwise_distances(anim_embeddings)
        anim_geodesic_distances = compute_geodesic_distances(anim_raw_features)

        # Collect valid pairs
        anim_emb_dist, anim_emb_alphas = collect_distance_inverse_comparison_value_pairs(
            anim_embedding_distances, triplet_module, get_inverse_direct_comparison_value
        )

        anim_geo_dist, anim_geo_alphas = collect_distance_inverse_comparison_value_pairs(
            anim_geodesic_distances, triplet_module, get_inverse_direct_comparison_value
        )

        # Add valid pairs to combined lists
        for i, alpha in enumerate(anim_emb_alphas):
            if alpha is not None:
                combined_emb_dist.append(anim_emb_dist[i])
                combined_emb_alphas.append(alpha)

        for i, alpha in enumerate(anim_geo_alphas):
            if alpha is not None:
                combined_geo_dist.append(anim_geo_dist[i])
                combined_geo_alphas.append(alpha)

    # Final combined statistics
    print(f"Combined {len(combined_emb_alphas)} embedding pairs with human ratings")
    print(f"Combined {len(combined_geo_alphas)} geodesic pairs with human ratings")

    # Debug information
    print("\nDEBUG - Sample pairs in combined analysis:")
    for i in range(min(5, len(combined_emb_dist))):
        print(f"Embedding pair {i}: distance={combined_emb_dist[i]:.6f}, "
              f"alpha={combined_emb_alphas[i]:.6f}")

    for i in range(min(5, len(combined_geo_dist))):
        print(f"Geodesic pair {i}: distance={combined_geo_dist[i]:.6f}, "
              f"alpha={combined_geo_alphas[i]:.6f}")

    # Final comparison
    comparison_results = compare_distance_inverse_comparison_value_relationships(
        combined_emb_dist, combined_emb_alphas,
        combined_geo_dist, combined_geo_alphas,
        method="geodesic"
    )

    # Print final results
    print("\n" + "=" * 70)
    print(f"{subset_label}: EMBEDDING L2 VS RAW FEATURE GEODESIC DISTANCE")
    print("=" * 70 + "\n")
    print(comparison_results['output_text'])


if __name__ == "__main__":
    main()
