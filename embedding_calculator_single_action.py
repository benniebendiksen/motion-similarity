"""
Motion Similarity Analysis Framework - Per-Module Analysis

This script analyzes motion data similarity for each animation type separately,
comparing embedding-based and raw feature-based approaches including geodesic distance
calculations for quaternion data. It processes each animation module individually,
providing comprehensive analysis of how well each method correlates with human perception.

The framework includes:
1. Loading and processing motion data for each animation type
2. Generating embeddings using a trained neural network
3. Extracting variable-length raw features directly from pickle files
4. Calculating L2 distances for embeddings and geodesic distances for raw features
5. Analyzing the relationship between distance metrics and human comparison data
6. Comparing the different approaches for each animation type separately

"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from dtaidistance import dtw_ndim
import tensorflow as tf
import torch
import pickle
import time
from itertools import combinations

# Add necessary paths
curr_path = os.getcwd()
sys.path.append(curr_path)
sys.path.append(os.path.join(curr_path, 'networks'))

# Import required modules
from networks.similarity_network import SimilarityNetwork
from networks.similarity_data_loader import SimilarityDataLoader
from networks.triplet_mining import TripletMining
from bvh_visualizing.datasetLoad import BVHDataset
from bvh_visualizing.bvhvisualize import DualBVHAnimator
from bvh_visualizing.bvh import BVH
from Config import Config
import src.organize_synthetic_data as osd


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


def collect_distance_inverse_comparison_value_pairs(distance_tuples, triplet_modules, get_inverse_comparison_value_func):
    """
    Collect distance and inverse comparison values from distance tuples.

    Args:
        distance_tuples: List of tuples (distance, key1, key2)
        triplet_modules: Triplet module object or list of triplet module objects
        get_inverse_comparison_value_func: Function to get inverse comparison value for a pair of keys

    Returns:
        tuple: (distances, inverse_comparison_values)
    """
    distances = []
    alphas = []

    for distance, key1, key2 in distance_tuples:
        alpha_value = get_inverse_comparison_value_func(key1, key2, triplet_modules)
        distances.append(distance)
        alphas.append(alpha_value)

    return distances, alphas


def create_triplet_module(anim_name, bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
                          squared_left_right_euc_dist, squared_class_neut_euc_dist, config):
    """
    Create a single triplet module for the specified animation.

    Args:
        anim_name: Name of the animation (e.g., "walking", "pointing", "picking")
        bool_drop_neutral_exemplar: Whether to drop neutral exemplar
        bool_fixed_neutral_embedding: Whether to fix neutral embedding
        squared_left_right_euc_dist: Whether to square left-right Euclidean distance
        squared_class_neut_euc_dist: Whether to square class-neutral Euclidean distance
        config: Configuration object

    Returns:
        TripletMining object for the specified animation
    """
    triplet_module = TripletMining(
        bool_drop_neutral_exemplar,
        bool_fixed_neutral_embedding,
        squared_left_right_euc_dist,
        squared_class_neut_euc_dist,
        anim_name,
        config
    )
    return triplet_module


def load_model(checkpoint_path, architecture_variant, config, data_loader, triplet_module):
    """
    Load a trained similarity network model from a PyTorch checkpoint file.

    Args:
        checkpoint_path: Path to the saved model weights (.pt file)
        architecture_variant: Architecture variant number for model configuration
        config: Configuration object containing model parameters
        data_loader: Data loader object for model initialization
        triplet_module: Triplet module for model initialization

    Returns:
        Loaded and initialized network model ready for inference
    """
    # Create and load the similarity network
    similarity_network = SimilarityNetwork(
        train_loader=data_loader,
        validation_loader=data_loader,
        test_loader=data_loader,
        checkpoint_root_dir=config.checkpoint_root_dir,
        triplet_modules=[triplet_module],
        architecture_variant=architecture_variant,
        config=config
    )

    # Load only the model state dict and not the optimizer state dict
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    similarity_network.network.load_state_dict(checkpoint['model_state_dict'])
    similarity_network.network.eval()
    print(f"Model loaded from {checkpoint_path}")

    return similarity_network.network


def generate_embeddings_from_dataloader(model, data_loader, similarity_dict, anim_name):
    """
    Generate embeddings using a PyTorch model and DataLoader.

    Args:
        model: Loaded PyTorch model
        data_loader: DataLoader object containing motion data
        similarity_dict: Dictionary of similarity data for this animation
        anim_name: Name of the animation being processed

    Returns:
        Dictionary mapping (action_type, effort_tuple) to embedding vectors
    """
    print("Generating embeddings...")

    # Get the mapping between batch indices and keys
    embedding_keys = []
    for class_tuple, _value in similarity_dict.items():
        new_key = (anim_name, class_tuple)
        embedding_keys.append(new_key)

    # Process the data through the model
    model.eval()
    embeddings = {}

    with torch.no_grad():
        for batch_features, _ in data_loader:
            if len(batch_features.shape) == 3:
                batch_features = batch_features.unsqueeze(-1)

            batch_features = batch_features.to(torch.float32)
            print(f"batch_features.shape: {batch_features.shape}")
            batch_embeddings = model(batch_features).cpu().numpy()
            print(f"batch_embeddings.shape: {batch_embeddings.shape}")

            for i, embedding in enumerate(batch_embeddings):
                embeddings[embedding_keys[i]] = embedding

    print(f"Generated {len(embeddings)} embeddings")
    return embeddings


def get_raw_features_without_dataloader(anim_name, config):
    """
    Directly load and extract raw features from the pickle file, completely bypassing
    the dataloader to preserve variable-length sequences.

    Args:
        anim_name: Name of the animation to load
        config: Configuration object with paths

    Returns:
        Dictionary mapping (action_type, effort_tuple) to raw feature vectors
    """
    # Directly open the pickle file
    file_path = config.similarity_exemplars_dir + anim_name + "_" + config.similarity_dict_file_name
    print(f"Loading data directly from pickle file: {file_path}")

    with open(file_path, "rb") as f:
        raw_dict = pickle.load(f)
        list_balanced_dict = osd.balance_single_exemplar_similarity_classes_by_frame_count(
            [raw_dict], 137)
        raw_dict = list_balanced_dict[0]

    # Print original dictionary information
    print(f"Loaded raw dictionary with {len(raw_dict)} keys")
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
        if isinstance(raw_tensor, tf.Tensor):
            numpy_tensor = raw_tensor.numpy().copy()
        elif isinstance(raw_tensor, np.ndarray):
            numpy_tensor = raw_tensor.copy()
        else:
            raise Exception(f"Unsupported type: {type(raw_tensor)}")
        key = (anim_name, class_tuple)
        raw_features[key] = numpy_tensor

    # Verify sequence lengths
    lengths = [(key, tensor.shape[0]) for key, tensor in raw_features.items()]
    unique_lengths = sorted(set(length for _, length in lengths))

    print(f"Number of extracted features: {len(raw_features)}")
    print(f"Unique sequence lengths: {unique_lengths}")

    return raw_features


# def calculate_pairwise_distances(embeddings):
#     """
#     Calculate pairwise squared Euclidean distances between all embeddings.
#
#     Args:
#         embeddings: Dictionary mapping (action_type, effort_tuple) to embedding vectors
#
#     Returns:
#         List of tuples (squared_distance, key1, key2) sorted by squared distance in ascending order
#     """
#     print("Calculating pairwise squared L2 distances for embeddings...")
#     distances = []
#     embedding_keys = list(embeddings.keys())
#     total_pairs = len(embedding_keys) * (len(embedding_keys) - 1) // 2
#
#     print(f"Processing {total_pairs} pairs across {len(embedding_keys)} embeddings")
#
#     # Optional: Track progress
#     progress_interval = max(1, total_pairs // 20)  # Show progress 20 times
#     pair_count = 0
#     start_time = time.time()
#
#     for i in range(len(embedding_keys)):
#         for j in range(i + 1, len(embedding_keys)):
#             key1 = embedding_keys[i]
#             key2 = embedding_keys[j]
#
#             # Calculate squared Euclidean distance
#             embedding1 = embeddings[key1]
#             embedding2 = embeddings[key2]
#             # Use sum of squared differences instead of norm
#             squared_distance = np.sum((embedding1 - embedding2) ** 2)
#
#             # Store as tuple (squared_distance, key1, key2)
#             distances.append((squared_distance, key1, key2))
#
#             # Update progress
#             pair_count += 1
#             if pair_count % progress_interval == 0:
#                 elapsed = time.time() - start_time
#                 percent = (pair_count / total_pairs) * 100
#                 # print(f"Progress: {pair_count}/{total_pairs} pairs ({percent:.1f}%) - Elapsed: {elapsed:.1f}s")
#
#     # Sort by squared distance (ascending)
#     distances.sort()
#     print(f"Calculated {len(distances)} squared L2 distances")
#
#     return distances


def calculate_pairwise_distances(embeddings):
    """
    Calculate pairwise Euclidean distances between all embeddings.

    Args:
        embeddings: Dictionary mapping (action_type, effort_tuple) to embedding vectors

    Returns:
        List of tuples (distance, key1, key2) sorted by distance in ascending order
    """
    print("Calculating pairwise L2 distances for embeddings...")
    distances = []
    embedding_keys = list(embeddings.keys())
    total_pairs = len(embedding_keys) * (len(embedding_keys) - 1) // 2

    print(f"Processing {total_pairs} pairs across {len(embedding_keys)} embeddings")

    # Optional: Track progress
    progress_interval = max(1, total_pairs // 20)  # Show progress 20 times
    pair_count = 0
    start_time = time.time()

    for i in range(len(embedding_keys)):
        for j in range(i + 1, len(embedding_keys)):
            key1 = embedding_keys[i]
            key2 = embedding_keys[j]

            # Calculate Euclidean distance
            embedding1 = embeddings[key1]
            embedding2 = embeddings[key2]
            distance = np.linalg.norm(embedding1 - embedding2)

            # Store as tuple (distance, key1, key2)
            distances.append((distance, key1, key2))

            # Update progress
            pair_count += 1
            if pair_count % progress_interval == 0:
                elapsed = time.time() - start_time
                percent = (pair_count / total_pairs) * 100
                # print(f"Progress: {pair_count}/{total_pairs} pairs ({percent:.1f}%) - Elapsed: {elapsed:.1f}s")

    # Sort by distance (ascending)
    distances.sort()
    print(f"Calculated {len(distances)} L2 distances")

    return distances

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
        if isinstance(sample, tf.Tensor):
            sample = sample.numpy()

        reshaped = sample.reshape(num_frames, num_joints, 4)
        norms = np.sqrt(np.sum(reshaped ** 2, axis=-1))

        # print(f"Quaternion norms for {key}:")
        # print(f"  Min norm: {np.min(norms):.6f}")
        # print(f"  Max norm: {np.max(norms):.6f}")
        # print(f"  Mean norm: {np.mean(norms):.6f}")
        # print(f"  Std norm: {np.std(norms):.6f}")

    # Now process all quaternions
    for key, sample in dict_raw_features.items():
        # Handle list case
        if isinstance(sample, list) and len(sample) > 0:
            sample = sample[0]

        # Convert to numpy if needed
        if isinstance(sample, tf.Tensor):
            sample = sample.numpy()
        elif not isinstance(sample, np.ndarray):
            raise Exception(f"Sample is not a tf.Tensor or np.ndarray but: {type(sample)}")

        # Reshape to (137, 28, 4)
        reshaped = sample.reshape(num_frames, num_joints, 4)

        # # Compute norms and normalize
        # norms = np.sqrt(np.sum(reshaped ** 2, axis=-1, keepdims=True))
        # normalized = reshaped / (norms + 1e-10)  # Add epsilon to avoid division by zero

        # Store the normalized version
        normalized_features[key] = reshaped

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
        # pair_count += 1
        # if pair_count % progress_interval == 0:
        #     elapsed = time.time() - start_time
        #     percent = (pair_count / total_pairs) * 100
        #     eta = (elapsed / pair_count) * (total_pairs - pair_count) if pair_count > 0 else 0
        #     print(
        #         f"Progress: {pair_count}/{total_pairs} pairs ({percent:.1f}%) - Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")

    # Sort by distance (ascending)
    distances.sort()
    print(f"Computed {len(distances)} geodesic distances")

    return distances


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
            # pair_count += 1
            # if pair_count % progress_interval == 0:
            #     elapsed = time.time() - start_time
            #     percent = (pair_count / total_pairs) * 100
            #     eta = (elapsed / pair_count) * (total_pairs - pair_count) if pair_count > 0 else 0
            #     print(f"Progress: {pair_count}/{total_pairs} pairs ({percent:.1f}%) - Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")

    # Sort by distance (ascending)
    distances.sort()
    print(f"Computed {len(distances)} DTW distances")

    return distances


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
        return None

    # Find the appropriate module for this animation
    for module in triplet_modules:
        if module.anim_name == action1:
            # Check if alpha_dataframes exists
            if not hasattr(module, 'alpha_dataframes') or module.alpha_dataframes is None:
                print(f"No alpha_dataframes found for {action1}")
                return None

            # Look for the pair in the efforts_tuples column
            df = module.alpha_dataframes

            pair_match = df[df['efforts_tuples'].apply(lambda x:
                                                       set(x) == set([effort1, effort2]) if isinstance(x,
                                                                                                       list) else False)]
            # print(f"Pair match found: {pair_match}")
            if pair_match.iloc[0]['selected0'] == 1 or pair_match.iloc[0]['selected1'] == 1:
                return None

            if not pair_match.empty:
                row = pair_match.iloc[0]

                # Check if we have a direct comparison value
                if 'direct_comparison_value' in row and not pd.isna(row['direct_comparison_value']):
                    # Return 1 minus the comparison value
                    return 1 - row['direct_comparison_value']

            return None

            # df = module.df_comparisons
            # pair_match = df[df['efforts_tuples'].apply(lambda x:
            #                                            set(x) == set([effort1, effort2]) if isinstance(x,
            #                                                                                            list) else False)]
            # if not pair_match.empty:
            #     row = pair_match.iloc[0]
            #
            #     # Check if we have a direct comparison value
            #     if 'count_normalized' in row and not pd.isna(row['count_normalized']):
            #         # Return 1 minus the comparison value
            #         return 1 - row['count_normalized']

            return None

    return None


def main():
    """
    Main execution function that analyzes each animation type separately
    and then performs an overall analysis across all animations.
    """
    # Initialize configuration
    config = Config()

    # Set up paths and model parameters
    architecture_variant = 0
    #checkpoint_path = os.path.join(config.checkpoint_root_dir,
    #                               f"{architecture_variant}_similarity_model_weights_epoch_079.pt")
    # checkpoint_path = os.path.join(config.checkpoint_root_dir,
    #                                f"{architecture_variant}_similarity_model_weights_epoch_033.pt")
    checkpoint_path = os.path.join(config.checkpoint_root_dir,
                                   f"{architecture_variant}_similarity_model_weights_epoch_032.pt")

    bool_drop_neutral_exemplar = False
    bool_fixed_neutral_embedding = False
    squared_left_right_euc_dist = False
    squared_class_neut_euc_dist = False

    # Animation types to process
    animations = ["walking", "pointing", "picking"]

    # Create containers for aggregated results
    all_embeddings = {}
    all_raw_features = {}
    all_triplet_modules = []

    # Process each animation individually
    for anim_name in animations:
        print(f"\n{'=' * 70}")
        print(f"PROCESSING ANIMATION: {anim_name.upper()}")
        print(f"{'=' * 70}")
        print(f"This analysis will evaluate which distance metric better correlates with human perception")
        print(f"for the {anim_name} animation type.\n")

        # PART 1: Create triplet module for this animation
        print("\n1. CREATING TRIPLET MODULE")
        print("-" * 50)
        triplet_module = create_triplet_module(
            anim_name,
            bool_drop_neutral_exemplar,
            bool_fixed_neutral_embedding,
            squared_left_right_euc_dist,
            squared_class_neut_euc_dist,
            config
        )
        # print(f"Created triplet module for {anim_name}")
        all_triplet_modules.append(triplet_module)  # Store for overall analysis

        # PART 2: Load data and generate embeddings
        print("\n2. LOADING DATA AND GENERATING EMBEDDINGS")
        print("-" * 50)

        # Load similarity data for this animation
        anim_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, anim_name, config)

        # Store original dictionary for raw feature extraction
        original_anim_similarity_dict = anim_similarity_dict_partition["train"]

        # Create a balanced copy for embedding generation
        balanced_anim_similarity_dict = dict(original_anim_similarity_dict)  # Deep copy
        balanced_anim_similarity_dict = osd.balance_single_exemplar_similarity_classes_by_frame_count(
            [balanced_anim_similarity_dict], 137)[0]

        # Create dataloader for the balanced dictionary
        balanced_data_loader = SimilarityDataLoader([balanced_anim_similarity_dict], config, False)

        # Load model
        model = load_model(checkpoint_path, architecture_variant, config, balanced_data_loader, triplet_module)

        # Generate embeddings
        embeddings = generate_embeddings_from_dataloader(model, balanced_data_loader, balanced_anim_similarity_dict,
                                                         anim_name)

        # PART 3: Extract raw features
        print("\n3. EXTRACTING RAW FEATURES")
        print("-" * 50)
        raw_features = get_raw_features_without_dataloader(anim_name, config)

        # Store embeddings and raw features for overall analysis
        for key, value in embeddings.items():
            all_embeddings[key] = value

        for key, value in raw_features.items():
            all_raw_features[key] = value

        # PART 4: Calculate distances
        print("\n4. CALCULATING DISTANCES")
        print("-" * 50)

        # Calculate L2 distances for embeddings
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
        print(f"FINAL ANALYSIS FOR {anim_name.upper()}: EMBEDDING L2 VS RAW FEATURE GEODESIC DISTANCE")
        print("=" * 70 + "\n")
        print(comparison_results['output_text'])

        # Print concise conclusion
        print("\n" + "=" * 70)
        print(
            f"CONCLUSION FOR {anim_name.upper()}: {comparison_results['summary']['stronger_method']} shows a stronger relationship with human perception")
        print("=" * 70)

    # OVERALL ANALYSIS ACROSS ALL ANIMATIONS
    print(f"\n{'=' * 70}")
    print(f"OVERALL ANALYSIS ACROSS ALL ANIMATIONS")
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
        # Recreate the necessary objects to get the pairs
        triplet_module = create_triplet_module(
            anim_name,
            bool_drop_neutral_exemplar,
            bool_fixed_neutral_embedding,
            squared_left_right_euc_dist,
            squared_class_neut_euc_dist,
            config
        )

        # Load similarity data for this animation
        anim_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, anim_name, config)
        original_anim_similarity_dict = anim_similarity_dict_partition["train"]

        # Create a balanced copy for embedding generation
        balanced_anim_similarity_dict = dict(original_anim_similarity_dict)  # Deep copy
        balanced_anim_similarity_dict = osd.balance_single_exemplar_similarity_classes_by_frame_count(
            [balanced_anim_similarity_dict], 137)[0]

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
    print("FINAL OVERALL ANALYSIS: EMBEDDING L2 VS RAW FEATURE GEODESIC DISTANCE")
    print("=" * 70 + "\n")
    print(comparison_results['output_text'])

    # Print concise conclusion
    print("\n" + "=" * 70)
    print(
        f"OVERALL CONCLUSION: {comparison_results['summary']['stronger_method']} shows a stronger relationship with human perception across all animations")
    print("=" * 70)

# def main():
#     """
#     Main execution function that analyzes each animation type separately:
#     1. Load and process data for the animation type
#     2. Generate embeddings using a trained neural network
#     3. Extract variable-length raw features directly from pickle files
#     4. Calculate L2 distances for embeddings and geodesic distances for raw features
#     5. Analyze which approach better correlates with human perception
#     """
#     # Initialize configuration
#     config = Config()
#
#     # Set up paths and model parameters
#     architecture_variant = 0
#     checkpoint_path = os.path.join(config.checkpoint_root_dir,
#                                    f"{architecture_variant}_similarity_model_weights_epoch_079.pt")
#
#     bool_drop_neutral_exemplar = False
#     bool_fixed_neutral_embedding = False
#     squared_left_right_euc_dist = False
#     squared_class_neut_euc_dist = True
#
#     # Animation types to process
#     animations = ["walking", "pointing", "picking"]
#
#     for anim_name in animations:
#         print(f"\n{'=' * 70}")
#         print(f"PROCESSING ANIMATION: {anim_name.upper()}")
#         print(f"{'=' * 70}")
#         print(f"This analysis will evaluate which distance metric better correlates with human perception")
#         print(f"for the {anim_name} animation type.\n")
#
#         # PART 1: Create triplet module for this animation
#         print("\n1. CREATING TRIPLET MODULE")
#         print("-" * 50)
#         triplet_module = create_triplet_module(
#             anim_name,
#             bool_drop_neutral_exemplar,
#             bool_fixed_neutral_embedding,
#             squared_left_right_euc_dist,
#             squared_class_neut_euc_dist,
#             config
#         )
#         print(f"Created triplet module for {anim_name}")
#
#         # PART 2: Load data and generate embeddings
#         print("\n2. LOADING DATA AND GENERATING EMBEDDINGS")
#         print("-" * 50)
#
#         # Load similarity data for this animation
#         anim_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, anim_name, config)
#
#         # Store original dictionary for raw feature extraction
#         original_anim_similarity_dict = anim_similarity_dict_partition["train"]
#
#         # Create a balanced copy for embedding generation
#         balanced_anim_similarity_dict = dict(original_anim_similarity_dict)  # Deep copy
#         balanced_anim_similarity_dict = osd.balance_single_exemplar_similarity_classes_by_frame_count(
#             [balanced_anim_similarity_dict], 137)[0]
#
#         # Create dataloader for the balanced dictionary
#         balanced_data_loader = SimilarityDataLoader([balanced_anim_similarity_dict], config, False)
#
#         # Load model
#         model = load_model(checkpoint_path, architecture_variant, config, balanced_data_loader, triplet_module)
#
#         # Generate embeddings
#         embeddings = generate_embeddings_from_dataloader(model, balanced_data_loader, balanced_anim_similarity_dict, anim_name)
#
#         # PART 3: Extract raw features
#         print("\n3. EXTRACTING RAW FEATURES")
#         print("-" * 50)
#         raw_features = get_raw_features_without_dataloader(anim_name, config)
#
#         # PART 4: Calculate distances
#         print("\n4. CALCULATING DISTANCES")
#         print("-" * 50)
#
#         # Calculate L2 distances for embeddings
#         embedding_distances = calculate_pairwise_distances(embeddings)
#
#         # Calculate geodesic distances for raw features
#         geodesic_distances = compute_geodesic_distances(raw_features)
#
#         # PART 5: Analyze relationships with human perception
#         print("\n5. ANALYZING RELATIONSHIPS WITH HUMAN PERCEPTION")
#         print("-" * 50)
#
#         # Collect distances and corresponding inverse comparison values
#         print("\nCollecting distance and inverse comparison value pairs...")
#         embedding_dist, embedding_inverse_comparison_values = collect_distance_inverse_comparison_value_pairs(
#             embedding_distances, triplet_module, get_inverse_direct_comparison_value
#         )
#
#         geo_dist, geo_inverse_comparison_values = collect_distance_inverse_comparison_value_pairs(
#             geodesic_distances, triplet_module, get_inverse_direct_comparison_value
#         )
#
#         # Count valid pairs (with human ratings)
#         valid_embedding_pairs = sum(1 for v in embedding_inverse_comparison_values if v is not None)
#         valid_geo_pairs = sum(1 for v in geo_inverse_comparison_values if v is not None)
#
#         print(f"Found {valid_embedding_pairs} embedding pairs with human ratings")
#         print(f"Found {valid_geo_pairs} geodesic pairs with human ratings")
#
#         # Debug: Print first 5 pairs to verify distances
#         print("\nDEBUG - Sample distance pairs going into analysis:")
#         for i in range(min(5, len(embedding_dist))):
#             if embedding_inverse_comparison_values[i] is not None:
#                 print(
#                     f"Embedding pair {i}: distance={embedding_dist[i]:.6f}, alpha={embedding_inverse_comparison_values[i]:.6f}")
#
#         for i in range(min(5, len(geo_dist))):
#             if geo_inverse_comparison_values[i] is not None:
#                 print(f"Geodesic pair {i}: distance={geo_dist[i]:.6f}, alpha={geo_inverse_comparison_values[i]:.6f}")
#
#         # Compare relationship between distances and human perception
#         comparison_results = compare_distance_inverse_comparison_value_relationships(
#             embedding_dist, embedding_inverse_comparison_values,
#             geo_dist, geo_inverse_comparison_values,
#             method="geodesic"
#         )
#
#         # Print final analysis results
#         print("\n" + "=" * 70)
#         print(f"FINAL ANALYSIS FOR {anim_name.upper()}: EMBEDDING L2 VS RAW FEATURE GEODESIC DISTANCE")
#         print("=" * 70 + "\n")
#         print(comparison_results['output_text'])
#
#         # Print concise conclusion
#         print("\n" + "=" * 70)
#         print(f"CONCLUSION FOR {anim_name.upper()}: {comparison_results['summary']['stronger_method']} shows a stronger relationship with human perception")
#         print("=" * 70)


if __name__ == "__main__":
    main()