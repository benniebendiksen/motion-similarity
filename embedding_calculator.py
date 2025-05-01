"""
Motion Similarity Framework - Overall Analysis

This script analyzes motion data similarity across all animation types collectively,
comparing embedding L2 distances with raw feature DTW distances. It handles both
within-module and cross-module comparisons to provide a comprehensive analysis.

The framework:
1. Loads and processes data for all animation types
2. Generates embeddings using a trained neural network with balanced data
3. Extracts variable-length raw features directly from pickle files
4. Calculates L2 distances for embeddings and DTW distances for raw features
5. Analyzes how well each approach correlates with human perception data
"""

from networks.similarity_network import SimilarityNetwork
from networks.similarity_data_loader import SimilarityDataLoader
from networks.triplet_mining import TripletMining
from bvh_visualizing.datasetLoad import BVHDataset
from bvh_visualizing.bvhvisualize import DualBVHAnimator
from bvh_visualizing.bvh import BVH
from Config import Config
import src.organize_synthetic_data as osd
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from itertools import combinations
from dtaidistance import dtw_ndim
import tensorflow as tf
import torch
import pickle
import time

# Add necessary paths
curr_path = os.getcwd()
sys.path.append(curr_path)
sys.path.append(os.path.join(curr_path, 'networks'))


def compare_distance_inverse_comparison_value_relationships(embedding_distances, embedding_alphas,
                                                            dtw_distances, dtw_alphas, method="DTW"):
    """
    Compare the relationship between distances and inverse comparison values for embedding and DTW methods.

    Args:
        embedding_distances: List of distance values from embedding method
        embedding_alphas: List of inverse comparison values corresponding to embedding_distances
        dtw_distances: List of distance values from raw feature DTW method
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


def collect_distance_inverse_comparison_value_pairs(distance_tuples, triplet_modules, get_alpha_value_func):
    """
    Collect distance and inverse comparison values from distance tuples.

    Args:
        distance_tuples: List of tuples (distance, key1, key2)
        triplet_modules: List of triplet module objects
        get_alpha_value_func: Function to get inverse comparison value for a pair of keys

    Returns:
        tuple: (distances, inverse_comparison_values)
    """
    distances = []
    alphas = []

    for distance, key1, key2 in distance_tuples:
        alpha_value = get_alpha_value_func(key1, key2, triplet_modules)
        distances.append(distance)
        alphas.append(alpha_value)

    return distances, alphas


def create_triplet_modules(anim_names, bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
                         squared_left_right_euc_dist, squared_class_neut_euc_dist, config):
    """
    Create triplet modules for each animation type.

    Args:
        anim_names: List of animation names to create modules for
        bool_drop_neutral_exemplar: Whether to drop neutral exemplar
        bool_fixed_neutral_embedding: Whether to fix neutral embedding
        squared_left_right_euc_dist: Whether to square left-right Euclidean distance
        squared_class_neut_euc_dist: Whether to square class-neutral Euclidean distance
        config: Configuration object

    Returns:
        List of TripletMining objects for specified animations
    """
    triplet_modules = []
    for anim_name in anim_names:
        triplet_module = TripletMining(
            bool_drop_neutral_exemplar,
            bool_fixed_neutral_embedding,
            squared_left_right_euc_dist,
            squared_class_neut_euc_dist,
            anim_name,
            config
        )
        triplet_modules.append(triplet_module)

    return triplet_modules


def load_model(checkpoint_path, architecture_variant, config, data_loader, triplet_modules):
    """
    Load a trained similarity network model from a PyTorch checkpoint file.

    Args:
        checkpoint_path: Path to the saved model weights (.pt file)
        architecture_variant: Architecture variant number
        config: Configuration object containing model parameters
        data_loader: Data loader object for model initialization
        triplet_modules: List of triplet modules for model initialization

    Returns:
        Loaded and initialized network model ready for inference
    """
    # Create and load the similarity network
    similarity_network = SimilarityNetwork(
        train_loader=data_loader,
        validation_loader=data_loader,
        test_loader=data_loader,
        checkpoint_root_dir=config.checkpoint_root_dir,
        triplet_modules=triplet_modules,
        architecture_variant=architecture_variant,
        config=config
    )

    # Load only the model state dict and not the optimizer state dict
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    similarity_network.network.load_state_dict(checkpoint['model_state_dict'])
    similarity_network.network.eval()
    print(f"Model loaded from {checkpoint_path}")

    return similarity_network.network

def convert_list_similarity_dicts_to_anim_class_tuple_dicts(list_similarity_dicts):
    """
    Convert a list of similarity dictionaries to a dictionary mapping (action_type, effort_tuple) to similarity values.

    Args:
        list_similarity_dicts: List of similarity dictionaries for all animations

    Returns:
        Dictionary mapping (action_type, effort_tuple) to similarity values
    """
    dict_similarity = {}
    idx_to_action = {0: "walking", 1: "pointing", 2: "picking"}

    for i, similarity_dict in enumerate(list_similarity_dicts):
        for (class_tuple, value) in similarity_dict.items():
            new_key = (idx_to_action[i], class_tuple)
            dict_similarity[new_key] = value

    return dict_similarity

def generate_embeddings_from_dataloader(model, data_loader, list_similarity_dicts):
    """
    Generate embeddings using a PyTorch model and DataLoader.
    This version handles multiple animation types at once.

    Args:
        model: Loaded PyTorch model
        data_loader: DataLoader object containing motion data
        list_similarity_dicts: List of similarity dictionaries for all animations

    Returns:
        Dictionary mapping (action_type, effort_tuple) to embedding vectors
    """
    print("Generating embeddings for all animation types...")

    # Get the mapping between batch indices and keys
    embedding_keys = []
    idx_to_action = {0: "walking", 1: "pointing", 2: "picking"}

    for i, similarity_dict in enumerate(list_similarity_dicts):
        for (class_tuple, _value) in similarity_dict.items():
            new_key = (idx_to_action[i], class_tuple)
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

    print(f"Generated {len(embeddings)} embeddings across all animations")
    return embeddings


def get_raw_features_without_dataloader(anim_names, config):
    """
    Directly load and extract raw features from pickle files for multiple animations,
    completely bypassing the dataloader to preserve variable-length sequences.

    Args:
        anim_names: List of animation names to load
        config: Configuration object with paths

    Returns:
        Dictionary mapping (action_type, effort_tuple) to raw feature vectors
    """
    raw_features = {}

    for anim_name in anim_names:
        # Directly open the pickle file
        file_path = config.similarity_exemplars_dir + anim_name + "_" + config.similarity_dict_file_name
        print(f"Loading data directly from pickle file: {file_path}")

        with open(file_path, "rb") as f:
            raw_dict = pickle.load(f)

        # Print original dictionary information for the first few entries
        print(f"Loaded {anim_name} raw dictionary with {len(raw_dict)} keys")

        sample_keys = list(raw_dict.keys())[:2]
        for key in sample_keys:
            exemplar = raw_dict[key]
            if exemplar:
                print(f"  Sample key {key}: Shape {exemplar[0].shape}")

        # Extract raw features
        for class_tuple, exemplar in raw_dict.items():
            if not exemplar:
                continue

            # Get the raw tensor
            raw_tensor = exemplar[0]

            # Convert to numpy for compatibility
            if isinstance(raw_tensor, tf.Tensor):
                numpy_tensor = raw_tensor.numpy().copy()
            else:
                raise Exception(f"Payload within similarity dict entry's list is not a tf tensor but: {type(raw_tensor)}")

            key = (anim_name, class_tuple)
            raw_features[key] = numpy_tensor

    # Report on the combined dataset
    lengths = [(key, tensor.shape[0]) for key, tensor in raw_features.items()]
    unique_lengths = sorted(set(length for _, length in lengths))

    print(f"Total extracted features: {len(raw_features)}")
    print(f"Unique sequence lengths: {unique_lengths}")

    # Count by animation type
    anim_counts = {}
    for (anim, _), _ in raw_features.items():
        anim_counts[anim] = anim_counts.get(anim, 0) + 1

    for anim, count in anim_counts.items():
        print(f"  {anim}: {count} motions")

    return raw_features


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
                print(f"Progress: {pair_count}/{total_pairs} pairs ({percent:.1f}%) - Elapsed: {elapsed:.1f}s")

    # Sort by distance (ascending)
    distances.sort()
    print(f"Calculated {len(distances)} L2 distances")

    return distances


def compute_geodesic_distances(dict_padded_raw_features):
    """
    Compute pairwise geodesic distances between multiple quaternion samples.

    Args:
        dict_padded_raw_features (dict): Dictionary mapping (action_type, effort_tuple)
                                to raw quaternion motion data of shape (137, 112).

    Returns:
        List of tuples (distance, key1, key2) sorted by distance in ascending order.
    """
    keys = list(dict_padded_raw_features.keys())
    num_samples = len(keys)
    num_frames = 137
    num_joints = 28

    distances = []
    total_pairs = num_samples * (num_samples - 1) // 2  # Total pairwise comparisons
    progress_interval = max(1, total_pairs // 10)  # Show progress 10 times
    pair_count = 0
    start_time = time.time()

    print(f"Computing geodesic distances for {total_pairs} pairs across {num_samples} samples...")

    # Track same-action vs cross-action pairs
    same_action_pairs = 0
    cross_action_pairs = 0

    for i, j in combinations(range(num_samples), 2):
        key1 = keys[i]
        key2 = keys[j]

        # Determine if this is a same-action or cross-action comparison
        if key1[0] == key2[0]:
            same_action_pairs += 1
        else:
            cross_action_pairs += 1

        # Get quaternion data
        sample_i = dict_padded_raw_features[key1]
        sample_j = dict_padded_raw_features[key2]

        # If it's a list with one element (from the similarity dict), take the first element
        if isinstance(sample_i, list) and len(sample_i) > 0:
            sample_i = sample_i[0]

        if isinstance(sample_j, list) and len(sample_j) > 0:
            sample_j = sample_j[0]

        # Convert to numpy for compatibility
        if isinstance(sample_i, tf.Tensor):
            sample_i = sample_i.numpy()
        elif not isinstance(sample_i, np.ndarray):
            raise Exception(f"sample_i is not a tf.Tensor or np.ndarray but: {type(sample_i)}")

        if isinstance(sample_j, tf.Tensor):
            sample_j = sample_j.numpy()
        elif not isinstance(sample_j, np.ndarray):
            raise Exception(f"sample_j is not a tf.Tensor or np.ndarray but: {type(sample_j)}")

        # Reshape to (137, 28, 4) for quaternion operations
        q1 = sample_i.reshape(num_frames, num_joints, 4)
        q2 = sample_j.reshape(num_frames, num_joints, 4)

        # Compute geodesic distances for all frames and joints
        distances_per_joint = 2 * np.arccos(np.clip(np.abs(np.sum(q1 * q2, axis=2)), -1.0, 1.0))  # Shape: (137, 28)
        mean_distance = np.mean(distances_per_joint)  # Aggregate over frames and joints

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
    print(
        f"Computed {len(distances)} geodesic distances ({same_action_pairs} same-action, {cross_action_pairs} cross-action)")

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

    # Track same-action vs cross-action pairs
    same_action_pairs = 0
    cross_action_pairs = 0

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key1 = keys[i]
            key2 = keys[j]

            # Track if this is a same-action or cross-action comparison
            if key1[0] == key2[0]:
                same_action_pairs += 1
            else:
                cross_action_pairs += 1

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
                print(f"Progress: {pair_count}/{total_pairs} pairs ({percent:.1f}%) - Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")

    # Sort by distance (ascending)
    distances.sort()
    print(f"Computed {len(distances)} DTW distances ({same_action_pairs} same-action, {cross_action_pairs} cross-action)")

    return distances


def get_inverse_direct_comparison_value(key1, key2, triplet_modules):
    """
    Get the inverse of direct comparison value (1 - value) for a pair of keys.
    This version handles both within-module and cross-module comparisons.

    Args:
        key1: First key (action_type, effort_tuple)
        key2: Second key (action_type, effort_tuple)
        triplet_modules: List of triplet module objects

    Returns:
        Inverse comparison value or None if not available
    """
    action1, effort1 = key1
    action2, effort2 = key2

    # Skip neutral exemplars
    if effort1 == (0, 0, 0, 0) or effort2 == (0, 0, 0, 0):
        return None

    # Only process pairs from the same animation type
    # Cross-animation comparisons don't have direct human ratings
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
            if not pair_match.empty:
                row = pair_match.iloc[0]

                # Check if we have a direct comparison value
                if 'direct_comparison_value' in row and not pd.isna(row['direct_comparison_value']):
                    # Return 1 minus the comparison value
                    return 1 - row['direct_comparison_value']

            return None

    return None


def main(pad_raw_features=True):
    """
    Main execution function that analyzes all animations together:
    1. Load and process data for all animation types
    2. Generate embeddings with balanced data for all animations
    3. Extract variable-length raw features directly from pickle files
    4. Calculate L2 distances for embeddings and DTW distances for raw features
    5. Compare which approach better correlates with human perception
    """
    # Initialize configuration
    config = Config()

    # Animation types to process
    animations = ["walking", "pointing", "picking"]

    print("\nOVERALL ANALYSIS: COMPARING EMBEDDING L2 VS RAW FEATURE DTW")
    print("=" * 70)
    print("This analysis will evaluate which distance metric better correlates with human perception")
    print("across all animation types, considering both within-module and cross-module comparisons.\n")

    # Set up paths and model parameters
    architecture_variant = 0
    # checkpoint_path = os.path.join(config.checkpoint_root_dir,
    #                               f"{architecture_variant}_similarity_model_weights_epoch_079.pt")
    checkpoint_path = os.path.join(config.checkpoint_root_dir,
                                   f"{architecture_variant}_similarity_model_weights_epoch_032.pt")

    bool_drop_neutral_exemplar = False
    bool_fixed_neutral_embedding = False
    squared_left_right_euc_dist = False
    squared_class_neut_euc_dist = True

    # PART 1: Create triplet modules for all animations
    print("\n1. CREATING TRIPLET MODULES FOR ALL ANIMATIONS")
    print("-" * 50)
    triplet_modules = create_triplet_modules(
        animations,
        bool_drop_neutral_exemplar,
        bool_fixed_neutral_embedding,
        squared_left_right_euc_dist,
        squared_class_neut_euc_dist,
        config
    )
    print(f"Created {len(triplet_modules)} triplet modules")

    # PART 2: Load data and generate embeddings
    print("\n2. LOADING DATA AND GENERATING EMBEDDINGS")
    print("-" * 50)

    # Load similarity data for all animations
    all_similarity_dicts = []
    list_balanced_similarity_dicts = []

    for anim_name in animations:
        anim_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, anim_name, config)
        original_dict = anim_similarity_dict_partition["train"]
        all_similarity_dicts.append(original_dict)

        # Create a balanced copy for embedding generation
        module_similarity_dict_copy = dict(original_dict)  # Deep copy

        list_balanced_similarity_dicts.append(module_similarity_dict_copy)

    # Apply balancing to all dictionaries together
    list_balanced_similarity_dicts = osd.balance_single_exemplar_similarity_classes_by_frame_count(list_balanced_similarity_dicts, 137)

    # Create a single dataloader for all balanced dictionaries
    print("\nCreating balanced dataloader for all animations")
    balanced_data_loader = SimilarityDataLoader(list_balanced_similarity_dicts, config, False)
    # Load model
    model = load_model(checkpoint_path, architecture_variant, config, balanced_data_loader, triplet_modules)
    # Generate embeddings for all animations using the balanced dataloader
    print("\nGenerating embeddings for all animations")
    embeddings = generate_embeddings_from_dataloader(model, balanced_data_loader, list_balanced_similarity_dicts)

    # PART 3: Get raw features with variable lengths
    print("\n3. EXTRACTING RAW FEATURES")
    print("-" * 50)
    if pad_raw_features:
        dict_raw_features = convert_list_similarity_dicts_to_anim_class_tuple_dicts(list_balanced_similarity_dicts)
    else:
        dict_raw_features = get_raw_features_without_dataloader(animations, config)
    # PART 4: Calculate distances
    print("\n4. CALCULATING DISTANCES")
    print("-" * 50)
    # Calculate L2 distances for embeddings
    embedding_distances = calculate_pairwise_distances(embeddings)

    # Calculate geodesic distances for raw features
    # (i.e., numpy arrays of size (137, 116) where
    # 116 -> 28 joints with rotations x 4 dimensions per quaternions = 112 + efforts = 116)
    geodesic_distances = compute_geodesic_distances(dict_raw_features)

    # PART 5: Analyze relationships with human perception
    print("\n5. ANALYZING RELATIONSHIPS WITH HUMAN PERCEPTION")
    print("-" * 50)

    # Collect distances and corresponding inverse comparison values
    print("\nCollecting distance and inverse comparison value pairs...")
    embedding_dist, embedding_inverse_comparison_values = collect_distance_inverse_comparison_value_pairs(
        embedding_distances, triplet_modules, get_inverse_direct_comparison_value
    )

    geo_dist, geo_inverse_comparison_values = collect_distance_inverse_comparison_value_pairs(
        geodesic_distances, triplet_modules, get_inverse_direct_comparison_value
    )

    # Count valid pairs (with human ratings)
    valid_embedding_pairs = sum(1 for v in embedding_inverse_comparison_values if v is not None)
    valid_dtw_pairs = sum(1 for v in geo_inverse_comparison_values if v is not None)

    print(f"Found {valid_embedding_pairs} embedding pairs with human ratings")
    print(f"Found {valid_dtw_pairs} DTW pairs with human ratings")

    # Compare relationship between distances and human perception
    comparison_results = compare_distance_inverse_comparison_value_relationships(
        embedding_dist, embedding_inverse_comparison_values,
        geo_dist, geo_inverse_comparison_values,
        method="geodesic"
    )

    # Print final analysis results
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS: EMBEDDING L2 VS RAW FEATURE GEODESIC DISTANCE")
    print("=" * 70 + "\n")
    print(comparison_results['output_text'])

    # Print concise conclusion
    print("\n" + "=" * 70)
    print(f"CONCLUSION: {comparison_results['summary']['stronger_method']} shows a stronger relationship with human perception")
    print("=" * 70)


if __name__ == "__main__":
    main()