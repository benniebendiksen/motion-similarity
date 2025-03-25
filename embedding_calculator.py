import os
import sys
import numpy as np
import tensorflow as tf
import torch
import pickle
from pathlib import Path

# Add necessary paths
curr_path = os.getcwd()
sys.path.append(curr_path)
sys.path.append(os.path.join(curr_path, 'networks'))

# Import required modules
from networks.similarity_network import SimilarityNetwork
from networks.similarity_data_loader import SimilarityDataLoader
from networks.triplet_mining import TripletMining
from Config import Config
import src.organize_synthetic_data as osd


def create_triplet_modules(list_anim_names, bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
                           squared_left_right_euc_dist, squared_class_neut_euc_dist, config):
    list_triplet_modules = []
    for anim_name in list_anim_names:
        list_triplet_modules.append((bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
                                     squared_left_right_euc_dist, squared_class_neut_euc_dist, anim_name, config))

    return list_triplet_modules


# def load_model(checkpoint_path, architecture_variant, config, data_loader, triplet_modules):
#     """
#     Load a trained similarity network model for generating embeddings.
#
#     Args:
#         checkpoint_path: Path to the saved model weights
#         architecture_variant: Architecture variant number
#         config: Configuration object
#
#     Returns:
#         Loaded network model
#     """
#
#     # Create and load the similarity network
#     similarity_network = SimilarityNetwork(
#         train_loader=data_loader,
#         validation_loader=data_loader,
#         test_loader=data_loader,
#         checkpoint_root_dir=config.checkpoint_root_dir,
#         triplet_modules=triplet_modules,
#         architecture_variant=architecture_variant,
#         config=config
#     )
#
#     # Load the trained weights
#     similarity_network.network.load_weights(checkpoint_path)
#     print(f"Loaded model weights from {checkpoint_path}")
#
#     return similarity_network.network

def load_model(checkpoint_path, architecture_variant, config, data_loader, triplet_modules):
    """
    Load a trained similarity network model from a .pt file.

    Args:
        checkpoint_path: Path to the saved model weights
        architecture_variant: Architecture variant number
        config: Configuration object

    Returns:
        Loaded network model
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
    print(f"Loaded model weights from {checkpoint_path}")

    return similarity_network.network


# def generate_embeddings_from_dataloader(model, data_loader, list_similarity_dicts):
#     """
#     Generate embeddings for all samples in a data dictionary using a dataloader.
#
#     Args:
#         model: Loaded network model
#         data_dict: Dictionary of similarity data
#         config: Configuration object
#
#     Returns:
#         Dictionary mapping (key, idx) to embedding vectors
#     """
#     print("Generating embeddings using dataloader...")
#
#     # Get the mapping between batch indices and keys
#     embedding_keys = []
#
#     # Create dictionary idx, class_key tuples to identify any given exemplar by its action and effort tuple
#     # i = 0 -> walking, i = 1 -> pointing, i = 2 -> picking
#     for i, similarity_dict in enumerate(list_similarity_dicts):
#         for _, (class_tuple, value) in enumerate(similarity_dict.items()):
#             new_key = (i, class_tuple)
#             embedding_keys.append(new_key)
#
#     # Get batch features from dataloader
#     batch_features, batch_labels = data_loader[0]  # Get the first (and only) batch
#
#     # Make sure the batch features have the right shape
#     if len(batch_features.shape) == 3:
#         print(f"data loader returned batch features of 3 dimensions, expanding to 4")
#         # Add channel dimension if it's missing
#         batch_features = tf.expand_dims(batch_features, -1)
#
#     # Generate embeddings for the entire batch at once
#     print(f"Generating embeddings for batch of shape {batch_features.shape}...")
#     batch_embeddings = model.predict(batch_features, verbose=1)
#
#     # Map the embeddings back to their keys
#     embeddings = {}
#     for i, embedding in enumerate(batch_embeddings):
#         embeddings[embedding_keys[i]] = embedding
#
#     print(f"Generated {len(embeddings)} embeddings")
#     return embeddings

def generate_embeddings_from_dataloader(model, data_loader, list_similarity_dicts):
    """
    Generate embeddings using a PyTorch model and DataLoader.

    Args:
        model: Loaded PyTorch model
        data_loader: DataLoader object
        list_similarity_dicts: List of similarity dictionaries

    Returns:
        Dictionary mapping (key, idx) to embedding vectors
    """
    print("Generating embeddings using dataloader...")

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
                print(f"Data loader returned batch features of 3 dimensions, expanding to 4")
                batch_features = batch_features.unsqueeze(-1)

            batch_features = batch_features.to(torch.float32)
            batch_embeddings = model(batch_features).cpu().numpy()

            for i, embedding in enumerate(batch_embeddings):
                embeddings[embedding_keys[i]] = embedding

    print(f"Generated {len(embeddings)} embeddings")
    return embeddings


def calculate_pairwise_distances(embeddings):
    """
    Calculate pairwise Euclidean distances between all embeddings.

    Args:
        embeddings: Dictionary mapping (key, idx) to embedding vectors

    Returns:
        List of tuples (distance, key1, key2)
    """
    # Calculate pairwise distances
    print("Calculating pairwise distances...")
    distances = []
    embedding_keys = list(embeddings.keys())

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

    # Sort by distance (ascending)
    distances.sort()

    return distances


def main():
    # Initialize configuration
    config = Config()

    # Set up paths and model parameters
    architecture_variant = 0
    #checkpoint_path = os.path.join(config.checkpoint_root_dir,
     #                              f"{architecture_variant}_similarity_model_weights.weights.h5")
    checkpoint_path = os.path.join(config.checkpoint_root_dir,
                                   f"{architecture_variant}_similarity_model_weights_epoch_074.pt")

    bool_drop_neutral_exemplar = False
    bool_fixed_neutral_embedding = False
    squared_left_right_euc_dist = True
    squared_class_neut_euc_dist = False

    # Create Triplet Module instances, each houses a within-cluster loss calculation
    triplet_modules = create_triplet_modules(["walking", "pointing", "picking"], bool_drop_neutral_exemplar,
                                             bool_fixed_neutral_embedding,
                                             squared_left_right_euc_dist, squared_class_neut_euc_dist, config)

    # load all similarity data into list of dicts
    walking_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, "walking", config)
    pointing_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, "pointing", config)
    picking_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, "picking", config)
    list_similarity_dicts = [walking_similarity_dict_partition["train"],
                             pointing_similarity_dict_partition["train"],
                             picking_similarity_dict_partition["train"]]
    list_similarity_dicts = osd.balance_single_exemplar_similarity_classes_by_frame_count(list_similarity_dicts)

    # Create dataloader object
    data_loader = SimilarityDataLoader(list_similarity_dicts, config, False)

    # Load model
    model = load_model(checkpoint_path, architecture_variant, config, data_loader, triplet_modules)

    # Generate embeddings using dataloader
    embeddings = generate_embeddings_from_dataloader(model, data_loader, list_similarity_dicts)

    # Calculate pairwise distances
    distances = calculate_pairwise_distances(embeddings)

    # Display results
    print(f"\nPairwise distances (sorted by distance, ascending):")
    print(f"{'Distance':<10} {'Key1':<25} {'Key2':<25}")
    print("-" * 60)

    for i, (distance, key1, key2) in enumerate(distances):

        # Only show top 100 results to avoid excessive output
        if i >= 99:
            # print(f"\n... and {len(distances) - 100} more pairs")
            if i == 14534:
                print(f"final pair: {distance:<10.4f} {str(key1):<25} {str(key2):<25}")
            if i == 14535:
                assert False, (f"erroneous end pair selected!!!")

        else:
            print(f"{distance:<10.4f} {str(key1):<25} {str(key2):<25}")

    # Save results to file
    # with open(f"pairwise_distances_{anim_name}_{partition}.txt", "w") as f:
    #     f.write(f"Pairwise distances (sorted by distance, ascending):\n")
    #     f.write(f"{'Distance':<10} {'Key1':<25} {'Key2':<25}\n")
    #     f.write("-" * 60 + "\n")
    #
    #     for distance, key1, key2 in distances:
    #         f.write(f"{distance:<10.4f} {str(key1):<25} {str(key2):<25}\n")
    #
    # print(f"\nSaved all {len(distances)} pairs to pairwise_distances_{anim_name}_{partition}.txt")


if __name__ == "__main__":
    main()
