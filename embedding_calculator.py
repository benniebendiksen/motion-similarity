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
from bvh_visualizing.datasetLoad import BVHDataset
from bvh_visualizing.bvhvisualize import BVHAnimator
from bvh_visualizing.bvh import BVH
from Config import Config
import src.organize_synthetic_data as osd


def convert_to_bvh(motion_data_root, motion_data_rots, original_bvh, dataset, out_file=None):
    """
    Convert the decoded motion back to bvh format.
    """
    bvh_out = original_bvh
    num_frames = bvh_out.numFrames()

    for frame_idx in range(num_frames):
        bvh_out.readFrame(frame_idx)

        # denormalize
        # root_pos = motion_data_root[0, frame_idx, 0, 0:3] * dataset.bb_size + dataset.bb_min
        # bvh_out.root.setGlobalPos(root_pos.tolist())

        bvh_out.root.setGlobalPos(motion_data_root[0, frame_idx, 0, 0:3].tolist())

        for joint_idx in range(bvh_out.numJoints()):
            bvh_out.jointById(joint_idx).setLocalRotQuat(motion_data_rots[0, frame_idx, joint_idx, :])

        bvh_out.writeFrame(frame_idx)

    if out_file:
        bvh_out.save(out_file)
    return bvh_out


def create_triplet_modules(list_anim_names, bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
                           squared_left_right_euc_dist, squared_class_neut_euc_dist, config):
    list_triplet_modules = []
    for anim_name in list_anim_names:
        list_triplet_modules.append((bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
                                     squared_left_right_euc_dist, squared_class_neut_euc_dist, anim_name, config))

    return list_triplet_modules


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


def visualize_embedding_pair(distance_tuples):
    for tuple in distance_tuples:
        key = tuple[1]
        # parse key and form valid bvh file name
        action = key[0]
        effort = key[1]
        bvh_file_name = f"{action}_{effort[0]}_{effort[1]}_{effort[2]}_{effort[3]}.bvh"
        print(f"action: {action}")
        # get directory based on action
        dataset_dir = action + "_perform_user_study_1"
        dataset = BVHDataset(directory="walking_perform_user_study_1")
        animation = BVH()
        a = animation.load(f"{dataset_dir}/{bvh_file_name}")
        motion_data = dataset.extract_root_and_rotations(animation)
        motion_data = torch.tensor(motion_data, dtype=torch.float32)
        motion_data = motion_data.unsqueeze(0)
        mot_root = motion_data[:, :, :1, :]
        mot_rots = motion_data[:, :, 1:, :]
        out = convert_to_bvh(mot_root, mot_rots, animation, "new.bvh")
        anim = BVHAnimator(out)


def main():
    # Initialize configuration
    config = Config()

    # Set up paths and model parameters
    architecture_variant = 0
    # checkpoint_path = os.path.join(config.checkpoint_root_dir,
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

    visualize_embedding_pair(distances[0:2])
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
