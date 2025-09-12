#!/usr/bin/env python3
"""
Complete integration script for running triplet training with pre-generated embeddings.

This script:
1. Loads your root and rotation embeddings
2. Combines them appropriately
3. Creates similarity datasets compatible with your existing triplet training
4. Runs the triplet training using embeddings as input

Usage:
    python run_embedding_triplet_training.py --embedding-dir ../datasets/lma_perform_encoded
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add current directory to path
curr_path = os.getcwd()
sys.path.append(curr_path)

# Import all necessary modules
from embedding_dataset import EmbeddingDataset, create_embedding_similarity_data
from src.organize_synthetic_data import load_similarity_data_from_embeddings, EmbeddingSimilarityDataLoader
from networks.similarity_network import make_embedding_aware
from networks.similarity_network import EmbeddingAwareSimilarityNetwork
from networks.triplet_mining import TripletMining
from Config import Config


def setup_embedding_training(embedding_dir, config, combination_method='concat'):
    """
    Set up all components for embedding-based triplet training.

    Returns:
        Tuple of (train_loader, val_loader, train_triplet_modules, val_triplet_modules)
    """
    print("Setting up embedding-based training...")

    # Load similarity data from embeddings
    try:
        walking_similarity_dict = load_similarity_data_from_embeddings(
            bool_drop=True,  # Drop neutral exemplar
            anim_name="walking",
            config=config,
            embedding_dir=embedding_dir,
            combination_method=combination_method,
            force_regenerate=True  # Force regeneration to ensure we use embeddings
        )["train"]

        print(f"Loaded {len(walking_similarity_dict)} effort classes from embeddings")

    except Exception as e:
        print(f"Error loading similarity data: {e}")
        print("Creating fresh similarity data from embeddings...")

        # Create similarity data from scratch
        output_path, walking_similarity_dict = create_embedding_similarity_data(
            embedding_dir,
            config.similarity_exemplars_dir,
            combination_method
        )
        print(f"Created {len(walking_similarity_dict)} effort classes")

    # Create simple train/val split
    def create_train_val_split(similarity_dict, val_ratio=0.4):
        """Simple train/val split for a single animation."""
        import random

        keys = [k for k in similarity_dict.keys() if k != (0, 0, 0, 0)]
        val_size = max(1, int(len(keys) * val_ratio))

        val_keys = set(random.sample(keys, val_size))
        train_keys = set(k for k in keys if k not in val_keys)

        # Always include neutral in both if it exists
        if (0, 0, 0, 0) in similarity_dict:
            train_keys.add((0, 0, 0, 0))
            val_keys.add((0, 0, 0, 0))

        return [train_keys], [val_keys]

    # Split data
    train_indices, val_indices = create_train_val_split(walking_similarity_dict)

    # Create data loaders
    list_similarity_dicts = [walking_similarity_dict]

    train_loader = EmbeddingSimilarityDataLoader(
        list_similarity_dicts, config, shuffle=True, valid_indices=train_indices
    )
    val_loader = EmbeddingSimilarityDataLoader(
        list_similarity_dicts, config, shuffle=False, valid_indices=val_indices
    )

    print(f"Created data loaders:")
    print(f"  Training: {train_loader.num_classes} classes")
    print(f"  Validation: {val_loader.num_classes} classes")
    print(f"  Embedding dimension: {train_loader.exemplar_dim}")

    # Create triplet mining modules
    train_triplet = TripletMining(
        bool_drop=True,
        bool_fixed=True,
        squared_left_right=False,
        squared_class_neut=False,
        anim_name="walking",
        config=config,
        valid_indices=train_indices[0]
    )

    val_triplet = TripletMining(
        bool_drop=True,
        bool_fixed=True,
        squared_left_right=False,
        squared_class_neut=False,
        anim_name="walking",
        config=config,
        valid_indices=val_indices[0]
    )

    return train_loader, val_loader, [train_triplet], [val_triplet]


def run_embedding_triplet_training(
        embedding_dir,
        combination_method='concat',
        use_perception_loss=False,
        use_adaptive_distance=False,
        scheduler='plateau',
        n_epochs=200
):
    """
    Run the complete embedding-based triplet training pipeline.
    """
    print("=" * 60)
    print("EMBEDDING-BASED TRIPLET TRAINING")
    print("=" * 60)

    # Initialize config
    config = Config()
    config.n_similarity_epochs = n_epochs
    config.ensure_directories_exist()

    # Setup training components
    train_loader, val_loader, train_triplet_modules, val_triplet_modules = setup_embedding_training(
        embedding_dir, config, combination_method
    )

    # Create embedding-aware similarity network
    print("Creating embedding-aware similarity network...")

    similarity_network = EmbeddingAwareSimilarityNetwork(
        train_loader=train_loader,
        validation_loader=val_loader,
        test_loader=val_loader,
        checkpoint_root_dir=config.checkpoint_root_dir,
        triplet_modules=train_triplet_modules,
        val_triplet_modules=val_triplet_modules,
        architecture_variant=0,
        config=config,
        lr_scheduler_type=scheduler,
        use_perception_loss=use_perception_loss,
        use_adaptive_distance=use_adaptive_distance
    )

    # Print training configuration
    print("\n" + "=" * 40)
    print("TRAINING CONFIGURATION")
    print("=" * 40)
    print(f"Embedding directory: {embedding_dir}")
    print(f"Combination method: {combination_method}")
    print(f"Training classes: {train_loader.num_classes}")
    print(f"Validation classes: {val_loader.num_classes}")
    print(f"Input embedding dim: {train_loader.exemplar_dim}")
    print(f"Output embedding size: {config.embedding_refinement_model_output_size}")
    print(f"Epochs: {n_epochs}")
    print(f"Perception loss: {use_perception_loss}")
    print(f"Adaptive distance: {use_adaptive_distance}")
    print(f"Scheduler: {scheduler}")
    print(f"Device: {similarity_network.device}")
    print("=" * 40)

    # Run training
    print("\nStarting training...")
    similarity_network.run_model_training()

    # Evaluate
    print("\nEvaluating model...")
    results = similarity_network.evaluate()

    # Print results
    print("\n" + "=" * 40)
    print("FINAL RESULTS")
    print("=" * 40)
    print(f"Test Loss: {results['test_loss']:.4f}")
    if 'pearson_correlation' in results:
        print(f"Pearson Correlation: {results['pearson_correlation']:.4f}")
        print(f"Spearman Correlation: {results['spearman_correlation']:.4f}")
        print(f"R² Score: {results['r2_score']:.4f}")
        print(f"Valid pairs: {results['num_pairs']}")
    print("=" * 40)

    return similarity_network, results


def main():
    parser = argparse.ArgumentParser(description='Run embedding-based triplet training')

    # Required arguments
    parser.add_argument('--embedding-dir', type=str, default='../datasets/lma_perform_walking_encoded',
                        help='Directory containing embedding files (.pt files)')

    # Optional arguments
    parser.add_argument('--combination-method', type=str, default='rots_only',
                        choices=['concat', 'weighted', 'root_only', 'rots_only'],
                        help='How to combine root and rotation embeddings')
    parser.add_argument('--use-perception-loss', action='store_true', default=False,
                        help='Use enhanced perception-aligned loss')
    parser.add_argument('--use-adaptive-distance', action='store_true', default=False,
                        help='Use adaptive distance module')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'step'],
                        help='Learning rate scheduler')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--test-loading', action='store_true',
                        help='Only test loading embeddings without training')

    args = parser.parse_args()

    # Validate embedding directory
    embedding_dir = Path(args.embedding_dir)
    if not embedding_dir.exists():
        print(f"Error: Embedding directory {embedding_dir} does not exist")
        sys.exit(1)

    # Check for embedding files
    root_files = list(embedding_dir.glob("*_root.pt"))
    rots_files = list(embedding_dir.glob("*_rots.pt"))

    if not root_files or not rots_files:
        print(f"Error: No embedding files found in {embedding_dir}")
        print(f"Expected files ending with '_root.pt' and '_rots.pt'")
        sys.exit(1)

    print(f"Found {len(root_files)} root and {len(rots_files)} rotation embeddings")

    # Test loading only
    if args.test_loading:
        print("Testing embedding loading...")
        dataset = EmbeddingDataset(embedding_dir)
        print(f"Successfully loaded embedding dataset with {len(dataset.embedding_pairs)} pairs")

        # Test creating similarity dict
        similarity_dict = dataset.create_similarity_dict(args.combination_method)
        print(f"Created similarity dict with {len(similarity_dict)} effort classes")

        for effort_tuple, embeddings in list(similarity_dict.items())[:3]:
            print(f"  {effort_tuple}: {len(embeddings)} embeddings, shape {embeddings[0].shape}")

        print("Loading test successful!")
        return

    # Run full training
    try:
        network, results = run_embedding_triplet_training(
            embedding_dir=args.embedding_dir,
            combination_method=args.combination_method,
            use_perception_loss=args.use_perception_loss,
            use_adaptive_distance=args.use_adaptive_distance,
            scheduler=args.scheduler,
            n_epochs=args.epochs
        )

        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
