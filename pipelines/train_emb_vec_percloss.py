"""
Modified run_motion_triplet_training.py to use pre-generated embeddings instead of raw motion data
"""

import os
import sys
import random
import argparse
import torch
import pandas as pd
import ast
import numpy as np

# Resolve project root from this file's location so the script works
# whether invoked from pipelines/ or from the project root.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'networks'))
from networks.similarity_network import SimilarityNetwork
from networks.triplet_mining import TripletMining
from Config import Config

# TensorFlow is optional — only used for the GPU availability check helper.
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

# Import our new embedding functions
from embedding_dataset import create_embedding_similarity_data
from src.organize_synthetic_data import load_similarity_data_from_embeddings, EmbeddingSimilarityDataLoader


def check_gpu_access():
    if not _TF_AVAILABLE:
        # Fall back to PyTorch GPU check when TensorFlow is not installed.
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("❌ No GPU detected by PyTorch (TensorFlow not installed).")
        return
    # Check if GPUs are available via TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ TensorFlow detected {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu}")

        # Set TensorFlow to use GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Run a small computation on the GPU
        with tf.device('/GPU:0'):
            a = tf.constant([1.0, 2.0, 3.0])
            b = tf.constant([4.0, 5.0, 6.0])
            c = a + b
        print(f"✅ GPU computation successful: {c.numpy()}")
    else:
        print("❌ No GPU detected by TensorFlow.")


def create_train_val_split(similarity_dicts, val_ratio=0.4):
    """
    Create training and validation indices for each animation type.
    """
    train_indices = []
    val_indices = []

    for anim_dict in similarity_dicts:
        # Get keys except neutral
        keys = [k for k in anim_dict.keys() if k != (0, 0, 0, 0)]

        # Determine validation set size
        val_size = max(1, int(len(keys) * val_ratio))

        # Randomly sample keys for validation
        val_keys = set(random.sample(keys, val_size))
        train_keys = set(k for k in keys if k not in val_keys)

        # Add neutral exemplar to both sets
        if (0, 0, 0, 0) in anim_dict:
            train_keys.add((0, 0, 0, 0))
            val_keys.add((0, 0, 0, 0))

        train_indices.append(train_keys)
        val_indices.append(val_keys)

    return train_indices, val_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train similarity network with embeddings')
    parser.add_argument('--use-perception-loss', action='store_true', help='Use enhanced perception-aligned loss')
    parser.add_argument('--use-adaptive-distance', action='store_true', help='Use adaptive distance module')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'step'],
                        help='Learning rate scheduler type')
    parser.add_argument('--animation', type=str, default='walking', choices=['walking', 'pointing', 'picking'],
                        help='Which animation type to train on')
    parser.add_argument('--task-index', type=str, help='Task index for distributed training')
    parser.add_argument('--embedding-dir', type=str, default='../datasets/lma_perform_walking_encoded',
                        help='Directory containing embedding files (_root.pt / _rots.pt pairs)')
    parser.add_argument('--combination-method', type=str, default='concat',
                        choices=['concat', 'weighted', 'root_only', 'rots_only'],
                        help='How to combine root and rotation embeddings')

    args = parser.parse_args()

    check_gpu_access()

    # Initialize configuration
    task_index = args.task_index if args.task_index else None
    config = Config(task_index)

    # Ensure required directories exist
    config.ensure_directories_exist()

    # Architecture variant
    arch_variant = int(config.num_task) if config.num_task else 0

    bool_drop_neutral_exemplar = True
    bool_fixed_neutral_embedding = True
    squared_left_right_euc_dist = False
    squared_class_neut_euc_dist = False

    print(f"\n=== EMBEDDING-BASED TRAINING CONFIGURATION ===")
    print(f"Animation: {args.animation}")
    print(f"Embedding directory: {args.embedding_dir}")
    print(f"Combination method: {args.combination_method}")
    print(f"Using perception loss: {args.use_perception_loss}")
    print(f"Using adaptive distance: {args.use_adaptive_distance}")
    print("=" * 50)

    # Load similarity data from embeddings instead of raw motion data
    try:
        walking_similarity_dict = load_similarity_data_from_embeddings(
            bool_drop_neutral_exemplar,
            "walking",
            config,
            args.embedding_dir,
            args.combination_method
        )["train"]

        print(f"Successfully loaded embedding-based similarity data")
        print(f"Number of effort classes: {len(walking_similarity_dict)}")

        # Print some example embeddings info
        for i, (effort_tuple, embeddings) in enumerate(list(walking_similarity_dict.items())[:3]):
            print(f"  Effort {effort_tuple}: {len(embeddings)} embeddings, shape: {embeddings[0].shape}")

    except Exception as e:
        print(f"Error loading similarity data from embeddings: {e}")
        print("Falling back to creating similarity data from embeddings...")

        # Create similarity data from embeddings
        try:
            output_dir = "../datasets/similarity_walking_embeddings_dict"
            output_path, similarity_dict = create_embedding_similarity_data(
                args.embedding_dir,
                output_dir,
                args.combination_method
            )
            walking_similarity_dict = similarity_dict
            print(f"Created similarity data with {len(walking_similarity_dict)} effort classes")
        except Exception as e2:
            print(f"Error creating similarity data: {e2}")
            sys.exit(1)

    # Use only walking for now (as requested)
    list_similarity_dicts = [walking_similarity_dict]
    animation_names = ['walking']

    # No need to balance frame counts since embeddings have fixed dimensions
    print("Skipping frame count balancing for embeddings...")

    # Create train/val split
    train_indices, val_indices = create_train_val_split(list_similarity_dicts)

    # Create data loaders using the embedding-specific loader
    print("Creating embedding-based data loaders...")
    train_loader = EmbeddingSimilarityDataLoader(list_similarity_dicts, config, True, train_indices)
    val_loader = EmbeddingSimilarityDataLoader(list_similarity_dicts, config, True, val_indices)

    print(f"Training loader: {train_loader.num_classes} classes, embedding dim: {train_loader.exemplar_dim}")
    print(f"Validation loader: {val_loader.num_classes} classes, embedding dim: {val_loader.exemplar_dim}")

    # Create training and validation triplet modules
    train_triplet_modules = []
    val_triplet_modules = []

    for i, anim_name in enumerate(animation_names):
        idx = 0 if len(animation_names) == 1 else i

        train_triplet = TripletMining(
            bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
            squared_left_right_euc_dist, squared_class_neut_euc_dist,
            anim_name, config, valid_indices=train_indices[idx]
        )

        val_triplet = TripletMining(
            bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
            squared_left_right_euc_dist, squared_class_neut_euc_dist,
            anim_name, config, valid_indices=val_indices[idx]
        )

        train_triplet_modules.append(train_triplet)
        val_triplet_modules.append(val_triplet)

    print(f"Created {len(train_triplet_modules)} triplet mining modules")

    # Create the similarity network.
    # NOTE: SimilarityNetwork.__init__ calls build_model() which builds a CNN
    # suitable for 2-D (frames × features) raw-motion inputs.  Because this
    # pipeline feeds flat 1-D AE embedding vectors we must immediately swap
    # the backbone to the MLP variant by calling build_embedding_model().
    print("Creating similarity network (embedding-input variant)...")
    similarity_network = SimilarityNetwork(
        train_loader=train_loader,
        validation_loader=val_loader,
        test_loader=val_loader,
        checkpoint_root_dir=config.checkpoint_root_dir,
        triplet_modules=train_triplet_modules,
        val_triplet_modules=val_triplet_modules,
        architecture_variant=arch_variant,
        config=config,
        lr_scheduler_type=args.scheduler,
        use_perception_loss=args.use_perception_loss,
        use_adaptive_distance=args.use_adaptive_distance
    )
    # Swap CNN backbone → MLP backbone for flat embedding vectors.
    similarity_network.build_embedding_model()

    # Print final training configuration
    print("\n=== Final Training Configuration ===")
    print(f"Architecture variant: {arch_variant}")
    print(f"Input embedding dimension: {train_loader.exemplar_dim}")
    print(f"Output embedding size: {config.embedding_refinement_model_output_size}")
    print(f"Number of training classes: {train_loader.num_classes}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Number of epochs: {config.n_similarity_epochs}")
    print(f"Using device: {similarity_network.device}")
    print("============================\n")

    # Train the model
    print("Starting training...")
    similarity_network.run_model_training()

    # Evaluate the model
    print("Evaluating model...")
    results = similarity_network.evaluate()

    # Print final results
    print("\n=== Final Evaluation Results ===")
    print(f"Test Loss: {results['test_loss']:.4f}")
    if 'pearson_correlation' in results:
        print(f"Pearson Correlation: {results['pearson_correlation']:.4f}")
        print(f"Spearman Correlation: {results['spearman_correlation']:.4f}")
        print(f"R² Score: {results['r2_score']:.4f}")
        print(f"Number of valid pairs: {results['num_pairs']}")
    print("==============================\n")