#!/usr/bin/env python3
"""
Enhanced motion triplet training with clustering-based neutral initialization.

This script extends the original run_motion_triplet_training.py to:
1. Learn neutral representations using clustering from raw motion data before training
2. Use these learned neutrals as initialized anchors during triplet training
3. Compare results with the embedding-based clustering approach
"""

import os
import sys
import random
import argparse
import torch
import numpy as np
from pathlib import Path
import logging

curr_path = os.getcwd()
sys.path.append(curr_path)
sys.path.append(curr_path + '\networks')

from networks.similarity_network import SimilarityNetwork
from networks.similarity_data_loader import SimilarityDataLoader
from networks.triplet_mining import TripletMining
from Config import Config
import src.organize_synthetic_data as osd
from neutral_clustering_motion import MotionNeutralRepresentationLearner, integrate_motion_neutrals

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_train_val_split(similarity_dicts_list, val_ratio=0.4):
    """
    Create deterministic training and validation indices for each animation type.

    Args:
        similarity_dicts_list: List of dictionaries containing class exemplars for each animation
        val_ratio: Ratio of classes to use for validation

    Returns:
        train_indices: List of sets containing training class indices for each animation
        val_indices: List of sets containing validation class indices for each animation
    """
    train_indices = []
    val_indices = []

    for anim_dict in similarity_dicts_list:
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


def learn_motion_neutrals(list_similarity_dicts, animation_names, config, clustering_config):
    """
    Learn neutral representations from raw motion data using clustering.
    
    Args:
        list_similarity_dicts: List of similarity dictionaries containing motion data
        animation_names: List of animation names
        config: Configuration object
        clustering_config: Dictionary with clustering parameters
        
    Returns:
        Dictionary mapping animation names to learned neutral motion sequences
    """
    logger.info("=" * 60)
    logger.info("LEARNING NEUTRAL REPRESENTATIONS FROM RAW MOTION")
    logger.info("=" * 60)
    
    # Create the learner
    learner = MotionNeutralRepresentationLearner(**clustering_config)
    
    # Prepare animation dictionaries
    animation_dicts = {}
    for i, anim_name in enumerate(animation_names):
        animation_dicts[anim_name] = list_similarity_dicts[i]
        logger.info(f"Prepared {anim_name}: {len(list_similarity_dicts[i])} effort classes")
    
    # Learn neutrals for all animations
    learned_neutrals = learner.learn_all_neutrals(animation_dicts, validate=True)
    
    # Save the learned neutrals
    neutral_save_path = Path(config.checkpoint_root_dir) / "learned_motion_neutrals.pkl"
    learner.save_neutrals(neutral_save_path)
    logger.info(f"Saved learned neutrals to {neutral_save_path}")
    
    # Print summary
    logger.info("\nLearned Neutral Representations Summary:")
    for anim_name, neutral in learned_neutrals.items():
        logger.info(f"  {anim_name}: shape {neutral.shape}")
        if anim_name in learner.cluster_stats:
            logger.info(f"    Silhouette score: {learner.cluster_stats[anim_name]['silhouette_score']:.3f}")
    
    return learned_neutrals, learner


def replace_neutrals_in_dicts(list_similarity_dicts, learned_neutrals, animation_names):
    """
    Replace the original neutral exemplars with learned ones in the similarity dictionaries.
    
    Args:
        list_similarity_dicts: List of similarity dictionaries
        learned_neutrals: Dictionary of learned neutral representations
        animation_names: List of animation names
        
    Returns:
        Modified list of similarity dictionaries
    """
    for i, anim_name in enumerate(animation_names):
        if anim_name in learned_neutrals:
            neutral_motion = learned_neutrals[anim_name]
            
            # Replace the neutral in the dictionary
            if (0, 0, 0, 0) in list_similarity_dicts[i]:
                # Store as a list with single element (matching the structure)
                list_similarity_dicts[i][(0, 0, 0, 0)] = [neutral_motion]
                logger.info(f"Replaced neutral for {anim_name} with learned representation")
            else:
                logger.warning(f"No neutral found in {anim_name} dictionary to replace")
    
    return list_similarity_dicts


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced motion triplet training with clustering-based neutral initialization'
    )
    
    # Original arguments
    parser.add_argument('--use-perception-loss', default=False,
                       help='Use enhanced perception-aligned loss')
    parser.add_argument('--use-adaptive-distance', default=False,
                       help='Use adaptive distance module')
    parser.add_argument('--scheduler', type=str, default='plateau', 
                       choices=['plateau', 'cosine', 'step'],
                       help='Learning rate scheduler type')
    parser.add_argument('--animation', type=str, default='all', 
                       choices=['all', 'walking', 'pointing', 'picking', 'walking_pointing'],
                       help='Which animation type to train on')
    parser.add_argument('--task-index', type=str, 
                       help='Task index for distributed training')
    
    # Clustering-specific arguments
    parser.add_argument('--use-clustering', default=True,
                       help='Use clustering-based neutral initialization')
    parser.add_argument('--n-clusters', type=int, default=5,
                       help='Number of clusters for K-means')
    parser.add_argument('--reduction-method', type=str, default='temporal_mean',
                       choices=['temporal_mean', 'temporal_stats', 'pca', 'pca_temporal'],
                       help='Method for reducing motion dimensionality')
    parser.add_argument('--pca-components', type=int, default=None,
                       help='Number of PCA components (if using PCA reduction)')
    parser.add_argument('--selection-strategy', type=str, default='centroid',
                       choices=['centroid', 'medoid', 'weighted'],
                       help='Strategy for selecting neutral from clusters')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of training epochs')
    
    args = parser.parse_args()

    # Initialize configuration
    task_index = args.task_index if args.task_index else None
    config = Config(task_index)
    
    # Override epochs if specified
    if args.epochs:
        config.n_similarity_epochs = args.epochs

    # Ensure required directories exist
    config.ensure_directories_exist()

    # Architecture variant
    arch_variant = int(config.num_task) if config.num_task else 0

    # Configuration for triplet mining
    bool_drop_neutral_exemplar = False  # Keep neutral, we'll replace it with learned one
    bool_fixed_neutral_embedding = False  # Will be learned during first forward pass
    squared_left_right_euc_dist = False
    squared_class_neut_euc_dist = False

    # Load similarity data for selected animations
    logger.info(f"Loading motion data for: {args.animation}")
    
    list_similarity_dicts = []
    animation_names = []
    
    if args.animation == 'walking':
        walking_dict = osd.load_similarity_data(False, "walking", config)["train"]
        list_similarity_dicts = [walking_dict]
        animation_names = ['walking']
        
    elif args.animation == 'pointing':
        pointing_dict = osd.load_similarity_data(False, "pointing", config)["train"]
        list_similarity_dicts = [pointing_dict]
        animation_names = ['pointing']
        
    elif args.animation == 'picking':
        picking_dict = osd.load_similarity_data(False, "picking", config)["train"]
        list_similarity_dicts = [picking_dict]
        animation_names = ['picking']
        
    elif args.animation == 'walking_pointing':
        walking_dict = osd.load_similarity_data(False, "walking", config)["train"]
        pointing_dict = osd.load_similarity_data(False, "pointing", config)["train"]
        list_similarity_dicts = [walking_dict, pointing_dict]
        animation_names = ['walking', 'pointing']
        
    else:  # 'all'
        walking_dict = osd.load_similarity_data(False, "walking", config)["train"]
        pointing_dict = osd.load_similarity_data(False, "pointing", config)["train"]
        picking_dict = osd.load_similarity_data(False, "picking", config)["train"]
        list_similarity_dicts = [walking_dict, pointing_dict, picking_dict]
        animation_names = ['walking', 'pointing', 'picking']

    # Learn clustering-based neutrals if requested
    learned_neutrals = None
    learner = None
    
    if args.use_clustering:
        logger.info("\n" + "=" * 60)
        logger.info("CLUSTERING-BASED NEUTRAL INITIALIZATION ENABLED")
        logger.info("=" * 60)
        
        # Prepare clustering configuration
        clustering_config = {
            'n_clusters': args.n_clusters,
            'reduction_method': args.reduction_method,
            'pca_components': args.pca_components,
            'selection_strategy': args.selection_strategy,
            'random_state': 42
        }
        
        logger.info(f"Clustering configuration: {clustering_config}")
        
        # Learn neutrals from motion data
        learned_neutrals, learner = learn_motion_neutrals(
            list_similarity_dicts,
            animation_names,
            config,
            clustering_config
        )
        
        # Replace original neutrals with learned ones
        list_similarity_dicts = replace_neutrals_in_dicts(
            list_similarity_dicts,
            learned_neutrals,
            animation_names
        )
        
        logger.info("Neutral replacement complete")
        logger.info("=" * 60 + "\n")

    # Balance frame counts
    list_similarity_dicts = osd.balance_single_exemplar_similarity_classes_by_frame_count(
        list_similarity_dicts, 137
    )

    # Create train/val split
    train_indices, val_indices = create_train_val_split(list_similarity_dicts)

    # Create data loaders
    train_loader = SimilarityDataLoader(list_similarity_dicts, config, True, train_indices)
    val_loader = SimilarityDataLoader(list_similarity_dicts, config, True, val_indices)

    # Create training and validation triplet modules
    train_triplet_modules = []
    val_triplet_modules = []

    for i, anim_name in enumerate(animation_names):
        idx = 0 if len(animation_names) == 1 else i

        train_triplet = TripletMining(
            bool_drop_neutral_exemplar, 
            bool_fixed_neutral_embedding,
            squared_left_right_euc_dist, 
            squared_class_neut_euc_dist,
            anim_name, 
            config, 
            valid_indices=train_indices[idx]
        )

        val_triplet = TripletMining(
            bool_drop_neutral_exemplar, 
            bool_fixed_neutral_embedding,
            squared_left_right_euc_dist, 
            squared_class_neut_euc_dist,
            anim_name, 
            config, 
            valid_indices=val_indices[idx]
        )

        train_triplet_modules.append(train_triplet)
        val_triplet_modules.append(val_triplet)

    # Create the similarity network
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

    # Print training configuration
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Animation(s): {args.animation}")
    logger.info(f"Architecture variant: {arch_variant}")
    logger.info(f"Using clustering-based neutrals: {args.use_clustering}")
    if args.use_clustering:
        logger.info(f"  Reduction method: {args.reduction_method}")
        logger.info(f"  N clusters: {args.n_clusters}")
        logger.info(f"  Selection strategy: {args.selection_strategy}")
    logger.info(f"Using perception-aligned loss: {args.use_perception_loss}")
    logger.info(f"Using adaptive distance: {args.use_adaptive_distance}")
    logger.info(f"Learning rate scheduler: {args.scheduler}")
    logger.info(f"Number of epochs: {config.n_similarity_epochs}")
    logger.info(f"Device: {similarity_network.device}")
    logger.info("=" * 60 + "\n")

    # Train the model
    logger.info("Starting training...")
    similarity_network.run_model_training()

    # Evaluate the model
    logger.info("\nEvaluating model...")
    results = similarity_network.evaluate()

    # Print final results
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Test Loss: {results['test_loss']:.4f}")
    logger.info(f"Pearson Correlation: {results['pearson_correlation']:.4f} "
               f"(p={results['pearson_p_value']:.4f})")
    logger.info(f"Spearman Correlation: {results['spearman_correlation']:.4f} "
               f"(p={results['spearman_p_value']:.4f})")
    logger.info(f"R² Score: {results['r2_score']:.4f}")
    logger.info(f"Valid pairs: {results['num_pairs']}")
    logger.info("=" * 60)
    
    # Save summary
    summary_path = Path(config.checkpoint_root_dir) / "motion_training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Enhanced Motion Triplet Training Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Animation(s): {args.animation}\n")
        f.write(f"Clustering enabled: {args.use_clustering}\n")
        if args.use_clustering:
            f.write(f"Clustering config:\n")
            f.write(f"  Reduction method: {args.reduction_method}\n")
            f.write(f"  N clusters: {args.n_clusters}\n")
            f.write(f"  Selection strategy: {args.selection_strategy}\n")
            if learner and learner.cluster_stats:
                f.write("\nClustering Statistics:\n")
                for anim_name, stats in learner.cluster_stats.items():
                    f.write(f"  {anim_name}: silhouette={stats['silhouette_score']:.3f}\n")
        f.write(f"\nTraining config:\n")
        f.write(f"  Epochs: {config.n_similarity_epochs}\n")
        f.write(f"  Perception loss: {args.use_perception_loss}\n")
        f.write(f"  Adaptive distance: {args.use_adaptive_distance}\n")
        f.write(f"\nFinal Results:\n")
        f.write(f"  Test Loss: {results['test_loss']:.4f}\n")
        f.write(f"  Pearson: {results['pearson_correlation']:.4f}\n")
        f.write(f"  Spearman: {results['spearman_correlation']:.4f}\n")
        f.write(f"  R²: {results['r2_score']:.4f}\n")
    
    logger.info(f"\nSaved training summary to {summary_path}")
    logger.info("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
