#!/usr/bin/env python3
"""
Enhanced triplet training with clustering-based neutral initialization.

This script extends the original run_embedding_triplet_training.py to:
1. Learn neutral representations using clustering before training
2. Use these learned neutrals as fixed anchors during triplet training  
3. Improve alignment with human perception ratings
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Add current directory to path
# Resolve project root from this file's location so the script works
# whether invoked from pipelines/ or from the project root.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'networks'))
# Import necessary modules
from embedding_dataset import EmbeddingDataset, create_embedding_similarity_data
from src.organize_synthetic_data import load_similarity_data_from_embeddings, EmbeddingSimilarityDataLoader
from networks.similarity_network import EmbeddingRefiningSimilarityNetwork
from networks.triplet_mining import TripletMining
from Config import Config
from neutral_clustering import NeutralRepresentationLearner, integrate_learned_neutrals

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedEmbeddingTrainer:
    """
    Enhanced trainer with clustering-based neutral initialization.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the enhanced trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.neutral_learner = None
        self.learned_neutrals = {}
        self.animation_dicts = {}  # populated by setup_clustering_based_training
        
    def setup_clustering_based_training(self,
                                       embedding_dirs: Dict[str, str],
                                       combination_method: str = 'rots_only',
                                       clustering_config: Optional[Dict] = None) -> Tuple:
        """
        Set up training with clustering-based neutral initialization.
        
        Args:
            embedding_dirs: Dictionary mapping animation names to embedding directories
            combination_method: How to combine embeddings
            clustering_config: Configuration for clustering
            
        Returns:
            Tuple of (train_loader, val_loader, train_triplet_modules, val_triplet_modules)
        """
        logger.info("Setting up clustering-based embedding training...")
        
        # Default clustering configuration
        if clustering_config is None:
            clustering_config = {
                'n_clusters': 5,  # Increased for better granularity
                'selection_strategy': 'centroid',
                'random_state': 42
            }
        
        # Initialize neutral learner
        self.neutral_learner = NeutralRepresentationLearner(**clustering_config)
        
        # Load similarity data for each animation
        animation_dicts = {}
        animation_names = []
        
        for anim_name, embedding_dir in embedding_dirs.items():
            logger.info(f"Loading {anim_name} embeddings from {embedding_dir}")
            
            try:
                similarity_dict = load_similarity_data_from_embeddings(
                    bool_drop=False,  # Don't drop neutral yet, we'll learn it
                    anim_name=anim_name,
                    config=self.config,
                    embedding_dir=embedding_dir,
                    combination_method=combination_method,
                    force_regenerate=True
                )["train"]
                
                animation_dicts[anim_name] = similarity_dict
                animation_names.append(anim_name)
                
                logger.info(f"Loaded {len(similarity_dict)} {anim_name} effort classes")
                
            except Exception as e:
                logger.error(f"Error loading {anim_name} data: {e}")
                raise
        
        # Learn neutral representations using clustering
        logger.info("Learning neutral representations using clustering...")
        self.learned_neutrals = self.neutral_learner.learn_all_neutrals(
            animation_dicts,
            validate=True
        )
        
        # Save learned neutrals for reproducibility
        neutral_save_path = Path(self.config.checkpoint_root_dir) / "learned_neutrals.pkl"
        self.neutral_learner.save_neutrals(neutral_save_path)
        
        # Replace existing neutral exemplars with learned ones
        for anim_name, neutral_rep in self.learned_neutrals.items():
            if anim_name in animation_dicts:
                # Remove old neutral if it exists
                if (0, 0, 0, 0) in animation_dicts[anim_name]:
                    del animation_dicts[anim_name][(0, 0, 0, 0)]
                
                # Add learned neutral as the neutral exemplar
                animation_dicts[anim_name][(0, 0, 0, 0)] = [neutral_rep]
                logger.info(f"Replaced {anim_name} neutral with learned representation")
        
        # Create train/validation split
        train_indices, val_indices = self._create_stratified_split(
            list(animation_dicts.values()),
            val_ratio=0.4
        )
        
        # Store for use in _create_triplet_modules_with_learned_neutrals
        self.animation_dicts = animation_dicts

        # Create data loaders
        list_similarity_dicts = list(animation_dicts.values())

        train_loader = EmbeddingSimilarityDataLoader(
            list_similarity_dicts, 
            self.config, 
            shuffle=True, 
            valid_indices=train_indices
        )
        
        val_loader = EmbeddingSimilarityDataLoader(
            list_similarity_dicts, 
            self.config, 
            shuffle=False, 
            valid_indices=val_indices
        )
        
        logger.info(f"Created data loaders:")
        logger.info(f"  Training: {train_loader.num_classes} classes")
        logger.info(f"  Validation: {val_loader.num_classes} classes")
        
        # Create triplet modules with learned neutrals
        train_triplet_modules = self._create_triplet_modules_with_learned_neutrals(
            animation_names,
            train_indices,
            is_training=True
        )
        
        val_triplet_modules = self._create_triplet_modules_with_learned_neutrals(
            animation_names,
            val_indices,
            is_training=False
        )
        
        return train_loader, val_loader, train_triplet_modules, val_triplet_modules
    
    def _create_stratified_split(self, 
                                similarity_dicts: List[Dict],
                                val_ratio: float = 0.5) -> Tuple[List, List]:
        """
        Create stratified train/validation split ensuring balanced effort representation.
        
        Args:
            similarity_dicts: List of similarity dictionaries
            val_ratio: Validation set ratio
            
        Returns:
            Tuple of (train_indices, val_indices)
        """
        train_indices = []
        val_indices = []
        
        for anim_dict in similarity_dicts:
            # Get all keys except neutral
            keys = [k for k in anim_dict.keys() if k != (0, 0, 0, 0)]
            
            # Group by effort magnitude for stratified sampling
            effort_groups = {}
            for key in keys:
                magnitude = sum(abs(e) for e in key)
                if magnitude not in effort_groups:
                    effort_groups[magnitude] = []
                effort_groups[magnitude].append(key)
            
            # Sample from each group proportionally
            train_keys = set()
            val_keys = set()
            
            for magnitude, group_keys in effort_groups.items():
                n_val = max(1, int(len(group_keys) * val_ratio))
                np.random.shuffle(group_keys)
                
                val_keys.update(group_keys[:n_val])
                train_keys.update(group_keys[n_val:])
            
            # Always include neutral in both sets
            if (0, 0, 0, 0) in anim_dict:
                train_keys.add((0, 0, 0, 0))
                val_keys.add((0, 0, 0, 0))
            
            train_indices.append(train_keys)
            val_indices.append(val_keys)
        
        return train_indices, val_indices
    
    def _create_triplet_modules_with_learned_neutrals(self,
                                                     animation_names: List[str],
                                                     indices: List[set],
                                                     is_training: bool = True) -> List:
        """
        Create triplet mining modules with learned neutral representations.
        
        Args:
            animation_names: List of animation names
            indices: Valid indices for each animation
            is_training: Whether these are training modules
            
        Returns:
            List of TripletMining modules
        """
        modules = []
        
        for i, anim_name in enumerate(animation_names):
            # Create module with appropriate settings
            module = TripletMining(
                bool_drop=False,  # Don't drop neutral, we'll use learned one
                bool_fixed=True,  # Fix the neutral representation
                squared_left_right=False,
                squared_class_neut=False,
                anim_name=anim_name,
                config=self.config,
                valid_indices=indices[i],
                exclude_neutral_completely=False,  # Use neutral in training
                preloaded_dict=self.animation_dicts.get(anim_name)
            )
            
            # Set the learned neutral representation
            if anim_name in self.learned_neutrals:
                neutral = self.learned_neutrals[anim_name]
                if not isinstance(neutral, torch.Tensor):
                    neutral = torch.tensor(neutral, dtype=torch.float32)
                
                module.neutral_embedding = neutral
                module.bool_fixed_neutral_embedding = True
                
                logger.info(f"Set learned neutral for {anim_name} ({'training' if is_training else 'validation'})")
            
            modules.append(module)
        
        return modules


def run_enhanced_triplet_training(
        embedding_dirs: Dict[str, str],
        combination_method: str = 'rots_only',
        clustering_config: Optional[Dict] = None,
        training_config: Optional[Dict] = None
) -> Tuple:
    """
    Run enhanced triplet training with clustering-based neutral initialization.
    
    Args:
        embedding_dirs: Dictionary mapping animation names to embedding directories
        combination_method: How to combine embeddings
        clustering_config: Configuration for clustering
        training_config: Configuration for training
        
    Returns:
        Tuple of (trained_network, results)
    """
    logger.info("=" * 60)
    logger.info("ENHANCED TRIPLET TRAINING WITH CLUSTERING-BASED NEUTRALS")
    logger.info("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Apply training configuration
    if training_config is None:
        training_config = {
            'n_epochs': 200,
            'scheduler': 'plateau',
            'use_perception_loss': False,
            'use_adaptive_distance': False
        }
    
    config.n_similarity_epochs = training_config.get('n_epochs', 200)
    config.ensure_directories_exist()
    
    # Create enhanced trainer
    trainer = EnhancedEmbeddingTrainer(config)
    
    # Setup training with clustering
    train_loader, val_loader, train_triplet_modules, val_triplet_modules = \
        trainer.setup_clustering_based_training(
            embedding_dirs,
            combination_method,
            clustering_config
        )
    
    # Create similarity network
    logger.info("Creating embedding-refining similarity network...")
    
    similarity_network = EmbeddingRefiningSimilarityNetwork(
        train_loader=train_loader,
        validation_loader=val_loader,
        test_loader=val_loader,
        checkpoint_root_dir=config.checkpoint_root_dir,
        triplet_modules=train_triplet_modules,
        val_triplet_modules=val_triplet_modules,
        architecture_variant=0,
        config=config,
        lr_scheduler_type=training_config.get('scheduler', 'plateau'),
        use_perception_loss=training_config.get('use_perception_loss', False),
        use_adaptive_distance=training_config.get('use_adaptive_distance', False)
    )
    
    # Print configuration summary
    logger.info("\n" + "=" * 40)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 40)
    logger.info(f"Embedding directories: {list(embedding_dirs.keys())}")
    logger.info(f"Combination method: {combination_method}")
    logger.info(f"Clustering strategy: {clustering_config}")
    logger.info(f"Training epochs: {config.n_similarity_epochs}")
    logger.info(f"Device: {similarity_network.device}")
    logger.info("=" * 40)
    
    # Run training
    logger.info("\nStarting training...")
    similarity_network.run_model_training()
    
    # Evaluate
    logger.info("\nEvaluating model...")
    results = similarity_network.evaluate()
    
    # Print final results
    logger.info("\n" + "=" * 40)
    logger.info("FINAL RESULTS")
    logger.info("=" * 40)
    logger.info(f"Test Loss: {results['test_loss']:.4f}")
    logger.info("=" * 40)
    
    # Save training summary
    summary_path = Path(config.checkpoint_root_dir) / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Enhanced Triplet Training Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Clustering Config: {clustering_config}\n")
        f.write(f"Training Config: {training_config}\n")
        f.write(f"Final Test Loss: {results['test_loss']:.4f}\n")
        
        # Write learned neutral statistics
        if trainer.neutral_learner and trainer.neutral_learner.cluster_stats:
            f.write("\nClustering Statistics:\n")
            for anim_name, stats in trainer.neutral_learner.cluster_stats.items():
                f.write(f"  {anim_name}: {stats}\n")
    
    logger.info(f"Saved training summary to {summary_path}")
    
    return similarity_network, results


def main():
    parser = argparse.ArgumentParser(description='Enhanced triplet training with clustering-based neutrals')
    
    # Embedding directories
    parser.add_argument('--walking-dir', type=str, 
                       default='../datasets/lma_perform_walking_ae_paired',
                       help='Directory for walking embeddings')
    parser.add_argument('--pointing-dir', type=str,
                       default='../datasets/lma_perform_pointing_ae_paired',
                       help='Directory for pointing embeddings')
    parser.add_argument('--picking-dir', type=str,
                       default='../datasets/lma_perform_picking_ae_paired',
                       help='Directory for picking embeddings')
    
    # Embedding configuration
    parser.add_argument('--combination-method', type=str, default='rots_only',
                       choices=['concat', 'weighted', 'root_only', 'rots_only'],
                       help='How to combine embeddings')
    
    # Clustering configuration
    parser.add_argument('--n-clusters', type=int, default=5,
                       help='Number of clusters for K-means')
    parser.add_argument('--selection-strategy', type=str, default='centroid',
                       choices=['centroid', 'medoid', 'weighted'],
                       help='Strategy for selecting neutral from clusters')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'step'],
                       help='Learning rate scheduler')
    parser.add_argument('--use-perception-loss', action='store_true',
                       help='Use perception-aligned loss')
    parser.add_argument('--use-adaptive-distance', action='store_true',
                       help='Use adaptive distance module')
    
    args = parser.parse_args()
    
    # Prepare embedding directories
    embedding_dirs = {
        'walking': args.walking_dir,
        'pointing': args.pointing_dir,
        'picking': args.picking_dir
    }
    
    # Validate directories
    for anim_name, dir_path in embedding_dirs.items():
        if not Path(dir_path).exists():
            logger.error(f"Directory {dir_path} for {anim_name} does not exist")
            sys.exit(1)
    
    # Prepare clustering configuration
    clustering_config = {
        'n_clusters': args.n_clusters,
        'selection_strategy': args.selection_strategy,
        'random_state': 42
    }
    
    # Prepare training configuration
    training_config = {
        'n_epochs': args.epochs,
        'scheduler': args.scheduler,
        'use_perception_loss': args.use_perception_loss,
        'use_adaptive_distance': args.use_adaptive_distance
    }
    
    # Run enhanced training
    try:
        network, results = run_enhanced_triplet_training(
            embedding_dirs,
            args.combination_method,
            clustering_config,
            training_config
        )
        
        logger.info("\nTraining completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
