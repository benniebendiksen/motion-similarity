#!/usr/bin/env python3
"""
Migration helper to integrate clustering-based neutral learning into existing code.
This can be imported and used with minimal changes to your current pipeline.
"""

import numpy as np
import torch
from pathlib import Path
import pickle
from typing import Dict, Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def integrate_clustering_into_existing_pipeline(config, existing_triplet_modules):
    """
    Quick integration function to add clustering to your existing pipeline.
    
    Usage in your run_embedding_triplet_training.py:
    
    from migration_helper import integrate_clustering_into_existing_pipeline
    
    # After creating your triplet modules but before training:
    train_triplet_modules = integrate_clustering_into_existing_pipeline(
        config, train_triplet_modules
    )
    """
    from neutral_clustering import NeutralRepresentationLearner
    
    # Load the embedding data for each animation
    animation_data = {}
    
    for module in existing_triplet_modules:
        anim_name = module.anim_name
        
        # Load embeddings for this animation
        embedding_file = Path(config.similarity_exemplars_dir) / f"{anim_name}_embeddings_rots_only_labels_exemplars_dict_local.pickle"
        
        if embedding_file.exists():
            with open(embedding_file, 'rb') as f:
                animation_data[anim_name] = pickle.load(f)
            logger.info(f"Loaded embeddings for {anim_name}")
        else:
            # Fallback to original data
            logger.warning(f"Embedding file not found for {anim_name}, using original data")
            animation_data[anim_name] = module.dict_similarity_classes_exemplars
    
    # Learn neutral representations
    learner = NeutralRepresentationLearner(
        n_clusters=5,
        selection_strategy='centroid'
    )
    
    learned_neutrals = learner.learn_all_neutrals(animation_data)
    
    # Apply to existing modules
    for module in existing_triplet_modules:
        if module.anim_name in learned_neutrals:
            neutral = learned_neutrals[module.anim_name]
            if not isinstance(neutral, torch.Tensor):
                neutral = torch.tensor(neutral, dtype=torch.float32)
            
            # Update the module
            module.neutral_embedding = neutral
            module.bool_fixed_neutral_embedding = True
            module.bool_drop_neutral_exemplar = False  # Don't drop, use learned
            
            logger.info(f"Updated {module.anim_name} with learned neutral")
    
    return existing_triplet_modules


def quick_fix_for_picking_performance(picking_triplet_module, config):
    """
    Specific fix for Picking performance issues.
    
    Usage:
    from migration_helper import quick_fix_for_picking_performance
    
    # For the picking module specifically:
    if triplet_module.anim_name == 'picking':
        triplet_module = quick_fix_for_picking_performance(triplet_module, config)
    """
    from neutral_clustering import NeutralRepresentationLearner
    
    # Load picking embeddings
    embedding_file = Path(config.similarity_exemplars_dir) / "picking_embeddings_rots_only_labels_exemplars_dict_local.pickle"
    
    if not embedding_file.exists():
        logger.warning("Picking embedding file not found")
        return picking_triplet_module
    
    with open(embedding_file, 'rb') as f:
        picking_data = pickle.load(f)
    
    # Use more clusters for picking (it seems to need finer granularity)
    learner = NeutralRepresentationLearner(
        n_clusters=7,  # More clusters for picking
        selection_strategy='weighted'  # Weighted strategy often works better for picking
    )
    
    neutral = learner.learn_neutral_representation(
        picking_data,
        'picking',
        validate_clustering=True
    )
    
    # Apply to module
    if not isinstance(neutral, torch.Tensor):
        neutral = torch.tensor(neutral, dtype=torch.float32)
    
    picking_triplet_module.neutral_embedding = neutral
    picking_triplet_module.bool_fixed_neutral_embedding = True
    picking_triplet_module.bool_drop_neutral_exemplar = False
    
    logger.info(f"Applied enhanced neutral to Picking module")
    
    return picking_triplet_module


def modify_existing_triplet_mining_class():
    """
    Minimal modifications needed for your existing TripletMining class.
    Add this method to your networks/triplet_mining.py:
    """
    code = '''
    def set_learned_neutral(self, learned_neutral_embedding):
        """
        Set a pre-learned neutral representation.
        
        Args:
            learned_neutral_embedding: numpy array or torch tensor
        """
        if not isinstance(learned_neutral_embedding, torch.Tensor):
            learned_neutral_embedding = torch.tensor(
                learned_neutral_embedding, 
                dtype=torch.float32
            )
        
        self.neutral_embedding = learned_neutral_embedding
        self.bool_fixed_neutral_embedding = True
        
        # Update tensor_dists_class_neut to use new neutral
        if hasattr(self, 'dict_similarity_classes_exemplars'):
            # Recalculate distances with new neutral
            self._recalculate_neutral_distances()
    
    def _recalculate_neutral_distances(self):
        """Recalculate class-neutral distances with new neutral."""
        # This would need to be called during training
        # when embeddings are available
        pass
    '''
    return code


# Minimal changes needed in your run_embedding_triplet_training.py:
INTEGRATION_EXAMPLE = """
# In your existing run_embedding_triplet_training.py, add these lines:

from neutral_clustering import NeutralRepresentationLearner

# After loading your similarity dictionaries but before creating triplet modules:

# Learn neutral representations
neutral_learner = NeutralRepresentationLearner(n_clusters=5, selection_strategy='centroid')
animation_dicts = {
    'walking': walking_similarity_dict,
    'pointing': pointing_similarity_dict,  
    'picking': picking_similarity_dict
}
learned_neutrals = neutral_learner.learn_all_neutrals(animation_dicts)

# When creating triplet modules, use the learned neutrals:
for i, anim_name in enumerate(animation_names):
    triplet_module = TripletMining(
        bool_drop=False,  # Don't drop neutral
        bool_fixed=True,  # Fix the neutral
        squared_left_right=False,
        squared_class_neut=False,
        anim_name=anim_name,
        config=config,
        valid_indices=train_indices[i]
    )
    
    # Set the learned neutral
    if anim_name in learned_neutrals:
        triplet_module.set_learned_neutral(learned_neutrals[anim_name])
    
    train_triplet_modules.append(triplet_module)
"""


def validate_improvements(old_results, new_results):
    """
    Helper to validate that the changes are improving results.
    
    Args:
        old_results: Dictionary of old Spearman correlations
        new_results: Dictionary of new Spearman correlations
    
    Returns:
        Dictionary with improvement metrics
    """
    improvements = {}
    
    for key in old_results:
        if key in new_results:
            old_val = old_results[key]
            new_val = new_results[key]
            improvement = new_val - old_val
            percent_improvement = (improvement / abs(old_val)) * 100 if old_val != 0 else 0
            
            improvements[key] = {
                'old': old_val,
                'new': new_val,
                'absolute_improvement': improvement,
                'percent_improvement': percent_improvement
            }
    
    # Print summary
    logger.info("=" * 50)
    logger.info("IMPROVEMENT SUMMARY")
    logger.info("=" * 50)
    
    for key, metrics in improvements.items():
        logger.info(f"{key}:")
        logger.info(f"  Old Spearman: {metrics['old']:.4f}")
        logger.info(f"  New Spearman: {metrics['new']:.4f}")
        logger.info(f"  Improvement:  {metrics['absolute_improvement']:.4f} "
                   f"({metrics['percent_improvement']:.1f}%)")
    
    avg_improvement = np.mean([m['absolute_improvement'] for m in improvements.values()])
    logger.info(f"\nAverage Improvement: {avg_improvement:.4f}")
    
    return improvements


# Specific configuration recommendations based on your results
RECOMMENDED_CONFIGS = {
    'walking': {
        'n_clusters': 5,
        'selection_strategy': 'centroid',
        'notes': 'Walking already performs reasonably well, standard config should work'
    },
    'pointing': {
        'n_clusters': 7,
        'selection_strategy': 'weighted',
        'notes': 'Pointing has lowest performance, needs more clusters and weighted strategy'
    },
    'picking': {
        'n_clusters': 6,
        'selection_strategy': 'medoid',
        'notes': 'Picking benefits from using actual samples (medoid) as neutral'
    }
}


if __name__ == "__main__":
    print("Migration Helper Loaded")
    print("=" * 50)
    print("QUICK START GUIDE:")
    print("=" * 50)
    print("\n1. Import the helper functions:")
    print("   from migration_helper import integrate_clustering_into_existing_pipeline")
    print("\n2. Apply to your existing modules:")
    print("   train_triplet_modules = integrate_clustering_into_existing_pipeline(config, train_triplet_modules)")
    print("\n3. Continue with your normal training")
    print("\nFor action-specific configurations, see RECOMMENDED_CONFIGS")
    print("\nFor minimal code changes needed, see INTEGRATION_EXAMPLE")
