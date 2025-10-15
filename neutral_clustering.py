#!/usr/bin/env python3
"""
Clustering-based neutral representation learning for motion embeddings.

This module implements K-means clustering to learn optimal neutral representations
for each action type (walking, pointing, picking) before triplet training.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple, Optional, Union
import pickle
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeutralRepresentationLearner:
    """
    Learn neutral representations for action types using clustering.
    
    This class implements a clustering-based approach to find optimal neutral
    representations for each action type, which can then be used as fixed
    anchors during triplet training.
    """
    
    def __init__(self, 
                 n_clusters: int = 3,
                 clustering_method: str = 'kmeans',
                 selection_strategy: str = 'centroid',
                 random_state: int = 42):
        """
        Initialize the neutral representation learner.
        
        Args:
            n_clusters: Number of clusters for K-means
            clustering_method: Clustering algorithm to use ('kmeans', 'spectral')
            selection_strategy: How to select neutral ('centroid', 'medoid', 'weighted')
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.selection_strategy = selection_strategy
        self.random_state = random_state
        
        # Storage for learned representations
        self.neutral_representations = {}
        self.cluster_models = {}
        self.cluster_stats = {}
    
    def learn_neutral_representation(self, 
                                    embeddings_dict: Dict[Tuple, List[np.ndarray]], 
                                    anim_name: str,
                                    validate_clustering: bool = True) -> np.ndarray:
        """
        Learn the neutral representation for a single animation type.
        
        Args:
            embeddings_dict: Dictionary mapping effort tuples to embedding lists
            anim_name: Name of the animation ('walking', 'pointing', 'picking')
            validate_clustering: Whether to validate clustering quality
            
        Returns:
            Learned neutral representation as numpy array
        """
        logger.info(f"Learning neutral representation for {anim_name}")
        
        # Prepare embeddings for clustering
        all_embeddings, effort_labels = self._prepare_embeddings(embeddings_dict)
        
        if all_embeddings.shape[0] < self.n_clusters:
            logger.warning(f"Not enough samples for clustering. Using mean representation.")
            neutral_rep = np.mean(all_embeddings, axis=0)
            self.neutral_representations[anim_name] = neutral_rep
            return neutral_rep
        
        # Perform clustering
        if self.clustering_method == 'kmeans':
            cluster_model = self._perform_kmeans(all_embeddings)
        else:
            raise NotImplementedError(f"Clustering method {self.clustering_method} not implemented")
        
        # Validate clustering if requested
        if validate_clustering and all_embeddings.shape[0] > self.n_clusters:
            score = self._validate_clustering(all_embeddings, cluster_model.labels_)
            logger.info(f"Clustering silhouette score for {anim_name}: {score:.3f}")
            self.cluster_stats[anim_name] = {'silhouette_score': score}
        
        # Select neutral representation based on strategy
        neutral_rep = self._select_neutral(cluster_model, all_embeddings, effort_labels)
        
        # Store results
        self.neutral_representations[anim_name] = neutral_rep
        self.cluster_models[anim_name] = cluster_model
        
        logger.info(f"Learned neutral representation shape: {neutral_rep.shape}")
        
        return neutral_rep
    
    def _prepare_embeddings(self, 
                           embeddings_dict: Dict[Tuple, List]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare embeddings for clustering by flattening the dictionary structure.
        
        Args:
            embeddings_dict: Dictionary of embeddings
            
        Returns:
            Tuple of (flattened embeddings array, effort labels array)
        """
        all_embeddings = []
        effort_labels = []
        
        for effort_tuple, embedding_list in embeddings_dict.items():
            # Skip the existing neutral if present
            if effort_tuple == (0, 0, 0, 0):
                continue
                
            for embedding in embedding_list:
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                elif not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                
                # Flatten if needed
                if len(embedding.shape) > 1:
                    embedding = embedding.flatten()
                    
                all_embeddings.append(embedding)
                effort_labels.append(effort_tuple)
        
        all_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        effort_labels = np.array(effort_labels)
        
        logger.info(f"Prepared {all_embeddings.shape[0]} embeddings for clustering")
        
        return all_embeddings, effort_labels
    
    def _perform_kmeans(self, embeddings: np.ndarray) -> KMeans:
        """
        Perform K-means clustering on embeddings.
        
        Args:
            embeddings: Array of embeddings to cluster
            
        Returns:
            Fitted KMeans model
        """
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        kmeans.fit(embeddings)
        
        # Log cluster sizes
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            logger.info(f"  Cluster {cluster_id}: {count} samples")
        
        return kmeans
    
    def _validate_clustering(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Validate clustering quality using silhouette score.
        
        Args:
            embeddings: Original embeddings
            labels: Cluster labels
            
        Returns:
            Silhouette score
        """
        try:
            score = silhouette_score(embeddings, labels)
            return score
        except Exception as e:
            logger.warning(f"Could not compute silhouette score: {e}")
            return -1.0
    
    def _select_neutral(self, 
                       cluster_model: KMeans, 
                       embeddings: np.ndarray,
                       effort_labels: np.ndarray) -> np.ndarray:
        """
        Select the neutral representation based on the configured strategy.
        
        Args:
            cluster_model: Fitted clustering model
            embeddings: Original embeddings
            effort_labels: Effort tuple labels for each embedding
            
        Returns:
            Selected neutral representation
        """
        if self.selection_strategy == 'centroid':
            # Find the cluster with lowest average effort magnitudes
            cluster_efforts = self._compute_cluster_efforts(
                cluster_model.labels_, effort_labels
            )
            
            # Select cluster with lowest total effort
            neutral_cluster_idx = np.argmin([
                np.sum(np.abs(efforts)) for efforts in cluster_efforts
            ])
            
            neutral_rep = cluster_model.cluster_centers_[neutral_cluster_idx]
            logger.info(f"Selected cluster {neutral_cluster_idx} as neutral (centroid strategy)")
            
        elif self.selection_strategy == 'medoid':
            # Find the actual sample closest to the cluster center
            neutral_cluster_idx = self._find_neutral_cluster(
                cluster_model.labels_, effort_labels
            )
            
            cluster_mask = cluster_model.labels_ == neutral_cluster_idx
            cluster_embeddings = embeddings[cluster_mask]
            
            # Find medoid (sample closest to centroid)
            center = cluster_model.cluster_centers_[neutral_cluster_idx]
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            medoid_idx = np.argmin(distances)
            
            neutral_rep = cluster_embeddings[medoid_idx]
            logger.info(f"Selected medoid from cluster {neutral_cluster_idx}")
            
        elif self.selection_strategy == 'weighted':
            # Weighted average of all cluster centers based on effort magnitudes
            weights = []
            for i in range(self.n_clusters):
                cluster_mask = cluster_model.labels_ == i
                cluster_efforts = effort_labels[cluster_mask]
                
                # Compute inverse of average effort magnitude as weight
                avg_effort = np.mean([np.sum(np.abs(e)) for e in cluster_efforts])
                weight = 1.0 / (1.0 + avg_effort)  # Avoid division by zero
                weights.append(weight)
            
            weights = np.array(weights)
            weights /= weights.sum()  # Normalize
            
            neutral_rep = np.sum(
                cluster_model.cluster_centers_ * weights[:, np.newaxis], 
                axis=0
            )
            logger.info(f"Created weighted neutral with weights: {weights}")
            
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
        
        return neutral_rep
    
    def _compute_cluster_efforts(self, 
                                labels: np.ndarray, 
                                effort_labels: np.ndarray) -> List[np.ndarray]:
        """
        Compute average effort values for each cluster.
        
        Args:
            labels: Cluster labels
            effort_labels: Effort tuples
            
        Returns:
            List of average effort arrays per cluster
        """
        cluster_efforts = []
        
        for i in range(self.n_clusters):
            cluster_mask = labels == i
            cluster_effort_tuples = effort_labels[cluster_mask]
            
            if len(cluster_effort_tuples) > 0:
                avg_efforts = np.mean(cluster_effort_tuples, axis=0)
            else:
                avg_efforts = np.zeros(4)
            
            cluster_efforts.append(avg_efforts)
        
        return cluster_efforts
    
    def _find_neutral_cluster(self, 
                            labels: np.ndarray, 
                            effort_labels: np.ndarray) -> int:
        """
        Find the cluster that best represents neutral motion.
        
        Args:
            labels: Cluster labels
            effort_labels: Effort tuples
            
        Returns:
            Index of the neutral cluster
        """
        min_effort_sum = float('inf')
        neutral_cluster = 0
        
        for i in range(self.n_clusters):
            cluster_mask = labels == i
            cluster_efforts = effort_labels[cluster_mask]
            
            if len(cluster_efforts) > 0:
                # Sum of absolute effort values
                effort_sum = np.sum(np.abs(cluster_efforts))
                
                if effort_sum < min_effort_sum:
                    min_effort_sum = effort_sum
                    neutral_cluster = i
        
        return neutral_cluster
    
    def learn_all_neutrals(self, 
                          animation_dicts: Dict[str, Dict],
                          validate: bool = True) -> Dict[str, np.ndarray]:
        """
        Learn neutral representations for all animation types.
        
        Args:
            animation_dicts: Dictionary mapping animation names to embedding dicts
            validate: Whether to validate clustering quality
            
        Returns:
            Dictionary mapping animation names to neutral representations
        """
        for anim_name, embeddings_dict in animation_dicts.items():
            self.learn_neutral_representation(
                embeddings_dict, 
                anim_name, 
                validate_clustering=validate
            )
        
        return self.neutral_representations
    
    def save_neutrals(self, filepath: Union[str, Path]) -> None:
        """
        Save learned neutral representations to disk.
        
        Args:
            filepath: Path to save the representations
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'neutral_representations': self.neutral_representations,
            'cluster_stats': self.cluster_stats,
            'config': {
                'n_clusters': self.n_clusters,
                'clustering_method': self.clustering_method,
                'selection_strategy': self.selection_strategy,
                'random_state': self.random_state
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Saved neutral representations to {filepath}")
    
    def load_neutrals(self, filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Load previously learned neutral representations.
        
        Args:
            filepath: Path to load representations from
            
        Returns:
            Dictionary of neutral representations
        """
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.neutral_representations = save_dict['neutral_representations']
        self.cluster_stats = save_dict.get('cluster_stats', {})
        
        # Restore config if available
        if 'config' in save_dict:
            config = save_dict['config']
            self.n_clusters = config.get('n_clusters', self.n_clusters)
            self.clustering_method = config.get('clustering_method', self.clustering_method)
            self.selection_strategy = config.get('selection_strategy', self.selection_strategy)
        
        logger.info(f"Loaded neutral representations from {filepath}")
        
        return self.neutral_representations


def integrate_learned_neutrals(triplet_mining_modules: List,
                              neutral_representations: Dict[str, np.ndarray],
                              device: str = 'cpu') -> None:
    """
    Integrate learned neutral representations into triplet mining modules.
    
    Args:
        triplet_mining_modules: List of TripletMining instances
        neutral_representations: Dictionary of learned neutrals
        device: Device to place tensors on
    """
    for module in triplet_mining_modules:
        if module.anim_name in neutral_representations:
            neutral = neutral_representations[module.anim_name]
            
            # Convert to tensor
            if not isinstance(neutral, torch.Tensor):
                neutral = torch.tensor(neutral, dtype=torch.float32)
            
            # Set as the fixed neutral embedding
            module.neutral_embedding = neutral.to(device)
            module.bool_fixed_neutral_embedding = True
            
            logger.info(f"Set learned neutral for {module.anim_name}")
