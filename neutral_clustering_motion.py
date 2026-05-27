#!/usr/bin/env python3
"""
Clustering-based neutral representation learning for RAW MOTION data.

This module adapts the embedding-based clustering approach to work directly
with raw motion tensors (quaternion rotations over time).
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple, Optional, Union
import pickle
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotionNeutralRepresentationLearner:
    """
    Learn neutral representations for action types using clustering on raw motion data.
    
    Since raw motion tensors are high-dimensional (frames × features), this class
    provides multiple strategies for dimensionality reduction before clustering.
    """
    
    def __init__(self, 
                 n_clusters: int = 5,
                 reduction_method: str = 'temporal_mean',
                 pca_components: Optional[int] = None,
                 selection_strategy: str = 'centroid',
                 random_state: int = 42):
        """
        Initialize the motion-based neutral representation learner.
        
        Args:
            n_clusters: Number of clusters for K-means
            reduction_method: How to reduce motion dimensionality
                - 'temporal_mean': Average across time dimension
                - 'temporal_stats': Use mean, std, min, max across time
                - 'pca': Apply PCA to flattened motion
                - 'pca_temporal': Apply PCA to temporal statistics
            pca_components: Number of PCA components (if using PCA)
            selection_strategy: How to select neutral ('centroid', 'medoid', 'weighted')
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.reduction_method = reduction_method
        self.pca_components = pca_components
        self.selection_strategy = selection_strategy
        self.random_state = random_state
        
        # Storage
        self.neutral_representations = {}
        self.cluster_models = {}
        self.cluster_stats = {}
        self.pca_models = {}
        
        logger.info(f"Initialized MotionNeutralRepresentationLearner:")
        logger.info(f"  Reduction method: {reduction_method}")
        logger.info(f"  N clusters: {n_clusters}")
        logger.info(f"  Selection strategy: {selection_strategy}")
    
    def learn_neutral_representation(self, 
                                    motion_dict: Dict[Tuple, List[np.ndarray]], 
                                    anim_name: str,
                                    validate_clustering: bool = True) -> np.ndarray:
        """
        Learn the neutral representation for a single animation type from raw motion.
        
        Args:
            motion_dict: Dictionary mapping effort tuples to lists of motion tensors
            anim_name: Name of the animation ('walking', 'pointing', 'picking')
            validate_clustering: Whether to validate clustering quality
            
        Returns:
            Learned neutral representation as numpy array (same shape as input motions)
        """
        logger.info(f"Learning motion-based neutral representation for {anim_name}")
        
        # Prepare motion data for clustering
        all_motions, effort_labels, original_shapes = self._prepare_motion_data(motion_dict)
        
        if len(all_motions) < self.n_clusters:
            logger.warning(f"Not enough samples ({len(all_motions)}) for clustering. Using mean representation.")
            neutral_rep = np.mean(all_motions, axis=0)
            self.neutral_representations[anim_name] = neutral_rep
            return neutral_rep
        
        # Reduce dimensionality for clustering
        reduced_features, feature_info = self._reduce_dimensionality(all_motions, anim_name)
        
        logger.info(f"Reduced motion from shape {all_motions[0].shape} to {reduced_features.shape[1]} features")
        
        # Perform clustering
        cluster_model = self._perform_kmeans(reduced_features)
        
        # Validate clustering if requested
        if validate_clustering and len(reduced_features) > self.n_clusters:
            score = self._validate_clustering(reduced_features, cluster_model.labels_)
            logger.info(f"Clustering silhouette score for {anim_name}: {score:.3f}")
            self.cluster_stats[anim_name] = {'silhouette_score': score}
        
        # Select neutral representation based on strategy
        neutral_rep_reduced = self._select_neutral(
            cluster_model, 
            reduced_features, 
            effort_labels
        )
        
        # Reconstruct to full motion space
        neutral_rep_full = self._reconstruct_to_motion_space(
            neutral_rep_reduced, 
            all_motions,
            cluster_model,
            feature_info,
            anim_name
        )
        
        # Store results
        self.neutral_representations[anim_name] = neutral_rep_full
        self.cluster_models[anim_name] = cluster_model
        
        logger.info(f"Learned neutral representation shape: {neutral_rep_full.shape}")
        
        return neutral_rep_full
    
    def _prepare_motion_data(self, 
                            motion_dict: Dict[Tuple, List]) -> Tuple[List[np.ndarray], List[Tuple], List]:
        """
        Prepare motion data for clustering by extracting from dictionary structure.
        
        Args:
            motion_dict: Dictionary of motion tensors
            
        Returns:
            Tuple of (list of motion arrays, effort labels, original shapes)
        """
        all_motions = []
        effort_labels = []
        original_shapes = []
        
        for effort_tuple, motion_list in motion_dict.items():
            # Skip the existing neutral if present
            if effort_tuple == (0, 0, 0, 0):
                continue
                
            for motion in motion_list:
                # Convert to numpy if needed
                if isinstance(motion, torch.Tensor):
                    motion = motion.cpu().numpy()
                elif not isinstance(motion, np.ndarray):
                    motion = np.array(motion)
                
                original_shapes.append(motion.shape)
                all_motions.append(motion)
                effort_labels.append(effort_tuple)
        
        effort_labels = np.array(effort_labels)
        
        logger.info(f"Prepared {len(all_motions)} motion sequences for clustering")
        if all_motions:
            logger.info(f"Motion shape: {all_motions[0].shape}")
        
        return all_motions, effort_labels, original_shapes
    
    def _reduce_dimensionality(self, 
                               motions: List[np.ndarray],
                               anim_name: str) -> Tuple[np.ndarray, Dict]:
        """
        Reduce dimensionality of motion data for clustering.
        
        Args:
            motions: List of motion arrays
            anim_name: Animation name (for storing PCA model if needed)
            
        Returns:
            Tuple of (reduced features, feature_info dict for reconstruction)
        """
        feature_info = {'method': self.reduction_method}
        
        if self.reduction_method == 'temporal_mean':
            # Simple: average across time dimension
            reduced = np.array([np.mean(motion, axis=0) for motion in motions])
            feature_info['original_shape'] = motions[0].shape
            
        elif self.reduction_method == 'temporal_stats':
            # Use multiple statistics across time
            features_list = []
            for motion in motions:
                mean_feat = np.mean(motion, axis=0)
                std_feat = np.std(motion, axis=0)
                min_feat = np.min(motion, axis=0)
                max_feat = np.max(motion, axis=0)
                features = np.concatenate([mean_feat, std_feat, min_feat, max_feat])
                features_list.append(features)
            reduced = np.array(features_list)
            feature_info['original_shape'] = motions[0].shape
            
        elif self.reduction_method == 'pca':
            # Flatten motions and apply PCA
            flattened = np.array([motion.flatten() for motion in motions])
            
            n_components = self.pca_components or min(flattened.shape[0] - 1, 50)
            pca = PCA(n_components=n_components, random_state=self.random_state)
            reduced = pca.fit_transform(flattened)
            
            self.pca_models[anim_name] = pca
            feature_info['pca_model'] = pca
            feature_info['original_shape'] = motions[0].shape
            
            logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
            
        elif self.reduction_method == 'pca_temporal':
            # Apply PCA to temporal statistics
            features_list = []
            for motion in motions:
                mean_feat = np.mean(motion, axis=0)
                std_feat = np.std(motion, axis=0)
                features = np.concatenate([mean_feat, std_feat])
                features_list.append(features)
            temporal_features = np.array(features_list)
            
            n_components = self.pca_components or min(temporal_features.shape[0] - 1, 50)
            pca = PCA(n_components=n_components, random_state=self.random_state)
            reduced = pca.fit_transform(temporal_features)
            
            self.pca_models[anim_name] = pca
            feature_info['pca_model'] = pca
            feature_info['original_shape'] = motions[0].shape
            feature_info['temporal_features'] = temporal_features
            
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction_method}")
        
        return reduced, feature_info
    
    def _perform_kmeans(self, features: np.ndarray) -> KMeans:
        """
        Perform K-means clustering on reduced features.
        
        Args:
            features: Array of reduced features to cluster
            
        Returns:
            Fitted KMeans model
        """
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        kmeans.fit(features)
        
        # Log cluster sizes
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            logger.info(f"  Cluster {cluster_id}: {count} samples")
        
        return kmeans
    
    def _validate_clustering(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Validate clustering quality using silhouette score.
        
        Args:
            features: Original features
            labels: Cluster labels
            
        Returns:
            Silhouette score
        """
        try:
            score = silhouette_score(features, labels)
            return score
        except Exception as e:
            logger.warning(f"Could not compute silhouette score: {e}")
            return -1.0
    
    def _select_neutral(self, 
                       cluster_model: KMeans, 
                       features: np.ndarray,
                       effort_labels: np.ndarray) -> np.ndarray:
        """
        Select the neutral representation in the reduced feature space.
        
        Args:
            cluster_model: Fitted clustering model
            features: Reduced features
            effort_labels: Effort tuple labels for each sample
            
        Returns:
            Selected neutral representation in reduced space
        """
        if self.selection_strategy == 'centroid':
            # Find cluster with lowest average effort magnitudes
            cluster_efforts = self._compute_cluster_efforts(
                cluster_model.labels_, effort_labels
            )
            
            neutral_cluster_idx = np.argmin([
                np.sum(np.abs(efforts)) for efforts in cluster_efforts
            ])
            
            neutral_rep = cluster_model.cluster_centers_[neutral_cluster_idx]
            logger.info(f"Selected cluster {neutral_cluster_idx} as neutral (centroid strategy)")
            
        elif self.selection_strategy == 'medoid':
            # Find actual sample closest to the neutral cluster center
            neutral_cluster_idx = self._find_neutral_cluster(
                cluster_model.labels_, effort_labels
            )
            
            cluster_mask = cluster_model.labels_ == neutral_cluster_idx
            cluster_features = features[cluster_mask]
            
            center = cluster_model.cluster_centers_[neutral_cluster_idx]
            distances = np.linalg.norm(cluster_features - center, axis=1)
            medoid_idx = np.argmin(distances)
            
            neutral_rep = cluster_features[medoid_idx]
            logger.info(f"Selected medoid from cluster {neutral_cluster_idx}")
            
        elif self.selection_strategy == 'weighted':
            # Weighted average of cluster centers
            weights = []
            for i in range(self.n_clusters):
                cluster_mask = cluster_model.labels_ == i
                cluster_efforts = effort_labels[cluster_mask]
                
                avg_effort = np.mean([np.sum(np.abs(e)) for e in cluster_efforts])
                weight = 1.0 / (1.0 + avg_effort)
                weights.append(weight)
            
            weights = np.array(weights)
            weights /= weights.sum()
            
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
                effort_sum = np.sum(np.abs(cluster_efforts))
                
                if effort_sum < min_effort_sum:
                    min_effort_sum = effort_sum
                    neutral_cluster = i
        
        return neutral_cluster
    
    def _reconstruct_to_motion_space(self,
                                    neutral_reduced: np.ndarray,
                                    original_motions: List[np.ndarray],
                                    cluster_model: KMeans,
                                    feature_info: Dict,
                                    anim_name: str) -> np.ndarray:
        """
        Reconstruct the neutral from reduced space back to full motion space.
        
        Args:
            neutral_reduced: Neutral in reduced feature space
            original_motions: Original motion sequences
            cluster_model: The clustering model
            feature_info: Information about the reduction method
            anim_name: Animation name
            
        Returns:
            Neutral representation in full motion space
        """
        method = feature_info['method']
        original_shape = feature_info['original_shape']
        
        if method == 'temporal_mean':
            # The reduced neutral is a temporal mean
            # Expand it to full sequence by repeating
            neutral_full = np.tile(neutral_reduced, (original_shape[0], 1))
            
        elif method == 'temporal_stats':
            # Extract the mean component (first quarter of features)
            n_features = len(neutral_reduced) // 4
            neutral_mean = neutral_reduced[:n_features]
            # Expand to full sequence
            neutral_full = np.tile(neutral_mean, (original_shape[0], 1))
            
        elif method == 'pca':
            # Inverse PCA transform
            pca = feature_info['pca_model']
            neutral_flattened = pca.inverse_transform(neutral_reduced.reshape(1, -1))
            neutral_full = neutral_flattened.reshape(original_shape)
            
        elif method == 'pca_temporal':
            # Inverse PCA transform then expand
            pca = feature_info['pca_model']
            neutral_temporal = pca.inverse_transform(neutral_reduced.reshape(1, -1))
            n_features = neutral_temporal.shape[1] // 2
            neutral_mean = neutral_temporal[0, :n_features]
            neutral_full = np.tile(neutral_mean, (original_shape[0], 1))
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Reconstructed neutral to shape: {neutral_full.shape}")
        
        return neutral_full
    
    def learn_all_neutrals(self, 
                          animation_dicts: Dict[str, Dict],
                          validate: bool = True) -> Dict[str, np.ndarray]:
        """
        Learn neutral representations for all animation types.
        
        Args:
            animation_dicts: Dictionary mapping animation names to motion dicts
            validate: Whether to validate clustering quality
            
        Returns:
            Dictionary mapping animation names to neutral representations
        """
        for anim_name, motion_dict in animation_dicts.items():
            self.learn_neutral_representation(
                motion_dict, 
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
                'reduction_method': self.reduction_method,
                'pca_components': self.pca_components,
                'selection_strategy': self.selection_strategy,
                'random_state': self.random_state
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Saved motion-based neutral representations to {filepath}")
    
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
        
        if 'config' in save_dict:
            config = save_dict['config']
            self.n_clusters = config.get('n_clusters', self.n_clusters)
            self.reduction_method = config.get('reduction_method', self.reduction_method)
            self.pca_components = config.get('pca_components', self.pca_components)
            self.selection_strategy = config.get('selection_strategy', self.selection_strategy)
        
        logger.info(f"Loaded motion-based neutral representations from {filepath}")
        
        return self.neutral_representations


def integrate_motion_neutrals(triplet_mining_modules: List,
                              neutral_representations: Dict[str, np.ndarray],
                              device: str = 'cpu') -> None:
    """
    Integrate learned motion-based neutral representations into triplet mining modules.
    
    Args:
        triplet_mining_modules: List of TripletMining instances
        neutral_representations: Dictionary of learned neutrals (in motion space)
        device: Device to place tensors on
    """
    for module in triplet_mining_modules:
        if module.anim_name in neutral_representations:
            neutral_motion = neutral_representations[module.anim_name]
            
            # The neutral is in motion space, but the module expects it after
            # passing through the network. For now, we'll store it as the
            # exemplar to be used, and it will get embedded during training.
            
            # Convert to tensor
            if not isinstance(neutral_motion, torch.Tensor):
                neutral_motion = torch.tensor(neutral_motion, dtype=torch.float32)
            
            # Update the neutral exemplar in the dictionary
            if (0, 0, 0, 0) in module.dict_similarity_classes_exemplars:
                module.dict_similarity_classes_exemplars[(0, 0, 0, 0)] = [neutral_motion]
                logger.info(f"Updated neutral exemplar for {module.anim_name} with learned motion")
            
            # Set flags
            module.bool_fixed_neutral_embedding = False  # Will be learned from the new exemplar
            module.bool_drop_neutral_exemplar = False
            
            logger.info(f"Set learned motion-based neutral for {module.anim_name}")
