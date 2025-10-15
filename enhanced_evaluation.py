#!/usr/bin/env python3
"""
Enhanced evaluation script for assessing embedding quality with learned neutrals.

This script evaluates the trained embeddings against human perception data,
with special focus on Spearman rank correlation which is more robust to outliers
and better captures ordinal relationships.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import logging
from typing import Dict, List, Tuple, Optional

# Add current directory
curr_path = os.getcwd()
sys.path.append(curr_path)

# Import necessary modules
from networks.similarity_network import EmbeddingRefiningSimilarityNetwork
from networks.triplet_mining import TripletMining
from Config import Config
import src.organize_synthetic_data as osd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedEvaluator:
    """
    Enhanced evaluator for embedding quality assessment.
    """
    
    def __init__(self, checkpoint_path: str, config: Config):
        """
        Initialize the evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration object
        """
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None
        self.results = {}
        
    def load_model(self) -> None:
        """Load the trained model from checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Recreate the model architecture
        input_dim = checkpoint.get('embedding_dim', 512)
        if isinstance(input_dim, tuple):
            input_dim = input_dim[0]
        
        from networks.similarity_network import EmbeddingSimilarityNetworkV0
        self.model = EmbeddingSimilarityNetworkV0(input_dim, self.config.embedding_refinement_model_output_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def evaluate_animation(self,
                          anim_name: str,
                          embeddings: Dict,
                          triplet_module: TripletMining,
                          subset: str = 'validation') -> Dict:
        """
        Evaluate embeddings for a single animation type.
        
        Args:
            anim_name: Animation name
            embeddings: Dictionary of embeddings
            triplet_module: TripletMining module with comparison data
            subset: Data subset to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {anim_name} ({subset} subset)")
        
        # Calculate pairwise distances
        embedding_list = []
        key_list = []
        
        for key, embedding in embeddings.items():
            if key[0] == anim_name:  # Filter by animation
                embedding_list.append(embedding)
                key_list.append(key)
        
        if not embedding_list:
            logger.warning(f"No embeddings found for {anim_name}")
            return {}
        
        # Convert to tensor
        embedding_tensor = torch.stack([torch.tensor(e) for e in embedding_list])
        
        # Calculate distances
        distances = self._calculate_pairwise_distances(embedding_tensor)
        
        # Get human perception values
        perception_pairs = self._get_perception_pairs(key_list, triplet_module)
        
        if not perception_pairs:
            logger.warning(f"No perception pairs found for {anim_name}")
            return {}
        
        # Calculate metrics
        metrics = self._calculate_metrics(distances, perception_pairs)
        
        # Store results
        self.results[f"{anim_name}_{subset}"] = metrics
        
        return metrics
    
    def _calculate_pairwise_distances(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Calculate pairwise L2 distances between embeddings.
        
        Args:
            embeddings: Tensor of embeddings
            
        Returns:
            Array of pairwise distances
        """
        n = embeddings.shape[0]
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(embeddings[i] - embeddings[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _get_perception_pairs(self,
                             key_list: List,
                             triplet_module: TripletMining) -> List[Tuple]:
        """
        Get perception values for embedding pairs.
        
        Args:
            key_list: List of embedding keys
            triplet_module: TripletMining module
            
        Returns:
            List of (distance_idx_i, distance_idx_j, perception_value) tuples
        """
        perception_pairs = []
        
        if not hasattr(triplet_module, 'df_comparisons'):
            return perception_pairs
        
        df = triplet_module.df_comparisons
        
        # Create mapping from effort tuples to indices
        effort_to_idx = {}
        for idx, key in enumerate(key_list):
            effort_tuple = key[1]  # Extract effort tuple from (anim_name, effort_tuple)
            effort_to_idx[effort_tuple] = idx
        
        # Extract perception values
        for _, row in df.iterrows():
            if row['selected0'] == 0 and row['selected1'] == 2:  # Left-right comparison
                efforts = row['efforts_tuples']
                
                if efforts[0] in effort_to_idx and efforts[1] in effort_to_idx:
                    idx1 = effort_to_idx[efforts[0]]
                    idx2 = effort_to_idx[efforts[1]]
                    
                    # Use inverse of count_normalized as distance target
                    perception_value = 1.0 - row['count_normalized']
                    
                    perception_pairs.append((idx1, idx2, perception_value))
        
        return perception_pairs
    
    def _calculate_metrics(self,
                          distances: np.ndarray,
                          perception_pairs: List[Tuple]) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            distances: Pairwise distance matrix
            perception_pairs: List of perception value tuples
            
        Returns:
            Dictionary of metrics
        """
        if not perception_pairs:
            return {}
        
        # Extract distances and perception values
        dist_values = []
        perc_values = []
        
        for idx1, idx2, perception in perception_pairs:
            dist_values.append(distances[idx1, idx2])
            perc_values.append(perception)
        
        dist_values = np.array(dist_values)
        perc_values = np.array(perc_values)
        
        # Normalize distances to [0, 1]
        if dist_values.max() > dist_values.min():
            dist_norm = (dist_values - dist_values.min()) / (dist_values.max() - dist_values.min())
        else:
            dist_norm = dist_values
        
        # Calculate metrics
        pearson_r, pearson_p = stats.pearsonr(dist_norm, perc_values)
        spearman_r, spearman_p = stats.spearmanr(dist_norm, perc_values)
        
        # Calculate R² score
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(dist_norm.reshape(-1, 1), perc_values)
        r2 = lr.score(dist_norm.reshape(-1, 1), perc_values)
        
        # Calculate Kendall's tau (another rank correlation metric)
        kendall_tau, kendall_p = stats.kendalltau(dist_norm, perc_values)
        
        metrics = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'kendall_tau': kendall_tau,
            'kendall_p': kendall_p,
            'r2_score': r2,
            'n_pairs': len(perception_pairs)
        }
        
        return metrics
    
    def compare_with_baseline(self,
                            raw_features: Dict,
                            method: str = 'geodesic') -> pd.DataFrame:
        """
        Compare embedding results with raw feature baseline.
        
        Args:
            raw_features: Dictionary of raw features
            method: Distance method for raw features
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for anim_subset, metrics in self.results.items():
            anim_name = anim_subset.split('_')[0]
            subset = anim_subset.split('_')[1]
            
            # Add embedding results
            comparison_data.append({
                'Animation': anim_name,
                'Subset': subset,
                'Method': 'Embedding_L2',
                'Pearson_r': metrics.get('pearson_r', 0),
                'Spearman_r': metrics.get('spearman_r', 0),
                'Kendall_tau': metrics.get('kendall_tau', 0),
                'R2_score': metrics.get('r2_score', 0),
                'N_pairs': metrics.get('n_pairs', 0)
            })
        
        return pd.DataFrame(comparison_data)
    
    def visualize_results(self, save_path: Optional[str] = None) -> None:
        """
        Visualize evaluation results.
        
        Args:
            save_path: Path to save the visualization
        """
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        # Prepare data for visualization
        animations = []
        spearman_scores = []
        pearson_scores = []
        
        for key, metrics in self.results.items():
            animations.append(key)
            spearman_scores.append(metrics.get('spearman_r', 0))
            pearson_scores.append(metrics.get('pearson_r', 0))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Spearman correlation plot
        x_pos = np.arange(len(animations))
        ax1.bar(x_pos, spearman_scores, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Animation/Subset')
        ax1.set_ylabel('Spearman Correlation')
        ax1.set_title('Spearman Rank Correlation with Human Perception')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(animations, rotation=45, ha='right')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(spearman_scores):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Pearson correlation plot
        ax2.bar(x_pos, pearson_scores, color='darkgreen', alpha=0.7)
        ax2.set_xlabel('Animation/Subset')
        ax2.set_ylabel('Pearson Correlation')
        ax2.set_title('Pearson Correlation with Human Perception')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(animations, rotation=45, ha='right')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(pearson_scores):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Embedding Quality Evaluation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, output_path: str) -> None:
        """
        Generate a detailed evaluation report.
        
        Args:
            output_path: Path to save the report
        """
        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("ENHANCED EMBEDDING EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Model Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Timestamp: {pd.Timestamp.now()}\n\n")
            
            # Summary statistics
            if self.results:
                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 40 + "\n")
                
                all_spearman = [m.get('spearman_r', 0) for m in self.results.values()]
                all_pearson = [m.get('pearson_r', 0) for m in self.results.values()]
                all_kendall = [m.get('kendall_tau', 0) for m in self.results.values()]
                
                f.write(f"Mean Spearman r: {np.mean(all_spearman):.4f} ± {np.std(all_spearman):.4f}\n")
                f.write(f"Mean Pearson r:  {np.mean(all_pearson):.4f} ± {np.std(all_pearson):.4f}\n")
                f.write(f"Mean Kendall τ:  {np.mean(all_kendall):.4f} ± {np.std(all_kendall):.4f}\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS BY ANIMATION\n")
            f.write("-" * 40 + "\n")
            
            for key in sorted(self.results.keys()):
                metrics = self.results[key]
                f.write(f"\n{key.upper()}:\n")
                f.write(f"  Spearman r: {metrics.get('spearman_r', 0):.4f} (p={metrics.get('spearman_p', 1):.4f})\n")
                f.write(f"  Pearson r:  {metrics.get('pearson_r', 0):.4f} (p={metrics.get('pearson_p', 1):.4f})\n")
                f.write(f"  Kendall τ:  {metrics.get('kendall_tau', 0):.4f} (p={metrics.get('kendall_p', 1):.4f})\n")
                f.write(f"  R² Score:   {metrics.get('r2_score', 0):.4f}\n")
                f.write(f"  N Pairs:    {metrics.get('n_pairs', 0)}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("RECOMMENDATIONS FOR IMPROVEMENT\n")
            f.write("-" * 40 + "\n")
            
            # Analyze results and provide recommendations
            worst_performing = min(self.results.items(), 
                                 key=lambda x: x[1].get('spearman_r', 0))
            
            f.write(f"Lowest performing: {worst_performing[0]}\n")
            f.write(f"Spearman r = {worst_performing[1].get('spearman_r', 0):.4f}\n\n")
            
            if worst_performing[1].get('spearman_r', 0) < 0.4:
                f.write("Recommendations:\n")
                f.write("1. Consider increasing the number of clusters for neutral learning\n")
                f.write("2. Try different selection strategies (medoid or weighted)\n")
                f.write("3. Increase training epochs or adjust learning rate\n")
                f.write("4. Experiment with different embedding combination methods\n")
        
        logger.info(f"Report saved to {output_path}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced evaluation of trained embeddings')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Directory for output files')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize configuration
    config = Config()
    
    # Create evaluator
    evaluator = EnhancedEvaluator(args.checkpoint, config)
    
    # Load model
    evaluator.load_model()
    
    # TODO: Add code here to load embeddings and evaluate each animation
    # This would require loading the validation data and triplet modules
    
    # Generate report
    report_path = output_dir / "evaluation_report.txt"
    evaluator.generate_report(str(report_path))
    
    # Visualize if requested
    if args.visualize:
        viz_path = output_dir / "evaluation_results.png"
        evaluator.visualize_results(str(viz_path))
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
