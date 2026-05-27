#!/usr/bin/env python3
"""
Utility script to compare learned neutral representations from both approaches:
1. Motion-based clustering (neutral_clustering_motion.py)
2. Embedding-based clustering (neutral_clustering.py)

This helps understand if both methods learn similar neutral patterns.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
import torch


def load_motion_neutrals(filepath):
    """Load motion-based learned neutrals."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['neutral_representations'], data.get('cluster_stats', {})


def load_embedding_neutrals(filepath):
    """Load embedding-based learned neutrals."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['neutral_representations'], data.get('cluster_stats', {})


def load_original_neutrals(config_path, anim_name):
    """Load original neutral from similarity dictionary."""
    import sys
    sys.path.append(str(Path.cwd()))
    from Config import Config
    import src.organize_synthetic_data as osd
    
    config = Config()
    similarity_dict = osd.load_similarity_data(False, anim_name, config)["train"]
    
    if (0, 0, 0, 0) in similarity_dict:
        original = similarity_dict[(0, 0, 0, 0)][0]
        if isinstance(original, torch.Tensor):
            original = original.cpu().numpy()
        return original
    return None


def compute_similarity_metrics(neutral1, neutral2, name1="Method 1", name2="Method 2"):
    """
    Compute similarity metrics between two neutral representations.
    
    Args:
        neutral1: First neutral representation
        neutral2: Second neutral representation
        name1: Name of first method
        name2: Name of second method
        
    Returns:
        Dictionary of similarity metrics
    """
    # Flatten if needed
    n1_flat = neutral1.flatten()
    n2_flat = neutral2.flatten()
    
    # Ensure same length
    min_len = min(len(n1_flat), len(n2_flat))
    n1_flat = n1_flat[:min_len]
    n2_flat = n2_flat[:min_len]
    
    # Compute metrics
    metrics = {}
    
    # Euclidean distance
    metrics['euclidean_distance'] = np.linalg.norm(n1_flat - n2_flat)
    
    # Cosine similarity
    metrics['cosine_similarity'] = 1 - cosine(n1_flat, n2_flat)
    
    # Pearson correlation
    if len(n1_flat) > 1:
        metrics['pearson_correlation'], _ = pearsonr(n1_flat, n2_flat)
    else:
        metrics['pearson_correlation'] = 0.0
    
    # Mean squared error
    metrics['mse'] = np.mean((n1_flat - n2_flat) ** 2)
    
    # Normalized difference (as percentage)
    metrics['normalized_diff'] = np.mean(np.abs(n1_flat - n2_flat)) / (np.mean(np.abs(n1_flat)) + 1e-8) * 100
    
    return metrics


def visualize_neutrals(neutrals_dict, save_dir, anim_name):
    """
    Visualize learned neutral representations.
    
    Args:
        neutrals_dict: Dictionary with keys like 'motion', 'embedding', 'original'
        save_dir: Directory to save plots
        anim_name: Animation name
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    n_methods = len(neutrals_dict)
    fig, axes = plt.subplots(n_methods, 2, figsize=(15, 5*n_methods))
    
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (method_name, neutral) in enumerate(neutrals_dict.items()):
        if neutral is None:
            continue
            
        # Temporal visualization (if 2D)
        if len(neutral.shape) == 2:
            # Plot as heatmap
            axes[idx, 0].imshow(neutral.T, aspect='auto', cmap='coolwarm')
            axes[idx, 0].set_title(f'{method_name} - Temporal Pattern')
            axes[idx, 0].set_xlabel('Frame')
            axes[idx, 0].set_ylabel('Feature')
            
            # Plot temporal mean
            temporal_mean = np.mean(neutral, axis=0)
            axes[idx, 1].plot(temporal_mean)
            axes[idx, 1].set_title(f'{method_name} - Feature Means')
            axes[idx, 1].set_xlabel('Feature Index')
            axes[idx, 1].set_ylabel('Mean Value')
            axes[idx, 1].grid(True)
            
        else:
            # 1D representation
            axes[idx, 0].plot(neutral)
            axes[idx, 0].set_title(f'{method_name} - Neutral Values')
            axes[idx, 0].set_xlabel('Feature Index')
            axes[idx, 0].set_ylabel('Value')
            axes[idx, 0].grid(True)
            
            # Distribution
            axes[idx, 1].hist(neutral, bins=50, alpha=0.7)
            axes[idx, 1].set_title(f'{method_name} - Value Distribution')
            axes[idx, 1].set_xlabel('Value')
            axes[idx, 1].set_ylabel('Frequency')
            axes[idx, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{anim_name}_neutral_comparison.png', dpi=150)
    print(f"Saved visualization to {save_dir / f'{anim_name}_neutral_comparison.png'}")
    plt.close()


def create_comparison_report(results, save_path):
    """
    Create a text report comparing neutral representations.
    
    Args:
        results: Dictionary with comparison results
        save_path: Path to save report
    """
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NEUTRAL REPRESENTATION COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for anim_name, anim_results in results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"Animation: {anim_name.upper()}\n")
            f.write(f"{'='*60}\n\n")
            
            # Clustering stats
            if 'motion_cluster_stats' in anim_results:
                f.write("Motion Clustering Statistics:\n")
                stats = anim_results['motion_cluster_stats']
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            if 'embedding_cluster_stats' in anim_results:
                f.write("Embedding Clustering Statistics:\n")
                stats = anim_results['embedding_cluster_stats']
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Shape information
            if 'shapes' in anim_results:
                f.write("Representation Shapes:\n")
                for method, shape in anim_results['shapes'].items():
                    f.write(f"  {method}: {shape}\n")
                f.write("\n")
            
            # Similarity metrics
            if 'similarities' in anim_results:
                f.write("Similarity Metrics:\n\n")
                for comparison, metrics in anim_results['similarities'].items():
                    f.write(f"  {comparison}:\n")
                    for metric_name, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"    {metric_name}: {value:.6f}\n")
                        else:
                            f.write(f"    {metric_name}: {value}\n")
                    f.write("\n")
            
            f.write("-" * 60 + "\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write("\nKey Findings:\n")
        f.write("1. Check cosine similarity - values > 0.8 indicate similar patterns\n")
        f.write("2. Check normalized difference - values < 20% indicate close match\n")
        f.write("3. Compare clustering silhouette scores - higher is better\n")
        f.write("4. Visual inspection recommended for full understanding\n")
    
    print(f"Saved comparison report to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare learned neutral representations from different methods'
    )
    
    parser.add_argument('--motion-neutrals', type=str,
                       default='checkpoints/learned_motion_neutrals.pkl',
                       help='Path to motion-based learned neutrals')
    parser.add_argument('--embedding-neutrals', type=str,
                       default='checkpoints/learned_neutrals.pkl',
                       help='Path to embedding-based learned neutrals')
    parser.add_argument('--animations', nargs='+', 
                       default=['walking', 'pointing', 'picking'],
                       help='Animations to compare')
    parser.add_argument('--output-dir', type=str, default='neutral_comparison',
                       help='Directory to save comparison results')
    parser.add_argument('--include-original', action='store_true',
                       help='Include original neutral from dataset')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("NEUTRAL REPRESENTATION COMPARISON")
    print("=" * 80)
    
    # Load neutrals
    motion_neutrals = None
    motion_stats = {}
    embedding_neutrals = None
    embedding_stats = {}
    
    if Path(args.motion_neutrals).exists():
        print(f"\nLoading motion-based neutrals from {args.motion_neutrals}")
        motion_neutrals, motion_stats = load_motion_neutrals(args.motion_neutrals)
        print(f"Found {len(motion_neutrals)} motion-based neutrals")
    else:
        print(f"\nWarning: Motion neutrals file not found: {args.motion_neutrals}")
    
    if Path(args.embedding_neutrals).exists():
        print(f"Loading embedding-based neutrals from {args.embedding_neutrals}")
        embedding_neutrals, embedding_stats = load_embedding_neutrals(args.embedding_neutrals)
        print(f"Found {len(embedding_neutrals)} embedding-based neutrals")
    else:
        print(f"Warning: Embedding neutrals file not found: {args.embedding_neutrals}")
    
    # Compare for each animation
    results = {}
    
    for anim_name in args.animations:
        print(f"\n{'='*60}")
        print(f"Analyzing: {anim_name.upper()}")
        print(f"{'='*60}")
        
        anim_results = {
            'shapes': {},
            'similarities': {}
        }
        
        # Collect neutrals for this animation
        neutrals_to_compare = {}
        
        if motion_neutrals and anim_name in motion_neutrals:
            neutrals_to_compare['motion'] = motion_neutrals[anim_name]
            anim_results['shapes']['motion'] = motion_neutrals[anim_name].shape
            if anim_name in motion_stats:
                anim_results['motion_cluster_stats'] = motion_stats[anim_name]
            print(f"  Motion neutral shape: {motion_neutrals[anim_name].shape}")
        
        if embedding_neutrals and anim_name in embedding_neutrals:
            neutrals_to_compare['embedding'] = embedding_neutrals[anim_name]
            anim_results['shapes']['embedding'] = embedding_neutrals[anim_name].shape
            if anim_name in embedding_stats:
                anim_results['embedding_cluster_stats'] = embedding_stats[anim_name]
            print(f"  Embedding neutral shape: {embedding_neutrals[anim_name].shape}")
        
        if args.include_original:
            try:
                original = load_original_neutrals(None, anim_name)
                if original is not None:
                    neutrals_to_compare['original'] = original
                    anim_results['shapes']['original'] = original.shape
                    print(f"  Original neutral shape: {original.shape}")
            except Exception as e:
                print(f"  Could not load original neutral: {e}")
        
        # Compute pairwise similarities
        neutral_names = list(neutrals_to_compare.keys())
        for i in range(len(neutral_names)):
            for j in range(i+1, len(neutral_names)):
                name1 = neutral_names[i]
                name2 = neutral_names[j]
                neutral1 = neutrals_to_compare[name1]
                neutral2 = neutrals_to_compare[name2]
                
                comparison_key = f"{name1} vs {name2}"
                print(f"\n  Comparing {name1} vs {name2}:")
                
                metrics = compute_similarity_metrics(neutral1, neutral2, name1, name2)
                anim_results['similarities'][comparison_key] = metrics
                
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        print(f"    {metric_name}: {value:.6f}")
        
        # Visualize
        visualize_neutrals(neutrals_to_compare, output_dir, anim_name)
        
        results[anim_name] = anim_results
    
    # Create comparison report
    report_path = output_dir / "comparison_report.txt"
    create_comparison_report(results, report_path)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Visualizations: {output_dir}/<animation>_neutral_comparison.png")
    print(f"  - Report: {report_path}")
    print("\nRecommendations:")
    print("  1. Check the visualizations to see if patterns are similar")
    print("  2. High cosine similarity (>0.8) suggests methods learn similar neutrals")
    print("  3. Compare with training results to see which leads to better performance")


if __name__ == '__main__':
    main()
