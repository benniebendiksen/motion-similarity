#!/usr/bin/env python3
"""
Walking-only clustered embedding triplet training.

Diagnostic counterpart to train_emb_vec_pointing.py — reproduces the
train_emb_vec_clustered.py pipeline restricted to walking to verify that
training violations decrease meaningfully when the same architecture and
neutral strategy is applied to a motion type where we know it works.

If walking-only shows monotonically decreasing violations and training loss,
the frozen-gradient problem observed for pointing is action-type-specific
rather than architectural.

Usage
-----
    python pipelines/train_emb_vec_walking.py \
        --combination-method rots_only \
        --epochs 100
"""

import os
import sys
import argparse
import logging
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'networks'))

from networks.similarity_network import EmbeddingRefiningSimilarityNetwork
from Config import Config
from train_emb_vec_clustered import EnhancedEmbeddingTrainer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_DATASETS = Path(_root).parent / "datasets"


def run_walking_clustered_training(
        walking_dir: str,
        combination_method: str = 'rots_only',
        n_epochs: int = 100,
        n_clusters: int = 5,
        use_perception_loss: bool = False,
) -> tuple:

    config = Config()
    config.n_similarity_epochs   = n_epochs
    config.early_stopping_warmup = 40
    config.ensure_directories_exist()

    clustering_config = {
        'n_clusters': n_clusters,
        'selection_strategy': 'centroid',
        'random_state': 42,
    }

    trainer = EnhancedEmbeddingTrainer(config)

    train_loader, val_loader, train_modules, val_modules = \
        trainer.setup_clustering_based_training(
            embedding_dirs={'walking': walking_dir},
            combination_method=combination_method,
            clustering_config=clustering_config,
        )

    logger.info("Creating network...")
    network = EmbeddingRefiningSimilarityNetwork(
        train_loader=train_loader,
        validation_loader=val_loader,
        test_loader=val_loader,
        checkpoint_root_dir=config.checkpoint_root_dir,
        triplet_modules=train_modules,
        val_triplet_modules=val_modules,
        architecture_variant=0,
        config=config,
        lr_scheduler_type='cosine',
        use_perception_loss=use_perception_loss,
        use_adaptive_distance=False,
    )

    logger.info("=" * 55)
    logger.info("WALKING-ONLY CLUSTERED TRAINING  (diagnostic)")
    logger.info(f"  combination_method   : {combination_method}")
    logger.info(f"  epochs               : {n_epochs}")
    logger.info(f"  n_clusters           : {n_clusters}")
    logger.info(f"  scheduler            : cosine")
    logger.info(f"  warmup               : {config.early_stopping_warmup}")
    logger.info(f"  patience             : {config.early_stopping_patience}")
    logger.info(f"  neutral_update_freq  : {config.neutral_update_frequency}")
    logger.info(f"  perception_loss      : {use_perception_loss}")
    logger.info(f"  device               : {network.device}")
    logger.info("=" * 55)

    network.run_model_training()

    # After run_model_training() the live network weights are the last training
    # epoch (early-stopping fires but does not call load_state_dict).
    # save_checkpoint(is_final=True) writes _best_model_state (best val) to
    # 0_final_embedding_model.pt without touching the network weights.
    # Save the last-epoch state separately for inference comparison.
    import torch, os
    last_epoch_path = os.path.join(
        config.checkpoint_root_dir, '0_last_epoch_embedding_model.pt'
    )
    torch.save({
        'model_state_dict': network.network.state_dict(),
        'epoch': 'last',
        'input_type': 'embeddings',
        'use_embeddings': True,
        'embedding_dim': train_loader.exemplar_dim,
        'output_dim': config.embedding_refinement_model_output_size,
    }, last_epoch_path)
    logger.info(f"Saved last-epoch weights → {last_epoch_path}")

    logger.info("Evaluating...")
    results = network.evaluate()
    logger.info(f"Test Loss: {results['test_loss']:.4f}")
    return network, results


def main():
    parser = argparse.ArgumentParser(
        description='Walking-only clustered embedding triplet training (diagnostic)'
    )
    parser.add_argument('--walking-dir', type=str,
                        default=str(_DATASETS / 'lma_perform_walking_ae_paired'))
    parser.add_argument('--combination-method', type=str, default='rots_only',
                        choices=['rots_only', 'concat', 'root_only'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n-clusters', type=int, default=5)
    parser.add_argument('--use-perception-loss', action='store_true')
    args = parser.parse_args()

    if not Path(args.walking_dir).exists():
        logger.error(f"Directory not found: {args.walking_dir}")
        sys.exit(1)

    try:
        run_walking_clustered_training(
            walking_dir=args.walking_dir,
            combination_method=args.combination_method,
            n_epochs=args.epochs,
            n_clusters=args.n_clusters,
            use_perception_loss=args.use_perception_loss,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
