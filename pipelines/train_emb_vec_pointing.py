#!/usr/bin/env python3
"""
Pointing-only clustered embedding triplet training.

Reproduces the train_emb_vec_clustered.py pipeline restricted to pointing,
using EnhancedEmbeddingTrainer directly so we can inject our own Config
(early_stopping_warmup=40, cosine scheduler) without touching the shared file.

The clustered pipeline:
  - NeutralRepresentationLearner: k-means over AE embeddings to find a
    semantically central neutral (not a zero vector or raw (0,0,0,0) exemplar)
  - Neutral placed first in the dict so EmbeddingSimilarityDataLoader
    exposes it at position 0 in every batch
  - update_output_space_neutrals: re-clusters in network output space every
    neutral_update_frequency epochs, keeping the anchor in the right space

Usage
-----
    python pipelines/train_emb_vec_pointing.py \
        --combination-method rots_only \
        --epochs 100
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'networks'))

from networks.similarity_network import EmbeddingRefiningSimilarityNetwork
from Config import Config, BatchStrategy
# Reuse EnhancedEmbeddingTrainer from the shared clustered pipeline unchanged.
from train_emb_vec_clustered import EnhancedEmbeddingTrainer

# Override the module-level BATCH_STRATEGY used by the loss functions to ALL.
# SEMI_HARD discards ~85% of pointing pairs (those where d(L,R) > d(L,N)),
# which is why training loss is frozen.  ALL lets every pair contribute.
import networks.custom_losses as _cl
_cl.BATCH_STRATEGY = BatchStrategy.ALL

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_DATASETS = Path(_root).parent / "datasets"


def run_pointing_clustered_training(
        pointing_dir: str,
        combination_method: str = 'rots_only',
        n_epochs: int = 100,
        n_clusters: int = 5,
        use_perception_loss: bool = False,
        dtw_weight: float = 0.0,
) -> tuple:

    config = Config()
    config.n_similarity_epochs   = n_epochs
    config.early_stopping_warmup = 40      # survive val=0 plateau + sawtooth
    config.dtw_loss_weight       = dtw_weight
    config.ensure_directories_exist()

    clustering_config = {
        'n_clusters': n_clusters,
        'selection_strategy': 'centroid',
        'random_state': 42,
    }

    trainer = EnhancedEmbeddingTrainer(config)

    train_loader, val_loader, train_modules, val_modules = \
        trainer.setup_clustering_based_training(
            embedding_dirs={'pointing': pointing_dir},
            combination_method=combination_method,
            clustering_config=clustering_config,
        )

    # DTW-fusion: precompute a frozen rank-normalized DTW matrix per module from
    # raw pointing features, aligned to each module's class order. No-op at λ=0.
    if dtw_weight and dtw_weight > 0.0:
        from networks.dtw_fusion import precompute_dtw_matrix
        sys.path.insert(0, str(Path(_here)))
        from infer_emb_vec import get_raw_features_without_dataloader
        logger.info(f"DTW-fusion enabled (λ={dtw_weight}); precomputing DTW matrices...")
        for mods in (train_modules, val_modules):
            for m in mods:
                vi = m.valid_indices
                raw = get_raw_features_without_dataloader(
                    "pointing", config, valid_indices=vi, balance_class_frame_counts=False)
                # strip the (anim, tuple) key down to the bare effort tuple
                raw_bare = {k[1] if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], str) else k: v
                            for k, v in raw.items()}
                m.dtw_loss_weight = dtw_weight
                precompute_dtw_matrix(m, raw_bare, anim_name="pointing")
    else:
        for mods in (train_modules, val_modules):
            for m in mods:
                m.dtw_loss_weight = 0.0
                m.dtw_distance_matrix = None

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
    logger.info("POINTING-ONLY CLUSTERED TRAINING")
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

    import torch, os
    last_path = os.path.join(config.checkpoint_root_dir, '0_last_epoch_embedding_model.pt')
    torch.save({
        'model_state_dict': network.network.state_dict(),
        'epoch': 'last',
        'input_type': 'embeddings',
        'use_embeddings': True,
        'embedding_dim': train_loader.exemplar_dim,
        'output_dim': config.embedding_refinement_model_output_size,
    }, last_path)
    logger.info(f"Saved last-epoch weights → {last_path}")

    logger.info("Evaluating...")
    results = network.evaluate()
    logger.info(f"Test Loss: {results['test_loss']:.4f}")
    return network, results


def main():
    parser = argparse.ArgumentParser(
        description='Pointing-only clustered embedding triplet training'
    )
    parser.add_argument('--pointing-dir', type=str,
                        default=str(_DATASETS / 'lma_perform_pointing_ae_paired'))
    parser.add_argument('--combination-method', type=str, default='rots_only',
                        choices=['rots_only', 'concat', 'root_only'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n-clusters', type=int, default=5)
    parser.add_argument('--use-perception-loss', action='store_true')
    parser.add_argument('--dtw-weight', type=float, default=0.0,
                        help='lambda for DTW-fusion alignment term (0 = pure triplet)')
    args = parser.parse_args()

    if not Path(args.pointing_dir).exists():
        logger.error(f"Directory not found: {args.pointing_dir}")
        sys.exit(1)

    try:
        run_pointing_clustered_training(
            pointing_dir=args.pointing_dir,
            combination_method=args.combination_method,
            n_epochs=args.epochs,
            n_clusters=args.n_clusters,
            use_perception_loss=args.use_perception_loss,
            dtw_weight=args.dtw_weight,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
