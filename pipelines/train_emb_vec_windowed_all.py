#!/usr/bin/env python3
"""
Full 3-action clustered embedding triplet training on windowed AE embeddings,
with optional DTW-fusion regularization.

Reuses EnhancedEmbeddingTrainer (k-means neutral + output-space re-clustering)
across walking/pointing/picking. Single-vector _emb.pt inputs (e.g. 4x256
windowed-concat AE latents) are auto-detected by EmbeddingDataset.

Usage:
    python pipelines/train_emb_vec_windowed_all.py --dtw-weight 0.5 --epochs 100
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
sys.path.insert(0, _here)

from networks.similarity_network import EmbeddingRefiningSimilarityNetwork
from Config import Config
from train_emb_vec_clustered import EnhancedEmbeddingTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_DATASETS = Path(_root).parent / "datasets"


def run(walking_dir, pointing_dir, picking_dir,
        n_epochs=100, n_clusters=5, dtw_weight=0.0):

    config = Config()
    config.n_similarity_epochs   = n_epochs
    config.early_stopping_warmup = 40
    config.dtw_loss_weight       = dtw_weight
    config.ensure_directories_exist()

    clustering_config = {'n_clusters': n_clusters, 'selection_strategy': 'centroid', 'random_state': 42}
    trainer = EnhancedEmbeddingTrainer(config)

    embedding_dirs = {'walking': walking_dir, 'pointing': pointing_dir, 'picking': picking_dir}
    train_loader, val_loader, train_modules, val_modules = \
        trainer.setup_clustering_based_training(embedding_dirs, 'rots_only', clustering_config)

    # DTW-fusion: precompute frozen rank-normalized DTW matrices per module from
    # each action's raw features, aligned to that module's class order. No-op at λ=0.
    if dtw_weight and dtw_weight > 0.0:
        from networks.dtw_fusion import precompute_dtw_matrix
        from infer_emb_vec import get_raw_features_without_dataloader
        logger.info(f"DTW-fusion enabled (λ={dtw_weight}); precomputing DTW matrices...")
        anim_order = ['walking', 'pointing', 'picking']
        for mods in (train_modules, val_modules):
            for m in mods:
                anim = getattr(m, 'anim_name', None)
                if anim not in anim_order:
                    m.dtw_loss_weight = 0.0; m.dtw_distance_matrix = None
                    continue
                raw = get_raw_features_without_dataloader(
                    anim, config, valid_indices=m.valid_indices, balance_class_frame_counts=False)
                raw_bare = {k[1] if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], str) else k: v
                            for k, v in raw.items()}
                m.dtw_loss_weight = dtw_weight
                precompute_dtw_matrix(m, raw_bare, anim_name=anim)
    else:
        for mods in (train_modules, val_modules):
            for m in mods:
                m.dtw_loss_weight = 0.0; m.dtw_distance_matrix = None

    logger.info("Creating network...")
    network = EmbeddingRefiningSimilarityNetwork(
        train_loader=train_loader, validation_loader=val_loader, test_loader=val_loader,
        checkpoint_root_dir=config.checkpoint_root_dir,
        triplet_modules=train_modules, val_triplet_modules=val_modules,
        architecture_variant=0, config=config,
        lr_scheduler_type='cosine', use_perception_loss=False, use_adaptive_distance=False)

    logger.info("=" * 55)
    logger.info("3-ACTION WINDOWED CLUSTERED TRAINING")
    logger.info(f"  epochs={n_epochs}  n_clusters={n_clusters}  dtw_weight={dtw_weight}")
    logger.info(f"  device={network.device}")
    logger.info("=" * 55)

    network.run_model_training()

    import torch
    last_path = os.path.join(config.checkpoint_root_dir, '0_last_epoch_embedding_model.pt')
    torch.save({
        'model_state_dict': network.network.state_dict(), 'epoch': 'last',
        'input_type': 'embeddings', 'use_embeddings': True,
        'embedding_dim': train_loader.exemplar_dim,
        'output_dim': config.embedding_refinement_model_output_size,
    }, last_path)
    logger.info(f"Saved last-epoch weights → {last_path}")

    logger.info("Evaluating...")
    results = network.evaluate()
    logger.info(f"Test Loss: {results['test_loss']:.4f}")
    return network, results


def main():
    ap = argparse.ArgumentParser(description='3-action windowed clustered triplet training')
    ap.add_argument('--walking-dir',  default=str(_DATASETS / 'lma_perform_walking_ae_full_win4'))
    ap.add_argument('--pointing-dir', default=str(_DATASETS / 'lma_perform_pointing_ae_full_win4'))
    ap.add_argument('--picking-dir',  default=str(_DATASETS / 'lma_perform_picking_ae_full_win4'))
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--n-clusters', type=int, default=5)
    ap.add_argument('--dtw-weight', type=float, default=0.0)
    args = ap.parse_args()

    for d in (args.walking_dir, args.pointing_dir, args.picking_dir):
        if not Path(d).exists():
            logger.error(f"Directory not found: {d}"); sys.exit(1)

    try:
        run(args.walking_dir, args.pointing_dir, args.picking_dir,
            n_epochs=args.epochs, n_clusters=args.n_clusters, dtw_weight=args.dtw_weight)
    except Exception:
        import traceback; traceback.print_exc(); sys.exit(1)


if __name__ == "__main__":
    main()
