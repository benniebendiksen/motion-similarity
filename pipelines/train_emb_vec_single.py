#!/usr/bin/env python3
"""
Single-action clustered embedding triplet training (any action), windowed-aware,
with optional DTW-fusion. Used for per-action N-window grid search.

Usage:
    python pipelines/train_emb_vec_single.py --action walking \
        --emb-dir ../datasets/lma_perform_walking_winp2 --dtw-weight 0.5 --epochs 100
"""
import os, sys, argparse, logging
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root); sys.path.insert(0, os.path.join(_root, 'networks')); sys.path.insert(0, _here)

from networks.similarity_network import EmbeddingRefiningSimilarityNetwork
from Config import Config
from train_emb_vec_clustered import EnhancedEmbeddingTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run(action, emb_dir, n_epochs=100, n_clusters=5, dtw_weight=0.0):
    config = Config()
    config.n_similarity_epochs   = n_epochs
    config.early_stopping_warmup = 40
    config.dtw_loss_weight       = dtw_weight
    config.ensure_directories_exist()

    clustering_config = {'n_clusters': n_clusters, 'selection_strategy': 'centroid', 'random_state': 42}
    trainer = EnhancedEmbeddingTrainer(config)
    train_loader, val_loader, train_modules, val_modules = \
        trainer.setup_clustering_based_training({action: emb_dir}, 'rots_only', clustering_config)

    if dtw_weight and dtw_weight > 0.0:
        from networks.dtw_fusion import precompute_dtw_matrix
        from infer_emb_vec import get_raw_features_without_dataloader
        for mods in (train_modules, val_modules):
            for m in mods:
                raw = get_raw_features_without_dataloader(
                    action, config, valid_indices=m.valid_indices, balance_class_frame_counts=False)
                raw_bare = {k[1] if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], str) else k: v
                            for k, v in raw.items()}
                m.dtw_loss_weight = dtw_weight
                precompute_dtw_matrix(m, raw_bare, anim_name=action)
    else:
        for mods in (train_modules, val_modules):
            for m in mods:
                m.dtw_loss_weight = 0.0; m.dtw_distance_matrix = None

    network = EmbeddingRefiningSimilarityNetwork(
        train_loader=train_loader, validation_loader=val_loader, test_loader=val_loader,
        checkpoint_root_dir=config.checkpoint_root_dir,
        triplet_modules=train_modules, val_triplet_modules=val_modules,
        architecture_variant=0, config=config,
        lr_scheduler_type='cosine', use_perception_loss=False, use_adaptive_distance=False)

    logger.info(f"SINGLE-ACTION {action} | dtw_weight={dtw_weight} | dim={train_loader.exemplar_dim}")
    network.run_model_training()
    results = network.evaluate()
    logger.info(f"Test Loss: {results['test_loss']:.4f}")
    return network, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--action', required=True, choices=['walking', 'pointing', 'picking'])
    ap.add_argument('--emb-dir', required=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--n-clusters', type=int, default=5)
    ap.add_argument('--dtw-weight', type=float, default=0.0)
    args = ap.parse_args()
    if not Path(args.emb_dir).exists():
        logger.error(f"Not found: {args.emb_dir}"); sys.exit(1)
    try:
        run(args.action, args.emb_dir, args.epochs, args.n_clusters, args.dtw_weight)
    except Exception:
        import traceback; traceback.print_exc(); sys.exit(1)


if __name__ == "__main__":
    main()
