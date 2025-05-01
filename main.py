"""
Program entry point. Loads data for, trains, and evaluates both effort and similarity networks.

If command line argument is provided, program assumes value represents a parallel job index and treats the
execution as a remote machine run. Otherwise, the program is assumed to be running locally.
"""

import os
import sys
import random
curr_path = os.getcwd()
sys.path.append(curr_path)
sys.path.append(curr_path + '\networks')
from networks.similarity_network import SimilarityNetwork
from networks.similarity_data_loader import SimilarityDataLoader
from networks.triplet_mining import TripletMining
from Config import Config
import src.organize_synthetic_data as osd
import tensorflow as tf


def check_gpu_access():
    # Check if GPUs are available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ TensorFlow detected {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu}")

        # Set TensorFlow to use GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Run a small computation on the GPU
        with tf.device('/GPU:0'):
            a = tf.constant([1.0, 2.0, 3.0])
            b = tf.constant([4.0, 5.0, 6.0])
            c = a + b
        print(f"✅ GPU computation successful: {c.numpy()}")
    else:
        print("❌ No GPU detected by TensorFlow.")


def create_train_val_split(similarity_dicts, val_ratio=0.2):
    """
    Create training and validation indices for each animation type.

    Args:
        similarity_dicts: List of dictionaries containing class exemplars for each animation
        val_ratio: Ratio of classes to use for validation

    Returns:
        train_indices: List of sets containing training class indices for each animation
        val_indices: List of sets containing validation class indices for each animation
    """
    train_indices = []
    val_indices = []

    for anim_dict in similarity_dicts:
        # Get keys except neutral
        keys = [k for k in anim_dict.keys() if k != (0, 0, 0, 0)]

        # Determine validation set size
        val_size = max(1, int(len(keys) * val_ratio))

        # Randomly sample keys for validation
        val_keys = set(random.sample(keys, val_size))
        train_keys = set(k for k in keys if k not in val_keys)

        # Add neutral exemplar to both sets
        if (0, 0, 0, 0) in anim_dict:
            train_keys.add((0, 0, 0, 0))
            val_keys.add((0, 0, 0, 0))

        train_indices.append(train_keys)
        val_indices.append(val_keys)

    return train_indices, val_indices


if __name__ == '__main__':
    check_gpu_access()

    # Initialize configuration with task index if provided (happens only for remote machine)
    task_index = sys.argv[1] if len(sys.argv) > 1 else None
    config = Config(task_index)

    # Ensure required directories exist
    config.ensure_directories_exist()

    # Architecture variant is either from task index or default
    arch_variant = int(config.num_task) if config.num_task else 0

    # load effort data and train effort network
    # batch_ids_partition, labels_dict = osd.load_data(rotations=True, velocities=False)
    # effort_train_generator = MotionDataGenerator(batch_ids_partition['train'], labels_dict,
    #                                              **conf.EFFORT_EXEMPLAR_GENERATOR_PARAMS)
    # effort_validation_generator = MotionDataGenerator(batch_ids_partition['validation'], labels_dict,
    #                                                   **conf.EFFORT_EXEMPLAR_GENERATOR_PARAMS)
    # effort_test_generator = MotionDataGenerator(batch_ids_partition['test'], labels_dict,
    #                                             **conf.EFFORT_EXEMPLAR_GENERATOR_PARAMS)
    # effort_network = EffortNetwork(train_generator=effort_train_generator,
    #                                validation_generator=effort_validation_generator,
    #                                test_generator=effort_test_generator, checkpoint_dir=checkpoint_dir)
    # start_time = time.time()
    # history = effort_network.run_model_training()
    # minutes_tot_time = (time.time() - start_time) / 60
    # effort_network.write_out_training_results(minutes_tot_time)
    # collect_job_metrics.collect_job_metrics()

    bool_drop_neutral_exemplar = False
    bool_fixed_neutral_embedding = False
    squared_left_right_euc_dist = False
    squared_class_neut_euc_dist = False

    # Load similarity data
    walking_similarity_dict = osd.load_similarity_data(bool_drop_neutral_exemplar, "walking", config)["train"]
    pointing_similarity_dict = osd.load_similarity_data(bool_drop_neutral_exemplar, "pointing", config)["train"]
    picking_similarity_dict = osd.load_similarity_data(bool_drop_neutral_exemplar, "picking", config)["train"]
    # assert False, "Check data integrity with embeddiing_calculator_single_action.py"
    list_similarity_dicts = [walking_similarity_dict, pointing_similarity_dict, picking_similarity_dict]
    list_similarity_dicts = osd.balance_single_exemplar_similarity_classes_by_frame_count(list_similarity_dicts, 137)

    # Create train/val split
    train_indices, val_indices = create_train_val_split(list_similarity_dicts)

    # Create data loaders
    train_loader = SimilarityDataLoader(list_similarity_dicts, config, True, train_indices)
    val_loader = SimilarityDataLoader(list_similarity_dicts, config, True, val_indices)

    # Create training triplet modules with validation filtering
    walking_train_triplet = TripletMining(
        bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
        squared_left_right_euc_dist, squared_class_neut_euc_dist,
        "walking", config, valid_indices=train_indices[0]
    )
    pointing_train_triplet = TripletMining(
        bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
        squared_left_right_euc_dist, squared_class_neut_euc_dist,
        "pointing", config, valid_indices=train_indices[1]
    )
    picking_train_triplet = TripletMining(
        bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
        squared_left_right_euc_dist, squared_class_neut_euc_dist,
        "picking", config, valid_indices=train_indices[2]
    )

    # Create validation triplet modules
    walking_val_triplet = TripletMining(
        bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
        squared_left_right_euc_dist, squared_class_neut_euc_dist,
        "walking", config, valid_indices=val_indices[0]
    )
    pointing_val_triplet = TripletMining(
        bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
        squared_left_right_euc_dist, squared_class_neut_euc_dist,
        "pointing", config, valid_indices=val_indices[1]
    )
    picking_val_triplet = TripletMining(
        bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
        squared_left_right_euc_dist, squared_class_neut_euc_dist,
        "picking", config, valid_indices=val_indices[2]
    )

    similarity_network = SimilarityNetwork(
        train_loader=train_loader,
        validation_loader=val_loader,
        test_loader=val_loader,
        checkpoint_root_dir=config.checkpoint_root_dir,
        triplet_modules=[walking_train_triplet, pointing_train_triplet, picking_train_triplet],
        val_triplet_modules=[walking_val_triplet, pointing_val_triplet, picking_val_triplet],
        architecture_variant=arch_variant,
        config=config
    )

    similarity_network.run_model_training()

    # similarity_network.evaluate()
    print(f"--------------------------------------------------------------------------")
    print(f"--------------------------------------------------------------------------")
    print(f"--------------------------------------------------------------------------")
