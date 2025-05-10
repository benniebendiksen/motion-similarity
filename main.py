# """
# Program entry point. Loads data for, trains, and evaluates both effort and similarity networks.
#
# If command line argument is provided, program assumes value represents a parallel job index and treats the
# execution as a remote machine run. Otherwise, the program is assumed to be running locally.
# """
#
# import os
# import sys
# import random
# curr_path = os.getcwd()
# sys.path.append(curr_path)
# sys.path.append(curr_path + '\networks')
# from networks.similarity_network import SimilarityNetwork
# from networks.similarity_data_loader import SimilarityDataLoader
# from networks.triplet_mining import TripletMining
# from Config import Config
# import src.organize_synthetic_data as osd
# import tensorflow as tf
#
#
# def check_gpu_access():
#     # Check if GPUs are available
#     gpus = tf.config.list_physical_devices('GPU')
#     if gpus:
#         print(f"✅ TensorFlow detected {len(gpus)} GPU(s):")
#         for gpu in gpus:
#             print(f"  - {gpu}")
#
#         # Set TensorFlow to use GPU memory growth
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#
#         # Run a small computation on the GPU
#         with tf.device('/GPU:0'):
#             a = tf.constant([1.0, 2.0, 3.0])
#             b = tf.constant([4.0, 5.0, 6.0])
#             c = a + b
#         print(f"✅ GPU computation successful: {c.numpy()}")
#     else:
#         print("❌ No GPU detected by TensorFlow.")
#
#
# def create_train_val_split(similarity_dicts, val_ratio=0.2):
#     """
#     Create training and validation indices for each animation type.
#
#     Args:
#         similarity_dicts: List of dictionaries containing class exemplars for each animation
#         val_ratio: Ratio of classes to use for validation
#
#     Returns:
#         train_indices: List of sets containing training class indices for each animation
#         val_indices: List of sets containing validation class indices for each animation
#     """
#     train_indices = []
#     val_indices = []
#
#     for anim_dict in similarity_dicts:
#         # Get keys except neutral
#         keys = [k for k in anim_dict.keys() if k != (0, 0, 0, 0)]
#
#         # Determine validation set size
#         val_size = max(1, int(len(keys) * val_ratio))
#
#         # Randomly sample keys for validation
#         val_keys = set(random.sample(keys, val_size))
#         train_keys = set(k for k in keys if k not in val_keys)
#
#         # Add neutral exemplar to both sets
#         if (0, 0, 0, 0) in anim_dict:
#             train_keys.add((0, 0, 0, 0))
#             val_keys.add((0, 0, 0, 0))
#
#         train_indices.append(train_keys)
#         val_indices.append(val_keys)
#
#     return train_indices, val_indices
#
#
# if __name__ == '__main__':
#     check_gpu_access()
#
#     # Initialize configuration with task index if provided (happens only for remote machine)
#     task_index = sys.argv[1] if len(sys.argv) > 1 else None
#     config = Config(task_index)
#
#     # Ensure required directories exist
#     config.ensure_directories_exist()
#
#     # Architecture variant is either from task index or default
#     arch_variant = int(config.num_task) if config.num_task else 0
#
#     # load effort data and train effort network
#     # batch_ids_partition, labels_dict = osd.load_data(rotations=True, velocities=False)
#     # effort_train_generator = MotionDataGenerator(batch_ids_partition['train'], labels_dict,
#     #                                              **conf.EFFORT_EXEMPLAR_GENERATOR_PARAMS)
#     # effort_validation_generator = MotionDataGenerator(batch_ids_partition['validation'], labels_dict,
#     #                                                   **conf.EFFORT_EXEMPLAR_GENERATOR_PARAMS)
#     # effort_test_generator = MotionDataGenerator(batch_ids_partition['test'], labels_dict,
#     #                                             **conf.EFFORT_EXEMPLAR_GENERATOR_PARAMS)
#     # effort_network = EffortNetwork(train_generator=effort_train_generator,
#     #                                validation_generator=effort_validation_generator,
#     #                                test_generator=effort_test_generator, checkpoint_dir=checkpoint_dir)
#     # start_time = time.time()
#     # history = effort_network.run_model_training()
#     # minutes_tot_time = (time.time() - start_time) / 60
#     # effort_network.write_out_training_results(minutes_tot_time)
#     # collect_job_metrics.collect_job_metrics()
#
#     bool_drop_neutral_exemplar = False
#     bool_fixed_neutral_embedding = False
#     squared_left_right_euc_dist = False
#     squared_class_neut_euc_dist = False
#
#     # Load similarity data
#     walking_similarity_dict = osd.load_similarity_data(bool_drop_neutral_exemplar, "walking", config)["train"]
#     pointing_similarity_dict = osd.load_similarity_data(bool_drop_neutral_exemplar, "pointing", config)["train"]
#     picking_similarity_dict = osd.load_similarity_data(bool_drop_neutral_exemplar, "picking", config)["train"]
#
#     # list_similarity_dicts = [walking_similarity_dict, pointing_similarity_dict, picking_similarity_dict]
#     list_similarity_dicts = [walking_similarity_dict]
#     list_similarity_dicts = osd.balance_single_exemplar_similarity_classes_by_frame_count(list_similarity_dicts, 137)
#
#     # Create train/val split
#     train_indices, val_indices = create_train_val_split(list_similarity_dicts)
#
#     # Create data loaders
#     train_loader = SimilarityDataLoader(list_similarity_dicts, config, True, train_indices)
#     val_loader = SimilarityDataLoader(list_similarity_dicts, config, True, val_indices)
#
#     # Create training triplet modules with validation filtering
#     walking_train_triplet = TripletMining(
#         bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
#         squared_left_right_euc_dist, squared_class_neut_euc_dist,
#         "walking", config, valid_indices=train_indices[0]
#     )
#     # pointing_train_triplet = TripletMining(
#     #     bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
#     #     squared_left_right_euc_dist, squared_class_neut_euc_dist,
#     #     "pointing", config, valid_indices=train_indices[1]
#     # )
#     # picking_train_triplet = TripletMining(
#     #     bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
#     #     squared_left_right_euc_dist, squared_class_neut_euc_dist,
#     #     "picking", config, valid_indices=train_indices[2]
#     # )
#
#     # Create validation triplet modules
#     walking_val_triplet = TripletMining(
#         bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
#         squared_left_right_euc_dist, squared_class_neut_euc_dist,
#         "walking", config, valid_indices=val_indices[0]
#     )
#     # pointing_val_triplet = TripletMining(
#     #     bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
#     #     squared_left_right_euc_dist, squared_class_neut_euc_dist,
#     #     "pointing", config, valid_indices=val_indices[1]
#     # )
#     # picking_val_triplet = TripletMining(
#     #     bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
#     #     squared_left_right_euc_dist, squared_class_neut_euc_dist,
#     #     "picking", config, valid_indices=val_indices[2]
#     # )
#
#     # similarity_network = SimilarityNetwork(
#     #     train_loader=train_loader,
#     #     validation_loader=val_loader,
#     #     test_loader=val_loader,
#     #     checkpoint_root_dir=config.checkpoint_root_dir,
#     #     triplet_modules=[walking_train_triplet, pointing_train_triplet, picking_train_triplet],
#     #     val_triplet_modules=[walking_val_triplet, pointing_val_triplet, picking_val_triplet],
#     #     architecture_variant=arch_variant,
#     #     config=config
#     # )
#
#     similarity_network = SimilarityNetwork(
#         train_loader=train_loader,
#         validation_loader=val_loader,
#         test_loader=val_loader,
#         checkpoint_root_dir=config.checkpoint_root_dir,
#         triplet_modules=[walking_train_triplet],
#         val_triplet_modules=[walking_val_triplet],
#         architecture_variant=arch_variant,
#         config=config
#     )
#
#     similarity_network.run_model_training()
#
#     # similarity_network.evaluate()
#     print(f"--------------------------------------------------------------------------")
#     print(f"--------------------------------------------------------------------------")
#     print(f"--------------------------------------------------------------------------")

"""
Program entry point. Loads data for, trains, and evaluates both effort and similarity networks.
Enhanced with perception-aligned loss function and adaptive distance metrics.

If command line argument is provided, program assumes value represents a parallel job index and treats the
execution as a remote machine run. Otherwise, the program is assumed to be running locally.
"""

import os
import sys
import random
import argparse
import torch
import pandas as pd
import ast
import numpy as np

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


def modify_alpha_matrices(module):
    """
    Modify the alpha matrices in a triplet mining module to focus all matrices on left-right relationships.
    This version properly handles the indices to ensure they're within bounds of the matrices.

    Args:
        module: A TripletMining module with populated alpha matrices

    Returns:
        The same module with modified alpha matrices
    """
    print(f"Modifying alpha matrices for {module.anim_name} to focus on left-right relationships...")

    # Get matrix sizes for validation
    matrix_size = module.matrix_alpha_left_right_right_left.shape[0]

    # Force a reprocessing of the alpha dataframe
    # The original df_alphas should be stored in the module after processing
    if hasattr(module, 'alpha_dataframes'):
        # Get all triplets from alpha_dataframes
        df_alphas = module.alpha_dataframes

        # Get dict_label_to_id
        dict_label_to_id = {class_label: idx for idx, class_label in
                            enumerate(module.dict_similarity_classes_exemplars.keys())}

        # For each row in df_alphas, extract left-right alphas and update all matrices
        for _, row in df_alphas.iterrows():
            efforts_tuple = row['efforts_tuples']

            # Skip if not in valid_indices
            if module.valid_indices is not None:
                if efforts_tuple[0] not in module.valid_indices or efforts_tuple[1] not in module.valid_indices:
                    continue

            # Skip neutral
            if efforts_tuple[0] == (0, 0, 0, 0) or efforts_tuple[1] == (0, 0, 0, 0) or efforts_tuple[0] == \
                    efforts_tuple[1]:
                continue

            # Get indices - make sure they're in dict_label_to_id
            if efforts_tuple[0] not in dict_label_to_id or efforts_tuple[1] not in dict_label_to_id:
                print(f"Warning: Effort tuple {efforts_tuple[0]} or {efforts_tuple[1]} not in dict_label_to_id")
                continue

            index_left = dict_label_to_id[efforts_tuple[0]]
            index_right = dict_label_to_id[efforts_tuple[1]]

            # Check if indices are within bounds
            if index_left >= matrix_size or index_right >= matrix_size:
                print(f"Warning: Indices {index_left}, {index_right} out of bounds for matrix size {matrix_size}")
                continue

            # Get left-right alphas
            left_right_alpha = row['alpha_0_2']
            right_left_alpha = row['alpha_2_0']

            # Set these values in the left-neutral and right-neutral matrices as well
            if row['alpha_0_1'] != 0 or row['alpha_1_0'] != 0:
                module.matrix_alpha_left_neut_neut_left[index_left, index_right] = left_right_alpha
                module.matrix_alpha_left_neut_neut_left[index_right, index_left] = right_left_alpha

                if left_right_alpha != 0:
                    module.matrix_bool_left_neut[index_left, index_right] = 1
                else:
                    module.matrix_bool_left_neut[index_left, index_right] = 0

                if right_left_alpha != 0:
                    module.matrix_bool_neut_left[index_right, index_left] = 1
                else:
                    module.matrix_bool_neut_left[index_right, index_left] = 0

            if row['alpha_2_1'] != 0 or row['alpha_1_2'] != 0:
                module.matrix_alpha_right_neut_neut_right[index_left, index_right] = left_right_alpha
                module.matrix_alpha_right_neut_neut_right[index_right, index_left] = right_left_alpha

                if left_right_alpha != 0:
                    module.matrix_bool_right_neut[index_left, index_right] = 1
                else:
                    module.matrix_bool_right_neut[index_left, index_right] = 0

                if right_left_alpha != 0:
                    module.matrix_bool_neut_right[index_right, index_left] = 1
                else:
                    module.matrix_bool_neut_right[index_right, index_left] = 0

    else:
        print(f"Warning: Could not find alpha_dataframes in module {module.anim_name}")

    return module


def modify_triplet_modules(triplet_modules):
    """
    Modify triplet mining modules to use left-right relationships in all matrices.
    This approach replaces the previous one that tried to modify the internal function.

    Args:
        triplet_modules: List of TripletMining modules

    Returns:
        The same list with modified modules
    """
    modified_modules = []

    for module in triplet_modules:
        # Modify the alpha matrices directly
        modified_module = modify_alpha_matrices(module)
        modified_modules.append(modified_module)

    return modified_modules


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train similarity network with enhanced perception alignment')
    parser.add_argument('--use-perception-loss', action='store_true', help='Use enhanced perception-aligned loss')
    parser.add_argument('--use-adaptive-distance', action='store_true', help='Use adaptive distance module')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'step'],
                        help='Learning rate scheduler type')
    parser.add_argument('--animation', type=str, default='all', choices=['all', 'walking', 'pointing', 'picking'],
                        help='Which animation type to train on (all or specific)')
    parser.add_argument('--task-index', type=str, help='Task index for distributed training')

    args = parser.parse_args()

    check_gpu_access()

    # Initialize configuration with task index if provided (happens only for remote machine)
    task_index = args.task_index if args.task_index else None
    config = Config(task_index)

    # Ensure required directories exist
    config.ensure_directories_exist()

    # Architecture variant is either from task index or default
    arch_variant = int(config.num_task) if config.num_task else 0

    bool_drop_neutral_exemplar = False
    bool_fixed_neutral_embedding = False
    squared_left_right_euc_dist = False
    squared_class_neut_euc_dist = False

    # Load similarity data for all animations
    walking_similarity_dict = osd.load_similarity_data(bool_drop_neutral_exemplar, "walking", config)["train"]

    # Select which animations to use based on command line argument
    if args.animation == 'walking':
        list_similarity_dicts = [walking_similarity_dict]
        animation_names = ['walking']
    elif args.animation == 'pointing':
        pointing_similarity_dict = osd.load_similarity_data(bool_drop_neutral_exemplar, "pointing", config)["train"]
        list_similarity_dicts = [pointing_similarity_dict]
        animation_names = ['pointing']
    elif args.animation == 'picking':
        picking_similarity_dict = osd.load_similarity_data(bool_drop_neutral_exemplar, "picking", config)["train"]
        list_similarity_dicts = [picking_similarity_dict]
        animation_names = ['picking']
    else:  # 'all'
        pointing_similarity_dict = osd.load_similarity_data(bool_drop_neutral_exemplar, "pointing", config)["train"]
        picking_similarity_dict = osd.load_similarity_data(bool_drop_neutral_exemplar, "picking", config)["train"]
        list_similarity_dicts = [walking_similarity_dict, pointing_similarity_dict, picking_similarity_dict]
        animation_names = ['walking', 'pointing', 'picking']

    # Balance frame counts
    list_similarity_dicts = osd.balance_single_exemplar_similarity_classes_by_frame_count(list_similarity_dicts, 137)

    # Create train/val split
    train_indices, val_indices = create_train_val_split(list_similarity_dicts)

    # Create data loaders
    train_loader = SimilarityDataLoader(list_similarity_dicts, config, True, train_indices)
    val_loader = SimilarityDataLoader(list_similarity_dicts, config, True, val_indices)

    # Create training and validation triplet modules
    train_triplet_modules = []
    val_triplet_modules = []

    for i, anim_name in enumerate(animation_names):
        idx = 0 if len(animation_names) == 1 else i

        train_triplet = TripletMining(
            bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
            squared_left_right_euc_dist, squared_class_neut_euc_dist,
            anim_name, config, valid_indices=train_indices[idx]
        )

        val_triplet = TripletMining(
            bool_drop_neutral_exemplar, bool_fixed_neutral_embedding,
            squared_left_right_euc_dist, squared_class_neut_euc_dist,
            anim_name, config, valid_indices=val_indices[idx]
        )

        train_triplet_modules.append(train_triplet)
        val_triplet_modules.append(val_triplet)

    # Modify triplet modules if using perception loss
    # if args.use_perception_loss:
    #     print("Using enhanced perception-aligned loss - modifying triplet mining modules...")
    #     train_triplet_modules = modify_triplet_modules(train_triplet_modules)
    #     val_triplet_modules = modify_triplet_modules(val_triplet_modules)

    # Create the similarity network with enhanced options
    similarity_network = SimilarityNetwork(
        train_loader=train_loader,
        validation_loader=val_loader,
        test_loader=val_loader,
        checkpoint_root_dir=config.checkpoint_root_dir,
        triplet_modules=train_triplet_modules,
        val_triplet_modules=val_triplet_modules,
        architecture_variant=arch_variant,
        config=config,
        lr_scheduler_type=args.scheduler,
        use_perception_loss=args.use_perception_loss,
        use_adaptive_distance=args.use_adaptive_distance
    )

    # Print training configuration
    print("\n=== Training Configuration ===")
    print(f"Animation(s): {args.animation}")
    print(f"Architecture variant: {arch_variant}")
    print(f"Using perception-aligned loss: {args.use_perception_loss}")
    print(f"Using adaptive distance: {args.use_adaptive_distance}")
    print(f"Learning rate scheduler: {args.scheduler}")
    print(f"Number of epochs: {config.n_similarity_epochs}")
    print(f"Number of training triplet modules: {len(train_triplet_modules)}")
    print(f"Using device: {similarity_network.device}")
    print("============================\n")

    # Train the model
    similarity_network.run_model_training()

    # Evaluate the model
    results = similarity_network.evaluate()

    # Print final results
    print("\n=== Final Evaluation Results ===")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Pearson Correlation: {results['pearson_correlation']:.4f} (p-value: {results['pearson_p_value']:.4f})")
    print(f"Spearman Correlation: {results['spearman_correlation']:.4f} (p-value: {results['spearman_p_value']:.4f})")
    print(f"R² Score: {results['r2_score']:.4f}")
    print(f"Number of valid pairs: {results['num_pairs']}")
    print("==============================\n")