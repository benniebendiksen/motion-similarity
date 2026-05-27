"""
static module for organizing triplet mining data as well as performing online triplet mining
"""
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import ast
import pickle


class TripletMining:
    def __init__(self, bool_drop, bool_fixed, squared_left_right, squared_class_neut, anim_name, config, valid_indices=None, exclude_neutral_completely=False, preloaded_dict=None):
        self.config = config
        self.anim_name = anim_name
        self.dict_similarity_classes_exemplars = {}
        self.matrix_alpha_left_right_right_left = None
        self.matrix_alpha_left_neut_neut_left = None
        self.matrix_alpha_right_neut_neut_right = None
        self.matrix_bool_left_right = None
        self.matrix_bool_right_left = None
        self.matrix_bool_left_neut = None
        self.matrix_bool_neut_left = None
        self.matrix_bool_right_neut = None
        self.matrix_bool_neut_right = None

        self.matrix_comparison_values_left_right = None
        self.matrix_comparison_values_left_neut = None
        self.matrix_comparison_values_right_neut = None
        self.matrix_comparison_bool_left_right = None
        self.matrix_comparison_bool_left_neut = None
        self.matrix_comparison_bool_right_neut = None

        self.num_states_drives = 0
        self.tensor_dists_left_right_right_left = None
        self.tensor_dists_class_neut = None
        self.neutral_embedding = None
        self.exclude_neutral_completely = exclude_neutral_completely
        self.anim_name = anim_name

        self.bool_drop_neutral_exemplar = bool_drop
        self.bool_fixed_neutral_embedding = bool_fixed
        self.squared_left_right_euc_dist = squared_left_right
        self.squared_class_neut_euc_dist = squared_class_neut

        # Store valid indices for filtering. A train/test split mechanism
        self.valid_indices = valid_indices

        # Initialize with proper batch_size for this module
        if self.valid_indices is not None:
            # For validation/training split
            self.batch_size = len(self.valid_indices)
        else:
            # Use the full batch size
            self.batch_size = self.config.similarity_per_anim_class_num

        # For pointing with complete neutral exclusion, we don't use class-neutral distances
        if self.exclude_neutral_completely:
            self.use_neutral_distances = False
        else:
            self.use_neutral_distances = True

        self.initialize_triplet_mining(anim_name, preloaded_dict=preloaded_dict)

    def initialize_triplet_mining(self, anim_name, preloaded_dict=None):
        """
        Initialize the triplet mining module's state variables.

        This function loads necessary data, sets up state variables, and performs preprocessing.

        Args:
            anim_name: str: name of the animation (e.g., "walking", "pointing", "picking")

        Returns:
            None
        """

        print("Initializing Triplet Mining module state variables")

        # Use a caller-supplied dict (embedding pipelines) or load from pickle (raw-motion pipelines).
        # Embedding pipelines pass preloaded_dict so the raw-motion pickle is never read,
        # avoiding type mismatches between AE embedding vectors and raw-motion arrays.
        if preloaded_dict is not None:
            print(f"  Using preloaded dict ({len(preloaded_dict)} classes) — skipping pickle load.")
            self.dict_similarity_classes_exemplars = preloaded_dict
        else:
            self.dict_similarity_classes_exemplars = pickle.load(open(
                self.config.similarity_exemplars_dir + anim_name + "_" + self.config.similarity_dict_file_name, "rb"))
        # print(f"Full dictionary classes: {len(self.dict_similarity_classes_exemplars.keys())}")

        # If valid_indices is provided, subset the dictionary
        if self.valid_indices is not None:
            original_dict = self.dict_similarity_classes_exemplars
            self.dict_similarity_classes_exemplars = {}

            # Only keep classes in valid_indices
            for key in self.valid_indices:
                if key in original_dict:
                    self.dict_similarity_classes_exemplars[key] = original_dict[key]

            # Always include neutral if not dropping it
            key_to_remove = (0, 0, 0, 0)
            if not self.bool_drop_neutral_exemplar:
                if key_to_remove not in self.valid_indices:
                    self.dict_similarity_classes_exemplars[key_to_remove] = original_dict[key_to_remove]
                self.num_states_drives = len(self.dict_similarity_classes_exemplars.keys()) - 1
            elif self.bool_drop_neutral_exemplar:
                if key_to_remove in self.valid_indices:
                    _removed_value = self.dict_similarity_classes_exemplars.pop(key_to_remove)
                self.num_states_drives = len(self.dict_similarity_classes_exemplars.keys())

        else:
        # Now check for the neutral key
            key_to_remove = (0, 0, 0, 0)
            # if key_to_remove not in self.dict_similarity_classes_exemplars:
            #     assert False, f"triplet_mining.py: Key '{key_to_remove}' not found in dict_similarity_classes_exemplars"

            # Calculate num_states_drives based on the actual dictionary (which now only has valid_indices)
            if self.bool_drop_neutral_exemplar:
                if key_to_remove in self.dict_similarity_classes_exemplars:
                    _removed_value = self.dict_similarity_classes_exemplars.pop(key_to_remove)
                    print(f"Removed key '{key_to_remove}' from dict_similarity_classes_exemplars")
                self.num_states_drives = len(self.dict_similarity_classes_exemplars.keys())
            else:
                self.num_states_drives = len(self.dict_similarity_classes_exemplars.keys()) - 1

        # print(f"triplet_mining:init: using {self.num_states_drives} states + drives for this module")

        # Now initialize matrices with the correct size based on actual num_states_drives
        (self.matrix_alpha_left_right_right_left,
         self.matrix_alpha_left_neut_neut_left,
         self.matrix_alpha_right_neut_neut_right,
         self.matrix_bool_left_right,
         self.matrix_bool_right_left,
         self.matrix_bool_left_neut,
         self.matrix_bool_neut_left,
         self.matrix_bool_right_neut,
         self.matrix_bool_neut_right) = [
            torch.zeros((self.num_states_drives, self.num_states_drives), dtype=torch.float32, requires_grad=False)
            for _ in range(9)
        ]

        (self.matrix_comparison_values_left_right,
         self.matrix_comparison_values_left_neut,
         self.matrix_comparison_values_right_neut,
         self.matrix_comparison_bool_left_right,
         self.matrix_comparison_bool_left_neut,
         self.matrix_comparison_bool_right_neut) = [
            torch.zeros((self.num_states_drives, self.num_states_drives), dtype=torch.float32, requires_grad=False)
            for _ in range(6)
        ]

        # Initialize tensor_dists_class_neut with the correct size
        self.tensor_dists_class_neut = torch.zeros(self.num_states_drives, dtype=torch.float32, requires_grad=False)
        self.neutral_embedding = torch.zeros(self.config.embedding_refinement_model_output_size, dtype=torch.float32, requires_grad=False)

        self.subset_global_dict()
        self.pre_process_comparisons_data(anim_name)

    def subset_global_dict(self):
        """
        Subsets the global dictionary of similarity classes based on data in the comparisons DataFrame.

        Args:
            None

        Returns:
            None
        """
        dict_label_to_id = {class_label: idx for idx, class_label in
                            enumerate(self.dict_similarity_classes_exemplars.keys())}
        # print(dict_label_to_id)

    # def extract_neutral_embedding(embeddings):
    #     """
    #     Assigns the neutral embedding from the network output, shape (embedding_size,), to instance attribute
    #
    #     Args:
    #         embeddings: tensor of shape (batch_size, embed_dim)
    #
    #     Returns:
    #         embeddings: tensor of shape (batch_size - 1, embed_dim)
    #     """
    #     # print(f'extract_neutral_embedding() called with embeddings of shape: {tf.shape(embeddings)}, neutral is: {embeddings[0]}')
    #     if not conf.bool_fixed_neutral_embedding:
    #         neutral_embedding.assign(embeddings[0])
    #         embeddings = embeddings[1:]
    #     return embeddings

    def zero_out_neutral_embedding(self, embeddings):
        if self.bool_drop_neutral_exemplar:
            modified_embeddings = embeddings
        else:
            modified_embeddings = embeddings[1:]

        return self.neutral_embedding, modified_embeddings

    def maintain_dynamic_neutral_embedding(self, embeddings):
        # print(f"Embeddings shape: {embeddings.shape}")  # Debugging line
        # print(f"Embeddings type: {type(embeddings)}")
        if self.bool_drop_neutral_exemplar:
            assert False, "triplet_mining.py: maintain_dynamic_neutral_embedding() called with bool_drop_neutral_exemplar set to True"
        # Assign the first embedding to neutral_embedding
        neutral_embedding = embeddings[0]

        # Remove the first element from the tensor
        modified_embeddings = embeddings[1:]

        return neutral_embedding, modified_embeddings

    def calculate_distances(self, embeddings):
        """Modified to handle pointing without neutral"""
        try:
            if self.anim_name == 'pointing' and self.exclude_neutral_completely:
                # For pointing, no neutral embedding exists - only calculate pairwise distances
                # All embeddings are non-neutral classes
                modified_embeddings = embeddings

                # Calculate left-right distances between all embeddings
                dot_product = torch.matmul(modified_embeddings, modified_embeddings.T)
                square_norm = torch.sum(modified_embeddings ** 2, dim=1)
                left_right_distances = square_norm.unsqueeze(1) + square_norm.unsqueeze(0) - 2.0 * dot_product
                left_right_distances = torch.clamp(left_right_distances, min=0.0)

                if not self.squared_left_right_euc_dist:
                    epsilon = 1e-12
                    left_right_distances = torch.sqrt(left_right_distances + epsilon)

                # For pointing, no class-neutral distances - set to zeros or a constant
                # This effectively removes the neutral anchor from the loss computation
                self.tensor_dists_class_neut = torch.zeros(modified_embeddings.shape[0],
                                                           dtype=torch.float32,
                                                           requires_grad=False)

                return left_right_distances
            else:
                # Original behavior for walking
                return self.calculate_distances_original(embeddings)
        except Exception as e:
            print(f"Error in calculate_distances: {e}")
            raise e

    def calculate_distances_original(self, embeddings):
        """
        Calculate both:
        1. 1D tensor of distances between class embeddings and the neutral embedding
        2. 2D matrix of distances between all class embeddings

        All distances are normalized to the range [0,1]
        """
        try:
            # Determine neutral and modified embeddings
            if self.bool_fixed_neutral_embedding:
                neutral_embedding, modified_embeddings = self.zero_out_neutral_embedding(embeddings)
            else:
                neutral_embedding, modified_embeddings = self.maintain_dynamic_neutral_embedding(embeddings)

            # Calculate left-right distances between all embeddings
            # Compute the dot product
            dot_product = torch.matmul(modified_embeddings, modified_embeddings.T)

            # Compute the squared norms
            square_norm = torch.sum(modified_embeddings ** 2, dim=1)

            # Compute pairwise squared Euclidean distances
            left_right_distances = square_norm.unsqueeze(1) + square_norm.unsqueeze(0) - 2.0 * dot_product

            # Clamp to ensure no negative distances due to floating-point errors
            left_right_distances = torch.clamp(left_right_distances, min=0.0)

            if not self.squared_left_right_euc_dist:
                # For non-squared distances, compute square root with epsilon for stability
                epsilon = 1e-12
                left_right_distances = torch.sqrt(left_right_distances + epsilon)

            # Calculate class-neutral distances using similar approach to left-right distances
            # Compute squared norm of neutral embedding
            neutral_square_norm = torch.sum(neutral_embedding ** 2)

            # Compute the dot product between modified embeddings and neutral embedding
            neutral_dot_product = torch.matmul(modified_embeddings, neutral_embedding)

            # Compute squared distances using the same formula as left-right
            class_neut_squared_dist = square_norm + neutral_square_norm - 2.0 * neutral_dot_product

            # Clamp to ensure no negative distances due to floating-point errors
            class_neut_squared_dist = torch.clamp(class_neut_squared_dist, min=0.0)

            # Apply sqrt if squared_class_neut_dist is True (notice this is inverted compared to left-right!)
            if not self.squared_class_neut_euc_dist:
                epsilon = 1e-12
                class_neut_dist = torch.sqrt(class_neut_squared_dist + epsilon)
            else:
                class_neut_dist = class_neut_squared_dist

            # ----- NORMALIZATION SECTION -----

            # 1. Normalize left-right distances to [0,1]
            if left_right_distances.numel() > 100000:  # Check if there's more than one element
                min_lr_dist = torch.min(left_right_distances)
                max_lr_dist = torch.max(left_right_distances)

                # Check to avoid division by zero
                if max_lr_dist > min_lr_dist:
                    # Normalize to [0,1]
                    left_right_distances = (left_right_distances - min_lr_dist) / (max_lr_dist - min_lr_dist)
                else:
                    # If all distances are the same, set to 0.5 (middle of range)
                    left_right_distances = torch.ones_like(left_right_distances) * 0.5

            # 2. Normalize class-neutral distances to [0,1]
            if class_neut_dist.numel() > 100000:  # Check if there's more than one element
                min_cn_dist = torch.min(class_neut_dist)
                max_cn_dist = torch.max(class_neut_dist)

                # Check to avoid division by zero
                if max_cn_dist > min_cn_dist:
                    # Normalize to [0,1]
                    normalized_class_neut_dist = (class_neut_dist - min_cn_dist) / (max_cn_dist - min_cn_dist)
                else:
                    # If all distances are the same, set to 0.5 (middle of range)
                    normalized_class_neut_dist = torch.ones_like(class_neut_dist) * 0.5

                # Store the normalized distances
                self.tensor_dists_class_neut = normalized_class_neut_dist
            else:
                # If there's only one element, just store it as is
                self.tensor_dists_class_neut = class_neut_dist

            return left_right_distances

        except Exception as e:
            print(f"Error in calculate_distances: {e}")
            raise e

    # def calculate_distances(self, embeddings):
    #     """
    #     Calculate both:
    #     1. 1D tensor of distances between class embeddings and the neutral embedding
    #     2. 2D matrix of distances between all class embeddings
    #     """
    #     try:
    #         # Determine neutral and modified embeddings
    #         if self.bool_fixed_neutral_embedding:
    #             neutral_embedding, modified_embeddings = self.zero_out_neutral_embedding(embeddings)
    #         else:
    #             neutral_embedding, modified_embeddings = self.maintain_dynamic_neutral_embedding(embeddings)
    #
    #         # Calculate left-right distances between all embeddings
    #         # Compute the dot product
    #         dot_product = torch.matmul(modified_embeddings, modified_embeddings.T)
    #
    #         # Compute the squared norms
    #         square_norm = torch.sum(modified_embeddings ** 2, dim=1)
    #
    #         # Compute pairwise squared Euclidean distances
    #         left_right_distances = square_norm.unsqueeze(1) + square_norm.unsqueeze(0) - 2.0 * dot_product
    #
    #         # Clamp to ensure no negative distances due to floating-point errors
    #         left_right_distances = torch.clamp(left_right_distances, min=0.0)
    #
    #         if not self.squared_left_right_euc_dist:
    #             # For non-squared distances, compute square root with epsilon for stability
    #             epsilon = 1e-12
    #             left_right_distances = torch.sqrt(left_right_distances + epsilon)
    #
    #         # Calculate class-neutral distances using similar approach to left-right distances
    #         # Compute squared norm of neutral embedding
    #         neutral_square_norm = torch.sum(neutral_embedding ** 2)
    #
    #         # Compute the dot product between modified embeddings and neutral embedding
    #         neutral_dot_product = torch.matmul(modified_embeddings, neutral_embedding)
    #
    #         # Compute squared distances using the same formula as left-right
    #         class_neut_squared_dist = square_norm + neutral_square_norm - 2.0 * neutral_dot_product
    #
    #         # Clamp to ensure no negative distances due to floating-point errors
    #         class_neut_squared_dist = torch.clamp(class_neut_squared_dist, min=0.0)
    #
    #         # Apply sqrt if squared_class_neut_dist is True (notice this is inverted compared to left-right!)
    #         # This matches your observation about what works well for training
    #         if not self.squared_class_neut_euc_dist:
    #             epsilon = 1e-12
    #             self.tensor_dists_class_neut = torch.sqrt(class_neut_squared_dist + epsilon)
    #         else:
    #             self.tensor_dists_class_neut = class_neut_squared_dist
    #
    #         return left_right_distances
    #
    #     except Exception as e:
    #         print(f"Error in calculate_distances: {e}")
    #         raise e

    def calculate_left_right_distances(self, embeddings):
        """Compute the 2D matrix of distances between all 56 class embeddings."""

        if self.bool_fixed_neutral_embedding:
            _neutral_embedding, modified_embeddings = self.zero_out_neutral_embedding(embeddings)
        else:
            _neutral_embedding, modified_embeddings = self.maintain_dynamic_neutral_embedding(embeddings)

        # print embeddings shape
        # print(f"Modified embeddings shape: {modified_embeddings.shape}")
        # print(f"Modified embeddings type: {type(modified_embeddings)}")
        # print(f"Neutral embedding shape: {_neutral_embedding.shape}")
        # print(f"Neutral embedding type: {type(_neutral_embedding)}")
        # Modified embeddings shape: torch.Size([45, 32])
        # Modified embeddings type: <class 'torch.Tensor'>
        # Neutral embedding shape: torch.Size([32])
        # Neutral embedding type: <class 'torch.Tensor'>


        # Compute the dot product
        dot_product = torch.matmul(modified_embeddings, modified_embeddings.T)

        # Compute the squared norms
        square_norm = torch.sum(modified_embeddings ** 2, dim=1)

        # Compute pairwise squared Euclidean distances
        distances = square_norm.unsqueeze(1) + square_norm.unsqueeze(0) - 2.0 * dot_product

        # Clamp to ensure no negative distances due to floating-point errors
        distances = torch.clamp(distances, min=0.0)

        if not self.squared_left_right_euc_dist:
            # For non-squared distances, compute square root
            # Add a small epsilon to all elements to avoid numerical instability
            epsilon = 1e-12  # Slightly larger epsilon for better stability
            sqrt_distances = torch.sqrt(distances + epsilon)
            return sqrt_distances
        else:
            # Return squared distances
            return distances

    def calculate_class_neut_distances(self, embeddings):
        """
        Calculate 1D tensor of either squared L2 norm or L2 norm of differences
        between class embeddings and the neutral embedding.
        """
        try:
            # Determine neutral and modified embeddings
            if self.bool_fixed_neutral_embedding:
                neutral_embedding, modified_embeddings = self.zero_out_neutral_embedding(embeddings)
            else:
                neutral_embedding, modified_embeddings = self.maintain_dynamic_neutral_embedding(embeddings)

                # Check that embeddings size matches what we expect
                # if self.valid_indices is not None and modified_embeddings.shape[0] != len(self.valid_indices):
                #     raise ValueError(f"Warning: Expected {len(self.valid_indices)} embeddings but got {modified_embeddings.shape[0]}")


            # Compute distances
            differences = modified_embeddings - neutral_embedding

            if self.squared_class_neut_euc_dist:
                # include numerical stability term
                epsilon = 1e-12
                self.tensor_dists_class_neut = torch.sqrt(torch.sum(differences ** 2, dim=1) + epsilon)
            else:
                # Euclidean distance
                self.tensor_dists_class_neut = torch.norm(differences, p=2, dim=1)
        except Exception as e:
            print(f"Error in calculate_class_neut_distances: {e}")
            raise e

    def pre_process_comparisons_data(self, anim_name):
        """
        Preprocess user comparison data and populate alpha matrices and masks based on the data. Filters based on valid indices.

        Args:
            anim_name: str: name of the animation (e.g., "walking", "pointing", "picking")

        Returns:
            None
        """

        def verify_comparison_data():
            # Check that the Dataframe has no repeated efforts_tuples values
            seen_tuples = set()
            for index, row in df_comparisons.iterrows():
                hashable_list = tuple(row['efforts_tuples'])
                # Check if the hashable list is already in the set
                if hashable_list in seen_tuples:
                    assert False, f"Duplicate df_comparison efforts_tuples row at index {index}: {row['efforts_tuples']}"
                seen_tuples.add(hashable_list)
                # Ensure efforts_left does not equal efforts_right
                assert row['efforts_tuples'][0] != row['efforts_tuples'][1], (f"efforts_left equals efforts_right at "
                                                                              f"index: {index}: "
                                                                              f"{row['efforts_tuples']}")
            # Check that the Dataframe does not include the neutral as a similarity class
            assert (0, 0, 0, 0) not in seen_tuples, "neutral similarity class included in comparisons data"
            # Check that the Dataframe has the correct number of similarity classes
            assert len(seen_tuples) == self.num_states_drives, (f"incomplete similarity class count in comparisons "
                                                           f"data: {len(seen_tuples)}")

        def get_all_left_right_comparisons(self):
            """
            Creates an easy-to-use data structure containing all left-right comparisons
            with their inverse values (1 - comparison_value) to use in contrastive loss.

            Returns:
                dict: Dictionary with keys:
                    'indices': List of tuples (left_idx, right_idx) for all valid comparisons
                    'targets': List of target distance values (1 - comparison_value)
                    'masks': Dictionary with left-right, left-neut, right-neut boolean masks
                    'values': Dictionary with the original comparison values
            """
            # Get all valid left-right comparison pairs
            valid_pairs = []
            target_distances = []

            # Dictionary to track masks
            masks = {
                'left_right': [],  # Original left-right comparisons
                'left_neut': [],  # Left-right pairs derived from left-neutral comparisons
                'right_neut': []  # Left-right pairs derived from right-neutral comparisons
            }

            # Dictionary to store original comparison values
            comparison_values = {
                'left_right': [],
                'left_neut': [],
                'right_neut': []
            }

            # Get dict mappings for reference
            dict_label_to_id = {class_label: idx for idx, class_label in
                                enumerate(self.dict_similarity_classes_exemplars.keys())}

            # Process left-right comparisons from the comparison dataframe
            for idx, row in self.df_comparisons.iterrows():
                if row['selected0'] == 0 and row['selected1'] == 2:  # This is a left-right comparison
                    # Get the effort tuples
                    efforts_tuple = row['efforts_tuples']

                    # Skip if either class is not in valid_indices
                    if self.valid_indices is not None:
                        if efforts_tuple[0] not in self.valid_indices or efforts_tuple[1] not in self.valid_indices:
                            continue

                    # Skip neutral exemplars
                    if efforts_tuple[0] == (0, 0, 0, 0) or efforts_tuple[1] == (0, 0, 0, 0):
                        continue

                    # Get indices for this pair
                    index_left = dict_label_to_id[efforts_tuple[0]]
                    index_right = dict_label_to_id[efforts_tuple[1]]

                    # Get the comparison value and its inverse (target distance)
                    comparison_value = row['count_normalized']
                    target_distance = 1.0 - comparison_value

                    # Store this pair and its target distance
                    valid_pairs.append((index_left, index_right))
                    target_distances.append(target_distance)

                    # Track which mask this belongs to
                    masks['left_right'].append(1)
                    masks['left_neut'].append(0)
                    masks['right_neut'].append(0)

                    # Store original comparison value
                    comparison_values['left_right'].append(comparison_value)
                    comparison_values['left_neut'].append(0.0)
                    comparison_values['right_neut'].append(0.0)

            # Also include the matrices for convenience
            comparison_data = {
                'indices': valid_pairs,
                'targets': target_distances,
                'masks': masks,
                'values': comparison_values,
                'matrix_values': {
                    'left_right': self.matrix_comparison_values_left_right,
                    'left_neut': self.matrix_comparison_values_left_neut,
                    'right_neut': self.matrix_comparison_values_right_neut
                },
                'matrix_bool': {
                    'left_right': self.matrix_comparison_bool_left_right,
                    'left_neut': self.matrix_comparison_bool_left_neut,
                    'right_neut': self.matrix_comparison_bool_right_neut
                }
            }

            return comparison_data

        def _generate_df_alphas(df_comparisons):
            """
            Complete implementation of alpha values generation for all six possible pairwise comparisons in a triplet.

            For each triplet (group of 3 rows) representing comparisons between left (0), neutral (1), and right (2),
            this function calculates all six possible alpha values:
            - alpha_0_1: Left-Neutral comparison with Left as anchor
            - alpha_1_0: Left-Neutral comparison with Neutral as anchor
            - alpha_0_2: Left-Right comparison with Left as anchor
            - alpha_2_0: Left-Right comparison with Right as anchor
            - alpha_1_2: Neutral-Right comparison with Neutral as anchor
            - alpha_2_1: Neutral-Right comparison with Right as anchor

            Args:
                df_comparisons: DataFrame with comparison data

            Returns:
                alpha_dataframes: DataFrame with all alpha values
            """
            comparisons_list = []
            selection_values = [0, 1, 2]  # Left, Neutral, Right

            # Initialize alpha columns for all six possible pairwise comparisons
            for pair in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]:
                df_comparisons[f'alpha_{pair[0]}_{pair[1]}'] = 0.0

            # Add column for direct comparison between 0-2 (Left-Right)
            df_comparisons['direct_comparison_value'] = np.nan
            # Add column to indicate most preferred pair
            df_comparisons['most_preferred_pair'] = None

            # important for inference script
            self.df_comparisons = df_comparisons

            # Process triplets (groups of 3 rows)
            for i in range(0, len(df_comparisons), 3):
                # Get the current triplet
                group = df_comparisons.iloc[i:i + 3]

                # Skip incomplete groups
                if len(group) < 3:
                    raise ValueError(f"Triplet group at index {i} is incomplete: {group}")

                # Find row with highest preference (maximum count_normalized)
                max_row = group.loc[group['count_normalized'].idxmax()]
                max_selected_0, max_selected_1 = max_row['selected0'], max_row['selected1']

                # Identify the third option (the one not in the preferred pair)
                negative_index = next(x for x in selection_values if x != max_selected_0 and x != max_selected_1)

                # Find all normalized preference values for each possible pairing
                pair_values = {}

                # Extract all normalized preference values from the group
                for _, row in group.iterrows():
                    s0, s1 = row['selected0'], row['selected1']
                    pair_values[(s0, s1)] = row['count_normalized']

                    # determine most preferred pair
                    if row['count_normalized'] == max_row['count_normalized']:
                        df_comparisons.at[max_row.name, 'most_preferred_pair'] = (s0, s1)

                # Calculate all six alpha values systematically

                # For 0-1 pair (Left-Neutral)
                if (0, 1) in pair_values and (0, 2) in pair_values:
                    # alpha_0_1: How much more Left is preferred with Neutral vs. with Right
                    df_comparisons.loc[max_row.name, 'alpha_0_1'] = pair_values.get((0, 1), 0) - pair_values.get((0, 2),
                                                                                                                 0)

                if (0, 1) in pair_values and (1, 2) in pair_values:
                    # alpha_1_0: How much more Neutral is preferred with Left vs. with Right
                    df_comparisons.loc[max_row.name, 'alpha_1_0'] = pair_values.get((0, 1), 0) - pair_values.get((1, 2),
                                                                                                                 0)

                # For 0-2 pair (Left-Right)
                if (0, 2) in pair_values and (0, 1) in pair_values:
                    # alpha_0_2: How much more Left is preferred with Right vs. with Neutral
                    df_comparisons.loc[max_row.name, 'alpha_0_2'] = pair_values.get((0, 2), 0) - pair_values.get((0, 1),
                                                                                                                 0)

                if (0, 2) in pair_values and (1, 2) in pair_values:
                    # alpha_2_0: How much more Right is preferred with Left vs. with Neutral
                    df_comparisons.loc[max_row.name, 'alpha_2_0'] = pair_values.get((0, 2), 0) - pair_values.get((1, 2),
                                                                                                                 0)

                # For 1-2 pair (Neutral-Right)
                if (1, 2) in pair_values and (0, 1) in pair_values:
                    # alpha_1_2: How much more Neutral is preferred with Right vs. with Left
                    df_comparisons.loc[max_row.name, 'alpha_1_2'] = pair_values.get((1, 2), 0) - pair_values.get((0, 1),
                                                                                                                 0)

                if (1, 2) in pair_values and (0, 2) in pair_values:
                    # alpha_2_1: How much more Right is preferred with Neutral vs. with Left
                    df_comparisons.loc[max_row.name, 'alpha_2_1'] = pair_values.get((1, 2), 0) - pair_values.get((0, 2),
                                                                                                                 0)

                # Store the direct comparison value (Left-Right)
                direct_comparison_row = group[(group['selected0'] == 0) & (group['selected1'] == 2)]
                if not direct_comparison_row.empty:
                    df_comparisons.loc[max_row.name, 'direct_comparison_value'] = direct_comparison_row.iloc[0][
                        'count_normalized']

                # Add this processed row to our results
                comparisons_list.append(df_comparisons.loc[max_row.name])

            # Create the final dataframe with alpha values
            alpha_dataframes = pd.DataFrame(comparisons_list)
            alpha_dataframes.reset_index(drop=True, inplace=True)
            # important for inference script
            self.alpha_dataframes = alpha_dataframes

            return alpha_dataframes

        def _populate_alpha_matrices_and_masks(df_alphas):
            """
            Populate the alpha matrices and their corresponding masks based on the data in the comparisons DataFrame.
            Modified to collect left-right alphas across all preference cases.

            This version:
            1. Uses matrix_alpha_left_right_right_left for the original left-right preference case
            2. Replaces matrix_alpha_left_neut_neut_left with a matrix containing left-right alphas from left-neutral preference cases
            3. Replaces matrix_alpha_right_neut_neut_right with a matrix containing left-right alphas from right-neutral preference cases

            Args:
               df_alphas: DataFrame with alpha values

            Returns:
               None
            """
            # Iterate over the rows of the comparisons DataFrame
            counter_df_alphas_rows = 0
            repeat_class_comparison_counter = 0
            equal_comparison_counter = 0
            zero_alphas_counter = 0
            unequal_comparison_counter = 0
            bool_swap_left_right = False

            # write out df_alphas to csv
            df_alphas.to_csv('py_df_alphas_walking.csv')

            for index, row in df_alphas.iterrows():
                counter_df_alphas_rows += 1
                efforts_tuple = row['efforts_tuples']

                # Skip if either class is not in valid_indices. A train/val split ensuring mechanism
                if self.valid_indices is not None:
                    if efforts_tuple[0] not in self.valid_indices or efforts_tuple[1] not in self.valid_indices:
                        continue

                # enforce constraint that i < j always corresponds to left, right / i > j to right, left effort_tuples.
                if dict_label_to_id[efforts_tuple[0]] > dict_label_to_id[efforts_tuple[1]]:
                    row['efforts_tuples'] = [efforts_tuple[1], efforts_tuple[0]]
                    bool_swap_left_right = True

                efforts_left = row['efforts_tuples'][0]
                efforts_right = row['efforts_tuples'][1]
                index_left = dict_label_to_id[efforts_left]
                index_right = dict_label_to_id[efforts_right]

                # Skip neutral or self comparisons
                if efforts_left == (0, 0, 0, 0) or efforts_left == efforts_right:
                    repeat_class_comparison_counter += 1
                    print(f"repeat class comparison at indices: {index_left} , {index_right}")
                    continue

                # grab most preferred pair
                most_preferred_pair = row['most_preferred_pair']

                # Case 1: left and right are most preferable (original case)
                if most_preferred_pair == (0, 2):
                    # Extract the two alpha values for the comparison
                    left_right_alpha = row['alpha_0_2']
                    right_left_alpha = row['alpha_2_0']

                    if bool_swap_left_right:
                        left_right_alpha = row['alpha_2_0']
                        right_left_alpha = row['alpha_0_2']

                    # Set boolean flags based on alpha values
                    if left_right_alpha == 0 and right_left_alpha == 0:
                        zero_alphas_counter += 1
                        bool_constant_left_right = 0
                        bool_constant_right_left = 0
                    elif left_right_alpha == 0:
                        bool_constant_left_right = 0
                        bool_constant_right_left = 1
                        unequal_comparison_counter += 1
                    elif right_left_alpha == 0:
                        bool_constant_left_right = 1
                        bool_constant_right_left = 0
                        unequal_comparison_counter += 1
                    else:
                        bool_constant_left_right = 1
                        bool_constant_right_left = 1
                        unequal_comparison_counter += 1

                    # Update matrices for the original left-right case
                    self.matrix_alpha_left_right_right_left[index_left, index_right] += left_right_alpha
                    self.matrix_bool_left_right[index_left, index_right] = bool_constant_left_right
                    self.matrix_alpha_left_right_right_left[index_right, index_left] += right_left_alpha
                    self.matrix_bool_right_left[index_right, index_left] = bool_constant_right_left

                # Case 2: left and neutral are most preferable
                elif most_preferred_pair == (1, 2):
                    # Instead of left-neutral alphas, extract the left-right alphas present in the row
                    # These are already calculated during _generate_df_alphas() for all cases
                    left_right_alpha = row[
                        'alpha_0_2']  # Alpha for left compared to right when left-neutral was preferred
                    right_left_alpha = row[
                        'alpha_2_0']  # Alpha for right compared to left when left-neutral was preferred

                    if bool_swap_left_right:
                        left_right_alpha = row['alpha_2_0']
                        right_left_alpha = row['alpha_0_2']

                    # Set boolean flags
                    if left_right_alpha == 0 and right_left_alpha == 0:
                        zero_alphas_counter += 1
                        bool_constant_left_right = 0
                        bool_constant_right_left = 0
                    elif left_right_alpha == 0:
                        bool_constant_left_right = 0
                        bool_constant_right_left = 1
                        unequal_comparison_counter += 1
                    elif right_left_alpha == 0:
                        bool_constant_left_right = 1
                        bool_constant_right_left = 0
                        unequal_comparison_counter += 1
                    else:
                        bool_constant_left_right = 1
                        bool_constant_right_left = 1
                        unequal_comparison_counter += 1

                    # Update left-neutral matrix with left-right alphas
                    # This replaces the matrix_alpha_left_neut_neut_left with left-right alphas
                    self.matrix_alpha_left_neut_neut_left[index_left, index_right] += left_right_alpha
                    self.matrix_bool_left_neut[index_left, index_right] = bool_constant_left_right
                    self.matrix_alpha_left_neut_neut_left[index_right, index_left] += right_left_alpha
                    self.matrix_bool_neut_left[index_right, index_left] = bool_constant_right_left

                # Case 3: right and neutral are most preferable
                elif most_preferred_pair == (0, 1):
                    # Extract left-right alphas instead of right-neutral alphas
                    left_right_alpha = row[
                        'alpha_0_2']  # Alpha for left compared to right when right-neutral was preferred
                    right_left_alpha = row[
                        'alpha_2_0']  # Alpha for right compared to left when right-neutral was preferred

                    if bool_swap_left_right:
                        left_right_alpha = row['alpha_2_0']
                        right_left_alpha = row['alpha_0_2']

                    # Set boolean flags
                    if left_right_alpha == 0 and right_left_alpha == 0:
                        zero_alphas_counter += 1
                        bool_constant_left_right = 0
                        bool_constant_right_left = 0
                    elif left_right_alpha == 0:
                        bool_constant_left_right = 0
                        bool_constant_right_left = 1
                        unequal_comparison_counter += 1
                    elif right_left_alpha == 0:
                        bool_constant_left_right = 1
                        bool_constant_right_left = 0
                        unequal_comparison_counter += 1
                    else:
                        bool_constant_left_right = 1
                        bool_constant_right_left = 1
                        unequal_comparison_counter += 1

                    # Update right-neutral matrix with left-right alphas
                    # This replaces the matrix_alpha_right_neut_neut_right with left-right alphas
                    self.matrix_alpha_right_neut_neut_right[index_left, index_right] += left_right_alpha
                    self.matrix_bool_right_neut[index_left, index_right] = bool_constant_left_right
                    self.matrix_alpha_right_neut_neut_right[index_right, index_left] += right_left_alpha
                    self.matrix_bool_neut_right[index_right, index_left] = bool_constant_right_left

                # Implies equal selection across all three pairs of a triplet
                else:
                    equal_comparison_counter += 1

                bool_swap_left_right = False

            print(f" Equal comparison counter: {equal_comparison_counter}")
            print(f"Unequal comparison counter: {unequal_comparison_counter}")
            print(f"total comparison rows: {counter_df_alphas_rows}")

        def _populate_comparison_values_matrices(df_comparisons):
            """
            Populates the comparison values matrices directly from the df_comparisons dataframe.

            This method extracts the count_normalized values for left-right, left-neut, and right-neut comparisons
            from the dataframe and populates the corresponding comparison matrices.

            Args:
                df_comparisons: DataFrame containing comparison data with count_normalized values

            Returns:
                None
            """
            # Reset matrices to zero
            self.matrix_comparison_values_left_right.zero_()
            self.matrix_comparison_values_left_neut.zero_()
            self.matrix_comparison_values_right_neut.zero_()
            self.matrix_comparison_bool_left_right.zero_()
            self.matrix_comparison_bool_left_neut.zero_()
            self.matrix_comparison_bool_right_neut.zero_()

            # Iterate through the dataframe in groups of 3 rows (each triplet)
            for i in range(0, len(df_comparisons), 3):
                group = df_comparisons.iloc[i:i + 3]

                # Skip incomplete groups
                if len(group) < 3:
                    continue

                # Get the efforts tuples from the first row (all rows of a triplet have the same efforts_tuples)
                efforts_tuples = group.iloc[0]['efforts_tuples']
                efforts_left = efforts_tuples[0]
                efforts_right = efforts_tuples[1]

                # Skip if neutral class or if left == right (should never happen according to verification checks)
                if efforts_left == (0, 0, 0, 0) or efforts_right == (0, 0, 0, 0) or efforts_left == efforts_right:
                    continue

                # Skip if either class is not in valid_indices (train/val split)
                if self.valid_indices is not None:
                    if efforts_left not in self.valid_indices or efforts_right not in self.valid_indices:
                        continue

                # Get the indices for the classes
                index_left = dict_label_to_id[efforts_left]
                index_right = dict_label_to_id[efforts_right]

                # Find the rows for each comparison type
                left_right_row = group[(group['selected0'] == 0) & (group['selected1'] == 2)]
                left_neut_row = group[(group['selected0'] == 0) & (group['selected1'] == 1)]
                right_neut_row = group[(group['selected0'] == 1) & (group['selected1'] == 2)]

                # Extract count_normalized values
                left_right_value = left_right_row['count_normalized'].iloc[0]
                left_neut_value = left_neut_row['count_normalized'].iloc[0]
                right_neut_value = right_neut_row['count_normalized'].iloc[0]

                # Populate the comparison values matrices
                self.matrix_comparison_values_left_right[index_left, index_right] = left_right_value
                self.matrix_comparison_values_left_neut[index_left, index_right] = left_neut_value
                self.matrix_comparison_values_right_neut[index_left, index_right] = right_neut_value

                # Also populate the comparison bool matrices (1 if value exists, 0 otherwise)
                self.matrix_comparison_bool_left_right[index_left, index_right] = 1.0 if left_right_value > 0.0 else 0.0
                self.matrix_comparison_bool_left_neut[index_left, index_right] = 1.0 if left_neut_value > 0.0 else 0.0
                self.matrix_comparison_bool_right_neut[index_left, index_right] = 1.0 if right_neut_value > 0.0 else 0.0



        # preprocess comparison data code execution starts here
        # read_in R generated similarity comparisons ratios csv
        aux_folder_path = (Path(__file__) / '../../aux').resolve()
        csv_similarity_ratios_path = aux_folder_path / f'{anim_name}_similarity_comparisons_ratios.csv'
        df_comparisons = pd.read_csv(csv_similarity_ratios_path)
        # print(f"Shape of df_comparisons: {df_comparisons.shape}")
        # Split the efforts_tuples values at the delimiter '_' and convert tokens to tuples
        df_comparisons['efforts_tuples'] = df_comparisons['efforts_tuples'].apply(
            lambda x: [tuple(ast.literal_eval(token)) for token in x.split('_')])
        set_comparison_classes = set([efforts_tuple[0] for efforts_tuple in df_comparisons['efforts_tuples']]).union(
            set([efforts_tuple[1] for efforts_tuple in df_comparisons['efforts_tuples']]))
        print(f"Loaded comparison data for: {anim_name} ...Num comparisons: {len(set_comparison_classes)}")

        dict_similarity_classes_exemplars = {key: value for key, value in
                                             self.dict_similarity_classes_exemplars.items() if
                                             key in
                                             set_comparison_classes}

        # Create a dictionary mapping the n similarity class labels to integers
        dict_label_to_id = {class_label: idx for idx, class_label in
                            enumerate(dict_similarity_classes_exemplars.keys())}
        dict_id_to_label = {idx: class_label for idx, class_label in
                            enumerate(dict_similarity_classes_exemplars.keys())}
        # print(f"reduced dict label to id len: {len(dict_label_to_id)}")
        # print(f"k,v of dict id to label: {dict_id_to_label.items()}")
        # verify_comparison_data()
        df_alphas = _generate_df_alphas(df_comparisons)
        _populate_alpha_matrices_and_masks(df_alphas)
        _populate_comparison_values_matrices(df_comparisons)
        # print(f"type of matrix_bool_left_right: {type(self.matrix_bool_left_right)}") -> class 'torch.Tensor
