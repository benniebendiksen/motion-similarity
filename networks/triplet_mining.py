"""
static module for organizing triplet mining data as well as performing online triplet mining
"""
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
import pandas as pd
import ast
import pickle


class TripletMining:
    def __init__(self, bool_drop, bool_fixed, squared_left_right, squared_class_neut, anim_name, config, valid_indices=None):
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

        self.bool_drop_neutral_exemplar = bool_drop
        self.bool_fixed_neutral_embedding = bool_fixed
        self.squared_left_right_euc_dist = squared_left_right
        self.squared_class_neut_dist = squared_class_neut

        # Store valid indices for filtering. A train/test split mechanism
        self.valid_indices = valid_indices

        # Initialize with proper batch_size for this module
        if self.valid_indices is not None:
            # For validation/training split
            self.batch_size = len(self.valid_indices)
        else:
            # Use the full batch size
            self.batch_size = self.config.similarity_per_anim_class_num

        self.initialize_triplet_mining(anim_name)

    def initialize_triplet_mining(self, anim_name):
        """
        Initialize the triplet mining module's state variables.

        This function loads necessary data, sets up state variables, and performs preprocessing.

        Args:
            anim_name: str: name of the animation (e.g., "walking", "pointing", "picking")

        Returns:
            None
        """

        print("Initializing Triplet Mining module state variables")

        # Load the full dictionary
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
            if not self.bool_drop_neutral_exemplar and key_to_remove not in self.valid_indices:
                if key_to_remove in original_dict:
                    self.dict_similarity_classes_exemplars[key_to_remove] = original_dict[key_to_remove]

        # Now check for the neutral key
        key_to_remove = (0, 0, 0, 0)
        if key_to_remove not in self.dict_similarity_classes_exemplars:
            assert False, f"triplet_mining.py: Key '{key_to_remove}' not found in dict_similarity_classes_exemplars"

        # Calculate num_states_drives based on the actual dictionary (which now only has valid_indices)
        if self.bool_drop_neutral_exemplar:
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
        self.neutral_embedding = torch.zeros(self.config.embedding_size, dtype=torch.float32, requires_grad=False)

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
        """
        Calculate both:
        1. 1D tensor of distances between class embeddings and the neutral embedding
        2. 2D matrix of distances between all class embeddings
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
            # This matches your observation about what works well for training
            if self.squared_class_neut_dist:
                epsilon = 1e-12
                self.tensor_dists_class_neut = torch.sqrt(class_neut_squared_dist + epsilon)
            else:
                self.tensor_dists_class_neut = class_neut_squared_dist

            return left_right_distances

        except Exception as e:
            print(f"Error in calculate_distances: {e}")
            raise e

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

            if self.squared_class_neut_dist:
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

        def _generate_df_alphas():
            """
            Processes user comparison data to generate alpha values and direct comparison metrics between similarity classes.

            This method analyzes triplets of comparisons (groups of 3 rows) from the input DataFrame 'df_comparisons',
            where each triplet contains all possible pairwise comparisons between three options (typically labeled as 0, 1, 2,
            representing left agent, neutral, and right agent, respectively). For each triplet, it:

            1. Identifies the most preferred pair (highest count_normalized value)
            2. Extracts the direct comparison value between options 0 and 2 when available
            3. Calculates two alpha values for the preferred pair:
               - alpha_positive_1_positive_2: How much more the first element is preferred when paired
                 with the second element compared to when it's paired with the third (negative) element
               - alpha_positive_2_positive_1: How much more the second element is preferred when paired
                 with the first element compared to when it's paired with the third (negative) element

            Alpha Value Calculation:
            For a preferred pair (A,B) with third option C:
            - alpha_A_B = count_normalized(A,B) - count_normalized(A,C)
            - alpha_B_A = count_normalized(A,B) - count_normalized(B,C)

            These alpha values quantify the strength of preference for each element in the pairing, relative to
            their preference when paired with the third element. Higher alpha values indicate a stronger preference
            effect when the two elements are paired together.

            The resulting DataFrame contains one row per triplet (keeping only the row with maximum count_normalized),
            and includes:
            - Original comparison data (efforts_tuples, selected motions, etc.)
            - Six possible alpha columns (alpha_0_1, alpha_1_0, alpha_0_2, alpha_2_0, alpha_1_2, alpha_2_1)
            - Direct comparison value between options 0 and 2 (direct_02_comparison)

            This DataFrame provides a comprehensive view of similarity relationships between effort tuples,
            enabling both direct and indirect comparison metrics for distance-based analysis.

            Returns:
                pandas.DataFrame: DataFrame containing processed comparison data with alpha values
                                  and direct comparison metrics

            """
            comparisons_list = []
            selection_values = [0, 1, 2]

            # Initialize new columns for pairwise comparison alpha values (two values created per pairwise comparison)
            df_comparisons['alpha_0_2'] = 0.0
            df_comparisons['alpha_2_0'] = 0.0
            df_comparisons['alpha_0_1'] = 0.0
            df_comparisons['alpha_2_1'] = 0.0
            df_comparisons['alpha_1_0'] = 0.0
            df_comparisons['alpha_1_2'] = 0.0

            # Add new column for the 0-2 count_normalized (where selected0=0, selected1=2)
            df_comparisons['direct_comparison_value'] = np.nan

            self.df_comparisons = df_comparisons

            # Iterate over three consecutive rows
            # selected_0 is either 0 (agent left) or 1 (neutral) and selected_1 is either 1 or 2 (agent right) (else we terminate)
            for i in range(0, len(df_comparisons), 3):
                group = df_comparisons.iloc[i:i + 3]

                # Find the row within a triplet with the maximum 'count_normalized' value, thereby establishing the
                # positive pair (i.e., selected0 and selected1 which could be (0,1), (0,2) or (1,2)
                max_row = group.loc[group['count_normalized'].idxmax()]

                # Extract the direct comparison value (selected0=0, selected1=2) if it exists
                direct_comparison_row = group[(group['selected0'] == 0) & (group['selected1'] == 2)]
                direct_comparison_value = None
                if not direct_comparison_row.empty:
                    direct_comparison_value = direct_comparison_row.iloc[0]['count_normalized']
                    # Store this value in the max_row
                    df_comparisons.loc[max_row.name, 'direct_comparison_value'] = direct_comparison_value

                max_selected_0, max_selected_1 = max_row['selected0'], max_row['selected1']
                negative_index = next(x for x in selection_values if x != max_selected_0 and x != max_selected_1)

                # find the anchor_positive ratio value under the cases in which anchor is each of the positive pair,
                # respectively, and positive is the negative class
                if max_selected_0 == 0:
                    # means either 0-2 ratio if max_selected_1 is 1 (i.e., 2 is negative), else 0-1 ratio
                    ratio_positive_1_negative = \
                        group.loc[(group['selected0'] == max_selected_0) & (group['selected1'] ==
                                                                            negative_index)].iloc[0][
                            'count_normalized']
                    # max_selected_1 being 1 means 2 is the negative index: 1-2 ratio
                    if max_selected_1 == 1:
                        ratio_positive_2_negative = \
                            group.loc[(group['selected0'] == max_selected_1) & (group['selected1'] ==
                                                                                negative_index)].iloc[
                                0][
                                'count_normalized']
                    elif max_selected_1 == 2:
                        ratio_positive_2_negative = \
                            group.loc[(group['selected0'] == negative_index) & (group['selected1'] ==
                                                                                max_selected_1)].iloc[
                                0][
                                'count_normalized']
                    else:
                        assert False, "selected1 is not 1 or 2"

                elif max_selected_0 == 1:
                    if max_selected_1 == 2:
                        ratio_positive_1_negative = \
                            group.loc[(group['selected0'] == negative_index) & (group['selected1'] ==
                                                                                max_selected_0)].iloc[
                                0][
                                'count_normalized']
                        ratio_positive_2_negative = \
                            group.loc[(group['selected0'] == negative_index) & (group['selected1'] ==
                                                                                max_selected_1)].iloc[
                                0][
                                'count_normalized']
                    else:
                        assert False, "selected1 is not 1 or 2"
                else:
                    assert False, "selected0 is not 0 or 1"

                # Alternatively to generating the two possible alpha values for a comparison (e.g., treating selected0 as anchor versus
                # treating selected1 as anchor), we can extract only the dominant alpha value for each comparison (i.e., max difference).
                diff_positive_1_anchor = max_row['count_normalized'] - ratio_positive_1_negative
                diff_positive_2_anchor = max_row['count_normalized'] - ratio_positive_2_negative
                if ratio_positive_1_negative < ratio_positive_2_negative:
                    alpha_positive_1_positive_2 = diff_positive_1_anchor
                    alpha_positive_2_positive_1 = diff_positive_2_anchor
                else:
                    alpha_positive_1_positive_2 = diff_positive_1_anchor
                    alpha_positive_2_positive_1 = diff_positive_2_anchor

                if max_row['efforts_tuples'] == '[-1,-1,-1,0]_[0,-1,-1,1]':
                    print(f"ALPHAS: {alpha_positive_1_positive_2} . {alpha_positive_2_positive_1}")

                # Concatenate selected0 and selected1 to pattern match the alpha anchor_positive column
                alpha_selected_0_selected_1_column = f"alpha_{max_selected_0}_{max_selected_1}"
                alpha_selected_1_selected_0_column = f"alpha_{max_selected_1}_{max_selected_0}"

                df_comparisons.loc[max_row.name, alpha_selected_0_selected_1_column] = alpha_positive_1_positive_2
                df_comparisons.loc[max_row.name, alpha_selected_1_selected_0_column] = alpha_positive_2_positive_1
                comparisons_list.append(df_comparisons.loc[max_row.name])

            alpha_dataframes = pd.DataFrame(comparisons_list)
            alpha_dataframes.reset_index(drop=True, inplace=True)

            # Filter out rows where direct_comparison_value is NaN
            # if 'direct_comparison_value' in alpha_dataframes.columns:
            #     # Remove NaN values - only keep rows with a valid direct comparison
            #     alpha_dataframes_filtered = alpha_dataframes.dropna(subset=['direct_comparison_value'])
            #     # If you need to keep all rows but want to indicate which ones have valid direct comparisons:
            #     # alpha_dataframes['has_direct_comparison'] = ~alpha_dataframes['direct_comparison_value'].isna()
            #
            #     # Optionally, you can also rename the column to something more descriptive
            #     alpha_dataframes.rename(columns={'direct_comparison_value': 'direct_02_comparison'}, inplace=True)

            return alpha_dataframes

        def _populate_comparison_values_matrices(df_comparisons):
            # We must grab count_normalized values for the left-right, left-neut, and right-neut comparisons
            pass

        def _populate_alpha_matrices_and_masks(df_alphas):
            """
            Populate the alpha matrices and their corresponding masks based on the data in the comparisons DataFrame.
            Masks indicate if positive pairs (and therefore the two corresponding triplets / alphas) are to be used for loss calculation.
            Valid indices are used to set zeros in the alpha matrices and masks for classes that are not part of the training or validation set.

            PyTorch version of the original TensorFlow implementation.

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

                # check that effort tuples correspond to values of self.valid_indices
                # print(f"efforts_tuple: {efforts_tuple[0]}, {efforts_tuple[1]}")
                # print(f"valid_indices: {self.valid_indices}")

                # Skip if either class is not in valid_indices. A train/val split ensuring mechanism
                if self.valid_indices is not None:
                    if efforts_tuple[0] not in self.valid_indices or efforts_tuple[1] not in self.valid_indices:
                        continue

                # enforce constraint that i < j always corresponds to left, right / i > j to right, left effort_tuples.
                # where labels are efforts_tuple values (efforts_tuple[0] < efforts_tuple[1] based on R's
                # pmin, pmax functions) and i and j are indices to dict_similarity_classes_exemplars.keys()
                if dict_label_to_id[efforts_tuple[0]] > dict_label_to_id[efforts_tuple[1]]:
                    row['efforts_tuples'] = [efforts_tuple[1], efforts_tuple[0]]
                    bool_swap_left_right = True

                efforts_left = row['efforts_tuples'][0]
                efforts_right = row['efforts_tuples'][1]
                index_left = dict_label_to_id[efforts_left]
                index_right = dict_label_to_id[efforts_right]

                ### temporary fix for erroneous similarity class, and for self to self comparison, in comparisons
                ### DataFrame
                if efforts_left == (0, 0, 0, 0) or efforts_left == efforts_right:
                    repeat_class_comparison_counter += 1
                    print(f"repeat class comparison at indices: {index_left} , {index_right}")
                    continue

                # alpha_tripletid1_tripletid2 denotes one of the two alpha values per triplet as a function of the two most similar cases.
                # For any comparison, left_index < right_index
                # left, right indices indicate location for left, neut anchor_positive alpha with respect to Left,
                # Neutral Matrix (and right, neut anchor_positive alpha with respect to Right, Neutral Matrix)
                # whereas right, left indices indicate location for neut, left anchor_positive alpha and neut,
                # right anchor_positive alpha, respectively.
                if row['alpha_0_2'] != 0:
                    # print(f"entered alpha_0_2 with alphas: {row['alpha_0_2']} and {row['alpha_2_0']}")
                    # extract the two alpha values for the comparison, abiding by constraint
                    left_right_alpha = row['alpha_0_2']
                    right_left_alpha = row['alpha_2_0']

                    if bool_swap_left_right:
                        left_right_alpha = row['alpha_2_0']
                        right_left_alpha = row['alpha_0_2']

                    if left_right_alpha == 0 and right_left_alpha == 0:
                        zero_alphas_counter += 1
                        # print(f"left_right_alpha, right_left_alpha, both alphas zero...counter: {zero_alphas_counter}")
                        bool_constant_left_right = 0
                        bool_constant_right_left = 0
                    elif left_right_alpha == 0:
                        # print(f"left_right_alpha: {right_left_alpha}")
                        bool_constant_left_right = 0
                        bool_constant_right_left = 1
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")
                    elif right_left_alpha == 0:
                        # print(f"right_left_alpha: {right_left_alpha}")
                        bool_constant_left_right = 1
                        bool_constant_right_left = 0
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")
                    else:
                        # assert False, "left_right_alpha and right_left_alpha are both non-zero"
                        bool_constant_left_right = 1
                        bool_constant_right_left = 1
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")

                    # PyTorch direct tensor indexing for in-place updating
                    self.matrix_alpha_left_right_right_left[index_left, index_right] += left_right_alpha
                    self.matrix_bool_left_right[index_left, index_right] = bool_constant_left_right
                    self.matrix_alpha_left_right_right_left[index_right, index_left] += right_left_alpha
                    self.matrix_bool_right_left[index_right, index_left] = bool_constant_right_left

                elif row['alpha_0_1'] != 0:
                    # print(f"entered alpha_0_1 with alphas: {row['alpha_0_1']} and {row['alpha_1_0']}")
                    left_neutral_alpha = row['alpha_0_1']
                    neutral_left_alpha = row['alpha_1_0']

                    if bool_swap_left_right:
                        # left_neutral_alpha = row['alpha_2_1']
                        # neutral_left_alpha = row['alpha_1_2']
                        left_neutral_alpha = row['alpha_1_0']
                        neutral_left_alpha = row['alpha_0_1']

                    if left_neutral_alpha == 0 and neutral_left_alpha == 0:
                        zero_alphas_counter += 1
                        # print(f"left_neut, neut_left, both alphas zero...counter: {zero_alphas_counter}")
                        bool_constant_left_neutral = 0
                        bool_constant_neutral_left = 0
                    elif left_neutral_alpha == 0:
                        # print(f"left_neutral_alpha: {neutral_left_alpha}")
                        bool_constant_left_neutral = 0
                        bool_constant_neutral_left = 1
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")
                    elif neutral_left_alpha == 0:
                        # print(f"neutral_left_alpha: {neutral_left_alpha}")
                        bool_constant_left_neutral = 1
                        bool_constant_neutral_left = 0
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")
                    else:
                        # assert False, "left_neutral_alpha and neutral_left_alpha are both non-zero"
                        bool_constant_left_neutral = 1
                        bool_constant_neutral_left = 1
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")

                    # PyTorch direct tensor indexing for in-place updating
                    self.matrix_alpha_left_neut_neut_left[index_left, index_right] += left_neutral_alpha
                    self.matrix_bool_left_neut[index_left, index_right] = bool_constant_left_neutral
                    self.matrix_alpha_left_neut_neut_left[index_right, index_left] += neutral_left_alpha
                    self.matrix_bool_neut_left[index_right, index_left] = bool_constant_neutral_left

                elif row['alpha_2_1'] != 0:
                    # print(f"entered alpha_2_1 with alphas: {row['alpha_2_1']} and {row['alpha_1_2']}")
                    right_neutral_alpha = row['alpha_2_1']
                    neutral_right_alpha = row['alpha_1_2']

                    if bool_swap_left_right:
                        # right_neutral_alpha = row['alpha_0_1']
                        # neutral_right_alpha = row['alpha_1_0']
                        right_neutral_alpha = row['alpha_1_2']
                        neutral_right_alpha = row['alpha_2_1']

                    if right_neutral_alpha == 0 and neutral_right_alpha == 0:
                        zero_alphas_counter += 1
                        # print(f"right_neut, neut_right, both alphas zero...counter: {zero_alphas_counter}")
                        bool_constant_right_neutral = 0
                        bool_constant_neutral_right = 0
                    elif right_neutral_alpha == 0:
                        # print(f"right_neutral_alpha: {neutral_right_alpha}")
                        bool_constant_right_neutral = 0
                        bool_constant_neutral_right = 1
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")
                    elif neutral_right_alpha == 0:
                        # print(f"neutral_right_alpha: {neutral_right_alpha}")
                        bool_constant_right_neutral = 1
                        bool_constant_neutral_right = 0
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")
                    else:
                        # assert False, "right_neutral_alpha and neutral_right_alpha are both non-zero"
                        bool_constant_right_neutral = 1
                        bool_constant_neutral_right = 1
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")

                    # PyTorch direct tensor indexing for in-place updating
                    self.matrix_alpha_right_neut_neut_right[index_left, index_right] += right_neutral_alpha
                    self.matrix_bool_right_neut[index_left, index_right] = bool_constant_right_neutral
                    self.matrix_alpha_right_neut_neut_right[index_right, index_left] += neutral_right_alpha
                    self.matrix_bool_neut_right[index_right, index_left] = bool_constant_neutral_right

                # implies equal selection (or no selection) across all three pairs of a triplet
                else:
                    equal_comparison_counter += 1

                bool_swap_left_right = False

            print(f" Equal comparison counter: {equal_comparison_counter}")
            print(f"Unequal comparison counter: {unequal_comparison_counter}")



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
        df_alphas = _generate_df_alphas()
        _populate_alpha_matrices_and_masks(df_alphas)
        _populate_comparison_values_matrices(df_comparisons)
        # print(f"type of matrix_bool_left_right: {type(self.matrix_bool_left_right)}") -> class 'torch.Tensor
