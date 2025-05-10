from Config import BATCH_STRATEGY
from Config import BatchStrategy
import tensorflow as tf
import torch


def calculate_contrastive_loss(y_true, y_pred, triplet_mining, batch_strategy=BATCH_STRATEGY):
    """
    Calculate contrastive loss using the comparison matrices populated with count_normalized values.

    This loss encourages embeddings to be positioned according to their similarity values from user studies,
    using the count_normalized values as target similarities.

    Args:
        y_true: Labels of the batch (not used directly but kept for API consistency)
        y_pred: Embeddings, tensor of shape (batch_size, embed_dim)
        triplet_mining: TripletMining object containing comparison matrices
        batch_strategy: Batch strategy to use (affects which comparisons are considered)

    Returns:
        losses: Contrastive loss tensor
    """
    # Calculate the pairwise distances between embeddings
    # This also updates tensor_dists_class_neut internally
    classes_distances = triplet_mining.calculate_distances(y_pred)

    # Initialize loss tensors
    left_right_loss = torch.zeros_like(classes_distances)
    left_neut_loss = torch.zeros_like(classes_distances)
    right_neut_loss = torch.zeros_like(classes_distances)

    # Calculate contrastive loss for left-right comparisons
    # Loss = (distance - target_similarity)²
    # Where target_similarity is derived from count_normalized values

    # Only consider pairs with valid comparisons (bool_matrix == 1)
    # valid_left_right = triplet_mining.matrix_comparison_bool_left_right > 0
    # if torch.any(valid_left_right):
    #     # Scale distances to be in a similar range as the count_normalized values [0,1]
    #     # This scaling factor might need tuning based on your specific data
    #     scaling_factor = 0.1
    #     scaled_distances = classes_distances * scaling_factor
    #
    #     # Compute the squared difference between scaled distances and target similarities
    #     target_similarities = triplet_mining.matrix_comparison_values_left_right
    #     diff_left_right = scaled_distances - target_similarities
    #     left_right_loss = torch.square(diff_left_right) * valid_left_right

    # Only consider pairs with valid comparisons (bool_matrix == 1)
    valid_left_right = triplet_mining.matrix_comparison_bool_left_right > 0
    if torch.any(valid_left_right) or not torch.any(valid_left_right):
        # Scale distances to be in a similar range as the count_normalized values [0,1]
        # This scaling factor might need tuning based on your specific data
        scaling_factor = 0.1
        scaled_distances = classes_distances

        # Compute the loss when scaled distances exceed target similarities
        target_similarities = triplet_mining.matrix_comparison_values_left_right

        # Apply ReLU to only penalize when scaled_distances > target_similarities
        # diff_left_right = torch.relu(scaled_distances - target_similarities)
        diff_left_right = scaled_distances - target_similarities

        # Apply this only to valid pairs
        # left_right_loss = torch.square(diff_left_right) * valid_left_right
        left_right_loss = torch.square(diff_left_right)

    # Calculate contrastive loss for left-neutral comparisons
    valid_left_neut = triplet_mining.matrix_comparison_bool_left_neut > 0
    if torch.any(valid_left_neut) or not torch.any(valid_left_neut):
        # Use tensor_dists_class_neut for left-neutral distances
        # Need to reshape for proper broadcasting
        left_neut_distances = triplet_mining.tensor_dists_class_neut.reshape(-1, 1)
        print(f"shape of left_neut_distances: {left_neut_distances.shape}")

        # Scale distances
        scaled_distances = left_neut_distances * scaling_factor

        # Compute squared difference
        target_similarities = triplet_mining.matrix_comparison_values_left_neut
        diff_left_neut = scaled_distances - target_similarities
        left_neut_loss = torch.square(diff_left_neut) * valid_left_neut

    # Calculate contrastive loss for right-neutral comparisons
    valid_right_neut = triplet_mining.matrix_comparison_bool_right_neut > 0
    if torch.any(valid_right_neut) or not torch.any(valid_right_neut):
        # Use tensor_dists_class_neut for right-neutral distances
        # Need to reshape for proper broadcasting
        right_neut_distances = triplet_mining.tensor_dists_class_neut.reshape(1, -1)

        # Scale distances
        scaled_distances = right_neut_distances * scaling_factor

        # Compute squared difference
        target_similarities = triplet_mining.matrix_comparison_values_right_neut
        diff_right_neut = scaled_distances - target_similarities
        right_neut_loss = torch.square(diff_right_neut) * valid_right_neut

    # Combine all loss components
    # total_loss = left_right_loss + left_neut_loss + right_neut_loss

    total_loss = left_right_loss

    # Handle different batch strategies
    if batch_strategy == BatchStrategy.HARD:
        # Only use the hardest (largest) losses
        max_loss_per_row = torch.max(total_loss, dim=1)[0]
        max_loss_per_col = torch.max(total_loss, dim=0)[0]
        losses = torch.max(torch.cat([max_loss_per_row, max_loss_per_col]))

    elif batch_strategy == BatchStrategy.SEMI_HARD:
        # Use semi-hard examples (losses that are positive but not too large)
        # First, get all positive losses
        positive_losses = torch.where(total_loss > 0, total_loss, torch.zeros_like(total_loss))

        # Get median of positive losses
        median_loss = torch.median(torch.where(positive_losses > 0, positive_losses,
                                               torch.ones_like(positive_losses) * float('inf')))

        # Use losses that are close to the median
        semi_hard_margin = 0.2  # This is a hyperparameter to tune
        semi_hard_mask = torch.abs(positive_losses - median_loss) < (median_loss * semi_hard_margin)

        # Apply mask and get average
        semi_hard_losses = positive_losses * semi_hard_mask
        losses = torch.sum(semi_hard_losses) / (torch.sum(semi_hard_mask) + 1e-8)  # avoid division by zero

    else:  # BatchStrategy.ALL
        # Use all valid losses
        # num_valid = torch.sum(
        #     triplet_mining.matrix_comparison_bool_left_right +
        #     triplet_mining.matrix_comparison_bool_left_neut +
        #     triplet_mining.matrix_comparison_bool_right_neut
        # )
        # losses = torch.sum(total_loss) / (num_valid + 1e-8)  # avoid division by zero
        losses = torch.mean(total_loss)

    return losses
# consider alternative loss function based on cosine similarity:
# self.loss = tf.keras.losses.CosineSimilarity(axis=1)
# ap_distance = self.loss(anchor, positive)
# an_distance = self.loss(anchor, negative)
# loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
def calculate_triplet_loss(y_true, y_pred, triplet_mining, batch_strategy=BATCH_STRATEGY):
    """Calculate 1D tensor of either squared L2 norm or L2 norm of differences between class embeddings and the
            neutral embedding of shape (embedding_size,), resulting in tensor of shape (batch_size,).

                    Args:
                        y_true: supposed 'labels' of the batch (i.e., class indexes), tensor of size (batch_size,
                            ) where each element is singleton tensor of an integer in the range [1, 57]. Only needed
                            to the end of sorting the batch embeddings (i.e., y_pred)
                        y_true contains 1-57 repeated three times.
                        y_pred: embeddings, tensor of shape (batch_size, embed_dim)
                        triplet_mining: TripletMining object, contains the necessary alpha matrices information
                            for a given action type
                        batch_strategy: Enum value indicating the batch strategy to use for triplet mining

                    Returns:
                        losses: tensor of shape (triplet_mining.num_states_drives, triplet_mining.num_states_drives)
            """

    classes_distances = triplet_mining.calculate_distances(y_pred)
    # triplet_mining.calculate_class_neut_distances(y_pred)

    # Reshape the 1D tensor into a 2D tensor with shape (1, 56) (will allow for broadcasting)
    # row_dists_class_neut = tf.reshape(triplet_mining.tensor_dists_class_neut,
    #                                   (1, triplet_mining.num_states_drives))
    # Reshape the 1D tensor dists_class_neut into a 2D tensor with shape (1, 56) for broadcasting
    row_dists_class_neut = triplet_mining.tensor_dists_class_neut.reshape(1, triplet_mining.num_states_drives)

    # print(f"Custom_losses:calculate_triplet_loss: row_dists_class_neut shape: {row_dists_class_neut.shape}")
    # Reshape the 1D tensor into a 2D tensor of shape (56, 1) (will allow for broadcasting)
    # column_dists_class_neut = tf.reshape(triplet_mining.tensor_dists_class_neut,
    #                                      (triplet_mining.num_states_drives, 1))
    column_dists_class_neut = triplet_mining.tensor_dists_class_neut.reshape(triplet_mining.num_states_drives, 1)

    # case 1: left and right are positives
    # subcase L = anchor
    diff_lr_ln = classes_distances - column_dists_class_neut
    # print(f"diff_lr_ln: {diff_lr_ln}")

    assert diff_lr_ln.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
        f"Shape mismatch: expected ({triplet_mining.num_states_drives}, {triplet_mining.num_states_drives}), but got {diff_lr_ln.shape}"


    # consider only cases where diff_lr_ln > 0 (i.e., l_n - l_r > 0)
    if batch_strategy == BatchStrategy.HARD:
        # Ensure no negative distances
        diff_lr_ln = torch.clamp(diff_lr_ln, min=0.0)
        diff_lr_rl_alpha = diff_lr_ln + triplet_mining.matrix_alpha_left_right_right_left
        diff_lr_alpha = diff_lr_rl_alpha * triplet_mining.matrix_bool_left_right  # Element-wise multiplication

        # Ensure no negative losses
        triplet_loss_L_R = diff_lr_alpha

        # Assert no negative losses
        assert torch.all(triplet_loss_L_R >= 0), "Negative losses exist"

    # consider only cases where diff_lr_ln < 0 yet triplet_loss_L_N > 0 (i.e., l_r - l_n < 0 and l_r - l_n + alpha > 0)
    elif batch_strategy == BatchStrategy.SEMI_HARD:
        # Replace tf.where with torch.where
        diff_lr_ln = torch.where(diff_lr_ln < 0, diff_lr_ln, torch.zeros_like(diff_lr_ln))

        # Addition is the same in PyTorch
        diff_lr_rl_alpha = diff_lr_ln + triplet_mining.matrix_alpha_left_right_right_left

        # Replace tf.multiply with element-wise multiplication
        diff_lr_alpha = diff_lr_rl_alpha * triplet_mining.matrix_bool_left_right

        # Replace tf.maximum with torch.clamp
        triplet_loss_L_R = torch.clamp(diff_lr_alpha, min=0.0)

    # ALL strategy for diff_lr_ln case
    else:
        # Addition is the same in PyTorch
        diff_lr_rl_alpha = diff_lr_ln + triplet_mining.matrix_alpha_left_right_right_left

        # Replace tf.multiply with element-wise multiplication
        diff_lr_alpha = diff_lr_rl_alpha * triplet_mining.matrix_bool_left_right

        # print(f"diff_lr_alpha: {diff_lr_alpha}")

        # Replace tf.maximum with torch.clamp
        triplet_loss_L_R = torch.clamp(diff_lr_alpha, min=0.0)

    # subcase R = anchor
    diff_rl_rn = classes_distances - row_dists_class_neut
    if batch_strategy == BatchStrategy.HARD:
        diff_rl_rn = torch.clamp(diff_rl_rn, min=0.0)

        diff_rl_rn_alpha = diff_rl_rn + triplet_mining.matrix_alpha_left_right_right_left

        diff_rl_alpha = diff_rl_rn_alpha * triplet_mining.matrix_bool_right_left

        triplet_loss_R_L = diff_rl_alpha

        # Replace TensorFlow assertion with PyTorch assertion
        # torch.all performs element-wise comparison and returns True if all elements are True
        assert torch.all(triplet_loss_R_L >= 0.0), "Negative losses exist"

    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_rl_rn = torch.where(diff_rl_rn < 0, diff_rl_rn, torch.zeros_like(diff_rl_rn))

        diff_rl_rn_alpha = diff_rl_rn + triplet_mining.matrix_alpha_left_right_right_left

        diff_rl_alpha = diff_rl_rn_alpha * triplet_mining.matrix_bool_right_left

        triplet_loss_R_L = torch.clamp(diff_rl_alpha, min=0.0)

    else:
        diff_rl_rn_alpha = diff_rl_rn + triplet_mining.matrix_alpha_left_right_right_left

        diff_rl_alpha = diff_rl_rn_alpha * triplet_mining.matrix_bool_right_left

        triplet_loss_R_L = torch.clamp(diff_rl_alpha, min=0.0)

    # case 2: left and neutral are positives
    # L = anchor, Neutral = positive, R = negative
    # diff_ln_lr = column_dists_class_neut - classes_distances
    # # Assert that the shape of diff_ln_lr is correct
    # assert diff_ln_lr.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
    #     f"Tensor diff_ln_lr has incorrect shape: {diff_ln_lr.shape}"
    #
    # if batch_strategy == BatchStrategy.HARD:
    #     diff_ln_lr = torch.where(diff_ln_lr > 0, diff_ln_lr, torch.zeros_like(diff_ln_lr))
    #
    #     diff_ln_lr_alpha = diff_ln_lr + triplet_mining.matrix_alpha_left_neut_neut_left
    #
    #     diff_ln_lr_alpha = diff_ln_lr_alpha * triplet_mining.matrix_bool_left_neut
    #
    #     triplet_loss_L_N = diff_ln_lr_alpha
    #
    #     # Assert that all values in triplet_loss_L_N are non-negative
    #     assert torch.all(triplet_loss_L_N >= 0.0), "Negative losses exist"
    #
    # # L = anchor, Neutral = positive, R = negative
    # elif batch_strategy == BatchStrategy.SEMI_HARD:
    #     diff_ln_lr = torch.where(diff_ln_lr < 0, diff_ln_lr, torch.zeros_like(diff_ln_lr))
    #
    #     diff_ln_lr_alpha = diff_ln_lr + triplet_mining.matrix_alpha_left_neut_neut_left
    #
    #     diff_ln_lr_alpha = diff_ln_lr_alpha * triplet_mining.matrix_bool_left_neut
    #
    #     triplet_loss_L_N = torch.where(diff_ln_lr_alpha > 0, diff_ln_lr_alpha, torch.zeros_like(diff_ln_lr_alpha))
    #
    # # L = anchor, Neutral = positive, R = negative
    # else:
    #     diff_ln_lr_alpha = diff_ln_lr + triplet_mining.matrix_alpha_right_neut_neut_right
    #
    #     diff_ln_lr_alpha = diff_ln_lr_alpha * triplet_mining.matrix_bool_left_neut
    #
    #     triplet_loss_L_N = torch.where(diff_ln_lr_alpha > 0, diff_ln_lr_alpha, torch.zeros_like(diff_ln_lr_alpha))
    #
    # diff_nl_nr = row_dists_class_neut - column_dists_class_neut
    # assert diff_nl_nr.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
    #     f"Tensor diff_nl_nr has incorrect shape: {diff_nl_nr.shape}"
    # # N = anchor, L = positive, R = negative
    # if batch_strategy == BatchStrategy.HARD:
    #     diff_nl_nr = torch.where(diff_nl_nr > 0, diff_nl_nr, torch.zeros_like(diff_nl_nr))
    #
    #     diff_nl_nr_alpha = diff_nl_nr + triplet_mining.matrix_alpha_left_neut_neut_left
    #
    #     diff_nl_nr_alpha = diff_nl_nr_alpha * triplet_mining.matrix_bool_neut_left
    #
    #     triplet_loss_N_L = diff_nl_nr_alpha
    #
    #     # Assert that all values in triplet_loss_N_L are non-negative
    #     assert torch.all(triplet_loss_N_L >= 0.0), "Negative losses exist"
    #
    # # N = anchor, L = positive, R = negative
    # elif batch_strategy == BatchStrategy.SEMI_HARD:
    #     diff_nl_nr = torch.where(diff_nl_nr < 0, diff_nl_nr, torch.zeros_like(diff_nl_nr))
    #
    #     diff_nl_nr_alpha = diff_nl_nr + triplet_mining.matrix_alpha_left_neut_neut_left
    #
    #     diff_nl_nr_alpha = diff_nl_nr_alpha * triplet_mining.matrix_bool_neut_left
    #
    #     triplet_loss_N_L = torch.where(diff_nl_nr_alpha > 0, diff_nl_nr_alpha, torch.zeros_like(diff_nl_nr_alpha))
    #
    # # N = anchor, L = positive, R = negative
    # else:
    #     diff_nl_nr_alpha = diff_nl_nr + triplet_mining.matrix_alpha_left_neut_neut_left
    #
    #     diff_nl_nr_alpha = diff_nl_nr_alpha * triplet_mining.matrix_bool_neut_left
    #
    #     triplet_loss_N_L = torch.where(diff_nl_nr_alpha > 0, diff_nl_nr_alpha, torch.zeros_like(diff_nl_nr_alpha))
    #
    # ### case 3: right and neutral are positives
    # # R = anchor
    # diff_rn_rl = row_dists_class_neut - torch.transpose(classes_distances, 0, 1)
    # if batch_strategy == BatchStrategy.HARD:
    #     # # remove negative losses
    #     diff_rn_rl = torch.where(diff_rn_rl > 0, diff_rn_rl, torch.zeros_like(diff_rn_rl))
    #
    #     # Addition works the same way in PyTorch
    #     diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining.matrix_alpha_right_neut_neut_right
    #
    #     # Replace tf.multiply with element-wise multiplication
    #     diff_r_n_r_l_alpha = diff_r_n_r_l_alpha * triplet_mining.matrix_bool_right_neut
    #
    #     # Assign to triplet_loss_R_N
    #     triplet_loss_R_N = diff_r_n_r_l_alpha
    #
    #     # Assert that all values in triplet_loss_R_N are non-negative
    #     assert torch.all(triplet_loss_R_N >= 0.0), "Negative losses exist"
    #
    # elif batch_strategy == BatchStrategy.SEMI_HARD:
    #     diff_rn_rl = torch.where(diff_rn_rl < 0, diff_rn_rl, torch.zeros_like(diff_rn_rl))
    #
    #     diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining.matrix_alpha_right_neut_neut_right
    #
    #     diff_r_n_r_l_alpha = diff_r_n_r_l_alpha * triplet_mining.matrix_bool_right_neut
    #
    #     triplet_loss_R_N = torch.where(diff_r_n_r_l_alpha > 0, diff_r_n_r_l_alpha, torch.zeros_like(diff_r_n_r_l_alpha))
    #
    # else:  # BatchStrategy.ALL
    #     diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining.matrix_alpha_right_neut_neut_right
    #
    #     diff_r_n_r_l_alpha = diff_r_n_r_l_alpha * triplet_mining.matrix_bool_right_neut
    #
    #     triplet_loss_R_N = torch.where(diff_r_n_r_l_alpha > 0, diff_r_n_r_l_alpha, torch.zeros_like(diff_r_n_r_l_alpha))
    #
    # # tf.debugging.assert_equal(triplet_loss_R_N, tf.maximum(triplet_loss_R_N, 0.0), message="Negative losses exist")
    # # Assert that all values in triplet_loss_R_N are non-negative
    # assert torch.all(triplet_loss_R_N >= 0.0), "Negative losses exist"
    #
    # # N = anchor
    # diff_nr_nl = column_dists_class_neut - row_dists_class_neut
    # assert diff_nr_nl.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
    #     f"Tensor diff_nr_nl has incorrect shape: {diff_nr_nl.shape}"
    # if batch_strategy == BatchStrategy.HARD:
    #     diff_nr_nl = torch.where(diff_nr_nl > 0, diff_nr_nl, torch.zeros_like(diff_nr_nl))
    #
    #     diff_nr_nl_alpha = diff_nr_nl + triplet_mining.matrix_alpha_right_neut_neut_right
    #
    #     diff_nr_nl_alpha = diff_nr_nl_alpha * triplet_mining.matrix_bool_neut_right
    #
    #     triplet_loss_N_R = diff_nr_nl_alpha
    #
    #     # Assert that all values in triplet_loss_N_R are non-negative
    #     assert torch.all(triplet_loss_N_R >= 0.0), "Negative losses exist"
    #
    # elif batch_strategy == BatchStrategy.SEMI_HARD:
    #     diff_nr_nl_alpha = diff_nr_nl + triplet_mining.matrix_alpha_right_neut_neut_right
    #
    #     diff_nr_nl_alpha = diff_nr_nl_alpha * triplet_mining.matrix_bool_neut_right
    #
    #     triplet_loss_N_R = torch.where(diff_nr_nl_alpha > 0, diff_nr_nl_alpha, torch.zeros_like(diff_nr_nl_alpha))
    #
    # else:
    #     diff_nr_nl_alpha = diff_nr_nl + triplet_mining.matrix_alpha_right_neut_neut_right
    #
    #     diff_nr_nl_alpha = diff_nr_nl_alpha * triplet_mining.matrix_bool_neut_right
    #
    #     triplet_loss_N_R = torch.where(diff_nr_nl_alpha > 0, diff_nr_nl_alpha, torch.zeros_like(diff_nr_nl_alpha))


    # losses = (triplet_loss_L_R + triplet_loss_R_L + triplet_loss_L_N + triplet_loss_N_L + triplet_loss_R_N +
    #           triplet_loss_N_R)
    losses = (triplet_loss_L_R + triplet_loss_R_L)
    # print(f"losses: {losses}")

    # Assert that no negative losses exist
    assert torch.all(losses >= 0.0), "Negative losses exist"

    # Assert that the shape of losses tensor is correct
    assert losses.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
        f"Comparisons loss tensor has incorrect shape: {losses.shape}"
    return losses


def create_batch_triplet_loss(triplet_mining_modules, module_start_indices=None, module_sizes=None):
    """
    Create a batch triplet loss function for use with multiple triplet mining modules.

    Args:
        triplet_mining_modules: List of TripletMining objects
        module_start_indices: List of starting indices for each module in the batch
        module_sizes: List of sizes for each module
    """

    def batch_triplet_loss(y_true, y_pred):
        # ... existing flattening and sorting ...

        # Calculate overall triplet loss across all modules
        overall_triplet_loss = torch.tensor(0.0, device=y_pred.device)
        valid_modules = 0

        for i, triplet_mining in enumerate(triplet_mining_modules):
            # Determine batch slice for this module
            if module_start_indices is not None and module_sizes is not None:
                start_idx = module_start_indices[i]
                end_idx = start_idx + module_sizes[i]
            else:
                # Fallback to original approach
                start_idx = i * triplet_mining.batch_size
                end_idx = (i + 1) * triplet_mining.batch_size

            # Ensure indices are within bounds
            end_idx = min(end_idx, y_true.shape[0])

            # Skip if we don't have enough data
            if end_idx - start_idx <= 1:
                continue

            # Extract the portion of the batch for this module
            y_true_module = y_true[start_idx:end_idx]
            y_pred_module = y_pred[start_idx:end_idx]

            # Calculate triplet losses for this module
            try:
                triplet_losses = calculate_triplet_loss(y_true_module, y_pred_module, triplet_mining)
                triplet_loss = torch.sum(triplet_losses)
                overall_triplet_loss += triplet_loss
                valid_modules += 1
            except Exception as e:
                print(f"Error in triplet loss calculation for module {i}: {e}")
                continue

        # Normalize by number of valid modules
        if valid_modules > 0:
            overall_triplet_loss = overall_triplet_loss / valid_modules
        else:
            return overall_triplet_loss + 1e-8

        return overall_triplet_loss

    return batch_triplet_loss
