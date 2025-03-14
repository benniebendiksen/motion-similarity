from conf import BATCH_STRATEGY
from conf import BatchStrategy
import tensorflow as tf
import torch


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

    classes_distances = triplet_mining.calculate_left_right_distances(y_pred)
    triplet_mining.calculate_class_neut_distances(y_pred)

    # Reshape the 1D tensor into a 2D tensor with shape (1, 56) (will allow for broadcasting)
    # row_dists_class_neut = tf.reshape(triplet_mining.tensor_dists_class_neut,
    #                                   (1, triplet_mining.num_states_drives))
    # Reshape the 1D tensor into a 2D tensor with shape (1, 56) for broadcasting
    row_dists_class_neut = triplet_mining.tensor_dists_class_neut.reshape(1, triplet_mining.num_states_drives)

    # print(f"Custom_losses:calculate_triplet_loss: row_dists_class_neut shape: {row_dists_class_neut.shape}")
    # Reshape the 1D tensor into a 2D tensor of shape (56, 1) (will allow for broadcasting)
    # column_dists_class_neut = tf.reshape(triplet_mining.tensor_dists_class_neut,
    #                                      (triplet_mining.num_states_drives, 1))
    column_dists_class_neut = triplet_mining.tensor_dists_class_neut.reshape(triplet_mining.num_states_drives, 1)

    # case 1: left and right are positives
    # L = anchor
    diff_lr_ln = classes_distances - column_dists_class_neut
    # print(f"Custom_losses:calculate_triplet_loss: diff_lr_ln shape: {diff_lr_ln.shape}")
    # tf.debugging.assert_shapes(
    #     [(tf.shape(diff_lr_ln), (tf.TensorShape([triplet_mining.num_states_drives,
    #                                              triplet_mining.num_states_drives]),))],
    #     message="Tensor diff_lr_ln has incorrect shape")

    assert diff_lr_ln.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
        f"Shape mismatch: expected ({triplet_mining.num_states_drives}, {triplet_mining.num_states_drives}), but got {diff_lr_ln.shape}"


    # consider only cases where diff_lr_ln > 0 (i.e., l_n - l_r > 0)
    if batch_strategy == BatchStrategy.HARD:
        # diff_lr_ln = tf.maximum(diff_lr_ln, 0.0)
        # diff_lr_rl_alpha = diff_lr_ln + triplet_mining.matrix_alpha_left_right_right_left
        # diff_lr_alpha = tf.multiply(diff_lr_rl_alpha, triplet_mining.matrix_bool_left_right)
        # # ensure no negative losses
        # triplet_loss_L_R = diff_lr_alpha
        #
        # tf.debugging.assert_equal(triplet_loss_L_R, tf.maximum(triplet_loss_L_R, 0.0), message="Negative losses exist")

        # Ensure no negative distances
        diff_lr_ln = torch.clamp(diff_lr_ln, min=0.0)
        diff_lr_rl_alpha = diff_lr_ln + triplet_mining.matrix_alpha_left_right_right_left
        diff_lr_alpha = diff_lr_rl_alpha * triplet_mining.matrix_bool_left_right  # Element-wise multiplication

        # Ensure no negative losses
        triplet_loss_L_R = diff_lr_alpha

        # Assert no negative losses
        assert torch.all(triplet_loss_L_R >= 0), "Negative losses exist"

    #TODO: Undergo Tensorflow to Pytorch Conversion for this batch strat case
    # consider only cases where diff_lr_ln < 0 yet triplet_loss_L_N > 0 (i.e., l_r - l_n < 0 and l_r - l_n + alpha > 0)
    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_lr_ln = tf.where(diff_lr_ln < 0, diff_lr_ln, 0)
        diff_lr_rl_alpha = diff_lr_ln + triplet_mining.matrix_alpha_left_right_right_left
        diff_lr_alpha = tf.multiply(diff_lr_rl_alpha, triplet_mining.matrix_bool_left_right)
        triplet_loss_L_R = tf.maximum(diff_lr_alpha, 0.0)
    # consider all cases where diff_lr_ln + alpha > 0 (i.e., dist(l_r) - dist(l_n) + alpha > 0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_lr_rl_alpha = diff_lr_ln + triplet_mining.matrix_alpha_left_right_right_left
        diff_lr_alpha = tf.multiply(diff_lr_rl_alpha, triplet_mining.matrix_bool_left_right)
        triplet_loss_L_R = tf.maximum(diff_lr_alpha, 0.0)

    # R = anchor
    diff_rl_rn = classes_distances - row_dists_class_neut
    if batch_strategy == BatchStrategy.HARD:
        # diff_rl_rn = tf.maximum(diff_rl_rn, 0.0)
        # diff_rl_rn_alpha = diff_rl_rn + triplet_mining.matrix_alpha_left_right_right_left
        # diff_rl_alpha = tf.multiply(diff_rl_rn_alpha, triplet_mining.matrix_bool_right_left)
        # triplet_loss_R_L = diff_rl_alpha
        #
        #
        # tf.debugging.assert_equal(triplet_loss_R_L, tf.maximum(triplet_loss_R_L, 0.0), message="Negative losses exist")
        # Replace tf.maximum with torch.clamp
        diff_rl_rn = torch.clamp(diff_rl_rn, min=0.0)

        # Addition works the same way in PyTorch
        diff_rl_rn_alpha = diff_rl_rn + triplet_mining.matrix_alpha_left_right_right_left

        # Replace tf.multiply with torch.mul or simply use *
        diff_rl_alpha = diff_rl_rn_alpha * triplet_mining.matrix_bool_right_left

        triplet_loss_R_L = diff_rl_alpha

        # Replace TensorFlow assertion with PyTorch assertion
        # torch.all performs element-wise comparison and returns True if all elements are True
        assert torch.all(triplet_loss_R_L >= 0.0), "Negative losses exist"


    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_rl_rn = tf.where(diff_rl_rn < 0, diff_rl_rn, 0)
        diff_rl_rn_alpha = diff_rl_rn + triplet_mining.matrix_alpha_left_right_right_left
        diff_rl_alpha = tf.multiply(diff_rl_rn_alpha, triplet_mining.matrix_bool_right_left)
        triplet_loss_R_L = tf.maximum(diff_rl_alpha, 0.0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_rl_rn_alpha = diff_rl_rn + triplet_mining.matrix_alpha_left_right_right_left
        diff_rl_alpha = tf.multiply(diff_rl_rn_alpha, triplet_mining.matrix_bool_right_left)
        triplet_loss_R_L = tf.maximum(diff_rl_alpha, 0.0)

    # case 2: left and neutral are positives
    # L = anchor
    diff_ln_lr = column_dists_class_neut - classes_distances
    # tf.debugging.assert_shapes([(tf.shape(diff_ln_lr), (tf.TensorShape([triplet_mining.num_states_drives,
    #                                                                     triplet_mining.num_states_drives]),))],
    #                            message="Tensor diff_ln_lr has incorrect shape")
    # Assert that the shape of diff_ln_lr is correct
    assert diff_ln_lr.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
        f"Tensor diff_ln_lr has incorrect shape: {diff_ln_lr.shape}"

    if batch_strategy == BatchStrategy.HARD:
        # diff_ln_lr = tf.where(diff_ln_lr > 0, diff_ln_lr, 0)
        # diff_ln_lr_alpha = diff_ln_lr + triplet_mining.matrix_alpha_left_neut_neut_left
        # # we care only for differences with corresponding alpha_left_neut values (i.e., relevant l_n cases)
        # diff_ln_lr_alpha = tf.multiply(diff_ln_lr_alpha, triplet_mining.matrix_bool_left_neut)
        # # ensure no negative losses
        # triplet_loss_L_N = diff_ln_lr_alpha
        # tf.debugging.assert_equal(triplet_loss_L_N, tf.maximum(triplet_loss_L_N, 0.0), message="Negative losses exist")
        # Replace tf.where with torch.where - note argument order is the same
        diff_ln_lr = torch.where(diff_ln_lr > 0, diff_ln_lr, torch.zeros_like(diff_ln_lr))

        # Addition works the same way in PyTorch
        diff_ln_lr_alpha = diff_ln_lr + triplet_mining.matrix_alpha_left_neut_neut_left

        # Replace tf.multiply with element-wise multiplication
        diff_ln_lr_alpha = diff_ln_lr_alpha * triplet_mining.matrix_bool_left_neut

        # Assign to triplet_loss_L_N
        triplet_loss_L_N = diff_ln_lr_alpha

        # Assert that all values in triplet_loss_L_N are non-negative
        assert torch.all(triplet_loss_L_N >= 0.0), "Negative losses exist"

    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_ln_lr = tf.where(diff_ln_lr < 0, diff_ln_lr, 0)
        diff_ln_lr_alpha = diff_ln_lr + triplet_mining.matrix_alpha_left_neut_neut_left
        diff_ln_lr_alpha = tf.multiply(diff_ln_lr_alpha, triplet_mining.matrix_bool_left_neut)
        triplet_loss_L_N = tf.where(diff_ln_lr_alpha > 0, diff_ln_lr_alpha, 0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_ln_lr_alpha = diff_ln_lr + triplet_mining.matrix_alpha_right_neut_neut_right
        diff_ln_lr_alpha = tf.multiply(diff_ln_lr_alpha, triplet_mining.matrix_bool_left_neut)
        triplet_loss_L_N = tf.where(diff_ln_lr_alpha > 0, diff_ln_lr_alpha, 0)
    # N = anchor
    # diff_nl_nr = column_dists_class_neut - row_dists_class_neut
    diff_nl_nr = row_dists_class_neut - column_dists_class_neut
    # tf.debugging.assert_shapes([(tf.shape(diff_nl_nr), (tf.TensorShape([triplet_mining.num_states_drives,
    #                                                                     triplet_mining.num_states_drives]),))],
    #                            message="Tensor diff_nl_nr has incorrect shape")
    # Assert that the shape of diff_nl_nr is correct
    assert diff_nl_nr.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
        f"Tensor diff_nl_nr has incorrect shape: {diff_nl_nr.shape}"
    if batch_strategy == BatchStrategy.HARD:
        # diff_nl_nr = tf.where(diff_nl_nr > 0, diff_nl_nr, 0)
        # diff_nl_nr_alpha = diff_nl_nr + triplet_mining.matrix_alpha_left_neut_neut_left
        # diff_nl_nr_alpha = tf.multiply(diff_nl_nr_alpha, triplet_mining.matrix_bool_neut_left)
        # triplet_loss_N_L = diff_nl_nr_alpha
        # tf.debugging.assert_equal(triplet_loss_N_L, tf.maximum(triplet_loss_N_L, 0.0), message="Negative losses exist")
        # Replace tf.where with torch.where
        diff_nl_nr = torch.where(diff_nl_nr > 0, diff_nl_nr, torch.zeros_like(diff_nl_nr))

        # Addition works the same way in PyTorch
        diff_nl_nr_alpha = diff_nl_nr + triplet_mining.matrix_alpha_left_neut_neut_left

        # Replace tf.multiply with element-wise multiplication
        diff_nl_nr_alpha = diff_nl_nr_alpha * triplet_mining.matrix_bool_neut_left

        # Assign to triplet_loss_N_L
        triplet_loss_N_L = diff_nl_nr_alpha

        # Assert that all values in triplet_loss_N_L are non-negative
        assert torch.all(triplet_loss_N_L >= 0.0), "Negative losses exist"

    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_nl_nr = tf.where(diff_nl_nr < 0, diff_nl_nr, 0)
        diff_nl_nr_alpha = diff_nl_nr + triplet_mining.matrix_alpha_left_neut_neut_left
        diff_nl_nr_alpha = tf.multiply(diff_nl_nr_alpha, triplet_mining.matrix_bool_neut_left)
        triplet_loss_N_L = tf.where(diff_nl_nr_alpha > 0, diff_nl_nr_alpha, 0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_nl_nr_alpha = diff_nl_nr + triplet_mining.matrix_alpha_left_neut_neut_left
        diff_nl_nr_alpha = tf.multiply(diff_nl_nr_alpha, triplet_mining.matrix_bool_neut_left)
        triplet_loss_N_L = tf.where(diff_nl_nr_alpha > 0, diff_nl_nr_alpha, 0)

    ### case 3: right and neutral are positives
    # R = anchor
    # diff_rn_rl = row_dists_class_neut - tf.transpose(classes_distances)
    diff_rn_rl = row_dists_class_neut - torch.transpose(classes_distances, 0, 1)
    if batch_strategy == BatchStrategy.HARD:
        # diff_rn_rl = tf.where(diff_rn_rl > 0, diff_rn_rl, 0)
        # diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining.matrix_alpha_right_neut_neut_right
        # diff_r_n_r_l_alpha = tf.multiply(diff_r_n_r_l_alpha, triplet_mining.matrix_bool_right_neut)
        # # remove negative losses
        # triplet_loss_R_N = diff_r_n_r_l_alpha
        # tf.debugging.assert_equal(triplet_loss_R_N, tf.maximum(triplet_loss_R_N, 0.0), message="Negative losses exist")
        # Replace tf.where with torch.where
        diff_rn_rl = torch.where(diff_rn_rl > 0, diff_rn_rl, torch.zeros_like(diff_rn_rl))

        # Addition works the same way in PyTorch
        diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining.matrix_alpha_right_neut_neut_right

        # Replace tf.multiply with element-wise multiplication
        diff_r_n_r_l_alpha = diff_r_n_r_l_alpha * triplet_mining.matrix_bool_right_neut

        # Assign to triplet_loss_R_N
        triplet_loss_R_N = diff_r_n_r_l_alpha

        # Assert that all values in triplet_loss_R_N are non-negative
        assert torch.all(triplet_loss_R_N >= 0.0), "Negative losses exist"

    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_rn_rl = tf.where(diff_rn_rl < 0, diff_rn_rl, 0)
        diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining.matrix_alpha_right_neut_neut_right
        diff_r_n_r_l_alpha = tf.multiply(diff_r_n_r_l_alpha, triplet_mining.matrix_bool_right_neut)
        triplet_loss_R_N = tf.where(diff_r_n_r_l_alpha > 0, diff_r_n_r_l_alpha, 0)
    else:
        diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining.matrix_alpha_right_neut_neut_right
        diff_r_n_r_l_alpha = tf.multiply(diff_r_n_r_l_alpha, triplet_mining.matrix_bool_right_neut)
        triplet_loss_R_N = tf.where(diff_r_n_r_l_alpha > 0, diff_r_n_r_l_alpha, 0)
    # check if diff_r_n_r_l_alpha and diff_rn_rl contain non-zero values in the same locations
    # condition = tf.math.logical_and(tf.math.not_equal(diff_rn_rl, 0), tf.math.not_equal(diff_r_n_r_l_alpha, 0))
    # tf.debugging.Assert(tf.math.reduce_all(condition),
    #                     message="Difference rn_rl tensors do not have non-zero elements in the same cells")
    # tf.debugging.assert_equal(triplet_loss_R_N,
    #                           tf.multiply(triplet_loss_R_N, triplet_mining.matrix_bool_right_neut),
    #                           message="Improper triplet_loss_R_N generation")


    # tf.debugging.assert_equal(triplet_loss_R_N, tf.maximum(triplet_loss_R_N, 0.0), message="Negative losses exist")
    # Assert that all values in triplet_loss_R_N are non-negative
    assert torch.all(triplet_loss_R_N >= 0.0), "Negative losses exist"
    # N = anchor
    diff_nr_nl = column_dists_class_neut - row_dists_class_neut
    # tf.debugging.assert_shapes(
    #     [(diff_nr_nl, (triplet_mining.num_states_drives, triplet_mining.num_states_drives))],
    #     message="Tensor diff_nr_nl has incorrect shape")
    # Assert that the shape of diff_nr_nl is correct
    assert diff_nr_nl.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
        f"Tensor diff_nr_nl has incorrect shape: {diff_nr_nl.shape}"

    if batch_strategy == BatchStrategy.HARD:
        # diff_nr_nl = tf.where(diff_nr_nl > 0, diff_nr_nl, 0)
        # diff_nr_nl_alpha = diff_nr_nl + triplet_mining.matrix_alpha_right_neut_neut_right
        # diff_nr_nl_alpha = tf.multiply(diff_nr_nl_alpha, triplet_mining.matrix_bool_neut_right)
        # triplet_loss_N_R = diff_nr_nl_alpha
        # tf.debugging.assert_equal(triplet_loss_N_R, tf.maximum(triplet_loss_N_R, 0.0), message="Negative losses exist")
        # Replace tf.where with torch.where
        diff_nr_nl = torch.where(diff_nr_nl > 0, diff_nr_nl, torch.zeros_like(diff_nr_nl))

        # Addition works the same way in PyTorch
        diff_nr_nl_alpha = diff_nr_nl + triplet_mining.matrix_alpha_right_neut_neut_right

        # Replace tf.multiply with element-wise multiplication
        diff_nr_nl_alpha = diff_nr_nl_alpha * triplet_mining.matrix_bool_neut_right

        # Assign to triplet_loss_N_R
        triplet_loss_N_R = diff_nr_nl_alpha

        # Assert that all values in triplet_loss_N_R are non-negative
        assert torch.all(triplet_loss_N_R >= 0.0), "Negative losses exist"

    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_nr_nl_alpha = diff_nr_nl + triplet_mining.matrix_alpha_right_neut_neut_right
        diff_nr_nl_alpha = tf.multiply(diff_nr_nl_alpha, triplet_mining.matrix_bool_neut_right)
        triplet_loss_N_R = tf.where(diff_nr_nl_alpha > 0, diff_nr_nl_alpha, 0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_nr_nl_alpha = diff_nr_nl + triplet_mining.matrix_alpha_right_neut_neut_right
        diff_nr_nl_alpha = tf.multiply(diff_nr_nl_alpha, triplet_mining.matrix_bool_neut_right)
        triplet_loss_N_R = tf.where(diff_nr_nl_alpha > 0, diff_nr_nl_alpha, 0)


    #print(f"Custom_losses:calculate_triplet_loss: triplet_loss_L_R shape: {triplet_loss_L_R}")
    losses = (triplet_loss_L_R + triplet_loss_R_L + triplet_loss_L_N + triplet_loss_N_L + triplet_loss_R_N +
              triplet_loss_N_R)
    # assertion check that no negative losses exist
    # tf.debugging.assert_equal(losses, tf.maximum(losses, 0.0), message="Negative losses exist")
    # tf.debugging.assert_shapes([(tf.shape(losses), (tf.TensorShape([triplet_mining.num_states_drives,
    #                                                                 triplet_mining.num_states_drives]),))],
    #                            message="Comparisons loss tensor has incorrect shape")
    # Assert that no negative losses exist
    assert torch.all(losses >= 0.0), "Negative losses exist"

    # Assert that the shape of losses tensor is correct
    assert losses.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
        f"Comparisons loss tensor has incorrect shape: {losses.shape}"
    return losses


def create_batch_triplet_loss(triplet_mining_modules):
    """
    Create a batch triplet loss function for use with multiple triplet mining modules.

    Args:
        triplet_mining_modules: List of TripletMining objects for different action types

    Returns:
        batch_triplet_loss: Function that computes the dual term triplet loss per batch
    """

    def batch_triplet_loss(y_true, y_pred):
        """Build triplet loss over a batch of embeddings.

        Args:
            y_true: supposed 'labels' of the batch (i.e., class indexes), tensor of size (batch_size,)
            y_pred: embeddings, tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        print(f"Custom_losses:create_batch_triplet_loss: y_true shape: {y_true.shape}")
        # Flatten and sort for proper processing
        y_true_flat = y_true.reshape(-1)

        # Get sorted indices
        _, sorted_indices = torch.sort(y_true_flat)

        # Sort labels and embeddings
        y_true = torch.gather(y_true, 0, sorted_indices)
        y_pred = torch.gather(y_pred, 0, sorted_indices.unsqueeze(1).expand(-1, y_pred.size(1)))

        # Calculate overall triplet loss across all modules
        overall_triplet_loss = torch.tensor(0.0, device=y_pred.device)

        for i, triplet_mining in enumerate(triplet_mining_modules):
            # Extract the portion of the batch for this module
            y_true_module = y_true[i * triplet_mining.batch_size:(i + 1) * triplet_mining.batch_size]
            y_pred_module = y_pred[i * triplet_mining.batch_size:(i + 1) * triplet_mining.batch_size]

            # Calculate triplet losses for this module
            triplet_losses = calculate_triplet_loss(y_true_module, y_pred_module, triplet_mining)
            print(f"Custom_losses:create_batch_triplet_loss: module {i} triplet_losses: {triplet_losses}")

            # Combine the losses
            triplet_loss = torch.mean(triplet_losses)
            overall_triplet_loss += triplet_loss

            # Cross-module comparisons
            for j, other_triplet_mining in enumerate(triplet_mining_modules):
                if i != j:
                    y_pred_other_module = y_pred[
                                          j * other_triplet_mining.batch_size:(j + 1) * other_triplet_mining.batch_size]

                    # Intra-module (anchor-positive) pairwise distances
                    intra_module_distances = torch.norm(
                        y_pred_module.unsqueeze(1) - y_pred_module.unsqueeze(0), dim=-1
                    )

                    # Inter-module (anchor-negative) distances
                    inter_module_distances = torch.norm(
                        y_pred_module.unsqueeze(1) - y_pred_other_module.unsqueeze(0), dim=-1
                    )

                    # For every anchor-positive pair, ensure distance to negatives is greater than to positives + margin
                    loss_term = torch.clamp(
                        intra_module_distances + 1 - torch.min(inter_module_distances, dim=1, keepdim=True)[0],
                        min=0
                    )
                    print(f"Custom_losses:create_batch_triplet_loss: module {i} vs. module {j} loss_term: {loss_term}")

                    overall_triplet_loss += torch.mean(loss_term)

        print(f"Custom_losses:create_batch_triplet_loss: overall_triplet_loss: {overall_triplet_loss}")
        return overall_triplet_loss

    print(f"Custom_losses:create_batch_triplet_loss: batch_triplet_loss: {batch_triplet_loss}")
    return batch_triplet_loss