from conf import BATCH_STRATEGY
from conf import BatchStrategy
import tensorflow as tf


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
    row_dists_class_neut = tf.reshape(triplet_mining.tensor_dists_class_neut,
                                      (1, triplet_mining.num_states_drives))
    # print(f"Custom_losses:calculate_triplet_loss: row_dists_class_neut shape: {row_dists_class_neut.shape}")
    # Reshape the 1D tensor into a 2D tensor of shape (56, 1) (will allow for broadcasting)
    column_dists_class_neut = tf.reshape(triplet_mining.tensor_dists_class_neut,
                                         (triplet_mining.num_states_drives, 1))

    # case 1: left and right are positives
    # L = anchor
    diff_lr_ln = classes_distances - column_dists_class_neut
    # print(f"Custom_losses:calculate_triplet_loss: diff_lr_ln shape: {diff_lr_ln.shape}")
    tf.debugging.assert_shapes(
        [(tf.shape(diff_lr_ln), (tf.TensorShape([triplet_mining.num_states_drives,
                                                 triplet_mining.num_states_drives]),))],
        message="Tensor diff_lr_ln has incorrect shape")
    # consider only cases where diff_lr_ln > 0 (i.e., l_n - l_r > 0)
    if batch_strategy == BatchStrategy.HARD:
        diff_lr_ln = tf.maximum(diff_lr_ln, 0.0)
        diff_lr_rl_alpha = diff_lr_ln + triplet_mining.matrix_alpha_left_right_right_left
        diff_lr_alpha = tf.multiply(diff_lr_rl_alpha, triplet_mining.matrix_bool_left_right)
        # ensure no negative losses
        triplet_loss_L_R = diff_lr_alpha

        tf.debugging.assert_equal(triplet_loss_L_R, tf.maximum(triplet_loss_L_R, 0.0), message="Negative losses exist")

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
        diff_rl_rn = tf.maximum(diff_rl_rn, 0.0)
        diff_rl_rn_alpha = diff_rl_rn + triplet_mining.matrix_alpha_left_right_right_left
        diff_rl_alpha = tf.multiply(diff_rl_rn_alpha, triplet_mining.matrix_bool_right_left)
        triplet_loss_R_L = diff_rl_alpha


        tf.debugging.assert_equal(triplet_loss_R_L, tf.maximum(triplet_loss_R_L, 0.0), message="Negative losses exist")


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
    tf.debugging.assert_shapes([(tf.shape(diff_ln_lr), (tf.TensorShape([triplet_mining.num_states_drives,
                                                                        triplet_mining.num_states_drives]),))],
                               message="Tensor diff_ln_lr has incorrect shape")
    if batch_strategy == BatchStrategy.HARD:
        diff_ln_lr = tf.where(diff_ln_lr > 0, diff_ln_lr, 0)
        diff_ln_lr_alpha = diff_ln_lr + triplet_mining.matrix_alpha_left_neut_neut_left
        # we care only for differences with corresponding alpha_left_neut values (i.e., relevant l_n cases)
        diff_ln_lr_alpha = tf.multiply(diff_ln_lr_alpha, triplet_mining.matrix_bool_left_neut)
        # ensure no negative losses
        triplet_loss_L_N = diff_ln_lr_alpha
        tf.debugging.assert_equal(triplet_loss_L_N, tf.maximum(triplet_loss_L_N, 0.0), message="Negative losses exist")
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
    tf.debugging.assert_shapes([(tf.shape(diff_nl_nr), (tf.TensorShape([triplet_mining.num_states_drives,
                                                                        triplet_mining.num_states_drives]),))],
                               message="Tensor diff_nl_nr has incorrect shape")
    if batch_strategy == BatchStrategy.HARD:
        diff_nl_nr = tf.where(diff_nl_nr > 0, diff_nl_nr, 0)
        diff_nl_nr_alpha = diff_nl_nr + triplet_mining.matrix_alpha_left_neut_neut_left
        diff_nl_nr_alpha = tf.multiply(diff_nl_nr_alpha, triplet_mining.matrix_bool_neut_left)
        triplet_loss_N_L = diff_nl_nr_alpha
        tf.debugging.assert_equal(triplet_loss_N_L, tf.maximum(triplet_loss_N_L, 0.0), message="Negative losses exist")
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
    diff_rn_rl = row_dists_class_neut - tf.transpose(classes_distances)
    if batch_strategy == BatchStrategy.HARD:
        diff_rn_rl = tf.where(diff_rn_rl > 0, diff_rn_rl, 0)
        diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining.matrix_alpha_right_neut_neut_right
        diff_r_n_r_l_alpha = tf.multiply(diff_r_n_r_l_alpha, triplet_mining.matrix_bool_right_neut)
        # remove negative losses
        triplet_loss_R_N = diff_r_n_r_l_alpha
        tf.debugging.assert_equal(triplet_loss_R_N, tf.maximum(triplet_loss_R_N, 0.0), message="Negative losses exist")
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
    tf.debugging.assert_equal(triplet_loss_R_N, tf.maximum(triplet_loss_R_N, 0.0), message="Negative losses exist")
    # N = anchor
    diff_nr_nl = column_dists_class_neut - row_dists_class_neut
    tf.debugging.assert_shapes(
        [(diff_nr_nl, (triplet_mining.num_states_drives, triplet_mining.num_states_drives))],
        message="Tensor diff_nr_nl has incorrect shape")
    if batch_strategy == BatchStrategy.HARD:
        diff_nr_nl = tf.where(diff_nr_nl > 0, diff_nr_nl, 0)
        diff_nr_nl_alpha = diff_nr_nl + triplet_mining.matrix_alpha_right_neut_neut_right
        diff_nr_nl_alpha = tf.multiply(diff_nr_nl_alpha, triplet_mining.matrix_bool_neut_right)
        triplet_loss_N_R = diff_nr_nl_alpha
        tf.debugging.assert_equal(triplet_loss_N_R, tf.maximum(triplet_loss_N_R, 0.0), message="Negative losses exist")
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
    tf.debugging.assert_equal(losses, tf.maximum(losses, 0.0), message="Negative losses exist")
    tf.debugging.assert_shapes([(tf.shape(losses), (tf.TensorShape([triplet_mining.num_states_drives,
                                                                    triplet_mining.num_states_drives]),))],
                               message="Comparisons loss tensor has incorrect shape")
    return losses


def create_batch_triplet_loss(triplet_mining_modules):
    def batch_triplet_loss(y_true, y_pred):
        """Build triplet loss over a batch of embeddings.

        Args:
            y_true: supposed 'labels' of the batch (i.e., class indexes), tensor of size (batch_size,), i.e., (57*3,)
            y_pred: embeddings, tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """

        y_true_flat = tf.reshape(y_true, [-1])
        print(f"custom_losses:calculate_triplet_loss: y_true pre sort: {y_true}")
        sorted_indices = tf.argsort(y_true_flat)
        print(f"custom_losses:calculate_triplet_loss: sorted_indices shape: {sorted_indices.shape}")
        print(f"custom_losses:calculate_triplet_loss: sorted_indices example: {sorted_indices[0]}")
        y_true = tf.gather(y_true, sorted_indices)
        y_pred = tf.gather(y_pred, sorted_indices)
        print(f"custom_losses:calculate_triplet_loss: y_true post sort: {y_true}")




        overall_triplet_loss = 0
        for i, triplet_mining in enumerate(triplet_mining_modules):
            # Index the batch labels and values with respect to the current TripletMining module
            y_true_module = y_true[i * triplet_mining.batch_size:(i + 1) * triplet_mining.batch_size]
            y_pred_module = y_pred[i * triplet_mining.batch_size:(i + 1) * triplet_mining.batch_size]

            triplet_losses = calculate_triplet_loss(y_true_module, y_pred_module, triplet_mining)
            # Combine the losses
            triplet_loss = tf.reduce_mean(triplet_losses)
            tf.debugging.assert_scalar(triplet_loss, message="Epoch loss is not a scalar")
            overall_triplet_loss += triplet_loss

            # Cross-module comparisons
            for j, other_triplet_mining in enumerate(triplet_mining_modules):
                if i != j:
                    y_pred_other_module = y_pred[
                                          j * other_triplet_mining.batch_size:(j + 1) * other_triplet_mining.batch_size]
                    # Intra-module (anchor-positive) pairwise distances
                    intra_module_distances = tf.norm(tf.expand_dims(y_pred_module, 1) - tf.expand_dims(y_pred_module, 0), axis=-1)
                    # Inter-module (anchor-negative) distances
                    inter_module_distances = tf.norm(tf.expand_dims(y_pred_module, 1) - tf.expand_dims(y_pred_other_module, 0), axis=-1)
                    # For every embedding in y_pred_module as anchor (A), compare intra-module distances with inter-module distances
                    # Ensure that for every anchor-positive pair (A, P), the distance A-N is greater than A-P + margin
                    loss_term = tf.maximum(intra_module_distances + 1 - tf.reduce_min(inter_module_distances, axis=1, keepdims=True), 0)

                    overall_triplet_loss += tf.reduce_mean(loss_term)

        return overall_triplet_loss

    return batch_triplet_loss