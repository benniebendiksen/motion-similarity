from Config import BATCH_STRATEGY
from Config import BatchStrategy
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score


class AdaptiveDistanceModule(nn.Module):
    """
    A learnable distance metric module that adapts to better match human perception.

    This module learns:
    1. Feature importance weights for each dimension of the embedding
    2. A non-linear transformation to map raw distances to perceptual distances
    """

    def __init__(self, embedding_dim):
        super().__init__()
        # Initialize feature weights near 1 with some variance to enable learning
        self.feature_weights = nn.Parameter(torch.ones(embedding_dim) + 0.1 * torch.randn(embedding_dim))

        # Non-linear mapping from raw distance to perceptual distance
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output in [0,1] range to match perception scores
        )

    def forward(self, x1, x2):
        """
        Calculate the adaptive distance between two sets of embeddings.

        Args:
            x1: First set of embeddings, shape [batch_size, embedding_dim]
            x2: Second set of embeddings, shape [batch_size, embedding_dim]

        Returns:
            Adaptive distances between embeddings, shape [batch_size]
        """
        # Ensure feature weights are positive (importance weights)
        positive_weights = F.softplus(self.feature_weights)

        # Calculate weighted squared differences
        weighted_diff = ((x1 - x2) ** 2) * positive_weights

        # Weighted Euclidean distance
        raw_distances = torch.sqrt(torch.sum(weighted_diff, dim=1, keepdim=True) + 1e-8)

        # Apply non-linear transformation to better match perception
        perceptual_distances = self.distance_mlp(raw_distances)

        return perceptual_distances.squeeze(-1)

    def pairwise_distances(self, embeddings):
        """
        Calculate pairwise adaptive distances for a batch of embeddings.

        Args:
            embeddings: Embeddings tensor of shape [batch_size, embedding_dim]

        Returns:
            Pairwise distances of shape [batch_size, batch_size]
        """
        batch_size = embeddings.shape[0]
        distances = torch.zeros((batch_size, batch_size), device=embeddings.device)

        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:  # Skip diagonal elements (distance to self)
                    distances[i, j] = self.forward(
                        embeddings[i].unsqueeze(0),
                        embeddings[j].unsqueeze(0)
                    )

        return distances


def calculate_triplet_loss(y_true, y_pred, triplet_mining, batch_strategy=BATCH_STRATEGY):
    """Calculate triplet loss using left-right embeddings across all preference cases.

    Modified to focus on learning embeddings between left and right classes, regardless of
    which pair was most preferable in the original human comparisons.

    Args:
        y_true: Labels of the batch (class indexes), tensor of size (batch_size,)
        y_pred: Embeddings, tensor of shape (batch_size, embed_dim)
        triplet_mining: TripletMining object with alpha matrices
        batch_strategy: Enum value indicating the batch strategy to use

    Returns:
        losses: Tensor of shape (triplet_mining.num_states_drives, triplet_mining.num_states_drives)
    """
    # Calculate distances between embeddings
    classes_distances = triplet_mining.calculate_distances(y_pred)

    # Reshape class-neutral distances for broadcasting
    row_dists_class_neut = triplet_mining.tensor_dists_class_neut.reshape(1, triplet_mining.num_states_drives)
    column_dists_class_neut = triplet_mining.tensor_dists_class_neut.reshape(triplet_mining.num_states_drives, 1)

    # -------------------------------------------------------------------------
    # Case 1: Original Left-Right preference case
    # -------------------------------------------------------------------------
    # subcase L = anchor
    diff_lr_ln = classes_distances - column_dists_class_neut

    assert diff_lr_ln.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
        f"Shape mismatch: expected ({triplet_mining.num_states_drives}, {triplet_mining.num_states_drives}), but got {diff_lr_ln.shape}"

    # Process based on batch strategy
    if batch_strategy == BatchStrategy.HARD:
        diff_lr_ln = torch.clamp(diff_lr_ln, min=0.0)

    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_lr_ln = torch.where(diff_lr_ln < 0, diff_lr_ln, torch.zeros_like(diff_lr_ln))

    # Apply alpha values and boolean masks
    diff_lr_rl_alpha = diff_lr_ln + triplet_mining.matrix_alpha_left_right_right_left
    diff_lr_alpha = diff_lr_rl_alpha * triplet_mining.matrix_bool_left_right
    triplet_loss_L_R = torch.clamp(diff_lr_alpha, min=0.0)

    # subcase R = anchor
    diff_rl_rn = classes_distances - row_dists_class_neut

    if batch_strategy == BatchStrategy.HARD:
        diff_rl_rn = torch.clamp(diff_rl_rn, min=0.0)

    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_rl_rn = torch.where(diff_rl_rn < 0, diff_rl_rn, torch.zeros_like(diff_rl_rn))

    diff_rl_rn_alpha = diff_rl_rn + triplet_mining.matrix_alpha_left_right_right_left
    diff_rl_alpha = diff_rl_rn_alpha * triplet_mining.matrix_bool_right_left
    triplet_loss_R_L = torch.clamp(diff_rl_alpha, min=0.0)

    # -------------------------------------------------------------------------
    # Case 2: Left-Right alphas from Left-Neutral preference case
    # -------------------------------------------------------------------------
    # The same diff_lr_ln and diff_rl_rn calculations can be reused
    # but with matrix_alpha_left_neut_neut_left which now contains left-right alphas

    # L = anchor (from Left-Neutral preference case)
    ln_diff_lr_alpha = diff_lr_ln + triplet_mining.matrix_alpha_left_neut_neut_left
    ln_diff_lr_alpha = ln_diff_lr_alpha * triplet_mining.matrix_bool_left_neut
    triplet_loss_L_R_from_LN = torch.clamp(ln_diff_lr_alpha, min=0.0)

    # R = anchor (from Left-Neutral preference case)
    ln_diff_rl_alpha = diff_rl_rn + triplet_mining.matrix_alpha_left_neut_neut_left
    ln_diff_rl_alpha = ln_diff_rl_alpha * triplet_mining.matrix_bool_neut_left
    triplet_loss_R_L_from_LN = torch.clamp(ln_diff_rl_alpha, min=0.0)

    # -------------------------------------------------------------------------
    # Case 3: Left-Right alphas from Right-Neutral preference case
    # -------------------------------------------------------------------------
    # L = anchor (from Right-Neutral preference case)
    rn_diff_lr_alpha = diff_lr_ln + triplet_mining.matrix_alpha_right_neut_neut_right
    rn_diff_lr_alpha = rn_diff_lr_alpha * triplet_mining.matrix_bool_right_neut
    triplet_loss_L_R_from_RN = torch.clamp(rn_diff_lr_alpha, min=0.0)

    # R = anchor (from Right-Neutral preference case)
    rn_diff_rl_alpha = diff_rl_rn + triplet_mining.matrix_alpha_right_neut_neut_right
    rn_diff_rl_alpha = rn_diff_rl_alpha * triplet_mining.matrix_bool_neut_right
    triplet_loss_R_L_from_RN = torch.clamp(rn_diff_rl_alpha, min=0.0)

    # -------------------------------------------------------------------------
    # Combine all losses
    # -------------------------------------------------------------------------
    # This now focuses solely on left-right relationships across all preference cases
    losses = (triplet_loss_L_R + triplet_loss_R_L +
              triplet_loss_L_R_from_LN + triplet_loss_R_L_from_LN +
              triplet_loss_L_R_from_RN + triplet_loss_R_L_from_RN)

    # Validate the final losses
    assert torch.all(losses >= 0.0), "Negative losses exist"
    assert losses.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
        f"Comparisons loss tensor has incorrect shape: {losses.shape}"

    return losses


def calculate_integrated_perception_loss(y_true, y_pred, triplet_mining, adaptive_distance_module=None,
                                         batch_strategy=BATCH_STRATEGY):
    """
    Integrated perception-aligned loss function that combines:
    1. Direct correlation optimization
    2. Adaptive distance metric (if provided)
    3. Original triplet constraints (as regularization)
    4. Ranking and ordinal relationship preservation

    Args:
        y_true: Labels of the batch (class indexes), tensor of size (batch_size,)
        y_pred: Embeddings, tensor of shape (batch_size, embed_dim)
        triplet_mining: TripletMining object with comparison matrices
        adaptive_distance_module: Optional AdaptiveDistanceModule instance
        batch_strategy: Enum value indicating the batch strategy to use

    Returns:
        loss: Scalar tensor containing the combined loss
    """
    # Calculate distances between embeddings
    if adaptive_distance_module is not None:
        # Use adaptive distance metric
        classes_distances = adaptive_distance_module.pairwise_distances(y_pred)
    else:
        # Use standard Euclidean distance
        classes_distances = triplet_mining.calculate_distances(y_pred)

    # Get the human perception data (inverse of comparison values)
    inverse_comparison_values = torch.ones_like(triplet_mining.matrix_comparison_values_left_right)
    mask = triplet_mining.matrix_comparison_bool_left_right > 0
    inverse_comparison_values[mask] = 1.0 - triplet_mining.matrix_comparison_values_left_right[mask]

    # Get valid comparison mask (where we have human data)
    valid_comparisons = triplet_mining.matrix_comparison_bool_left_right > 0

    # If no valid comparisons, fall back to standard triplet loss
    if not torch.any(valid_comparisons):
        return calculate_triplet_loss(y_true, y_pred, triplet_mining, batch_strategy)

    # Extract valid distances and perception values
    valid_distances = classes_distances[valid_comparisons]
    valid_perception = inverse_comparison_values[valid_comparisons]

    # Normalize distances to [0,1] range
    if len(valid_distances) > 0:
        min_dist = torch.min(valid_distances)
        max_dist = torch.max(valid_distances)

        if max_dist > min_dist:
            normalized_distances = (valid_distances - min_dist) / (max_dist - min_dist)
        else:
            normalized_distances = torch.zeros_like(valid_distances)
    else:
        normalized_distances = valid_distances

    # Component 1: Direct MSE loss to align distances with perception
    alignment_loss = torch.mean(torch.square(normalized_distances - valid_perception))

    # Component 2: Pearson correlation loss
    dist_mean = torch.mean(normalized_distances)
    perc_mean = torch.mean(valid_perception)

    dist_centered = normalized_distances - dist_mean
    perc_centered = valid_perception - perc_mean

    numerator = torch.sum(dist_centered * perc_centered)
    denominator = torch.sqrt(torch.sum(dist_centered ** 2) * torch.sum(perc_centered ** 2) + 1e-8)
    correlation = numerator / denominator
    correlation_loss = -correlation  # Negative to maximize correlation

    # Component 3: Ranking preservation loss
    n = valid_distances.shape[0]
    rank_loss = torch.tensor(0.0, device=valid_distances.device)

    if n > 1:
        # Get all pairwise combinations
        i_indices, j_indices = torch.triu_indices(n, n, offset=1)

        # Extract pairs
        dist_i = normalized_distances[i_indices]
        dist_j = normalized_distances[j_indices]
        perc_i = valid_perception[i_indices]
        perc_j = valid_perception[j_indices]

        # Determine desired ordering
        should_be_greater = perc_i > perc_j + 0.05  # Add small threshold to avoid noise
        should_be_less = perc_i < perc_j - 0.05

        # Calculate ranking violations
        margin = 0.1

        # Where perception i > j, ensure distance i > j + margin
        greater_violations = torch.relu(margin - (dist_i - dist_j))
        greater_loss = torch.mean(greater_violations[should_be_greater]) if torch.any(should_be_greater) else 0.0

        # Where perception i < j, ensure distance i < j - margin
        less_violations = torch.relu(margin - (dist_j - dist_i))
        less_loss = torch.mean(less_violations[should_be_less]) if torch.any(should_be_less) else 0.0

        rank_loss = greater_loss + less_loss

    # Component 4: Use original triplet loss as a regularizer
    original_loss = calculate_triplet_loss(y_true, y_pred, triplet_mining, batch_strategy)
    structure_loss = torch.mean(original_loss)

    # Component 5: Instance contrast loss for neutral
    # Ensure neutral is different from all class embeddings
    neutral_distances = triplet_mining.tensor_dists_class_neut
    min_neutral_distance = 0.5
    neutral_violations = torch.relu(min_neutral_distance - neutral_distances)
    neutral_loss = torch.mean(neutral_violations)

    # Weight the different loss components (can be tuned)
    alignment_weight = 2.0  # Direct alignment with perception
    correlation_weight = 1.0  # Correlation maximization
    rank_weight = 2.0  # Ordinal relationship preservation
    structure_weight = 0.5  # Embedding structure preservation
    neutral_weight = 0.2  # Neutral contrast term

    # Combine all loss components
    total_loss = (
            alignment_weight * alignment_loss +
            correlation_weight * correlation_loss +
            rank_weight * rank_loss +
            structure_weight * structure_loss +
            neutral_weight * neutral_loss
    )

    return total_loss


def create_batch_triplet_loss(triplet_mining_modules, module_start_indices=None, module_sizes=None):
    """
    Create a batch triplet loss function for use with multiple triplet mining modules.

    Args:
        triplet_mining_modules: List of TripletMining objects
        module_start_indices: List of starting indices for each module in the batch
        module_sizes: List of sizes for each module
    """

    def batch_triplet_loss(y_true, y_pred):
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


def create_batch_integrated_loss(triplet_mining_modules, adaptive_distance_module=None, module_start_indices=None,
                                 module_sizes=None):
    """
    Create a batch loss function using the integrated perception-aligned loss.

    Args:
        triplet_mining_modules: List of TripletMining objects
        adaptive_distance_module: Optional AdaptiveDistanceModule instance
        module_start_indices: List of starting indices for each module in the batch
        module_sizes: List of sizes for each module
    """

    def batch_integrated_loss(y_true, y_pred):
        # Calculate overall loss across all modules
        overall_loss = torch.tensor(0.0, device=y_pred.device)
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

            # Calculate integrated loss for this module
            try:
                module_loss = calculate_integrated_perception_loss(
                    y_true_module,
                    y_pred_module,
                    triplet_mining,
                    adaptive_distance_module
                )
                overall_loss += module_loss
                valid_modules += 1
            except Exception as e:
                print(f"Error in loss calculation for module {i}: {e}")
                continue

        # Normalize by number of valid modules
        if valid_modules > 0:
            overall_loss = overall_loss / valid_modules

        return overall_loss

    return batch_integrated_loss


# Helper functions for analysis
def compute_correlation_metrics(distances, perception_values):
    """
    Compute correlation metrics between distances and perception values.

    Args:
        distances: List or array of distance values
        perception_values: List or array of perception values

    Returns:
        Dictionary with correlation metrics
    """
    # Convert to numpy for calculation
    distances = np.array(distances)
    perception_values = np.array(perception_values)

    # Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(distances, perception_values)

    # Spearman rank correlation
    spearman_corr, spearman_p = stats.spearmanr(distances, perception_values)

    # R² score (coefficient of determination)
    r2 = r2_score(perception_values, distances)

    return {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'r2_score': r2
    }


# def calculate_contrastive_loss(y_true, y_pred, triplet_mining, batch_strategy=BATCH_STRATEGY):
#     """
#     Calculate contrastive loss using the comparison matrices populated with count_normalized values.
#
#     This loss encourages embeddings to be positioned according to their similarity values from user studies,
#     using the count_normalized values as target similarities.
#
#     Args:
#         y_true: Labels of the batch (not used directly but kept for API consistency)
#         y_pred: Embeddings, tensor of shape (batch_size, embed_dim)
#         triplet_mining: TripletMining object containing comparison matrices
#         batch_strategy: Batch strategy to use (affects which comparisons are considered)
#
#     Returns:
#         losses: Contrastive loss tensor
#     """
#     # Calculate the pairwise distances between embeddings
#     # This also updates tensor_dists_class_neut internally
#     classes_distances = triplet_mining.calculate_distances(y_pred)
#
#     # Initialize loss tensors
#     left_right_loss = torch.zeros_like(classes_distances)
#     left_neut_loss = torch.zeros_like(classes_distances)
#     right_neut_loss = torch.zeros_like(classes_distances)
#
#     # Calculate contrastive loss for left-right comparisons
#     # Loss = (distance - target_similarity)²
#     # Where target_similarity is derived from count_normalized values
#
#     # Only consider pairs with valid comparisons (bool_matrix == 1)
#     # valid_left_right = triplet_mining.matrix_comparison_bool_left_right > 0
#     # if torch.any(valid_left_right):
#     #     # Scale distances to be in a similar range as the count_normalized values [0,1]
#     #     # This scaling factor might need tuning based on your specific data
#     #     scaling_factor = 0.1
#     #     scaled_distances = classes_distances * scaling_factor
#     #
#     #     # Compute the squared difference between scaled distances and target similarities
#     #     target_similarities = triplet_mining.matrix_comparison_values_left_right
#     #     diff_left_right = scaled_distances - target_similarities
#     #     left_right_loss = torch.square(diff_left_right) * valid_left_right
#
#     # Only consider pairs with valid comparisons (bool_matrix == 1)
#     valid_left_right = triplet_mining.matrix_comparison_bool_left_right > 0
#     if torch.any(valid_left_right) or not torch.any(valid_left_right):
#         # Scale distances to be in a similar range as the count_normalized values [0,1]
#         # This scaling factor might need tuning based on your specific data
#         scaling_factor = 0.1
#         scaled_distances = classes_distances
#
#         # Compute the loss when scaled distances exceed target similarities
#         target_similarities = triplet_mining.matrix_comparison_values_left_right
#
#         # Apply ReLU to only penalize when scaled_distances > target_similarities
#         # diff_left_right = torch.relu(scaled_distances - target_similarities)
#         diff_left_right = scaled_distances - target_similarities
#
#         # Apply this only to valid pairs
#         # left_right_loss = torch.square(diff_left_right) * valid_left_right
#         left_right_loss = torch.square(diff_left_right)
#
#     # Calculate contrastive loss for left-neutral comparisons
#     valid_left_neut = triplet_mining.matrix_comparison_bool_left_neut > 0
#     if torch.any(valid_left_neut) or not torch.any(valid_left_neut):
#         # Use tensor_dists_class_neut for left-neutral distances
#         # Need to reshape for proper broadcasting
#         left_neut_distances = triplet_mining.tensor_dists_class_neut.reshape(-1, 1)
#         print(f"shape of left_neut_distances: {left_neut_distances.shape}")
#
#         # Scale distances
#         scaled_distances = left_neut_distances * scaling_factor
#
#         # Compute squared difference
#         target_similarities = triplet_mining.matrix_comparison_values_left_neut
#         diff_left_neut = scaled_distances - target_similarities
#         left_neut_loss = torch.square(diff_left_neut) * valid_left_neut
#
#     # Calculate contrastive loss for right-neutral comparisons
#     valid_right_neut = triplet_mining.matrix_comparison_bool_right_neut > 0
#     if torch.any(valid_right_neut) or not torch.any(valid_right_neut):
#         # Use tensor_dists_class_neut for right-neutral distances
#         # Need to reshape for proper broadcasting
#         right_neut_distances = triplet_mining.tensor_dists_class_neut.reshape(1, -1)
#
#         # Scale distances
#         scaled_distances = right_neut_distances * scaling_factor
#
#         # Compute squared difference
#         target_similarities = triplet_mining.matrix_comparison_values_right_neut
#         diff_right_neut = scaled_distances - target_similarities
#         right_neut_loss = torch.square(diff_right_neut) * valid_right_neut
#
#     # Combine all loss components
#     # total_loss = left_right_loss + left_neut_loss + right_neut_loss
#
#     total_loss = left_right_loss
#
#     # Handle different batch strategies
#     if batch_strategy == BatchStrategy.HARD:
#         # Only use the hardest (largest) losses
#         max_loss_per_row = torch.max(total_loss, dim=1)[0]
#         max_loss_per_col = torch.max(total_loss, dim=0)[0]
#         losses = torch.max(torch.cat([max_loss_per_row, max_loss_per_col]))
#
#     elif batch_strategy == BatchStrategy.SEMI_HARD:
#         # Use semi-hard examples (losses that are positive but not too large)
#         # First, get all positive losses
#         positive_losses = torch.where(total_loss > 0, total_loss, torch.zeros_like(total_loss))
#
#         # Get median of positive losses
#         median_loss = torch.median(torch.where(positive_losses > 0, positive_losses,
#                                                torch.ones_like(positive_losses) * float('inf')))
#
#         # Use losses that are close to the median
#         semi_hard_margin = 0.2  # This is a hyperparameter to tune
#         semi_hard_mask = torch.abs(positive_losses - median_loss) < (median_loss * semi_hard_margin)
#
#         # Apply mask and get average
#         semi_hard_losses = positive_losses * semi_hard_mask
#         losses = torch.sum(semi_hard_losses) / (torch.sum(semi_hard_mask) + 1e-8)  # avoid division by zero
#
#     else:  # BatchStrategy.ALL
#         # Use all valid losses
#         # num_valid = torch.sum(
#         #     triplet_mining.matrix_comparison_bool_left_right +
#         #     triplet_mining.matrix_comparison_bool_left_neut +
#         #     triplet_mining.matrix_comparison_bool_right_neut
#         # )
#         # losses = torch.sum(total_loss) / (num_valid + 1e-8)  # avoid division by zero
#         losses = torch.mean(total_loss)
#
#     return losses
# # consider alternative loss function based on cosine similarity:
# # self.loss = tf.keras.losses.CosineSimilarity(axis=1)
# # ap_distance = self.loss(anchor, positive)
# # an_distance = self.loss(anchor, negative)
# # loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
# # def calculate_triplet_loss(y_true, y_pred, triplet_mining, batch_strategy=BATCH_STRATEGY):
# #     """Calculate 1D tensor of either squared L2 norm or L2 norm of differences between class embeddings and the
# #             neutral embedding of shape (embedding_size,), resulting in tensor of shape (batch_size,).
# #
# #                     Args:
# #                         y_true: supposed 'labels' of the batch (i.e., class indexes), tensor of size (batch_size,
# #                             ) where each element is singleton tensor of an integer in the range [1, 57]. Only needed
# #                             to the end of sorting the batch embeddings (i.e., y_pred)
# #                         y_true contains 1-57 repeated three times.
# #                         y_pred: embeddings, tensor of shape (batch_size, embed_dim)
# #                         triplet_mining: TripletMining object, contains the necessary alpha matrices information
# #                             for a given action type
# #                         batch_strategy: Enum value indicating the batch strategy to use for triplet mining
# #
# #                     Returns:
# #                         losses: tensor of shape (triplet_mining.num_states_drives, triplet_mining.num_states_drives)
# #             """
# #
# #     classes_distances = triplet_mining.calculate_distances(y_pred)
# #     # triplet_mining.calculate_class_neut_distances(y_pred)
# #
# #     # Reshape the 1D tensor into a 2D tensor with shape (1, 56) (will allow for broadcasting)
# #     # row_dists_class_neut = tf.reshape(triplet_mining.tensor_dists_class_neut,
# #     #                                   (1, triplet_mining.num_states_drives))
# #     # Reshape the 1D tensor dists_class_neut into a 2D tensor with shape (1, 56) for broadcasting
# #     row_dists_class_neut = triplet_mining.tensor_dists_class_neut.reshape(1, triplet_mining.num_states_drives)
# #
# #     # print(f"Custom_losses:calculate_triplet_loss: row_dists_class_neut shape: {row_dists_class_neut.shape}")
# #     # Reshape the 1D tensor into a 2D tensor of shape (56, 1) (will allow for broadcasting)
# #     # column_dists_class_neut = tf.reshape(triplet_mining.tensor_dists_class_neut,
# #     #                                      (triplet_mining.num_states_drives, 1))
# #     column_dists_class_neut = triplet_mining.tensor_dists_class_neut.reshape(triplet_mining.num_states_drives, 1)
# #
# #     # case 1: left and right are positives
# #     # subcase L = anchor
# #     diff_lr_ln = classes_distances - column_dists_class_neut
# #     # print(f"diff_lr_ln: {diff_lr_ln}")
# #
# #     assert diff_lr_ln.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
# #         f"Shape mismatch: expected ({triplet_mining.num_states_drives}, {triplet_mining.num_states_drives}), but got {diff_lr_ln.shape}"
# #
# #
# #     # consider only cases where diff_lr_ln > 0 (i.e., l_n - l_r > 0)
# #     if batch_strategy == BatchStrategy.HARD:
# #         # Ensure no negative distances
# #         diff_lr_ln = torch.clamp(diff_lr_ln, min=0.0)
# #         diff_lr_rl_alpha = diff_lr_ln + triplet_mining.matrix_alpha_left_right_right_left
# #         diff_lr_alpha = diff_lr_rl_alpha * triplet_mining.matrix_bool_left_right  # Element-wise multiplication
# #
# #         # Ensure no negative losses
# #         triplet_loss_L_R = diff_lr_alpha
# #
# #         # Assert no negative losses
# #         assert torch.all(triplet_loss_L_R >= 0), "Negative losses exist"
# #
# #     # consider only cases where diff_lr_ln < 0 yet triplet_loss_L_N > 0 (i.e., l_r - l_n < 0 and l_r - l_n + alpha > 0)
# #     elif batch_strategy == BatchStrategy.SEMI_HARD:
# #         # Replace tf.where with torch.where
# #         diff_lr_ln = torch.where(diff_lr_ln < 0, diff_lr_ln, torch.zeros_like(diff_lr_ln))
# #
# #         # Addition is the same in PyTorch
# #         diff_lr_rl_alpha = diff_lr_ln + triplet_mining.matrix_alpha_left_right_right_left
# #
# #         # Replace tf.multiply with element-wise multiplication
# #         diff_lr_alpha = diff_lr_rl_alpha * triplet_mining.matrix_bool_left_right
# #
# #         # Replace tf.maximum with torch.clamp
# #         triplet_loss_L_R = torch.clamp(diff_lr_alpha, min=0.0)
# #
# #     # ALL strategy for diff_lr_ln case
# #     else:
# #         # Addition is the same in PyTorch
# #         diff_lr_rl_alpha = diff_lr_ln + triplet_mining.matrix_alpha_left_right_right_left
# #
# #         # Replace tf.multiply with element-wise multiplication
# #         diff_lr_alpha = diff_lr_rl_alpha * triplet_mining.matrix_bool_left_right
# #
# #         # print(f"diff_lr_alpha: {diff_lr_alpha}")
# #
# #         # Replace tf.maximum with torch.clamp
# #         triplet_loss_L_R = torch.clamp(diff_lr_alpha, min=0.0)
# #
# #     # subcase R = anchor
# #     diff_rl_rn = classes_distances - row_dists_class_neut
# #     if batch_strategy == BatchStrategy.HARD:
# #         diff_rl_rn = torch.clamp(diff_rl_rn, min=0.0)
# #
# #         diff_rl_rn_alpha = diff_rl_rn + triplet_mining.matrix_alpha_left_right_right_left
# #
# #         diff_rl_alpha = diff_rl_rn_alpha * triplet_mining.matrix_bool_right_left
# #
# #         triplet_loss_R_L = diff_rl_alpha
# #
# #         # Replace TensorFlow assertion with PyTorch assertion
# #         # torch.all performs element-wise comparison and returns True if all elements are True
# #         assert torch.all(triplet_loss_R_L >= 0.0), "Negative losses exist"
# #
# #     elif batch_strategy == BatchStrategy.SEMI_HARD:
# #         diff_rl_rn = torch.where(diff_rl_rn < 0, diff_rl_rn, torch.zeros_like(diff_rl_rn))
# #
# #         diff_rl_rn_alpha = diff_rl_rn + triplet_mining.matrix_alpha_left_right_right_left
# #
# #         diff_rl_alpha = diff_rl_rn_alpha * triplet_mining.matrix_bool_right_left
# #
# #         triplet_loss_R_L = torch.clamp(diff_rl_alpha, min=0.0)
# #
# #     else:
# #         diff_rl_rn_alpha = diff_rl_rn + triplet_mining.matrix_alpha_left_right_right_left
# #
# #         diff_rl_alpha = diff_rl_rn_alpha * triplet_mining.matrix_bool_right_left
# #
# #         triplet_loss_R_L = torch.clamp(diff_rl_alpha, min=0.0)
# #
# #     # case 2: left and neutral are positives
# #     # L = anchor, Neutral = positive, R = negative
# #     diff_ln_lr = column_dists_class_neut - classes_distances
# #     # Assert that the shape of diff_ln_lr is correct
# #     assert diff_ln_lr.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
# #         f"Tensor diff_ln_lr has incorrect shape: {diff_ln_lr.shape}"
# #
# #     if batch_strategy == BatchStrategy.HARD:
# #         diff_ln_lr = torch.where(diff_ln_lr > 0, diff_ln_lr, torch.zeros_like(diff_ln_lr))
# #
# #         diff_ln_lr_alpha = diff_ln_lr + triplet_mining.matrix_alpha_left_neut_neut_left
# #
# #         diff_ln_lr_alpha = diff_ln_lr_alpha * triplet_mining.matrix_bool_left_neut
# #
# #         triplet_loss_L_N = diff_ln_lr_alpha
# #
# #         # Assert that all values in triplet_loss_L_N are non-negative
# #         assert torch.all(triplet_loss_L_N >= 0.0), "Negative losses exist"
# #
# #     # L = anchor, Neutral = positive, R = negative
# #     elif batch_strategy == BatchStrategy.SEMI_HARD:
# #         diff_ln_lr = torch.where(diff_ln_lr < 0, diff_ln_lr, torch.zeros_like(diff_ln_lr))
# #
# #         diff_ln_lr_alpha = diff_ln_lr + triplet_mining.matrix_alpha_left_neut_neut_left
# #
# #         diff_ln_lr_alpha = diff_ln_lr_alpha * triplet_mining.matrix_bool_left_neut
# #
# #         triplet_loss_L_N = torch.where(diff_ln_lr_alpha > 0, diff_ln_lr_alpha, torch.zeros_like(diff_ln_lr_alpha))
# #
# #     # L = anchor, Neutral = positive, R = negative
# #     else:
# #         diff_ln_lr_alpha = diff_ln_lr + triplet_mining.matrix_alpha_right_neut_neut_right
# #
# #         diff_ln_lr_alpha = diff_ln_lr_alpha * triplet_mining.matrix_bool_left_neut
# #
# #         triplet_loss_L_N = torch.where(diff_ln_lr_alpha > 0, diff_ln_lr_alpha, torch.zeros_like(diff_ln_lr_alpha))
# #
# #     diff_nl_nr = row_dists_class_neut - column_dists_class_neut
# #     assert diff_nl_nr.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
# #         f"Tensor diff_nl_nr has incorrect shape: {diff_nl_nr.shape}"
# #     # N = anchor, L = positive, R = negative
# #     if batch_strategy == BatchStrategy.HARD:
# #         diff_nl_nr = torch.where(diff_nl_nr > 0, diff_nl_nr, torch.zeros_like(diff_nl_nr))
# #
# #         diff_nl_nr_alpha = diff_nl_nr + triplet_mining.matrix_alpha_left_neut_neut_left
# #
# #         diff_nl_nr_alpha = diff_nl_nr_alpha * triplet_mining.matrix_bool_neut_left
# #
# #         triplet_loss_N_L = diff_nl_nr_alpha
# #
# #         # Assert that all values in triplet_loss_N_L are non-negative
# #         assert torch.all(triplet_loss_N_L >= 0.0), "Negative losses exist"
# #
# #     # N = anchor, L = positive, R = negative
# #     elif batch_strategy == BatchStrategy.SEMI_HARD:
# #         diff_nl_nr = torch.where(diff_nl_nr < 0, diff_nl_nr, torch.zeros_like(diff_nl_nr))
# #
# #         diff_nl_nr_alpha = diff_nl_nr + triplet_mining.matrix_alpha_left_neut_neut_left
# #
# #         diff_nl_nr_alpha = diff_nl_nr_alpha * triplet_mining.matrix_bool_neut_left
# #
# #         triplet_loss_N_L = torch.where(diff_nl_nr_alpha > 0, diff_nl_nr_alpha, torch.zeros_like(diff_nl_nr_alpha))
# #
# #     # N = anchor, L = positive, R = negative
# #     else:
# #         diff_nl_nr_alpha = diff_nl_nr + triplet_mining.matrix_alpha_left_neut_neut_left
# #
# #         diff_nl_nr_alpha = diff_nl_nr_alpha * triplet_mining.matrix_bool_neut_left
# #
# #         triplet_loss_N_L = torch.where(diff_nl_nr_alpha > 0, diff_nl_nr_alpha, torch.zeros_like(diff_nl_nr_alpha))
# #
# #     ### case 3: right and neutral are positives
# #     # R = anchor
# #     diff_rn_rl = row_dists_class_neut - torch.transpose(classes_distances, 0, 1)
# #     if batch_strategy == BatchStrategy.HARD:
# #         # # remove negative losses
# #         diff_rn_rl = torch.where(diff_rn_rl > 0, diff_rn_rl, torch.zeros_like(diff_rn_rl))
# #
# #         # Addition works the same way in PyTorch
# #         diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining.matrix_alpha_right_neut_neut_right
# #
# #         # Replace tf.multiply with element-wise multiplication
# #         diff_r_n_r_l_alpha = diff_r_n_r_l_alpha * triplet_mining.matrix_bool_right_neut
# #
# #         # Assign to triplet_loss_R_N
# #         triplet_loss_R_N = diff_r_n_r_l_alpha
# #
# #         # Assert that all values in triplet_loss_R_N are non-negative
# #         assert torch.all(triplet_loss_R_N >= 0.0), "Negative losses exist"
# #
# #     elif batch_strategy == BatchStrategy.SEMI_HARD:
# #         diff_rn_rl = torch.where(diff_rn_rl < 0, diff_rn_rl, torch.zeros_like(diff_rn_rl))
# #
# #         diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining.matrix_alpha_right_neut_neut_right
# #
# #         diff_r_n_r_l_alpha = diff_r_n_r_l_alpha * triplet_mining.matrix_bool_right_neut
# #
# #         triplet_loss_R_N = torch.where(diff_r_n_r_l_alpha > 0, diff_r_n_r_l_alpha, torch.zeros_like(diff_r_n_r_l_alpha))
# #
# #     else:  # BatchStrategy.ALL
# #         diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining.matrix_alpha_right_neut_neut_right
# #
# #         diff_r_n_r_l_alpha = diff_r_n_r_l_alpha * triplet_mining.matrix_bool_right_neut
# #
# #         triplet_loss_R_N = torch.where(diff_r_n_r_l_alpha > 0, diff_r_n_r_l_alpha, torch.zeros_like(diff_r_n_r_l_alpha))
# #
# #     # tf.debugging.assert_equal(triplet_loss_R_N, tf.maximum(triplet_loss_R_N, 0.0), message="Negative losses exist")
# #     # Assert that all values in triplet_loss_R_N are non-negative
# #     assert torch.all(triplet_loss_R_N >= 0.0), "Negative losses exist"
# #
# #     # N = anchor
# #     diff_nr_nl = column_dists_class_neut - row_dists_class_neut
# #     assert diff_nr_nl.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
# #         f"Tensor diff_nr_nl has incorrect shape: {diff_nr_nl.shape}"
# #     if batch_strategy == BatchStrategy.HARD:
# #         diff_nr_nl = torch.where(diff_nr_nl > 0, diff_nr_nl, torch.zeros_like(diff_nr_nl))
# #
# #         diff_nr_nl_alpha = diff_nr_nl + triplet_mining.matrix_alpha_right_neut_neut_right
# #
# #         diff_nr_nl_alpha = diff_nr_nl_alpha * triplet_mining.matrix_bool_neut_right
# #
# #         triplet_loss_N_R = diff_nr_nl_alpha
# #
# #         # Assert that all values in triplet_loss_N_R are non-negative
# #         assert torch.all(triplet_loss_N_R >= 0.0), "Negative losses exist"
# #
# #     elif batch_strategy == BatchStrategy.SEMI_HARD:
# #         diff_nr_nl_alpha = diff_nr_nl + triplet_mining.matrix_alpha_right_neut_neut_right
# #
# #         diff_nr_nl_alpha = diff_nr_nl_alpha * triplet_mining.matrix_bool_neut_right
# #
# #         triplet_loss_N_R = torch.where(diff_nr_nl_alpha > 0, diff_nr_nl_alpha, torch.zeros_like(diff_nr_nl_alpha))
# #
# #     else:
# #         diff_nr_nl_alpha = diff_nr_nl + triplet_mining.matrix_alpha_right_neut_neut_right
# #
# #         diff_nr_nl_alpha = diff_nr_nl_alpha * triplet_mining.matrix_bool_neut_right
# #
# #         triplet_loss_N_R = torch.where(diff_nr_nl_alpha > 0, diff_nr_nl_alpha, torch.zeros_like(diff_nr_nl_alpha))
# #
# #
# #     losses = (triplet_loss_L_R + triplet_loss_R_L + triplet_loss_L_N + triplet_loss_N_L + triplet_loss_R_N +
# #               triplet_loss_N_R)
# #     # losses = (triplet_loss_L_R + triplet_loss_R_L)
# #     # print(f"losses: {losses}")
# #
# #     # Assert that no negative losses exist
# #     assert torch.all(losses >= 0.0), "Negative losses exist"
# #
# #     # Assert that the shape of losses tensor is correct
# #     assert losses.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
# #         f"Comparisons loss tensor has incorrect shape: {losses.shape}"
# #     return losses
#
# def calculate_triplet_loss(y_true, y_pred, triplet_mining, batch_strategy=BATCH_STRATEGY):
#     """Calculate triplet loss using left-right embeddings across all preference cases.
#
#     Modified to focus on learning embeddings between left and right classes, regardless of
#     which pair was most preferable in the original human comparisons.
#
#     Args:
#         y_true: Labels of the batch (class indexes), tensor of size (batch_size,)
#         y_pred: Embeddings, tensor of shape (batch_size, embed_dim)
#         triplet_mining: TripletMining object with alpha matrices
#         batch_strategy: Enum value indicating the batch strategy to use
#
#     Returns:
#         losses: Tensor of shape (triplet_mining.num_states_drives, triplet_mining.num_states_drives)
#     """
#     # Calculate distances between embeddings
#     classes_distances = triplet_mining.calculate_distances(y_pred)
#
#     # Reshape class-neutral distances for broadcasting
#     row_dists_class_neut = triplet_mining.tensor_dists_class_neut.reshape(1, triplet_mining.num_states_drives)
#     column_dists_class_neut = triplet_mining.tensor_dists_class_neut.reshape(triplet_mining.num_states_drives, 1)
#
#     # -------------------------------------------------------------------------
#     # Case 1: Original Left-Right preference case
#     # -------------------------------------------------------------------------
#     # subcase L = anchor
#     diff_lr_ln = classes_distances - column_dists_class_neut
#
#     assert diff_lr_ln.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
#         f"Shape mismatch: expected ({triplet_mining.num_states_drives}, {triplet_mining.num_states_drives}), but got {diff_lr_ln.shape}"
#
#     # Process based on batch strategy
#     if batch_strategy == BatchStrategy.HARD:
#         diff_lr_ln = torch.clamp(diff_lr_ln, min=0.0)
#
#     elif batch_strategy == BatchStrategy.SEMI_HARD:
#         diff_lr_ln = torch.where(diff_lr_ln < 0, diff_lr_ln, torch.zeros_like(diff_lr_ln))
#
#     # Apply alpha values and boolean masks
#     diff_lr_rl_alpha = diff_lr_ln + triplet_mining.matrix_alpha_left_right_right_left
#     diff_lr_alpha = diff_lr_rl_alpha * triplet_mining.matrix_bool_left_right
#     triplet_loss_L_R = torch.clamp(diff_lr_alpha, min=0.0)
#
#     # subcase R = anchor
#     diff_rl_rn = classes_distances - row_dists_class_neut
#
#     if batch_strategy == BatchStrategy.HARD:
#         diff_rl_rn = torch.clamp(diff_rl_rn, min=0.0)
#
#     elif batch_strategy == BatchStrategy.SEMI_HARD:
#         diff_rl_rn = torch.where(diff_rl_rn < 0, diff_rl_rn, torch.zeros_like(diff_rl_rn))
#
#     diff_rl_rn_alpha = diff_rl_rn + triplet_mining.matrix_alpha_left_right_right_left
#     diff_rl_alpha = diff_rl_rn_alpha * triplet_mining.matrix_bool_right_left
#     triplet_loss_R_L = torch.clamp(diff_rl_alpha, min=0.0)
#
#     # -------------------------------------------------------------------------
#     # Case 2: Left-Right alphas from Left-Neutral preference case
#     # -------------------------------------------------------------------------
#     # The same diff_lr_ln and diff_rl_rn calculations can be reused
#     # but with matrix_alpha_left_neut_neut_left which now contains left-right alphas
#
#     # L = anchor (from Left-Neutral preference case)
#     ln_diff_lr_alpha = diff_lr_ln + triplet_mining.matrix_alpha_left_neut_neut_left
#     ln_diff_lr_alpha = ln_diff_lr_alpha * triplet_mining.matrix_bool_left_neut
#     triplet_loss_L_R_from_LN = torch.clamp(ln_diff_lr_alpha, min=0.0)
#
#     # R = anchor (from Left-Neutral preference case)
#     ln_diff_rl_alpha = diff_rl_rn + triplet_mining.matrix_alpha_left_neut_neut_left
#     ln_diff_rl_alpha = ln_diff_rl_alpha * triplet_mining.matrix_bool_neut_left
#     triplet_loss_R_L_from_LN = torch.clamp(ln_diff_rl_alpha, min=0.0)
#
#     # -------------------------------------------------------------------------
#     # Case 3: Left-Right alphas from Right-Neutral preference case
#     # -------------------------------------------------------------------------
#     # L = anchor (from Right-Neutral preference case)
#     rn_diff_lr_alpha = diff_lr_ln + triplet_mining.matrix_alpha_right_neut_neut_right
#     rn_diff_lr_alpha = rn_diff_lr_alpha * triplet_mining.matrix_bool_right_neut
#     triplet_loss_L_R_from_RN = torch.clamp(rn_diff_lr_alpha, min=0.0)
#
#     # R = anchor (from Right-Neutral preference case)
#     rn_diff_rl_alpha = diff_rl_rn + triplet_mining.matrix_alpha_right_neut_neut_right
#     rn_diff_rl_alpha = rn_diff_rl_alpha * triplet_mining.matrix_bool_neut_right
#     triplet_loss_R_L_from_RN = torch.clamp(rn_diff_rl_alpha, min=0.0)
#
#     # -------------------------------------------------------------------------
#     # Combine all losses
#     # -------------------------------------------------------------------------
#     # This now focuses solely on left-right relationships across all preference cases
#     losses = (triplet_loss_L_R + triplet_loss_R_L +
#               triplet_loss_L_R_from_LN + triplet_loss_R_L_from_LN +
#               triplet_loss_L_R_from_RN + triplet_loss_R_L_from_RN)
#
#     # Validate the final losses
#     assert torch.all(losses >= 0.0), "Negative losses exist"
#     assert losses.shape == (triplet_mining.num_states_drives, triplet_mining.num_states_drives), \
#         f"Comparisons loss tensor has incorrect shape: {losses.shape}"
#
#     return losses
#
#
# def calculate_integrated_perception_loss(y_true, y_pred, triplet_mining, adaptive_distance_module=None,
#                                          batch_strategy=BATCH_STRATEGY):
#     """
#     Integrated perception-aligned loss function that combines:
#     1. Direct correlation optimization
#     2. Adaptive distance metric (if provided)
#     3. Original triplet constraints (as regularization)
#     4. Ranking and ordinal relationship preservation
#
#     This function is designed to directly optimize for alignment with human perception
#     while maintaining the structural properties of the embedding space.
#
#     Args:
#         y_true: Labels of the batch (class indexes), tensor of size (batch_size,)
#         y_pred: Embeddings, tensor of shape (batch_size, embed_dim)
#         triplet_mining: TripletMining object with comparison matrices
#         adaptive_distance_module: Optional AdaptiveDistanceModule instance
#         batch_strategy: Enum value indicating the batch strategy to use
#
#     Returns:
#         loss: Scalar tensor containing the combined loss
#     """
#     # Calculate distances between embeddings
#     if adaptive_distance_module is not None:
#         # Use adaptive distance metric
#         classes_distances = adaptive_distance_module.pairwise_distances(y_pred)
#     else:
#         # Use standard Euclidean distance
#         classes_distances = triplet_mining.calculate_distances(y_pred)
#
#     # Get the human perception data (inverse of comparison values)
#     inverse_comparison_values = torch.ones_like(triplet_mining.matrix_comparison_values_left_right)
#     mask = triplet_mining.matrix_comparison_bool_left_right > 0
#     inverse_comparison_values[mask] = 1.0 - triplet_mining.matrix_comparison_values_left_right[mask]
#
#     # Get valid comparison mask (where we have human data)
#     valid_comparisons = triplet_mining.matrix_comparison_bool_left_right > 0
#
#     # If no valid comparisons, fall back to standard triplet loss
#     if not torch.any(valid_comparisons):
#         return calculate_triplet_loss(y_true, y_pred, triplet_mining, batch_strategy)
#
#     # Extract valid distances and perception values
#     valid_distances = classes_distances[valid_comparisons]
#     valid_perception = inverse_comparison_values[valid_comparisons]
#
#     # Normalize distances to [0,1] range
#     if len(valid_distances) > 0:
#         min_dist = torch.min(valid_distances)
#         max_dist = torch.max(valid_distances)
#
#         if max_dist > min_dist:
#             normalized_distances = (valid_distances - min_dist) / (max_dist - min_dist)
#         else:
#             normalized_distances = torch.zeros_like(valid_distances)
#     else:
#         normalized_distances = valid_distances
#
#     # Component 1: Direct MSE loss to align distances with perception
#     alignment_loss = torch.mean(torch.square(normalized_distances - valid_perception))
#
#     # Component 2: Pearson correlation loss
#     dist_mean = torch.mean(normalized_distances)
#     perc_mean = torch.mean(valid_perception)
#
#     dist_centered = normalized_distances - dist_mean
#     perc_centered = valid_perception - perc_mean
#
#     numerator = torch.sum(dist_centered * perc_centered)
#     denominator = torch.sqrt(torch.sum(dist_centered ** 2) * torch.sum(perc_centered ** 2) + 1e-8)
#     correlation = numerator / denominator
#     correlation_loss = -correlation  # Negative to maximize correlation
#
#     # Component 3: Ranking preservation loss
#     n = valid_distances.shape[0]
#     rank_loss = torch.tensor(0.0, device=valid_distances.device)
#
#     if n > 1:
#         # Get all pairwise combinations
#         i_indices, j_indices = torch.triu_indices(n, n, offset=1)
#
#         # Extract pairs
#         dist_i = normalized_distances[i_indices]
#         dist_j = normalized_distances[j_indices]
#         perc_i = valid_perception[i_indices]
#         perc_j = valid_perception[j_indices]
#
#         # Determine desired ordering
#         should_be_greater = perc_i > perc_j + 0.05  # Add small threshold to avoid noise
#         should_be_less = perc_i < perc_j - 0.05
#
#         # Calculate ranking violations
#         margin = 0.1
#
#         # Where perception i > j, ensure distance i > j + margin
#         greater_violations = torch.relu(margin - (dist_i - dist_j))
#         greater_loss = torch.mean(greater_violations[should_be_greater]) if torch.any(should_be_greater) else 0.0
#
#         # Where perception i < j, ensure distance i < j - margin
#         less_violations = torch.relu(margin - (dist_j - dist_i))
#         less_loss = torch.mean(less_violations[should_be_less]) if torch.any(should_be_less) else 0.0
#
#         rank_loss = greater_loss + less_loss
#
#     # Component 4: Penalize disagreement with left-right matrix values
#     # This ensures the learned embedding respects the alphas from our modified triplet mining
#
#     # Modified left-right relationship loss (uses all three matrices that now contain left-right relationships)
#     # for different preference cases
#
#     # Reshape class-neutral distances for broadcasting
#     row_dists_class_neut = triplet_mining.tensor_dists_class_neut.reshape(1, triplet_mining.num_states_drives)
#     column_dists_class_neut = triplet_mining.tensor_dists_class_neut.reshape(triplet_mining.num_states_drives, 1)
#
#     # Diff terms for left/right as anchor
#     diff_lr_ln = classes_distances - column_dists_class_neut
#     diff_rl_rn = classes_distances - row_dists_class_neut
#
#     # Get combined alpha matrices and masks for all preference cases
#     combined_alphas = (
#             triplet_mining.matrix_alpha_left_right_right_left +
#             triplet_mining.matrix_alpha_left_neut_neut_left +
#             triplet_mining.matrix_alpha_right_neut_neut_right
#     )
#
#     combined_mask = (
#                             triplet_mining.matrix_bool_left_right +
#                             triplet_mining.matrix_bool_right_left +
#                             triplet_mining.matrix_bool_left_neut +
#                             triplet_mining.matrix_bool_neut_left +
#                             triplet_mining.matrix_bool_right_neut +
#                             triplet_mining.matrix_bool_neut_right
#                     ) > 0
#
#     # Process diff terms with combined alphas and masks
#     lr_loss = torch.where(
#         combined_mask,
#         torch.relu(diff_lr_ln + combined_alphas),
#         torch.zeros_like(diff_lr_ln)
#     )
#
#     rl_loss = torch.where(
#         combined_mask,
#         torch.relu(diff_rl_rn + combined_alphas),
#         torch.zeros_like(diff_rl_rn)
#     )
#
#     # Aggregate structure loss
#     structure_loss = torch.mean(lr_loss + rl_loss)
#
#     # Component 5: Instance contrast loss to ensure neutral is different from all classes
#     # This prevents collapse of the embedding space
#
#     # Get neutral embedding distances to all class embeddings
#     neutral_distances = triplet_mining.tensor_dists_class_neut
#
#     # Define minimum distance that neutral should be from any class
#     min_neutral_distance = 0.5
#
#     # Calculate violation for each class
#     neutral_violations = torch.relu(min_neutral_distance - neutral_distances)
#
#     # Compute average violation
#     neutral_loss = torch.mean(neutral_violations)
#
#     # Weight the different loss components (can be tuned)
#     alignment_weight = 2.0  # Direct alignment with perception
#     correlation_weight = 1.0  # Correlation maximization
#     rank_weight = 2.0  # Ordinal relationship preservation
#     structure_weight = 0.5  # Embedding structure preservation
#     neutral_weight = 0.2  # Neutral contrast term
#
#     # Combine all loss components
#     total_loss = (
#             alignment_weight * alignment_loss +
#             correlation_weight * correlation_loss +
#             rank_weight * rank_loss +
#             structure_weight * structure_loss +
#             neutral_weight * neutral_loss
#     )
#
#     # Log loss components for debugging (comment out in production)
#     # print(f"Alignment: {alignment_loss.item():.4f}, Correlation: {correlation_loss.item():.4f}, "
#     #      f"Rank: {rank_loss.item():.4f}, Structure: {structure_loss.item():.4f}, Neutral: {neutral_loss.item():.4f}")
#
#     return total_loss
#
#
# def create_batch_integrated_loss(triplet_mining_modules, adaptive_distance_module=None, module_start_indices=None,
#                                  module_sizes=None):
#     """
#     Create a batch loss function using the integrated perception-aligned loss.
#
#     Args:
#         triplet_mining_modules: List of TripletMining objects
#         adaptive_distance_module: Optional AdaptiveDistanceModule instance
#         module_start_indices: List of starting indices for each module in the batch
#         module_sizes: List of sizes for each module
#     """
#
#     def batch_integrated_loss(y_true, y_pred):
#         # Calculate overall loss across all modules
#         overall_loss = torch.tensor(0.0, device=y_pred.device)
#         valid_modules = 0
#
#         for i, triplet_mining in enumerate(triplet_mining_modules):
#             # Determine batch slice for this module
#             if module_start_indices is not None and module_sizes is not None:
#                 start_idx = module_start_indices[i]
#                 end_idx = start_idx + module_sizes[i]
#             else:
#                 # Fallback to original approach
#                 start_idx = i * triplet_mining.batch_size
#                 end_idx = (i + 1) * triplet_mining.batch_size
#
#             # Ensure indices are within bounds
#             end_idx = min(end_idx, y_true.shape[0])
#
#             # Skip if we don't have enough data
#             if end_idx - start_idx <= 1:
#                 continue
#
#             # Extract the portion of the batch for this module
#             y_true_module = y_true[start_idx:end_idx]
#             y_pred_module = y_pred[start_idx:end_idx]
#
#             # Calculate integrated loss for this module
#             try:
#                 module_loss = calculate_integrated_perception_loss(
#                     y_true_module,
#                     y_pred_module,
#                     triplet_mining,
#                     adaptive_distance_module
#                 )
#                 overall_loss += module_loss
#                 valid_modules += 1
#             except Exception as e:
#                 print(f"Error in loss calculation for module {i}: {e}")
#                 continue
#
#         # Normalize by number of valid modules
#         if valid_modules > 0:
#             overall_loss = overall_loss / valid_modules
#
#         return overall_loss
#
#     return batch_integrated_loss
#
#
# def create_batch_triplet_loss(triplet_mining_modules, module_start_indices=None, module_sizes=None):
#     """
#     Create a batch triplet loss function for use with multiple triplet mining modules.
#
#     Args:
#         triplet_mining_modules: List of TripletMining objects
#         module_start_indices: List of starting indices for each module in the batch
#         module_sizes: List of sizes for each module
#     """
#
#     def batch_triplet_loss(y_true, y_pred):
#         # ... existing flattening and sorting ...
#
#         # Calculate overall triplet loss across all modules
#         overall_triplet_loss = torch.tensor(0.0, device=y_pred.device)
#         valid_modules = 0
#
#         for i, triplet_mining in enumerate(triplet_mining_modules):
#             # Determine batch slice for this module
#             if module_start_indices is not None and module_sizes is not None:
#                 start_idx = module_start_indices[i]
#                 end_idx = start_idx + module_sizes[i]
#             else:
#                 # Fallback to original approach
#                 start_idx = i * triplet_mining.batch_size
#                 end_idx = (i + 1) * triplet_mining.batch_size
#
#             # Ensure indices are within bounds
#             end_idx = min(end_idx, y_true.shape[0])
#
#             # Skip if we don't have enough data
#             if end_idx - start_idx <= 1:
#                 continue
#
#             # Extract the portion of the batch for this module
#             y_true_module = y_true[start_idx:end_idx]
#             y_pred_module = y_pred[start_idx:end_idx]
#
#             # Calculate triplet losses for this module
#             try:
#                 triplet_losses = calculate_triplet_loss(y_true_module, y_pred_module, triplet_mining)
#                 # triplet_losses = calculate_contrastive_loss(y_true_module, y_pred_module, triplet_mining)
#                 triplet_loss = torch.mean(triplet_losses)
#                 overall_triplet_loss += triplet_loss
#                 valid_modules += 1
#             except Exception as e:
#                 print(f"Error in triplet loss calculation for module {i}: {e}")
#                 continue
#
#             # Cross-module comparisons
#             # for j, other_triplet_mining in enumerate(triplet_mining_modules):
#             #     if i != j:
#             #         # Get other module's data
#             #         if module_start_indices is not None and module_sizes is not None:
#             #             other_start_idx = module_start_indices[j]
#             #             other_end_idx = other_start_idx + module_sizes[j]
#             #         else:
#             #             other_start_idx = j * other_triplet_mining.batch_size
#             #             other_end_idx = (j + 1) * other_triplet_mining.batch_size
#             #
#             #         other_end_idx = min(other_end_idx, y_true.shape[0])
#             #
#             #         if other_end_idx - other_start_idx <= 0:
#             #             continue
#             #
#             #         y_pred_other_module = y_pred[other_start_idx:other_end_idx]
#             #
#             #         # Intra-module (anchor-positive) pairwise distances
#             #         intra_module_distances = torch.norm(
#             #             y_pred_module.unsqueeze(1) - y_pred_module.unsqueeze(0), dim=-1
#             #         )
#             #
#             #         # Inter-module (anchor-negative) distances
#             #         inter_module_distances = torch.norm(
#             #             y_pred_module.unsqueeze(1) - y_pred_other_module.unsqueeze(0), dim=-1
#             #         )
#             #
#             #         # For every anchor-positive pair, ensure distance to negatives is greater than to positives + margin
#             #         loss_term = torch.clamp(
#             #             intra_module_distances + 1 - torch.min(inter_module_distances, dim=1, keepdim=True)[0],
#             #             min=0
#             #         )
#             #         # print(f"Custom_losses:create_batch_triplet_loss: module {i} vs. module {j} loss_term: {loss_term}")
#             #
#             #         overall_triplet_loss += torch.mean(loss_term)
#
#             # for j, other_triplet_mining in enumerate(triplet_mining_modules):
#             #     if i != j:
#             #         # Get other module's data
#             #         if module_start_indices is not None and module_sizes is not None:
#             #             other_start_idx = module_start_indices[j]
#             #             other_end_idx = other_start_idx + module_sizes[j]
#             #         else:
#             #             other_start_idx = j * other_triplet_mining.batch_size
#             #             other_end_idx = (j + 1) * other_triplet_mining.batch_size
#             #
#             #         other_end_idx = min(other_end_idx, y_true.shape[0])
#             #
#             #         if other_end_idx - other_start_idx <= 0:
#             #             continue
#             #
#             #         y_pred_other_module = y_pred[other_start_idx:other_end_idx]
#             #
#             #         # Intra-module distances for current module
#             #         intra_module_distances_A = torch.norm(
#             #             y_pred_module.unsqueeze(1) - y_pred_module.unsqueeze(0), dim=-1
#             #         )
#             #         mask_A = torch.eye(intra_module_distances_A.shape[0], device=intra_module_distances_A.device)
#             #         masked_intra_module_distances_A = intra_module_distances_A.masked_fill(mask_A == 1, float('inf'))
#             #
#             #         # Intra-module distances for other module
#             #         intra_module_distances_B = torch.norm(
#             #             y_pred_other_module.unsqueeze(1) - y_pred_other_module.unsqueeze(0), dim=-1
#             #         )
#             #         mask_B = torch.eye(intra_module_distances_B.shape[0], device=intra_module_distances_B.device)
#             #         masked_intra_module_distances_B = intra_module_distances_B.masked_fill(mask_B == 1, float('inf'))
#             #
#             #         # Inter-module distances
#             #         inter_module_distances = torch.norm(
#             #             y_pred_module.unsqueeze(1) - y_pred_other_module.unsqueeze(0), dim=-1
#             #         )
#             #
#             #         # Penalties from module A's perspective
#             #         penalties_A = torch.zeros_like(inter_module_distances)
#             #
#             #         # Calculate penalties for each element in module A
#             #         for a_idx in range(y_pred_module.shape[0]):
#             #             # Get this element's distances to all elements in module B
#             #             inter_dists = inter_module_distances[a_idx]  # Shape: [module_B_size]
#             #
#             #             # Get this element's distances to all elements in its own module
#             #             intra_dists = masked_intra_module_distances_A[a_idx]  # Shape: [module_A_size]
#             #
#             #             # For each element in module B
#             #             for b_idx in range(y_pred_other_module.shape[0]):
#             #                 inter_dist = inter_dists[b_idx]
#             #
#             #                 # Find violations where inter_dist < intra_dist
#             #                 violations = torch.relu(intra_dists - inter_dist)
#             #
#             #                 # Sum the violations
#             #                 penalties_A[a_idx, b_idx] = torch.sum(violations)
#             #
#             #         # Penalties from module B's perspective
#             #         penalties_B = torch.zeros_like(inter_module_distances)
#             #
#             #         # Calculate penalties for each element in module B
#             #         for b_idx in range(y_pred_other_module.shape[0]):
#             #             # Get this element's distances to all elements in module A
#             #             inter_dists = inter_module_distances[:, b_idx]  # Shape: [module_A_size]
#             #
#             #             # Get this element's distances to all elements in its own module
#             #             intra_dists = masked_intra_module_distances_B[b_idx]  # Shape: [module_B_size]
#             #
#             #             # For each element in module A
#             #             for a_idx in range(y_pred_module.shape[0]):
#             #                 inter_dist = inter_dists[a_idx]
#             #
#             #                 # Find violations where inter_dist < intra_dist
#             #                 violations = torch.relu(intra_dists - inter_dist)
#             #
#             #                 # Sum the violations
#             #                 penalties_B[a_idx, b_idx] = torch.sum(violations)
#             #
#             #         # Total penalties
#             #         penalties = penalties_A + penalties_B
#             #
#             #         # Add penalties to the overall loss
#             #         overall_triplet_loss += torch.sum(penalties)
#
#         # Normalize by number of valid modules
#         if valid_modules > 0:
#             overall_triplet_loss = overall_triplet_loss / valid_modules
#         else:
#             return overall_triplet_loss + 1e-8
#
#         return overall_triplet_loss
#
#     return batch_triplet_loss
