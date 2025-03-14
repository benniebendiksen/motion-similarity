from keras import layers

"""Define functions to create the triplet loss with online triplet mining."""

import tensorflow as tf


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss

def build_model(is_training, images, params):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(params.num_channels, 3, padding='same', input_shape=images.shape[1:]))
    if params.use_batch_norm:
        model.add(layers.BatchNormalization(momentum=params.bn_momentum))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(params.num_channels * 2, 3, padding='same'))
    if params.use_batch_norm:
        model.add(layers.BatchNormalization(momentum=params.bn_momentum))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2, 2))

    assert model.layers[-1].output_shape[1:] == [7, 7, params.num_channels * 2]

    model.add(layers.Flatten())

    model.add(layers.Dense(params.embedding_size))

    return model


def triplet_loss(labels, embeddings, params):
    if params.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=params.margin, squared=params.squared)
    elif params.triplet_strategy == "batch_semi_hard":
        pass
    elif params.triplet_strategy == "batch_combined_hard_semi_hard":
        pass
        raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))
    return loss


def train(model, train_dataset, params):
    optimizer = tf.keras.optimizers.Adam(params.learning_rate)
    embedding_mean_norm = tf.keras.metrics.Mean(name='embedding_mean_norm')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            embeddings = model(images, training=True)
            loss = triplet_loss(labels, embeddings, params)
            embedding_mean_norm.update_state(tf.reduce_mean(tf.norm(embeddings, axis=1)))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(params.num_epochs):
        for images, labels in train_dataset:
            loss = train_step(images, labels)
            tf.print("Epoch:", epoch + 1, "Loss:", loss)

    tf.print("Mean Embedding Norm:", embedding_mean_norm.result())


def test(model, test_dataset):
    accuracy = tf.keras.metrics.Accuracy()

    @tf.function
    def test_step(images, labels):
        embeddings = model(images, training=False)
        loss = triplet_loss(labels, embeddings, params)
        predicted_labels = tf.argmax(embeddings, axis=1)
        accuracy.update_state(labels, predicted_labels)
        return loss

    for images, labels in test_dataset:
        loss = test_step(images, labels)
        tf.print("Test Loss:", loss)

    tf.print("Accuracy:", accuracy.result())


if __name__ == '__main__':
    tf.random.set_seed(230)
    tf.keras.backend.clear_session()

    params = {
        "learning_rate": 1e-4,
        "batch_size": 64,
        "num_epochs": 200,

        "num_channels": 32,
        "use_batch_norm": False,
        "bn_momentum": 0.9,
        "embedding_size": 64,
        "triplet_strategy": "batch_semi_hard",
        "squared": False,

        "exemplar_feature_size": 87,
        "exemplar_frame_count": 100,
        "num_labels": 57,
        "train_size": 50000,
        "eval_size": 10000,

        "num_parallel_calls": 4,
        "save_summary_steps": 50
    }


    # Convert the dictionary to a namespace object
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


    params = Namespace(**params)

    # Create the model
    model = build_model(params)

    # Type tf.data.Dataset
    train_dataset = None#TODO
    test_dataset = None#TODO

    # Train the model
    train(model, train_dataset, params)

    # Evaluate the model on the test set
    test(model, test_dataset)