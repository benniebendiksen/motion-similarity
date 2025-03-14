# from keras.src.callbacks.model_checkpoint import ModelCheckpoint
# from keras.src.layers import Conv2D
# from keras.src.layers import BatchNormalization
# from keras.src.layers import Dense
# from keras.src.layers.pooling.max_pooling2d import MaxPooling2D
# from keras.src.layers import Dropout
# from keras.src.layers import Flatten
# from keras.src.optimizers import Adam
# from keras.src.models import Sequential

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import networks.custom_losses as custom_losses
from keras import callbacks
import logging
import os
import time

logging.basicConfig(level=logging.DEBUG,
                    filename=os.path.basename(__file__) + '.log',
                    format="{asctime} [{levelname:8}] {process} {thread} {module}: {message}",
                    style="{")
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)


class TrainingLogger(callbacks.Callback):
    def __init__(self, model_name):
        super().__init__()
        self.start_time = None
        self.logger = logging.getLogger(model_name)
        handler = logging.FileHandler(f'training_{model_name}.log')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.start_time
        loss = logs.get('loss', 'N/A')
        self.logger.info(f"Epoch {epoch + 1}: Durée = {duration:.2f}s, Perte = {loss}")


# Model variants
class SimilarityNetworkV0(nn.Module):
    def __init__(self, input_shape, embedding_size):
        super(SimilarityNetworkV0, self).__init__()

        # Input shape should be (batch_size, channels, height, width)
        # Original input was (batch_size, height, width, channels)

        # Layer 1: Conv -> Dropout -> BatchNorm
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.2)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2: Conv -> Dropout -> BatchNorm
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.dropout2 = nn.Dropout(0.2)
        self.batch_norm_2 = nn.BatchNorm2d(64)

        # Layer 3: Conv -> BatchNorm
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Trace size changes after convolutions and pooling
        # Start with input dimensions
        height, width = input_shape[0], input_shape[1]

        # After first conv (no padding): (height-2, width-2, 32)
        height, width = height - 2, width - 2

        # After first pooling: (height/2, width/2, 32)
        height, width = height // 2, width // 2

        # After second conv (no padding): (height-2, width-2, 64)
        height, width = height - 2, width - 2

        # After third conv (no padding): (height-2, width-2, 128)
        height, width = height - 2, width - 2

        # After second pooling: (height/2, width/2, 128)
        height, width = height // 2, width // 2

        self.fc_input_size = height * width * 128

        # Final dense layer
        self.fc = nn.Linear(self.fc_input_size, embedding_size)
        self.dropout3 = nn.Dropout(0.2)

    def forward(self, x):
        # Reshape input: (batch, height, width, channels) -> (batch, channels, height, width)
        # PyTorch expects channels first
        x = x.permute(0, 3, 1, 2)

        # Layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.batch_norm_1(x)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.batch_norm_2(x)

        # Layer 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.pool2(x)

        # Flatten
        x = x.view(-1, self.fc_input_size)

        # Final dense layer
        x = self.fc(x)
        x = self.dropout3(x)

        return x


# Main class that integrates the model variants
class SimilarityNetwork:
    """
    The SimilarityNetwork class defines our own triplet similarity network.

    This class inherits from the `Utilities` class and is used to build, compile, and train a similarity learning model.

    Args:
        train_loader: The PyTorch DataLoader for the training dataset.
        validation_loader: The PyTorch DataLoader for the validation dataset.
        test_loader: The PyTorch DataLoader for the test dataset.
        checkpoint_root_dir: The directory where model checkpoints will be saved.
        triplet_modules: A list of TripletMining modules.
        architecture_variant: The architecture variant to use (0, 1, 2, 3, 4).
        config: The configuration object.
    """

    def __init__(self, train_loader, validation_loader, test_loader, checkpoint_root_dir, triplet_modules,
                 architecture_variant, config):
        super().__init__()
        self.config = config
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.exemplar_dim = train_loader.exemplar_dim
        print(f"SIM NETWORK CLASS: exemplar dim: {self.exemplar_dim}")
        self.triplet_modules = triplet_modules
        self.architecture_variant = architecture_variant
        self.checkpoint_dir = checkpoint_root_dir
        self.embedding_size = self.config.embedding_size

        # Use GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Setup logger
        self.logger = TrainingLogger(f"variant_{self.architecture_variant}")

        # Build the model
        self.build_model()

    def build_model(self):
        input_shape = (self.exemplar_dim[0], self.exemplar_dim[1])

        if self.architecture_variant == 0:
            self.network = SimilarityNetworkV0(input_shape, self.embedding_size)
        elif self.architecture_variant == 1:
            # For simplicity, reusing variant 0
            self.network = SimilarityNetworkV0(input_shape, self.embedding_size)
        elif self.architecture_variant in [2, 3, 4]:
            # For simplicity, reusing variant 0
            self.network = SimilarityNetworkV0(input_shape, self.embedding_size)

        # Move model to device
        self.network = self.network.to(self.device)

        # Print model summary
        print(self.network)

        # Setup loss function and optimizer
        self.criterion = custom_losses.create_batch_triplet_loss(self.triplet_modules)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.0001, betas=(0.5, 0.999))

    def save_checkpoint(self, epoch=None):
        """Save model checkpoint"""
        if epoch:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{self.architecture_variant}_similarity_model_weights_epoch_{epoch:03d}.pt"
            )
        else:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{self.architecture_variant}_similarity_model_weights.pt"
            )

        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

        print(f"Model weights saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def run_model_training(self):
        """
        Train the neural network on the training dataset.
        """
        best_loss = float('inf')

        for epoch in range(self.config.n_similarity_epochs):
            self.logger.on_epoch_begin(epoch)

            # Training phase
            self.network.train()
            running_loss = 0.0
            batch_count = 0

            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.network(inputs)

                # Calculate loss
                loss = self.criterion(labels, outputs)
                print(f"Batch Loss: {loss.item()}")

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                batch_count += 1

            # Calculate average loss for this epoch
            epoch_loss = running_loss / batch_count

            # Validation phase
            self.network.eval()
            val_loss = 0.0
            val_batch_count = 0

            with torch.no_grad():
                for inputs, labels in self.validation_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.network(inputs)
                    loss = self.criterion(labels, outputs)

                    val_loss += loss.item()
                    val_batch_count += 1

            # Calculate average validation loss
            val_epoch_loss = val_loss / val_batch_count if val_batch_count > 0 else 0

            # Log the epoch results
            # self.logger.on_epoch_end(epoch, epoch_loss)
            print(f"Epoch {epoch + 1}: Training Loss = {epoch_loss:.4f}, Validation Loss = {val_epoch_loss:.4f}")

            # Save checkpoint if this is the best model so far
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                self.save_checkpoint(epoch + 1)

        # Save final model
        self.save_checkpoint()

    def evaluate(self):
        """
        Evaluate the model on the test dataset.
        """
        self.network.eval()
        test_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.network(inputs)
                loss = self.criterion(labels, outputs)

                test_loss += loss.item()
                batch_count += 1

        # Calculate average test loss
        avg_test_loss = test_loss / batch_count if batch_count > 0 else 0
        print(f"Test Loss: {avg_test_loss:.4f}")

        return avg_test_loss


# keras version
# class SimilarityNetwork_2:
#     """
#     The SimilarityNetwork class defines our own triplet similarity network.
#
#     Args:
#         train_loader: The data loader for the training dataset.
#         validation_loader: The data loader for the validation dataset.
#         test_loader: The data loader for the test dataset.
#         checkpoint_root_dir: The directory where model checkpoints will be saved.
#
#     Attributes:
#         train_set: The training dataset loader.
#         validation_set: The validation dataset loader.
#         test_set: The test dataset loader.
#         exemplar_dim: The dimensions of the exemplar data.
#         checkpoint_dir: The directory for saving model checkpoints.
#         embedding_size: The size of the embedding produced by the network.
#         callbacks: List of Keras callbacks for training.
#     """
#
#     def __init__(self, train_loader, validation_loader, test_loader, checkpoint_root_dir, triplet_modules,
#                  architecture_variant, config):
#         super().__init__()
#         # train_generator's job is to randomly, and lazily, load batches from disk
#         # SIM NETWORK CLASS: exemplar dim: (137, 88)
#         self.config = config
#         self.train_set = train_loader
#         self.validation_set = validation_loader
#         self.test_set = test_loader
#         self.exemplar_dim = train_loader.exemplar_dim
#         print(f"SIM NETWORK CLASS: exemplar dim: {self.exemplar_dim}")
#         self.triplet_modules = triplet_modules
#         self.architecture_variant = architecture_variant
#         self.checkpoint_dir = checkpoint_root_dir
#         self.embedding_size = self.config.embedding_size
#         self.callbacks = []
#         # callbacks.EarlyStopping(monitor='batch_triplet_loss', patience=10, restore_best_weights=True, mode='min',
#         #                                     verbose=1)
#         checkpoint_callback = ModelCheckpoint(
#             filepath=os.path.join(self.checkpoint_dir,
#                                   f"{self.architecture_variant}_similarity_model_weights_epoch_{{epoch:03d}}.weights.h5"),
#             save_weights_only=True,
#             save_best_only=True,  # Save only when validation loss improves
#             monitor="batch_triplet_loss",  # Ensure it tracks validation loss
#             mode="min",  # Save when val_loss decreases
#             verbose=1
#         )
#         self.callbacks.extend([TrainingLogger(f"variant_{self.architecture_variant}")])
#         self.network = Sequential()
#         self.build_model()
#         self.compile_model()
#
#     def build_model(self):
#         input_shape = (self.exemplar_dim[0], self.exemplar_dim[1], 1)  # (132, 88, 1)
#         self.network = Sequential()
#
#         if self.architecture_variant == 0:
#             # # Output size: (132,88,32)
#             self.network.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=input_shape))
#             self.network.add(Dropout(0.2))
#             self.network.add(BatchNormalization())
#             # Output size: (66,44,32)
#             # self.network.add(MaxPool2D(2, 2))
#             self.network.add(MaxPooling2D((2, 2)))
#             # Output size: (64,42,64)
#
#             self.network.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
#             self.network.add(Dropout(0.2))
#             self.network.add(BatchNormalization())
#
#             self.network.add(Conv2D(128, 3, strides=1, activation='relu'))
#             self.network.add(BatchNormalization())
#
#             # Output size: (32,21,64)
#             # self.network.add(MaxPool2D(2, 2))
#             self.network.add(MaxPooling2D((2, 2)))
#             # Output size: (132 * 88 * 32) = 43008
#             self.network.add(Flatten())
#             # Output size: (32)
#             self.network.add(Dense(self.embedding_size))
#             self.network.add(Dropout(0.2))
#             # Print the model summary
#             self.network.summary()
#
#         elif self.architecture_variant == 1:
#             # Feature extraction with deeper convolution
#             self.network.add(Conv2D(32, 3, strides=1, activation='relu', padding='same', input_shape=input_shape))
#             self.network.add(BatchNormalization())
#
#             self.network.add(Conv2D(64, 3, strides=1, activation='relu'))
#             self.network.add(BatchNormalization())
#
#             self.network.add(Conv2D(128, 3, strides=1, activation='relu'))
#             self.network.add(BatchNormalization())
#
#             # Downsampling with MaxPooling (instead of strided convolution)
#             # self.network.add(MaxPool2D(pool_size=(2, 2)))
#             self.network.add(MaxPooling2D((2, 2)))
#             # Output shape: (66, 44, 128) - Spatial size halved
#
#             # Further feature extraction
#             self.network.add(Conv2D(256, 3, strides=1, activation='relu'))
#             self.network.add(BatchNormalization())
#
#             self.network.add(Conv2D(256, 3, strides=1, activation='relu'))
#             self.network.add(BatchNormalization())
#
#             # Another MaxPooling for further downsampling
#             # self.network.add(MaxPool2D(pool_size=(2, 2)))
#             self.network.add(MaxPooling2D((2, 2)))
#             # Output shape: (33, 22, 256) - Spatial size halved again
#
#             # Flatten instead of GlobalAveragePooling
#             self.network.add(Flatten())
#
#             # Bottleneck dense layer
#             self.network.add(Dense(self.embedding_size))
#             self.network.add(Dropout(0.2))  # Retaining dropout from the original model
#
#             # Print the model summary
#             self.network.summary()
#
#         elif self.architecture_variant == 2:
#             self.network.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=input_shape))
#             self.network.add(Dropout(0.2))
#             self.network.add(BatchNormalization())
#             # Output size: (66,44,32)
#             # self.network.add(MaxPool2D(2, 2))
#             self.network.add(MaxPooling2D((2, 2)))
#             # Output size: (64,42,64)
#             self.network.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
#             self.network.add(Dropout(0.2))
#             self.network.add(BatchNormalization())
#
#             self.network.add(Conv2D(128, 3, strides=1, activation='relu'))
#             self.network.add(BatchNormalization())
#
#             # Output size: (32,21,64)
#             # self.network.add(MaxPool2D(2, 2))
#             self.network.add(MaxPooling2D((2, 2)))
#             # Output size: (132 * 88 * 32) = 43008
#             self.network.add(Flatten())
#             # Output size: (32)
#             self.network.add(Dense(self.embedding_size))
#             self.network.add(Dropout(0.2))
#             # Print the model summary
#             self.network.summary()
#             # self.build_efficient_residual_network()
#
#         elif self.architecture_variant == 3:
#             self.network.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=input_shape))
#             self.network.add(Dropout(0.2))
#             self.network.add(BatchNormalization())
#             # Output size: (66,44,32)
#             # self.network.add(MaxPool2D(2, 2))
#             self.network.add(MaxPooling2D((2, 2)))
#             # Output size: (64,42,64)
#             self.network.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
#             self.network.add(Dropout(0.2))
#             self.network.add(BatchNormalization())
#
#             self.network.add(Conv2D(128, 3, strides=1, activation='relu'))
#             self.network.add(BatchNormalization())
#
#             # Output size: (32,21,64)
#             # self.network.add(MaxPool2D(2, 2))
#             self.network.add(MaxPooling2D((2, 2)))
#             # Output size: (132 * 88 * 32) = 43008
#             self.network.add(Flatten())
#             # Output size: (32)
#             self.network.add(Dense(self.embedding_size))
#             self.network.add(Dropout(0.2))
#             # Print the model summary
#             self.network.summary()
#             # self.build_lightweight_network()
#
#         elif self.architecture_variant == 4:
#             self.network.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=input_shape))
#             self.network.add(Dropout(0.2))
#             self.network.add(BatchNormalization())
#             # Output size: (66,44,32)
#             # self.network.add(MaxPool2D(2, 2))
#             self.network.add(MaxPooling2D((2, 2)))
#             # Output size: (64,42,64)
#             self.network.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
#             self.network.add(Dropout(0.2))
#             self.network.add(BatchNormalization())
#
#             self.network.add(Conv2D(128, 3, strides=1, activation='relu'))
#             self.network.add(BatchNormalization())
#
#             # Output size: (32,21,64)
#             # self.network.add(MaxPool2D(2, 2))
#             self.network.add(MaxPooling2D((2, 2)))
#             # Output size: (132 * 88 * 32) = 43008
#             self.network.add(Flatten())
#             # Output size: (32)
#             self.network.add(Dense(self.embedding_size))
#             self.network.add(Dropout(0.2))
#             # Print the model summary
#             self.network.summary()
#             # self.build_attention_network()

    # Variant 2: Efficient Network with Residual Connections
    # def build_efficient_residual_network(self):
    #     input_shape = (self.exemplar_dim[0], self.exemplar_dim[1], 1)  # (132, 88, 1)
    #     input_layer = Input(shape=input_shape)
    #
    #     # Initial convolution
    #     x = Conv2D(32, 3, strides=2, activation='relu', padding='same')(input_layer)
    #     x = BatchNormalization()(x)
    #
    #     # First residual block
    #     residual = x
    #     x = Conv2D(64, 3, strides=1, activation='relu', padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = Conv2D(64, 3, strides=1, padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = Add()([x, residual])
    #     x = Activation('relu')(x)
    #
    #     # Efficient spatial pyramid pooling
    #     b1 = GlobalAveragePooling2D()(x)
    #     b1 = Dense(64)(b1)
    #     b1 = Reshape((1, 1, 64))(b1)
    #     b1 = UpSampling2D(size=(x.shape[1], x.shape[2]))(b1)
    #
    #     b2 = AveragePooling2D(pool_size=(2, 2))(x)
    #     b2 = Conv2D(64, 1, activation='relu')(b2)
    #     b2 = UpSampling2D(size=(2, 2))(b2)
    #
    #     x = Concatenate()([x, b1, b2])
    #
    #     # Final layers
    #     x = Conv2D(128, 1, activation='relu')(x)
    #     x = GlobalAveragePooling2D()(x)
    #     x = Dense(self.embedding_size)(x)
    #
    #     return Model(inputs=input_layer, outputs=x)

    # Variant 3: Lightweight MobileNet-style Architecture
    # def build_lightweight_network(self):
    #     def depthwise_separable_conv(x, filters, stride=1):
    #         x = DepthwiseConv2D(3, strides=stride, padding='same')(x)
    #         x = BatchNormalization()(x)
    #         x = Activation('relu')(x)
    #         x = Conv2D(filters, 1, strides=1, padding='same')(x)
    #         x = BatchNormalization()(x)
    #         return Activation('relu')(x)
    #
    #     input_shape = (self.exemplar_dim[0], self.exemplar_dim[1], 1)  # (132, 88, 1)
    #     input_layer = Input(shape=input_shape)
    #
    #     x = Conv2D(32, 3, strides=2, padding='same')(input_layer)
    #     x = BatchNormalization()(x)
    #     x = Activation('relu')(x)
    #
    #     x = depthwise_separable_conv(x, 64)
    #     x = depthwise_separable_conv(x, 128, stride=2)
    #     x = depthwise_separable_conv(x, 128)
    #     x = depthwise_separable_conv(x, 256, stride=2)
    #     x = depthwise_separable_conv(x, 256)
    #
    #     x = GlobalAveragePooling2D()(x)
    #     x = Dense(self.embedding_size)(x)
    #
    #     return Model(inputs=input_layer, outputs=x)

    # Variant 4: Channel Attention Network
    # def build_attention_network(self):
    #     def channel_attention(x, ratio=8):
    #         channel = x.shape[-1]
    #
    #         avg_pool = GlobalAveragePooling2D()(x)
    #         avg_pool = Reshape((1, 1, channel))(avg_pool)
    #         avg_pool = Dense(channel // ratio, activation='relu')(avg_pool)
    #         avg_pool = Dense(channel, activation='sigmoid')(avg_pool)
    #
    #         max_pool = GlobalMaxPooling2D()(x)
    #         max_pool = Reshape((1, 1, channel))(max_pool)
    #         max_pool = Dense(channel // ratio, activation='relu')(max_pool)
    #         max_pool = Dense(channel, activation='sigmoid')(max_pool)
    #
    #         attention = Add()([avg_pool, max_pool])
    #
    #         return Multiply()([x, attention])
    #
    #     input_shape = (self.exemplar_dim[0], self.exemplar_dim[1], 1)  # (132, 88, 1)
    #     input_layer = Input(shape=input_shape)
    #
    #     x = Conv2D(32, 3, strides=2, activation='relu', padding='same')(input_layer)
    #     x = BatchNormalization()(x)
    #     x = channel_attention(x)
    #
    #     x = Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = channel_attention(x)
    #
    #     x = Conv2D(128, 3, strides=1, activation='relu', padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = channel_attention(x)
    #
    #     x = GlobalAveragePooling2D()(x)
    #     x = Dense(self.embedding_size)(x)
    #
    #     return Model(inputs=input_layer, outputs=x)

    # def build_model(self):
    #     input_shape = (self.exemplar_dim[0], self.exemplar_dim[1], 1)  # (132, 88, 1)
    #     inputs = Input(shape=input_shape)
    #
    #     # Feature extraction
    #     x = Conv2D(32, 3, strides=1, activation='relu', padding='same')(inputs)
    #     x = BatchNormalization()(x)
    #     x = Conv2D(64, 3, strides=1, activation='relu', padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = Conv2D(128, 3, strides=1, activation='relu', padding='same', dilation_rate=2)(x)
    #     x = BatchNormalization()(x)
    #
    #     # Downsampling
    #     x = MaxPool2D(pool_size=(2, 2))(x)
    #
    #     # Further feature extraction
    #     x = Conv2D(256, 3, strides=1, activation='relu', padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = Conv2D(256, 3, strides=1, activation='relu', padding='same')(x)
    #     x = BatchNormalization()(x)
    #
    #     # Another downsampling
    #     x = MaxPool2D(pool_size=(2, 2))(x)
    #
    #     # **Attention mechanism**
    #     attention = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)  # Generate attention map
    #     x = Multiply()([x, attention])  # Element-wise multiplication of feature maps with attention weights
    #
    #     # Global Average Pooling
    #     x = GlobalAveragePooling2D()(x)
    #
    #     x = Flatten()(x)
    #
    #     # Bottleneck dense layer
    #     x = Dense(self.embedding_size)(x)
    #     x = Dropout(0.2)(x)
    #
    #     # Build model
    #     self.network = Model(inputs, x)
    #
    #     # Print the model summary
    #     self.network.summary()

    # def compile_model(self):
    #     """
    #     Compile the neural network with a custom triplet loss function.
    #
    #     Compiles the neural network using the Adam optimizer and a custom triplet loss function.
    #     Also sets up the network for training.
    #
    #     Args:
    #         None
    #
    #     Returns:
    #         None
    #     """
    #     opt = Adam(learning_rate=0.0001, beta_1=0.5)
    #     try:
    #         custom_loss_closure = custom_losses.create_batch_triplet_loss(self.triplet_modules)
    #         self.network.compile(optimizer=opt, loss=custom_loss_closure,
    #                              metrics=[custom_loss_closure], run_eagerly=True)
    #     except RuntimeError:
    #         custom_loss_closure = custom_losses.create_batch_triplet_loss(self.triplet_modules)
    #         self.network.compile(optimizer=opt, loss=custom_loss_closure,
    #                              metrics=[custom_loss_closure])
    #     finally:
    #         self.network.summary()

    # def run_model_training(self):
    #     """
    #    Train the neural network on the training dataset.
    #
    #    Trains the neural network on the training dataset and uses the validation dataset for monitoring.
    #
    #    Args:
    #        None
    #
    #    Returns:
    #        None
    #    """
    #
    #     self.network.fit(self.train_set, validation_data=self.validation_set, epochs=self.config.n_similarity_epochs,
    #                      steps_per_epoch=self.train_set.__len__(),
    #                      validation_steps=self.validation_set.__len__(), callbacks=self.callbacks)
    #
    #     weights_path = os.path.join(self.checkpoint_dir,
    #                                 f"{self.architecture_variant}_similarity_model_weights.weights.h5")
    #     self.network.save_weights(weights_path)
    #     print(f"Model weights saved to {weights_path}")
    #
    # def evaluate(self):
    #     """
    #     Evaluate the model on the test dataset.
    #
    #     Evaluates the trained model on the test dataset attribute.
    #
    #     Args:
    #         None
    #
    #     Returns:
    #         None
    #     """
    #     self.network.evaluate(self.test_set, steps=self.test_set.__len__())
