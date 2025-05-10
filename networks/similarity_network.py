import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import networks.custom_losses as custom_losses
from keras import callbacks
import logging
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy import stats

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


# Enhanced version of SimilarityNetwork that includes the adaptive distance module
class EnhancedSimilarityNetwork(nn.Module):
    """
    Enhanced similarity network with adaptive distance metric.

    This wraps an existing network and adds an adaptive distance module
    to better align with human perception.
    """

    def __init__(self, base_network, embedding_dim):
        super().__init__()
        self.base_network = base_network
        self.adaptive_distance = custom_losses.AdaptiveDistanceModule(embedding_dim)

    def forward(self, x):
        """Pass input through the base network to get embeddings"""
        return self.base_network(x)

    def compute_distances(self, embeddings):
        """Compute pairwise distances using adaptive metric"""
        return self.adaptive_distance.pairwise_distances(embeddings)


# Model variants - keep original
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


# Main class that integrates the model variants with enhanced correlation loss
class SimilarityNetwork:
    """
    The enhanced SimilarityNetwork class with perception-aligned loss functions.

    This class inherits from the `Utilities` class and is used to build, compile, and train a similarity learning model
    optimized for alignment with human perception data.

    Args:
        train_loader: The PyTorch DataLoader for the training dataset.
        validation_loader: The PyTorch DataLoader for the validation dataset.
        test_loader: The PyTorch DataLoader for the test dataset.
        checkpoint_root_dir: The directory where model checkpoints will be saved.
        triplet_modules: A list of TripletMining modules.
        architecture_variant: The architecture variant to use (0, 1, 2, 3, 4).
        config: The configuration object.
        val_triplet_modules: Optional, separate triplet modules for validation.
        lr_scheduler_type: Type of learning rate scheduler ('plateau', 'cosine', 'step').
        use_perception_loss: Whether to use the enhanced perception-aligned loss.
        use_adaptive_distance: Whether to use the adaptive distance module.
    """

    def __init__(self, train_loader, validation_loader, test_loader, checkpoint_root_dir, triplet_modules,
                 architecture_variant, config, val_triplet_modules=None, lr_scheduler_type='plateau',
                 use_perception_loss=True, use_adaptive_distance=True):
        super().__init__()
        self.config = config
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.exemplar_dim = train_loader.exemplar_dim
        print(f"SIM NETWORK CLASS: exemplar dim: {self.exemplar_dim}")

        # Store training and validation triplet modules
        self.train_triplet_modules = triplet_modules
        self.val_triplet_modules = val_triplet_modules if val_triplet_modules else triplet_modules

        self.architecture_variant = architecture_variant
        self.checkpoint_dir = checkpoint_root_dir
        self.embedding_size = self.config.embedding_size

        # Add learning rate scheduler type
        self.lr_scheduler_type = lr_scheduler_type
        self.initial_lr = 0.0001

        # Enhanced loss options
        self.use_perception_loss = use_perception_loss
        self.use_adaptive_distance = use_adaptive_distance and use_perception_loss  # Adaptive distance requires perception loss

        # Use GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Setup logger
        self.logger = TrainingLogger(f"variant_{self.architecture_variant}")

        # Training history tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'correlation': [],
            'r2_score': [],
            'learning_rate': []
        }

        # Build the model
        self.build_model()

    def build_model(self):
        input_shape = (self.exemplar_dim[0], self.exemplar_dim[1])

        # Create base network based on architecture variant
        if self.architecture_variant == 0:
            base_network = SimilarityNetworkV0(input_shape, self.embedding_size)
        elif self.architecture_variant == 1:
            # For simplicity, reusing variant 0
            base_network = SimilarityNetworkV0(input_shape, self.embedding_size)
        elif self.architecture_variant in [2, 3, 4]:
            # For simplicity, reusing variant 0
            base_network = SimilarityNetworkV0(input_shape, self.embedding_size)

        # Wrap base network with adaptive distance module if requested
        if self.use_adaptive_distance:
            self.network = EnhancedSimilarityNetwork(base_network, self.embedding_size)
            self.adaptive_distance_module = self.network.adaptive_distance
        else:
            self.network = base_network
            self.adaptive_distance_module = None

        # Move model to device
        self.network = self.network.to(self.device)

        # Print model summary
        # print(self.network)

        # Create loss functions with appropriate module information
        if self.use_perception_loss:
            self.train_criterion = custom_losses.create_batch_integrated_loss(
                self.train_triplet_modules,
                self.adaptive_distance_module,
                self.train_loader.module_start_indices,
                self.train_loader.module_sizes
            )

            self.val_criterion = custom_losses.create_batch_integrated_loss(
                self.val_triplet_modules,
                self.adaptive_distance_module,
                self.validation_loader.module_start_indices,
                self.validation_loader.module_sizes
            )
        else:
            # Original triplet loss
            self.train_criterion = custom_losses.create_batch_triplet_loss(
                self.train_triplet_modules,
                self.train_loader.module_start_indices,
                self.train_loader.module_sizes
            )

            self.val_criterion = custom_losses.create_batch_triplet_loss(
                self.val_triplet_modules,
                self.validation_loader.module_start_indices,
                self.validation_loader.module_sizes
            )

        # For backward compatibility
        self.criterion = self.train_criterion

        # Setup optimizer with the initial learning rate
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.initial_lr, betas=(0.5, 0.999))

        # Setup the learning rate scheduler based on the type
        if self.lr_scheduler_type == 'plateau':
            # Reduce learning rate when validation loss plateaus
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.9,  # Multiply LR by this factor
                patience=5,  # Number of epochs with no improvement
                verbose=True,  # Print message when LR is reduced
                min_lr=1e-6  # Minimum LR
            )
        elif self.lr_scheduler_type == 'cosine':
            # Cosine annealing scheduler
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.n_similarity_epochs,  # Max number of iterations
                eta_min=1e-6  # Minimum LR
            )
        elif self.lr_scheduler_type == 'step':
            # Step decay scheduler
            self.scheduler = StepLR(
                self.optimizer,
                step_size=10,  # Decay LR every step_size epochs
                gamma=0.1  # Multiply LR by gamma
            )
        else:
            self.scheduler = None
            print(f"Warning: Unknown scheduler type '{self.lr_scheduler_type}'. No scheduler will be used.")

    def save_checkpoint(self, epoch=None, correlation=None, r2=None):
        """Save model checkpoint with additional metrics"""
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
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'correlation': correlation,
            'r2_score': r2,
            'use_adaptive_distance': self.use_adaptive_distance,
            'history': self.history
        }, checkpoint_path)

        print(f"Model weights saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Model loaded from {checkpoint_path}")
        if 'correlation' in checkpoint and checkpoint['correlation']:
            print(f"Checkpoint correlation: {checkpoint['correlation']:.4f}")
        if 'r2_score' in checkpoint and checkpoint['r2_score']:
            print(f"Checkpoint R²: {checkpoint['r2_score']:.4f}")

    def calculate_correlation_metrics(self, loader, triplet_modules):
        """
        Calculate correlation between embedding distances and human perception.
        Uses direct access to comparison data from the triplet module's dataframes.
        """
        self.network.eval()

        # For correlation analysis
        all_distances = []
        all_perceptions = []

        with torch.no_grad():
            for batch_features, batch_labels in loader:
                # Move batch to device
                batch_features = batch_features.to(self.device)

                # Forward pass to get embeddings
                embeddings = self.network(batch_features)

                # Process each triplet module
                for triplet_module in triplet_modules:
                    try:
                        # Calculate distances
                        if self.use_adaptive_distance:
                            # For adaptive distance, handle neutral embedding
                            if not triplet_module.bool_drop_neutral_exemplar and embeddings.shape[0] > 1:
                                # Skip neutral embedding (always first)
                                non_neutral_embeddings = embeddings[1:]
                                distances = self.adaptive_distance_module.pairwise_distances(non_neutral_embeddings)
                            else:
                                distances = self.adaptive_distance_module.pairwise_distances(embeddings)
                        else:
                            # Use triplet module's internal distance calculation
                            distances = triplet_module.calculate_distances(embeddings)

                        # Check if we have the df_comparisons DataFrame in the triplet module
                        if not hasattr(triplet_module, 'df_comparisons') or triplet_module.df_comparisons is None:
                            print(f"Warning: No df_comparisons found in triplet module for {triplet_module.anim_name}")
                            continue

                        # Get the dictionary mapping from class indices to effort tuples
                        dict_id_to_label = {idx: class_label for idx, class_label in
                                            enumerate(triplet_module.dict_similarity_classes_exemplars.keys())}

                        # Get all valid pairs with human perception data
                        valid_pairs = []

                        # For each combination of labels, look up direct comparison data
                        n = len(dict_id_to_label)
                        for i in range(n):
                            for j in range(i + 1, n):
                                # Get the effort tuples for these indices
                                effort_i = dict_id_to_label[i]
                                effort_j = dict_id_to_label[j]

                                # Skip neutral pairs
                                if effort_i == (0, 0, 0, 0) or effort_j == (0, 0, 0, 0):
                                    continue

                                # Find direct comparison rows where selected0=0 and selected1=2
                                df = triplet_module.df_comparisons

                                # Find rows that have this pair of effort tuples
                                pair_match = df[df['efforts_tuples'].apply(lambda x:
                                                                           set(x) == set(
                                                                               [effort_i, effort_j]) if isinstance(x,
                                                                                                                   list) else False)]

                                if pair_match.empty:
                                    continue

                                # Find the specific row with selected0=0 (left) and selected1=2 (right)
                                target_row = pair_match[(pair_match['selected0'] == 0) & (pair_match['selected1'] == 2)]

                                if not target_row.empty and 'count_normalized' in target_row.columns:
                                    # Get the comparison value
                                    comparison_value = target_row['count_normalized'].iloc[0]

                                    # Get the distance between these embeddings
                                    if i < distances.shape[0] and j < distances.shape[0]:
                                        distance = distances[i, j].item()

                                        # Store the pair for correlation analysis
                                        valid_pairs.append((distance, 1.0 - comparison_value))

                        # Add all valid pairs to our collections
                        for distance, perception in valid_pairs:
                            all_distances.append(distance)
                            all_perceptions.append(perception)

                    except Exception as e:
                        print(f"Error in correlation calculation: {e}")
                        continue

        # Calculate correlation metrics if we have data
        if len(all_distances) > 0 and len(all_perceptions) > 0:
            # Pearson correlation
            pearson_corr, pearson_p = stats.pearsonr(all_distances, all_perceptions)

            # Spearman rank correlation
            spearman_corr, spearman_p = stats.spearmanr(all_distances, all_perceptions)

            # R² score
            r2 = r2_score(all_perceptions, all_distances)

            return {
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'r2_score': r2,
                'num_pairs': len(all_distances)
            }
        else:
            return {
                'pearson_correlation': 0.0,
                'pearson_p_value': 1.0,
                'spearman_correlation': 0.0,
                'spearman_p_value': 1.0,
                'r2_score': 0.0,
                'num_pairs': 0
            }

    def plot_training_history(self, save_path=None):
        """
        Plot training history metrics.

        Args:
            save_path: Path to save the plot (if None, display plot)
        """
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot training and validation loss
        axes[0, 0].plot(self.history['train_loss'], label='Training Loss')
        if self.history['val_loss']:
            # Calculate validation epochs
            val_epochs = list(range(0, len(self.history['train_loss']),
                                    max(1, len(self.history['train_loss']) // len(self.history['val_loss']))))[
                         :len(self.history['val_loss'])]
            axes[0, 0].plot(val_epochs, self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot correlation if available
        if self.history['correlation']:
            axes[0, 1].plot(val_epochs[:len(self.history['correlation'])], self.history['correlation'],
                            label='Pearson Correlation')
            axes[0, 1].set_title('Correlation with Human Perception')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Correlation')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # Plot R² if available
        if self.history['r2_score']:
            axes[1, 0].plot(val_epochs[:len(self.history['r2_score'])], self.history['r2_score'], label='R² Score')
            axes[1, 0].set_title('R² with Human Perception')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('R²')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Plot learning rate
        if self.history['learning_rate']:
            axes[1, 1].plot(self.history['learning_rate'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            axes[1, 1].set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def run_model_training(self):
        """
        Train the neural network with separate training and validation phases.
        Enhanced with correlation tracking and more detailed logging.
        """
        best_loss = float('inf')
        best_correlation = -1.0
        validation_frequency = 5  # Validate every N epochs

        for epoch in range(self.config.n_similarity_epochs):
            self.logger.on_epoch_begin(epoch)

            # Training phase
            self.network.train()
            running_loss = 0.0
            batch_count = 0

            for inputs, labels in self.train_loader:
                print(f"Train batch size: {inputs.shape[0]}")
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.network(inputs)

                # Calculate loss using training criterion
                loss = self.train_criterion(labels, outputs)
                print(f"Train Batch Loss: {loss.item()}")

                correlation_metrics = self.calculate_correlation_metrics(
                    self.train_loader,
                    self.train_triplet_modules
                )

                correlation = correlation_metrics['pearson_correlation']
                r2 = correlation_metrics['r2_score']

                print(
                    f"Training Correlation: {correlation:.4f}, R²: {r2:.4f}, Pairs: {correlation_metrics['num_pairs']}")

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                batch_count += 1

            # Calculate average loss for this epoch
            epoch_loss = running_loss / batch_count if batch_count > 0 else 0
            self.history['train_loss'].append(epoch_loss)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            print(f"Epoch {epoch + 1}: Training Loss = {epoch_loss:.4f}")
            # Calculate correlation metrics if using perception loss
            # if self.use_perception_loss:
            #     correlation_metrics = self.calculate_correlation_metrics(
            #         self.train_loader,
            #         self.train_triplet_modules
            #     )
            #
            #     correlation = correlation_metrics['pearson_correlation']
            #     r2 = correlation_metrics['r2_score']
            #
            #     print(f"Training Correlation: {correlation:.4f}, R²: {r2:.4f}, Pairs: {correlation_metrics['num_pairs']}")

            # Validation phase (run periodically to save time)
            run_validation = (epoch + 1) % validation_frequency == 0 or epoch == 0 or epoch == self.config.n_similarity_epochs - 1

            if run_validation:
                self.network.eval()
                val_loss = 0.0
                val_batch_count = 0

                with torch.no_grad():
                    for inputs, labels in self.validation_loader:
                        print(f"Validation batch size: {inputs.shape[0]}")
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        outputs = self.network(inputs)

                        # Use validation criterion
                        loss = self.val_criterion(labels, outputs)
                        print(f"Val Batch Loss: {loss.item()}")

                        val_loss += loss.item()
                        val_batch_count += 1

                # Calculate average validation loss
                val_epoch_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
                self.history['val_loss'].append(val_epoch_loss)

                # Calculate correlation metrics if using perception loss
                if self.use_perception_loss:
                    val_correlation_metrics = self.calculate_correlation_metrics(
                        self.validation_loader,
                        self.val_triplet_modules
                    )

                    val_correlation = val_correlation_metrics['pearson_correlation']
                    val_r2 = val_correlation_metrics['r2_score']

                    self.history['correlation'].append(val_correlation)
                    self.history['r2_score'].append(val_r2)

                    print(
                        f"Validation Correlation: {val_correlation:.4f}, R²: {val_r2:.4f}, Pairs: {val_correlation_metrics['num_pairs']}")
                else:
                    val_correlation = 0
                    val_r2 = 0

                # Step the scheduler if it's a plateau scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        # Use correlation for scheduler if perception loss is enabled
                        if self.use_perception_loss:
                            # Negative correlation because scheduler uses min mode (higher correlation is better)
                            self.scheduler.step(-val_correlation)
                        else:
                            self.scheduler.step(val_epoch_loss)
                    else:
                        self.scheduler.step()

                # Display current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Current Learning Rate: {current_lr:.8f}")

                # Save checkpoint if this is the best model by loss
                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    self.save_checkpoint(epoch + 1, val_correlation, val_r2)
                    print(f"New best model by loss! Val Loss: {val_epoch_loss:.4f}")

                # Save separate checkpoint if best correlation (only if using perception loss)
                if self.use_perception_loss and val_correlation > best_correlation:
                    best_correlation = val_correlation
                    # Save with special name to indicate best correlation
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir,
                        f"{self.architecture_variant}_best_correlation_epoch_{epoch + 1:03d}.pt"
                    )
                    torch.save({
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'epoch': epoch + 1,
                        'correlation': val_correlation,
                        'r2_score': val_r2,
                        'use_adaptive_distance': self.use_adaptive_distance,
                        'history': self.history
                    }, checkpoint_path)
                    print(f"New best model by correlation! Correlation: {val_correlation:.4f}, R²: {val_r2:.4f}")

                # Print summary
                print(f"Epoch {epoch + 1}: Training Loss = {epoch_loss:.4f}, Validation Loss = {val_epoch_loss:.4f}")
                if self.use_perception_loss:
                    print(f"Correlation: {val_correlation:.4f}, R²: {val_r2:.4f}")

            # Plot training history every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.plot_training_history(os.path.join(self.checkpoint_dir, f"training_history_epoch_{epoch + 1}.png"))

        # Save final model
        self.save_checkpoint(self.config.n_similarity_epochs)

        # Generate final training history plot
        self.plot_training_history(os.path.join(self.checkpoint_dir, "final_training_history.png"))

    def evaluate(self):
        """
        Evaluate the model on the test dataset.
        Fixed to handle size mismatches properly.
        """
        self.network.eval()
        test_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                try:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.network(inputs)

                    # Calculate loss
                    try:
                        loss = self.criterion(labels, outputs)
                        test_loss += loss.item()
                        batch_count += 1
                    except Exception as e:
                        print(f"Error in loss calculation during evaluation: {e}")
                        continue
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    continue

        # Calculate average test loss
        avg_test_loss = test_loss / batch_count if batch_count > 0 else 0
        print(f"Test Loss: {avg_test_loss:.4f}")

        # Calculate correlation metrics
        try:
            correlation_metrics = self.calculate_correlation_metrics(self.test_loader, self.train_triplet_modules)
        except Exception as e:
            print(f"Error calculating correlation metrics: {e}")
            correlation_metrics = {
                'pearson_correlation': 0.0,
                'pearson_p_value': 1.0,
                'spearman_correlation': 0.0,
                'spearman_p_value': 1.0,
                'r2_score': 0.0,
                'num_pairs': 0
            }

        print(f"Test Correlation: {correlation_metrics['pearson_correlation']:.4f}")
        print(f"Test Spearman Correlation: {correlation_metrics['spearman_correlation']:.4f}")
        print(f"Test R²: {correlation_metrics['r2_score']:.4f}")
        print(f"Number of valid pairs: {correlation_metrics['num_pairs']}")

        return {
            'test_loss': avg_test_loss,
            **correlation_metrics
        }


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
# import networks.custom_losses as custom_losses
# from keras import callbacks
# import logging
# import os
# import time
#
# logging.basicConfig(level=logging.DEBUG,
#                     filename=os.path.basename(__file__) + '.log',
#                     format="{asctime} [{levelname:8}] {process} {thread} {module}: {message}",
#                     style="{")
# logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
# logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
#
#
# class TrainingLogger(callbacks.Callback):
#     def __init__(self, model_name):
#         super().__init__()
#         self.start_time = None
#         self.logger = logging.getLogger(model_name)
#         handler = logging.FileHandler(f'training_{model_name}.log')
#         formatter = logging.Formatter('%(asctime)s - %(message)s')
#         handler.setFormatter(formatter)
#         self.logger.addHandler(handler)
#         self.logger.setLevel(logging.INFO)
#
#     def on_epoch_begin(self, epoch, logs=None):
#         self.start_time = time.time()
#
#     def on_epoch_end(self, epoch, logs=None):
#         duration = time.time() - self.start_time
#         loss = logs.get('loss', 'N/A')
#         self.logger.info(f"Epoch {epoch + 1}: Durée = {duration:.2f}s, Perte = {loss}")
#
#
# # Model variants
# class SimilarityNetworkV0(nn.Module):
#     def __init__(self, input_shape, embedding_size):
#         super(SimilarityNetworkV0, self).__init__()
#
#         # Input shape should be (batch_size, channels, height, width)
#         # Original input was (batch_size, height, width, channels)
#
#         # Layer 1: Conv -> Dropout -> BatchNorm
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
#         self.dropout1 = nn.Dropout(0.2)
#         self.batch_norm_1 = nn.BatchNorm2d(32)
#         self.pool1 = nn.MaxPool2d(2, 2)
#
#         # Layer 2: Conv -> Dropout -> BatchNorm
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
#         self.dropout2 = nn.Dropout(0.2)
#         self.batch_norm_2 = nn.BatchNorm2d(64)
#
#         # Layer 3: Conv -> BatchNorm
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.pool2 = nn.MaxPool2d(2, 2)
#
#         # Trace size changes after convolutions and pooling
#         # Start with input dimensions
#         height, width = input_shape[0], input_shape[1]
#
#         # After first conv (no padding): (height-2, width-2, 32)
#         height, width = height - 2, width - 2
#
#         # After first pooling: (height/2, width/2, 32)
#         height, width = height // 2, width // 2
#
#         # After second conv (no padding): (height-2, width-2, 64)
#         height, width = height - 2, width - 2
#
#         # After third conv (no padding): (height-2, width-2, 128)
#         height, width = height - 2, width - 2
#
#         # After second pooling: (height/2, width/2, 128)
#         height, width = height // 2, width // 2
#
#         self.fc_input_size = height * width * 128
#
#         # Final dense layer
#         self.fc = nn.Linear(self.fc_input_size, embedding_size)
#         self.dropout3 = nn.Dropout(0.2)
#
#     def forward(self, x):
#         # Reshape input: (batch, height, width, channels) -> (batch, channels, height, width)
#         # PyTorch expects channels first
#         x = x.permute(0, 3, 1, 2)
#
#         # Layer 1
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.dropout1(x)
#         x = self.batch_norm_1(x)
#         x = self.pool1(x)
#
#         # Layer 2
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.batch_norm_2(x)
#
#         # Layer 3
#         x = self.conv3(x)
#         x = F.relu(x)
#         x = self.bn3(x)
#         x = self.pool2(x)
#
#         # Flatten
#         x = x.view(-1, self.fc_input_size)
#
#         # Final dense layer
#         x = self.fc(x)
#         x = self.dropout3(x)
#
#         return x
#
#
# # Main class that integrates the model variants
# class SimilarityNetwork:
#     """
#     The SimilarityNetwork class defines our own triplet similarity network.
#
#     This class inherits from the `Utilities` class and is used to build, compile, and train a similarity learning model.
#
#     Args:
#         train_loader: The PyTorch DataLoader for the training dataset.
#         validation_loader: The PyTorch DataLoader for the validation dataset.
#         test_loader: The PyTorch DataLoader for the test dataset.
#         checkpoint_root_dir: The directory where model checkpoints will be saved.
#         triplet_modules: A list of TripletMining modules.
#         architecture_variant: The architecture variant to use (0, 1, 2, 3, 4).
#         config: The configuration object.
#     """
#
#     def __init__(self, train_loader, validation_loader, test_loader, checkpoint_root_dir, triplet_modules,
#                  architecture_variant, config, val_triplet_modules=None, lr_scheduler_type='plateau'):
#         super().__init__()
#         self.config = config
#         self.train_loader = train_loader
#         self.validation_loader = validation_loader
#         self.test_loader = test_loader
#         self.exemplar_dim = train_loader.exemplar_dim
#         print(f"SIM NETWORK CLASS: exemplar dim: {self.exemplar_dim}")
#
#         # Store training and validation triplet modules
#         self.train_triplet_modules = triplet_modules
#         self.val_triplet_modules = val_triplet_modules if val_triplet_modules else triplet_modules
#
#         self.architecture_variant = architecture_variant
#         self.checkpoint_dir = checkpoint_root_dir
#         self.embedding_size = self.config.embedding_size
#
#         # Add learning rate scheduler type
#         self.lr_scheduler_type = lr_scheduler_type
#         self.initial_lr = 0.0001
#
#         # Use GPU if available
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#         # Setup logger
#         self.logger = TrainingLogger(f"variant_{self.architecture_variant}")
#
#         # Build the model
#         self.build_model()
#
#     def build_model(self):
#         input_shape = (self.exemplar_dim[0], self.exemplar_dim[1])
#
#         if self.architecture_variant == 0:
#             self.network = SimilarityNetworkV0(input_shape, self.embedding_size)
#         elif self.architecture_variant == 1:
#             # For simplicity, reusing variant 0
#             self.network = SimilarityNetworkV0(input_shape, self.embedding_size)
#         elif self.architecture_variant in [2, 3, 4]:
#             # For simplicity, reusing variant 0
#             self.network = SimilarityNetworkV0(input_shape, self.embedding_size)
#
#         # Move model to device
#         self.network = self.network.to(self.device)
#
#         # Print model summary
#         # print(self.network)
#
#         # Create separate loss functions with module information
#         self.train_criterion = custom_losses.create_batch_triplet_loss(
#             self.train_triplet_modules,
#             self.train_loader.module_start_indices,
#             self.train_loader.module_sizes
#         )
#
#         self.val_criterion = custom_losses.create_batch_triplet_loss(
#             self.val_triplet_modules,
#             self.validation_loader.module_start_indices,
#             self.validation_loader.module_sizes
#         )
#
#         # For backward compatibility
#         self.criterion = self.train_criterion
#
#         # Setup optimizer with the initial learning rate
#         self.optimizer = optim.Adam(self.network.parameters(), lr=self.initial_lr, betas=(0.5, 0.999))
#
#         # Setup the learning rate scheduler based on the type
#         if self.lr_scheduler_type == 'plateau':
#             # Reduce learning rate when validation loss plateaus
#             self.scheduler = ReduceLROnPlateau(
#                 self.optimizer,
#                 mode='min',
#                 factor=0.9,  # Multiply LR by this factor
#                 patience=5,  # Number of epochs with no improvement
#                 verbose=True,  # Print message when LR is reduced
#                 min_lr=1e-6  # Minimum LR
#             )
#         elif self.lr_scheduler_type == 'cosine':
#             # Cosine annealing scheduler
#             self.scheduler = CosineAnnealingLR(
#                 self.optimizer,
#                 T_max=self.config.n_similarity_epochs,  # Max number of iterations
#                 eta_min=1e-6  # Minimum LR
#             )
#         elif self.lr_scheduler_type == 'step':
#             # Step decay scheduler
#             self.scheduler = StepLR(
#                 self.optimizer,
#                 step_size=10,  # Decay LR every step_size epochs
#                 gamma=0.1  # Multiply LR by gamma
#             )
#         else:
#             self.scheduler = None
#             print(f"Warning: Unknown scheduler type '{self.lr_scheduler_type}'. No scheduler will be used.")
#
#     def save_checkpoint(self, epoch=None):
#         """Save model checkpoint"""
#         if epoch:
#             checkpoint_path = os.path.join(
#                 self.checkpoint_dir,
#                 f"{self.architecture_variant}_similarity_model_weights_epoch_{epoch:03d}.pt"
#             )
#         else:
#             checkpoint_path = os.path.join(
#                 self.checkpoint_dir,
#                 f"{self.architecture_variant}_similarity_model_weights.pt"
#             )
#
#         torch.save({
#             'model_state_dict': self.network.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
#             'epoch': epoch
#         }, checkpoint_path)
#
#         print(f"Model weights saved to {checkpoint_path}")
#
#     def load_checkpoint(self, checkpoint_path):
#         """Load model checkpoint"""
#         checkpoint = torch.load(checkpoint_path)
#         self.network.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
#     def run_model_training(self):
#         """
#         Train the neural network with separate training and validation phases.
#         """
#         best_loss = float('inf')
#
#         for epoch in range(self.config.n_similarity_epochs):
#             self.logger.on_epoch_begin(epoch)
#
#             # Training phase
#             self.network.train()
#             running_loss = 0.0
#             batch_count = 0
#
#             for inputs, labels in self.train_loader:
#                 print(f"Train batch size: {inputs.shape[0]}")
#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device)
#
#                 # Zero the parameter gradients
#                 self.optimizer.zero_grad()
#
#                 # Forward pass
#                 outputs = self.network(inputs)
#
#                 # Calculate loss using training criterion
#                 loss = self.train_criterion(labels, outputs)
#                 print(f"Train Batch Loss: {loss.item()}")
#
#                 # Backward pass and optimize
#                 loss.backward()
#                 self.optimizer.step()
#
#                 running_loss += loss.item()
#                 batch_count += 1
#
#             # Calculate average loss for this epoch
#             epoch_loss = running_loss / batch_count if batch_count > 0 else 0
#             print(f"Epoch {epoch + 1}: Training Loss = {epoch_loss:.4f}")
#
#             # Validation phase
#             self.network.eval()
#             val_loss = 0.0
#             val_batch_count = 0
#
#             with torch.no_grad():
#                 for inputs, labels in self.validation_loader:
#                     print(f"Validation batch size: {inputs.shape[0]}")
#                     inputs = inputs.to(self.device)
#                     labels = labels.to(self.device)
#
#                     outputs = self.network(inputs)
#
#                     # Use validation criterion
#                     loss = self.val_criterion(labels, outputs)
#                     print(f"Val Batch Loss: {loss.item()}")
#
#                     val_loss += loss.item()
#                     val_batch_count += 1
#
#             # Calculate average validation loss
#             val_epoch_loss = val_loss / val_batch_count
#
#             # Step the scheduler if it's a plateau scheduler
#             if self.scheduler:
#                 if isinstance(self.scheduler, ReduceLROnPlateau):
#                     self.scheduler.step(val_epoch_loss)
#                 else:
#                     self.scheduler.step()
#
#             # Display current learning rate
#             current_lr = self.optimizer.param_groups[0]['lr']
#             print(f"Current Learning Rate: {current_lr:.8f}")
#
#             # Log or print the epoch results
#             # self.logger.on_epoch_end(epoch, epoch_loss)
#             print(f"Epoch {epoch + 1}: Training Loss = {epoch_loss:.4f}, Validation Loss = {val_epoch_loss:.4f}")
#
#             # Save checkpoint if this is the best model so far
#             if val_epoch_loss < best_loss:
#                 best_loss = val_epoch_loss
#                 self.save_checkpoint(epoch + 1)
#
#         # Save final model
#         self.save_checkpoint(self.config.n_similarity_epochs)
#
#     def evaluate(self):
#         """
#         Evaluate the model on the test dataset.
#         """
#         self.network.eval()
#         test_loss = 0.0
#         batch_count = 0
#
#         with torch.no_grad():
#             for inputs, labels in self.test_loader:
#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device)
#
#                 outputs = self.network(inputs)
#                 loss = self.criterion(labels, outputs)
#
#                 test_loss += loss.item()
#                 batch_count += 1
#
#         # Calculate average test loss
#         avg_test_loss = test_loss / batch_count if batch_count > 0 else 0
#         print(f"Test Loss: {avg_test_loss:.4f}")
#
#         return avg_test_loss

