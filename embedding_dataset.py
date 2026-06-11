"""
Dataset class for loading pre-generated embeddings from motion autoencoder.
"""
import os
import torch
import pickle
import numpy as np
from pathlib import Path


class EmbeddingDataset:
    def __init__(self, embedding_dir, motion_dir=None):
        """
        Dataset for loading pre-generated embeddings.

        Args:
            embedding_dir: Directory containing the .pt embedding files
            motion_dir: Optional directory containing original motion files (for filename mapping)
        """
        self.embedding_dir = Path(embedding_dir)
        self.motion_dir = Path(motion_dir) if motion_dir else None

        # Load all embedding files
        self.embedding_files = list(self.embedding_dir.glob("*.pt"))
        self.root_files = [f for f in self.embedding_files if "_root.pt" in f.name]
        self.rots_files = [f for f in self.embedding_files if "_rots.pt" in f.name]
        # Single-vector embeddings (e.g. unified VAE latent) are saved as _emb.pt.
        # When present, the dataset runs in single-embedding mode: there is no
        # root/rots split, so all combination_methods return the same vector.
        self.emb_files = [f for f in self.embedding_files if "_emb.pt" in f.name]
        self.single_embedding_mode = len(self.emb_files) > 0 and len(self.root_files) == 0

        # Generate valid states and drives
        valid_effort_tuples = self._generate_valid_effort_tuples()

        # Create mapping from base filename to embeddings, filtered by valid efforts
        self.embedding_pairs = {}
        self.effort_mapping = {}

        if self.single_embedding_mode:
            print(f"Found {len(self.emb_files)} single-vector (_emb.pt) embeddings — single-embedding mode")
            for emb_file in self.emb_files:
                base_name = emb_file.name.replace("_emb.pt", "")
                effort_tuple = self._extract_effort_tuple(base_name)
                if effort_tuple in valid_effort_tuples:
                    self.embedding_pairs[base_name] = {'emb_path': emb_file}
                    self.effort_mapping[base_name] = effort_tuple
                else:
                    print(f"Skipping {base_name}: effort tuple {effort_tuple} not in valid states/drives")
        else:
            print(f"Found {len(self.root_files)} root embeddings and {len(self.rots_files)} rotation embeddings")
            for root_file in self.root_files:
                base_name = root_file.name.replace("_root.pt", "")
                rots_file = self.embedding_dir / f"{base_name}_rots.pt"

                if rots_file.exists():
                    # Extract and validate effort values
                    effort_tuple = self._extract_effort_tuple(base_name)

                    # Only keep files with valid effort combinations
                    if effort_tuple in valid_effort_tuples:
                        self.embedding_pairs[base_name] = {
                            'root_path': root_file,
                            'rots_path': rots_file
                        }
                        self.effort_mapping[base_name] = effort_tuple
                    else:
                        print(f"Skipping {base_name}: effort tuple {effort_tuple} not in valid states/drives")

        print(f"Successfully paired {len(self.embedding_pairs)} embedding pairs after filtering")
        print(f"Kept files with {len(set(self.effort_mapping.values()))} unique effort combinations")

    @staticmethod
    def _generate_valid_effort_tuples():
        """
        Generate set of valid effort tuples (states, drives, and neutral).

        States: tuples with exactly 2 zeros (polarized in 2 efforts)
        Drives: tuples with exactly 1 zero (polarized in 3 efforts)
        Neutral: tuple with all zeros

        Returns:
            set of valid effort tuples
        """
        valid_tuples = set()
        effort_vals = [-1, 0, 1]

        # Generate all possible combinations
        import itertools
        for combo in itertools.product(effort_vals, repeat=4):
            zero_count = combo.count(0)

            # Include states (2 zeros), drives (1 zero), and neutral (4 zeros)
            if zero_count in [1, 2, 4]:
                valid_tuples.add(combo)

        return valid_tuples

    def _extract_effort_tuple(self, base_name):
        """
        Extract effort values from filename.

        Args:
            base_name: filename without extension (e.g., "Walking_-1_0_1_0")

        Returns:
            tuple of effort values or (0, 0, 0, 0) if parsing fails
        """
        try:
            # Split by underscore and look for numeric values
            name_parts = base_name.split('_')

            # Find consecutive numeric parts (could be negative)
            numeric_parts = []
            for part in name_parts:
                try:
                    # Try to convert to int (handles -1, 0, 1)
                    val = int(part)
                    if val in [-1, 0, 1]:
                        numeric_parts.append(val)
                except ValueError:
                    # If we've started collecting numbers and hit non-numeric, stop
                    if numeric_parts and len(numeric_parts) < 4:
                        numeric_parts = []

            # We should have exactly 4 effort values
            if len(numeric_parts) == 4:
                return tuple(numeric_parts)
            else:
                print(f"Warning: Could not extract 4 effort values from {base_name}")
                return (0, 0, 0, 0)  # Default neutral

        except Exception as e:
            print(f"Warning: Error parsing efforts from {base_name}: {e}")
            return (0, 0, 0, 0)
    # def __init__(self, embedding_dir, motion_dir=None):
    #     """
    #
    #     Dataset for loading pre-generated embeddings.
    #
    #     Args:
    #         embedding_dir: Directory containing the .pt embedding files
    #         motion_dir: Optional directory containing original motion files (for filename mapping)
    #     """
    #     self.embedding_dir = Path(embedding_dir)
    #     self.motion_dir = Path(motion_dir) if motion_dir else None
    #
    #     # Load all embedding files
    #     self.embedding_files = list(self.embedding_dir.glob("*.pt"))
    #     self.root_files = [f for f in self.embedding_files if "_root.pt" in f.name]
    #     self.rots_files = [f for f in self.embedding_files if "_rots.pt" in f.name]
    #
    #     print(f"Found {len(self.root_files)} root embeddings and {len(self.rots_files)} rotation embeddings")
    #
    #     # Create mapping from base filename to embeddings
    #     self.embedding_pairs = {}
    #     for root_file in self.root_files:
    #         base_name = root_file.name.replace("_root.pt", "")
    #         rots_file = self.embedding_dir / f"{base_name}_rots.pt"
    #
    #         if rots_file.exists():
    #             self.embedding_pairs[base_name] = {
    #                 'root_path': root_file,
    #                 'rots_path': rots_file
    #             }
    #
    #     print(f"Successfully paired {len(self.embedding_pairs)} embedding pairs")
    #
    #     # Extract effort values from filenames
    #     self.effort_mapping = {}
    #     for base_name in self.embedding_pairs.keys():
    #         try:
    #             # Assuming filename format: "prefix_effort1_effort2_effort3_effort4.bvh"
    #             name_parts = base_name.split('_')
    #             if len(name_parts) >= 4:
    #                 # Take last 4 parts as effort values
    #                 efforts = [float(part) for part in name_parts[-4:]]
    #                 self.effort_mapping[base_name] = tuple(efforts)
    #             else:
    #                 print(f"Warning: Could not extract efforts from {base_name}")
    #                 self.effort_mapping[base_name] = (0, 0, 0, 0)  # Default neutral
    #         except ValueError as e:
    #             print(f"Warning: Error parsing efforts from {base_name}: {e}")
    #             self.effort_mapping[base_name] = (0, 0, 0, 0)
    #
    #     print(f"Extracted effort values for {len(self.effort_mapping)} files")

    def load_embedding_pair(self, base_name):
        """Load both root and rotation embeddings for a given base filename."""
        if base_name not in self.embedding_pairs:
            raise KeyError(f"No embedding pair found for {base_name}")

        pair = self.embedding_pairs[base_name]
        root_embedding = torch.load(pair['root_path'])
        rots_embedding = torch.load(pair['rots_path'])

        return root_embedding, rots_embedding

    def get_combined_embedding(self, base_name, combination_method='concat'):
        """
        Get combined embedding for a given base filename.

        Args:
            base_name: Base filename (without _root.pt or _rots.pt suffix)
            combination_method: How to combine root and rotation embeddings
                - 'concat': Simple concatenation
                - 'weighted': Weighted combination (you can modify weights)
                - 'root_only': Use only root embedding
                - 'rots_only': Use only rotation embedding

        Returns:
            Combined embedding tensor
        """
        # Single-embedding mode: one unified vector, combination_method is moot.
        if getattr(self, 'single_embedding_mode', False):
            emb = torch.load(self.embedding_pairs[base_name]['emb_path'], weights_only=True)
            if emb.dim() > 1:
                emb = emb.squeeze()
            return emb

        root_emb, rots_emb = self.load_embedding_pair(base_name)

        # Ensure embeddings are 1D
        if root_emb.dim() > 1:
            root_emb = root_emb.squeeze()
        if rots_emb.dim() > 1:
            rots_emb = rots_emb.squeeze()

        if combination_method == 'concat':
            return torch.cat([root_emb, rots_emb])
        elif combination_method == 'weighted':
            # You can adjust these weights based on importance
            root_weight = 0.3
            rots_weight = 0.7
            # Ensure both embeddings have same size for weighted combination
            if root_emb.size(0) != rots_emb.size(0):
                min_size = min(root_emb.size(0), rots_emb.size(0))
                root_emb = root_emb[:min_size]
                rots_emb = rots_emb[:min_size]
            return root_weight * root_emb + rots_weight * rots_emb
        elif combination_method == 'root_only':
            return root_emb
        elif combination_method == 'rots_only':
            return rots_emb
        else:
            raise ValueError(f"Unknown combination method: {combination_method}")

    def create_similarity_dict(self, combination_method='concat'):
        """
        Create a similarity dictionary compatible with existing triplet training code.

        Returns:
            Dictionary mapping effort tuples to embeddings, compatible with existing code structure
        """
        similarity_dict = {}

        for base_name, effort_tuple in self.effort_mapping.items():
            try:
                combined_embedding = self.get_combined_embedding(base_name, combination_method)

                # Convert to format expected by existing code (list containing the embedding)
                if effort_tuple not in similarity_dict:
                    similarity_dict[effort_tuple] = []

                similarity_dict[effort_tuple].append(combined_embedding.numpy())

            except Exception as e:
                print(f"Error processing {base_name}: {e}")
                continue

        print(f"Created similarity dict with {len(similarity_dict)} effort classes")
        for effort_tuple, embeddings in similarity_dict.items():
            print(f"  Effort {effort_tuple}: {len(embeddings)} embeddings, shape {embeddings[0].shape}")

        return similarity_dict

    def save_similarity_dict(self, output_path, combination_method='concat'):
        """Save the similarity dictionary to a pickle file."""
        similarity_dict = self.create_similarity_dict(combination_method)

        with open(output_path, 'wb') as f:
            pickle.dump(similarity_dict, f)

        print(f"Saved similarity dictionary to {output_path}")
        return similarity_dict


def create_embedding_similarity_data(embedding_dir, output_dir, combination_method='concat'):
    """
    Create similarity data from embeddings and save it in the format expected by your training code.

    Args:
        embedding_dir: Directory containing your .pt embedding files
        output_dir: Directory to save the processed similarity data
        combination_method: How to combine root and rotation embeddings

    Returns:
        Path to the saved similarity dictionary
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    dataset = EmbeddingDataset(embedding_dir)

    # Create and save similarity dictionary
    output_path = output_dir / "walking_dict_class_label_to_exemplars.pkl"
    similarity_dict = dataset.save_similarity_dict(output_path, combination_method)

    return output_path, similarity_dict


if __name__ == "__main__":
    # Example usage
    embedding_dir = "../datasets/lma_perform_walking_encoded"
    output_dir = "../datasets/similarity_walking_embeddings_dict"

    output_path, similarity_dict = create_embedding_similarity_data(
        embedding_dir,
        output_dir,
        combination_method='concat'
    )

    print(f"Similarity data saved to: {output_path}")
    print(f"Created {len(similarity_dict)} effort classes")