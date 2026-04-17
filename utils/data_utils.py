"""
Data utilities for training and inference.
Handles dataset preparation and batch processing.
"""

import numpy as np
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split


class AudioDataset:
    """Dataset handler for audio files and labels."""
    
    def __init__(self, data_dir, sr=22050):
        """
        Initialize audio dataset.
        
        Args:
            data_dir: Directory containing audio files
            sr: Sample rate
        """
        self.data_dir = Path(data_dir)
        self.sr = sr
        self.files = []
        self.labels = []
        self.metadata = {}
    
    def load_from_directory(self, organize_by_chord=True):
        """
        Load audio files from directory structure.
        
        Args:
            organize_by_chord: If True, expects subdirectories named by chord
        
        Returns:
            dataset: List of (audio_path, chord_label) tuples
        """
        dataset = []
        
        if organize_by_chord:
            # Expect structure: data_dir/Cmaj/*.mp3, data_dir/Amin/*.mp3, etc.
            for chord_dir in self.data_dir.iterdir():
                if chord_dir.is_dir():
                    chord_label = chord_dir.name
                    for audio_file in chord_dir.glob("*.wav"):
                        dataset.append((str(audio_file), chord_label))
                    for audio_file in chord_dir.glob("*.mp3"):
                        dataset.append((str(audio_file), chord_label))
        else:
            # Flat structure with labels in filenames
            for audio_file in self.data_dir.glob("*.wav"):
                # Extract label from filename: e.g., "Cmaj_sample1.wav"
                label = audio_file.stem.split("_")[0]
                dataset.append((str(audio_file), label))
        
        self.files = [f[0] for f in dataset]
        self.labels = [f[1] for f in dataset]
        
        return dataset
    
    def split_dataset(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split dataset into train/val/test.
        
        Returns:
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset
        """
        # First split: separate test set
        train_indices, test_indices = train_test_split(
            range(len(self.files)),
            test_size=test_size,
            random_state=random_state
        )
        
        # Second split: separate validation from training
        val_fraction = val_size / (1 - test_size)
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=val_fraction,
            random_state=random_state
        )
        
        train_data = [(self.files[i], self.labels[i]) for i in train_indices]
        val_data = [(self.files[i], self.labels[i]) for i in val_indices]
        test_data = [(self.files[i], self.labels[i]) for i in test_indices]
        
        return train_data, val_data, test_data
    
    def load_audio(self, filepath):
        """Load single audio file."""
        try:
            y, sr = librosa.load(filepath, sr=self.sr)
            return y
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None


class FeatureBatcher:
    """Batch feature extraction for efficient training."""
    
    def __init__(self, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
        """Initialize feature extraction parameters."""
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_features_batch(self, audio_files, max_samples=None):
        """
        Extract mel spectrogram features from audio files in batch.
        
        Args:
            audio_files: List of audio file paths
            max_samples: Maximum number of samples to process
        
        Returns:
            features: Array of shape (n_samples, n_mels, n_frames, 1)
            valid_indices: Indices of successfully processed files
        """
        features = []
        valid_indices = []
        
        for idx, audio_file in enumerate(audio_files[:max_samples]):
            try:
                y, _ = librosa.load(audio_file, sr=self.sr)
                S = librosa.feature.melspectrogram(
                    y=y, sr=self.sr, n_mels=self.n_mels,
                    n_fft=self.n_fft, hop_length=self.hop_length
                )
                S_db = librosa.power_to_db(S, ref=np.max)
                # Normalize
                S_db = (S_db - np.mean(S_db)) / (np.std(S_db) + 1e-6)
                # Add channel dimension
                S_db = np.expand_dims(S_db, axis=-1)
                
                features.append(S_db)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Skipping {audio_file}: {e}")
                continue
        
        return np.array(features), valid_indices
    
    def pad_or_truncate(self, features, target_length=40):
        """
        Pad or truncate mel spectrograms to target length.
        
        Args:
            features: Array of shape (n, n_mels, n_frames, 1)
            target_length: Target number of frames
        
        Returns:
            processed_features: Padded/truncated features
        """
        n_samples = features.shape[0]
        n_mels = features.shape[1]
        
        processed = np.zeros((n_samples, n_mels, target_length, 1))
        
        for i, feat in enumerate(features):
            n_frames = feat.shape[1]
            if n_frames >= target_length:
                # Truncate from middle or randomly
                start = (n_frames - target_length) // 2
                processed[i] = feat[:, start:start + target_length, :]
            else:
                # Pad with zeros
                pad_width = ((0, 0), (0, target_length - n_frames), (0, 0))
                processed[i] = np.pad(feat, pad_width, mode='constant')
        
        return processed


class ChordEncoder:
    """Encode chord labels to one-hot and vice versa."""
    
    def __init__(self, num_classes=24):
        """
        Initialize chord encoder.
        
        Args:
            num_classes: Number of chord classes
        """
        self.num_classes = num_classes
        self.notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.chord_to_idx = {}
        self.idx_to_chord = {}
        self._build_mapping()
    
    def _build_mapping(self):
        """Build chord name to index mapping."""
        idx = 0
        # Major chords
        for note in self.notes:
            chord_name = f"{note}maj"
            self.chord_to_idx[chord_name] = idx
            self.idx_to_chord[idx] = chord_name
            idx += 1
        
        # Minor chords
        for note in self.notes:
            chord_name = f"{note}min"
            self.chord_to_idx[chord_name] = idx
            self.idx_to_chord[idx] = chord_name
            idx += 1
    
    def encode(self, chord_names):
        """
        Encode chord names to one-hot vectors.
        
        Args:
            chord_names: List of chord names or single chord name
        
        Returns:
            one_hot: One-hot encoded vectors
        """
        if isinstance(chord_names, str):
            chord_names = [chord_names]
        
        one_hot = np.zeros((len(chord_names), self.num_classes))
        for i, chord in enumerate(chord_names):
            idx = self.chord_to_idx.get(chord, 0)
            one_hot[i, idx] = 1
        
        return one_hot
    
    def decode(self, one_hot_or_idx):
        """
        Decode one-hot vectors or indices to chord names.
        
        Args:
            one_hot_or_idx: One-hot vectors or indices
        
        Returns:
            chord_names: List of chord names
        """
        if one_hot_or_idx.ndim > 1:
            indices = np.argmax(one_hot_or_idx, axis=1)
        else:
            indices = one_hot_or_idx
        
        return [self.idx_to_chord.get(int(idx), 'unknown') for idx in indices]
