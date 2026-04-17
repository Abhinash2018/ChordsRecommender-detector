"""
Deep learning models for automatic chord classification.
Uses CNN/LSTM architecture for robust chord recognition from audio features.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import joblib


class ChordClassifierModel:
    """CNN-LSTM model for chord classification from spectral features."""
    
    def __init__(self, input_shape=(128, 40, 1), num_classes=24):
        """
        Initialize chord classifier.
        
        Args:
            input_shape: Shape of input features (mel_bands, time_steps, channels)
            num_classes: Number of chord classes (12 major + 12 minor = 24)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.scaler = StandardScaler()
        self.chord_labels = self._get_chord_labels()
        
    def _get_chord_labels(self):
        """Get list of chord class labels."""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chords = []
        for note in notes:
            chords.append(f"{note}maj")
        for note in notes:
            chords.append(f"{note}min")
        return chords
    
    def build_model(self):
        """Build CNN-LSTM architecture for chord classification."""
        model = models.Sequential([
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Reshape for LSTM
            layers.Reshape((-1, 128 * 16)),  # Adjust based on pooling
            
            # LSTM layers
            layers.LSTM(256, return_sequences=True, dropout=0.3),
            layers.LSTM(128, dropout=0.3),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Train the chord classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels (one-hot encoded)
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
        
        Returns:
            history: Training history
        """
        if self.model is None:
            self.build_model()
        
        # Normalize features
        X_train_shape = X_train.shape
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_flat = self.scaler.fit_transform(X_train_flat)
        X_train = X_train_flat.reshape(X_train_shape)
        
        if X_val is not None:
            X_val_shape = X_val.shape
            X_val_flat = X_val.reshape(-1, X_val.shape[-1])
            X_val_flat = self.scaler.transform(X_val_flat)
            X_val = X_val_flat.reshape(X_val_shape)
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict(self, X, threshold=0.7):
        """
        Predict chord class from features.
        
        Args:
            X: Input features
            threshold: Confidence threshold for prediction
        
        Returns:
            predictions: List of (chord_name, confidence)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Normalize
        X_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_flat = self.scaler.transform(X_flat)
        X = X_flat.reshape(X_shape)
        
        # Predict
        y_pred = self.model.predict(X, verbose=0)
        
        predictions = []
        for pred in y_pred:
            chord_idx = np.argmax(pred)
            confidence = pred[chord_idx]
            
            if confidence >= threshold:
                chord_name = self.chord_labels[chord_idx]
                predictions.append((chord_name, float(confidence)))
            else:
                predictions.append(('unknown', float(confidence)))
        
        return predictions
    
    def save(self, filepath):
        """Save model to disk."""
        if self.model is not None:
            self.model.save(f"{filepath}_model.keras")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
    
    def load(self, filepath):
        """Load model from disk."""
        self.model = keras.models.load_model(f"{filepath}_model.keras")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")


class ChordEmbedding:
    """Chord embedding layer for harmonic analysis."""
    
    def __init__(self, embedding_dim=16):
        """
        Initialize chord embedding.
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.embeddings = self._create_chord_embeddings()
    
    def _create_chord_embeddings(self):
        """Create harmonic embeddings for all 24 chords."""
        # Create embeddings based on harmonic relationships
        embeddings = {}
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for i, note in enumerate(notes):
            # Major chord embedding
            major_chord = f"{note}maj"
            embeddings[major_chord] = self._harmonic_embedding(i, mode='major')
            
            # Minor chord embedding
            minor_chord = f"{note}min"
            embeddings[minor_chord] = self._harmonic_embedding(i, mode='minor')
        
        return embeddings
    
    def _harmonic_embedding(self, root_idx, mode='major', dim=16):
        """Create harmonic embedding for a chord."""
        embed = np.zeros(dim)
        
        # Root note
        embed[0] = np.cos(2 * np.pi * root_idx / 12)
        embed[1] = np.sin(2 * np.pi * root_idx / 12)
        
        # Third interval
        third_idx = (root_idx + (4 if mode == 'major' else 3)) % 12
        embed[2] = np.cos(2 * np.pi * third_idx / 12)
        embed[3] = np.sin(2 * np.pi * third_idx / 12)
        
        # Fifth interval
        fifth_idx = (root_idx + 7) % 12
        embed[4] = np.cos(2 * np.pi * fifth_idx / 12)
        embed[5] = np.sin(2 * np.pi * fifth_idx / 12)
        
        # Mode indicator
        embed[6] = 1 if mode == 'major' else -1
        
        return embed / np.linalg.norm(embed)
    
    def get_embedding(self, chord_name):
        """Get embedding vector for a chord."""
        return self.embeddings.get(chord_name, np.zeros(self.embedding_dim))
    
    def chord_distance(self, chord1, chord2):
        """Compute distance between two chords."""
        emb1 = self.get_embedding(chord1)
        emb2 = self.get_embedding(chord2)
        return np.linalg.norm(emb1 - emb2)
    
    def find_similar_chords(self, chord_name, k=5):
        """Find k most similar chords."""
        distances = []
        for other_chord in self.embeddings.keys():
            if other_chord != chord_name:
                dist = self.chord_distance(chord_name, other_chord)
                distances.append((other_chord, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]


def extract_mel_spectrogram_features(y, sr, n_mels=128, n_fft=2048, hop_length=512):
    """
    Extract mel spectrogram features for model input.
    
    Args:
        y: Audio time series
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT size
        hop_length: Hop length
    
    Returns:
        X: Features array of shape (n_mels, n_frames, 1)
    """
    import librosa
    
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Normalize
    S_db = (S_db - np.mean(S_db)) / (np.std(S_db) + 1e-6)
    
    # Add channel dimension
    X = np.expand_dims(S_db, axis=-1)
    
    return X
