"""
Training script for chord classification model.
Can be run standalone to train on custom datasets.

Usage:
    python train_model.py --data_dir ./training_data --epochs 50 --batch_size 32
"""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

from utils.chord_classifier import ChordClassifierModel, extract_mel_spectrogram_features
from utils.data_utils import AudioDataset, FeatureBatcher, ChordEncoder


def train_model(data_dir, epochs=50, batch_size=32, test_split=0.2):
    """
    Train chord classification model on audio dataset.
    
    Args:
        data_dir: Directory containing training audio files
        epochs: Number of training epochs
        batch_size: Batch size for training
        test_split: Fraction of data to use for testing
    """
    print("🎼 Chord Classifier Training Pipeline")
    print("=" * 50)
    
    # 1. Load dataset
    print("\n📂 Loading dataset...")
    dataset = AudioDataset(data_dir, sr=22050)
    files_labels = dataset.load_from_directory(organize_by_chord=True)
    
    if len(files_labels) == 0:
        print("❌ No audio files found. Check data directory structure.")
        return
    
    print(f"✅ Found {len(files_labels)} audio files")
    
    # 2. Get unique chord labels
    unique_chords = sorted(set(label for _, label in files_labels))
    num_classes = len(unique_chords)
    print(f"✅ Number of chord classes: {num_classes}")
    print(f"   Classes: {', '.join(unique_chords)}")
    
    # 3. Split dataset
    print("\n📊 Splitting dataset...")
    train_files = [f[0] for f in files_labels]
    train_labels = [f[1] for f in files_labels]
    
    # 4. Extract features
    print("\n🎵 Extracting mel spectrogram features...")
    batcher = FeatureBatcher(sr=22050, n_mels=128, n_fft=2048, hop_length=512)
    X, valid_indices = batcher.extract_features_batch(train_files)
    
    print(f"✅ Extracted features shape: {X.shape}")
    
    # Pad/truncate to consistent length
    X = batcher.pad_or_truncate(X, target_length=40)
    print(f"✅ Padded features shape: {X.shape}")
    
    # Filter labels to valid files
    y_labels = [train_labels[i] for i in valid_indices]
    
    # 5. Encode labels
    print("\n🏷️  Encoding labels...")
    encoder = ChordEncoder(num_classes=num_classes)
    
    # Re-map labels if not all 24 standard chords
    chord_to_idx = {chord: idx for idx, chord in enumerate(unique_chords)}
    y_encoded = np.zeros((len(y_labels), num_classes))
    for i, label in enumerate(y_labels):
        idx = chord_to_idx.get(label, 0)
        y_encoded[i, idx] = 1
    
    print(f"✅ Encoded labels shape: {y_encoded.shape}")
    
    # 6. Train/test split
    print("\n🔀 Train/test split...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_split, random_state=42
    )
    
    # Further split training into train/val
    val_split = 0.15
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=42
    )
    
    print(f"✅ Training set: {X_train.shape}")
    print(f"✅ Validation set: {X_val.shape}")
    print(f"✅ Test set: {X_test.shape}")
    
    # 7. Build and train model
    print("\n🏗️  Building model...")
    model = ChordClassifierModel(
        input_shape=(128, 40, 1),
        num_classes=num_classes
    )
    model.build_model()
    
    print(f"✅ Model built successfully")
    print("\n📈 Training model...")
    
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # 8. Evaluate
    print("\n📊 Evaluating model...")
    test_loss, test_acc = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {test_acc:.4f}")
    print(f"✅ Test Loss: {test_loss:.4f}")
    
    # 9. Save model
    print("\n💾 Saving model...")
    model.save("chord_classifier_model")
    print("✅ Model saved successfully")
    
    # 10. Summary
    print("\n" + "=" * 50)
    print("✅ Training Complete!")
    print(f"   - Classes trained: {num_classes}")
    print(f"   - Final accuracy: {test_acc:.2%}")
    print(f"   - Model saved to: chord_classifier_model_*")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train chord classification model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./training_data",
        help="Directory containing training audio files"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing"
    )
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_split=args.test_split
    )


if __name__ == "__main__":
    main()
