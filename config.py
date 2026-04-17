"""
Configuration file for the Sing2Chords application.
Centralized settings for audio processing, models, and UI.
"""

# Audio Processing Parameters
AUDIO_CONFIG = {
    "sample_rate": 22050,
    "mono": True,
    "target_db": -20,  # Normalize to -20 dB
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 50,
    "fmax": 8000,
}

# Pitch Detection Parameters
PITCH_CONFIG = {
    "model": "tiny",  # 'tiny' or 'full' for CREPE
    "hop_length": 512,
    "fmin": 50,  # C1
    "fmax": 2000,  # C7
    "threshold": 0.1,  # Voicing threshold
    "confidence_threshold": 0.6,
}

# Key Detection Parameters
KEY_CONFIG = {
    "algorithm": "krumhansl",
    "window_size": 40,  # frames
    "hop_length": 10,  # frames
}

# Chord Recognition Parameters
CHORD_CONFIG = {
    "num_classes": 24,  # 12 major + 12 minor
    "confidence_threshold": 0.7,
    "model_path": "./models/chord_classifier_model",
}

# Feature Extraction Parameters
FEATURE_CONFIG = {
    "mel_spectrogram": {
        "n_mels": 128,
        "n_fft": 2048,
        "hop_length": 512,
    },
    "chroma": {
        "n_chroma": 12,
        "n_fft": 2048,
        "hop_length": 512,
    },
    "loudness": {
        "hop_length": 512,
    },
}

# Vibrato Detection Parameters
VIBRATO_CONFIG = {
    "fmin": 4.0,  # Hz
    "fmax": 8.0,  # Hz
    "min_duration": 0.1,  # seconds
}

# UI Parameters
UI_CONFIG = {
    "max_file_mb": 20,
    "default_theme": "light",
    "layout": "wide",
    "show_metrics": True,
    "show_advanced": False,
}

# Mood Analysis Parameters
MOOD_CONFIG = {
    "brightness_threshold_low": 0.3,
    "brightness_threshold_high": 0.6,
    "energy_threshold_low": 0.3,
    "energy_threshold_high": 0.7,
}

# Harmonic Analysis Parameters
HARMONIC_CONFIG = {
    "detect_chord_changes": True,
    "min_chord_duration": 0.5,  # seconds
    "chord_change_threshold": 0.3,  # cosine distance
}

# Model Architecture Parameters
MODEL_CONFIG = {
    "input_shape": (128, 40, 1),
    "conv_filters": [32, 64, 128],
    "lstm_units": [256, 128],
    "dense_units": [64, 32],
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_normalization": True,
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
}

# Training Parameters
TRAINING_CONFIG = {
    "epochs": 50,
    "batch_size": 32,
    "validation_split": 0.15,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
}

# Inference Parameters
INFERENCE_CONFIG = {
    "batch_processing": True,
    "batch_size": 64,
    "padding_mode": "constant",
    "target_length": 40,
}

# Chord Progression Templates
CHORD_PROGRESSIONS = {
    "pop": {
        "name": "I–V–vi–IV (pop)",
        "degrees": [1, 5, 6, 4],
        "description": "Modern pop standard progression"
    },
    "classic": {
        "name": "I–IV–V (classic)",
        "degrees": [1, 4, 5],
        "description": "Classic three-chord progression"
    },
    "ballad": {
        "name": "vi–IV–I–V (ballad)",
        "degrees": [6, 4, 1, 5],
        "description": "Emotional ballad movement"
    },
    "jazz": {
        "name": "ii–V–I (jazz)",
        "degrees": [2, 5, 1],
        "description": "Jazz standard changes"
    },
    "50s": {
        "name": "I–vi–IV–V (1950s)",
        "degrees": [1, 6, 4, 5],
        "description": "Classic 1950s progression"
    },
    "modern": {
        "name": "I–IV–vi–V (modern)",
        "degrees": [1, 4, 6, 5],
        "description": "Contemporary alternative progression"
    },
}

# Mood-Based Chord Recommendations
MOOD_PROGRESSIONS = {
    "bright": {
        "major": {
            "degrees": [1, 5, 6, 4],
            "desc": "Uplifting pop progression"
        },
        "minor": {
            "degrees": [1, 5, 4, 1],
            "desc": "Hopeful minor progression"
        }
    },
    "dark": {
        "major": {
            "degrees": [1, 4, 5, 1],
            "desc": "Moody major progression"
        },
        "minor": {
            "degrees": [1, 6, 3, 7],
            "desc": "Dark atmospheric progression"
        }
    },
    "energetic": {
        "major": {
            "degrees": [1, 5, 1, 4],
            "desc": "High-energy rock progression"
        },
        "minor": {
            "degrees": [1, 4, 5, 1],
            "desc": "Powerful minor progression"
        }
    },
    "calm": {
        "major": {
            "degrees": [1, 6, 4, 5],
            "desc": "Peaceful major progression"
        },
        "minor": {
            "degrees": [1, 6, 3, 7],
            "desc": "Serene minor progression"
        }
    }
}

# Krumhansl-Schmuckler Key Profiles
KEY_PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
}

# Note and Chord Mappings
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

CHORD_QUALITY = {
    "major": "maj",
    "minor": "min",
    "diminished": "dim",
    "augmented": "aug",
    "major7": "maj7",
    "minor7": "min7",
    "dominant7": "7",
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "sing2chords.log",
}


def get_config(section=None):
    """
    Retrieve configuration by section.
    
    Args:
        section: Configuration section name (e.g., 'audio', 'model')
    
    Returns:
        Configuration dict or full config if section is None
    """
    config = {
        "audio": AUDIO_CONFIG,
        "pitch": PITCH_CONFIG,
        "key": KEY_CONFIG,
        "chord": CHORD_CONFIG,
        "feature": FEATURE_CONFIG,
        "vibrato": VIBRATO_CONFIG,
        "ui": UI_CONFIG,
        "mood": MOOD_CONFIG,
        "harmonic": HARMONIC_CONFIG,
        "model": MODEL_CONFIG,
        "training": TRAINING_CONFIG,
        "inference": INFERENCE_CONFIG,
    }
    
    if section and section in config:
        return config[section]
    elif section is None:
        return config
    else:
        raise ValueError(f"Unknown configuration section: {section}")
