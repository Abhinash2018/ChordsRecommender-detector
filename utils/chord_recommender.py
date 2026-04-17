# utils/chord_recommender.py
"""
Advanced chord recommendation system with mood analysis and harmonic patterns.
Recommends chord progressions based on key, mode, and detected mood.
"""

import music21 as m21
import numpy as np
from collections import Counter


MAJOR_DEGREE_CHORDS = {
    1: lambda t: t,
    2: lambda t: t + "m",
    3: lambda t: t + "m",
    4: lambda t: t,
    5: lambda t: t,
    6: lambda t: t + "m",
    7: lambda t: t + "dim",
}

MINOR_DEGREE_CHORDS = {
    1: lambda t: t + "m",
    2: lambda t: t + "dim",
    3: lambda t: t,
    4: lambda t: t + "m",
    5: lambda t: t + "m",
    6: lambda t: t,
    7: lambda t: t,
}


def chord_name_from_degree(tonic, degree, mode='major'):
    """
    Return chord name (e.g., 'G', 'Am') for tonic and scale degree (1..7)
    """
    scale = m21.scale.MajorScale(tonic) if mode == 'major' else m21.scale.MinorScale(tonic)
    pitch = scale.pitchFromDegree(degree)
    base = pitch.name  # e.g., 'G' or 'G#'
    if mode == 'major':
        chord = MAJOR_DEGREE_CHORDS[degree](base)
    else:
        chord = MINOR_DEGREE_CHORDS[degree](base)
    return chord


def recommend_progressions(tonic, mode='major'):
    """
    Return a list of recommended progressions with human-readable names.
    """
    # list of progression templates (degrees)
    templates = [
        {"name": "I–V–vi–IV (pop)", "degrees": [1, 5, 6, 4]},
        {"name": "I–IV–V (classic)", "degrees": [1, 4, 5]},
        {"name": "vi–IV–I–V (ballad)", "degrees": [6, 4, 1, 5]},
        {"name": "ii–V–I (jazz-ish)", "degrees": [2, 5, 1]},
        {"name": "I–vi–IV–V (50s)", "degrees": [1, 6, 4, 5]},
        {"name": "I–IV–vi–V (modern)", "degrees": [1, 4, 6, 5]},
    ]
    results = []
    for t in templates:
        chords = [chord_name_from_degree(tonic, d, mode) for d in t["degrees"]]
        results.append({"name": t["name"], "chords": chords})
    return results


def analyze_mood(loudness, spectral_centroid, chroma_features, key_confidence):
    """
    Analyze audio characteristics to estimate mood.
    
    Args:
        loudness: RMS energy over time
        spectral_centroid: Spectral centroid over time
        chroma_features: Chroma feature matrix
        key_confidence: Confidence of key detection
    
    Returns:
        mood: 'bright', 'dark', 'energetic', 'calm', or 'neutral'
        mood_score: Float between 0 and 1
    """
    mood_features = {}
    
    # Loudness analysis
    loudness_mean = np.mean(loudness)
    loudness_std = np.std(loudness)
    
    # Spectral analysis
    sc_mean = np.mean(spectral_centroid) if len(spectral_centroid) > 0 else 0
    
    # Harmonic analysis from chroma
    chroma_variance = np.var(np.mean(chroma_features, axis=1))
    
    # Determine mood
    brightness = sc_mean / 11000  # normalized by typical max freq
    energy = loudness_mean
    harmony = key_confidence
    
    if brightness > 0.6:
        mood = 'bright'
    elif brightness < 0.3:
        mood = 'dark'
    elif energy > 0.7:
        mood = 'energetic'
    elif energy < 0.3:
        mood = 'calm'
    else:
        mood = 'neutral'
    
    mood_score = (brightness + energy + harmony) / 3
    
    return mood, mood_score


def recommend_mood_aware_chords(tonic, mode, mood):
    """
    Recommend chord progressions based on detected mood.
    
    Args:
        tonic: Tonic note (e.g., 'C')
        mode: 'major' or 'minor'
        mood: Mood classification
    
    Returns:
        progression: Recommended chord progression as list
        description: Description of the progression
    """
    # Mood-based progression selection
    mood_progressions = {
        'bright': {
            'major': {'degrees': [1, 5, 6, 4], 'desc': 'Uplifting pop progression'},
            'minor': {'degrees': [1, 5, 4, 1], 'desc': 'Hopeful minor progression'},
        },
        'dark': {
            'major': {'degrees': [1, 4, 5, 1], 'desc': 'Moody major progression'},
            'minor': {'degrees': [1, 6, 3, 7], 'desc': 'Dark atmospheric progression'},
        },
        'energetic': {
            'major': {'degrees': [1, 5, 1, 4], 'desc': 'High-energy rock progression'},
            'minor': {'degrees': [1, 4, 5, 1], 'desc': 'Powerful minor progression'},
        },
        'calm': {
            'major': {'degrees': [1, 6, 4, 5], 'desc': 'Peaceful major progression'},
            'minor': {'degrees': [1, 6, 3, 7], 'desc': 'Serene minor progression'},
        },
        'neutral': {
            'major': {'degrees': [1, 4, 5, 1], 'desc': 'Classic major progression'},
            'minor': {'degrees': [1, 4, 5, 1], 'desc': 'Standard minor progression'},
        },
    }
    
    prog_info = mood_progressions.get(mood, mood_progressions['neutral'])[mode]
    chords = [chord_name_from_degree(tonic, d, mode) for d in prog_info['degrees']]
    
    return chords, prog_info['desc']


def detect_harmonic_patterns(chroma_features, times, window_size=8):
    """
    Detect harmonic patterns and chord changes in chroma features.
    
    Args:
        chroma_features: Chroma feature matrix (12, n_frames)
        times: Time array
        window_size: Window size for change detection
    
    Returns:
        chord_changes: List of (time, change_strength) tuples
    """
    if chroma_features.shape[1] < window_size:
        return []
    
    # Compute chroma difference across frames
    chroma_diff = np.zeros(chroma_features.shape[1] - 1)
    for i in range(len(chroma_diff)):
        c1 = chroma_features[:, i]
        c2 = chroma_features[:, i + 1]
        # Cosine distance
        c1_norm = c1 / (np.linalg.norm(c1) + 1e-10)
        c2_norm = c2 / (np.linalg.norm(c2) + 1e-10)
        chroma_diff[i] = 1 - np.dot(c1_norm, c2_norm)
    
    # Find peaks (chord changes)
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(chroma_diff, height=np.mean(chroma_diff) + np.std(chroma_diff))
    
    chord_changes = [(times[p], chroma_diff[p]) for p in peaks if p < len(times)]
    
    return sorted(chord_changes, key=lambda x: x[1], reverse=True)[:5]


def chord_diagram_for(chord_name):
    """
    Return a simple ASCII chord fingering hint for common open chords.
    """
    diagrams = {
        "C": "C (x32010)",
        "G": "G (320003)",
        "D": "D (xx0232)",
        "Em": "Em (022000)",
        "Am": "Am (x02210)",
        "E": "E (022100)",
        "A": "A (x02220)",
        "D7": "D7 (xx0212)",
        "Dm": "Dm (xx0231)",
        "Gm": "Gm (355333)",
        "F": "F (133211)",
        "Bm": "Bm (x24432)",
    }
    # try full match then base name fallback
    if chord_name in diagrams:
        return diagrams[chord_name]
    # consider stripping minor suffix and try
    base = chord_name.replace("m", "")
    return diagrams.get(base, f"{chord_name} (no diagram available)")


def get_guitar_voicings(chord_name):
    """
    Return alternative voicings for a chord (different fret positions).
    """
    voicings = {
        "C": ["x32010", "x3201", "332010"],
        "G": ["320003", "320033", "3x0033"],
        "D": ["xx0232", "xx0323", "x54x32"],
        "Em": ["022000", "0220003", "2x0000"],
        "Am": ["x02210", "x0221", "5x555"],
    }
    return voicings.get(chord_name, [chord_diagram_for(chord_name)])
