# utils/key_detection.py
"""
Advanced key and harmonic analysis from audio features.
Detects musical key, scale, and harmonic patterns.
"""

import numpy as np
from collections import Counter
import librosa
import music21 as m21


# Pitch class templates (Krumhansl-Schmuckler key profiles)
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

CHROMATIC_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def hz_to_pitch_class(hz):
    """Convert frequency (Hz) to pitch class (0-11)."""
    midi = librosa.hz_to_midi(hz)
    return int(midi) % 12


def hz_to_note_name(hz):
    """Convert frequency (Hz) to note name (e.g., 'C#')."""
    midi = librosa.hz_to_midi(hz)
    return librosa.midi_to_note(midi, octave=False)


def chroma_vector_to_key(chroma_features):
    """
    Estimate key from chroma features using Krumhansl-Schmuckler algorithm.
    
    Args:
        chroma_features: Array of shape (12, n_frames) from librosa
    
    Returns:
        tonic: Tonic note name (e.g., 'C')
        mode: 'major' or 'minor'
        confidence: Confidence score (0-1)
    """
    # Average chroma over time
    chroma_mean = np.mean(chroma_features, axis=1)
    chroma_mean = chroma_mean / np.sum(chroma_mean)  # normalize
    
    best_correlation = -1
    best_tonic = 'C'
    best_mode = 'major'
    
    # Try all 12 pitch classes
    for shift in range(12):
        # Rotate major profile
        major_corr = np.corrcoef(
            np.roll(MAJOR_PROFILE, shift),
            chroma_mean
        )[0, 1]
        
        if major_corr > best_correlation:
            best_correlation = major_corr
            best_tonic = CHROMATIC_NOTES[shift]
            best_mode = 'major'
        
        # Rotate minor profile
        minor_corr = np.corrcoef(
            np.roll(MINOR_PROFILE, shift),
            chroma_mean
        )[0, 1]
        
        if minor_corr > best_correlation:
            best_correlation = minor_corr
            best_tonic = CHROMATIC_NOTES[shift]
            best_mode = 'minor'
    
    return best_tonic, best_mode, max(0, best_correlation)


def estimate_key_from_notes(f0, sr=None):
    """
    Estimate key from pitch contour (f0 values in Hz).
    
    Args:
        f0: Array of Hz values with NaN for unvoiced
        sr: Sample rate (optional, not used here)
    
    Returns:
        tonic: Tonic note name (e.g., 'C')
        mode: 'major' or 'minor'
        top_notes: List of most frequent note names
    """
    # Drop NaNs and convert Hz to pitch class
    voiced = f0[~np.isnan(f0)]
    if len(voiced) == 0:
        return "C", "major", []
    
    pitch_classes = [hz_to_pitch_class(hz) for hz in voiced]
    note_names = [hz_to_note_name(hz) for hz in voiced]
    
    # Count pitch class frequencies
    pc_counts = Counter(pitch_classes)
    note_counts = Counter(note_names)
    
    # Create chroma vector
    chroma_vector = np.zeros(12)
    for pc, count in pc_counts.items():
        chroma_vector[pc] = count
    
    # Normalize
    if np.sum(chroma_vector) > 0:
        chroma_vector = chroma_vector / np.sum(chroma_vector)
    
    # Detect key using profile correlation
    tonic, mode, confidence = _estimate_key_from_chroma(chroma_vector)
    
    # Return most common notes
    top_notes = [n for n, _ in note_counts.most_common()]
    
    return tonic, mode, top_notes


def _estimate_key_from_chroma(chroma_vector):
    """
    Internal function to estimate key from normalized chroma vector.
    """
    best_correlation = -1
    best_tonic = 'C'
    best_mode = 'major'
    
    for shift in range(12):
        major_corr = np.corrcoef(
            np.roll(MAJOR_PROFILE, shift),
            chroma_vector
        )[0, 1]
        
        if major_corr > best_correlation:
            best_correlation = major_corr
            best_tonic = CHROMATIC_NOTES[shift]
            best_mode = 'major'
        
        minor_corr = np.corrcoef(
            np.roll(MINOR_PROFILE, shift),
            chroma_vector
        )[0, 1]
        
        if minor_corr > best_correlation:
            best_correlation = minor_corr
            best_tonic = CHROMATIC_NOTES[shift]
            best_mode = 'minor'
    
    return best_tonic, best_mode, max(0, best_correlation)


def detect_scale_degrees(tonic, mode, f0):
    """
    Classify detected notes as scale degrees.
    
    Args:
        tonic: Tonic note (e.g., 'C')
        mode: 'major' or 'minor'
        f0: Array of Hz values with NaN for unvoiced
    
    Returns:
        scale_degrees: List of (degree, frequency) tuples
    """
    # Define scales (interval patterns from tonic)
    major_intervals = {0: 'I', 2: 'II', 4: 'III', 5: 'IV', 7: 'V', 9: 'VI', 11: 'VII'}
    minor_intervals = {0: 'i', 2: 'ii', 3: 'III', 5: 'v', 7: 'VI', 8: 'vii', 10: 'VII'}
    
    # Find tonic pitch class
    tonic_pc = CHROMATIC_NOTES.index(tonic.replace('b', '-').split('-')[0])
    
    # Filter out NaN values and convert Hz to pitch class
    voiced = f0[~np.isnan(f0)]
    pc_counts = Counter([hz_to_pitch_class(hz) for hz in voiced])
    
    scale_degrees = []
    intervals = major_intervals if mode == 'major' else minor_intervals
    
    for interval, degree in intervals.items():
        pc = (tonic_pc + interval) % 12
        count = pc_counts.get(pc, 0)
        if count > 0:
            scale_degrees.append((degree, count))
    
    return sorted(scale_degrees, key=lambda x: x[1], reverse=True)
                       