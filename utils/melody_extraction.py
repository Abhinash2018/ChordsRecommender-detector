# utils/melody_extraction.py
"""
Extract melody (fundamental frequency over time) using CREPE if available,
otherwise fallback to librosa.pyin. Includes vibrato detection and note segmentation.
Returns: times (s), f0 (Hz array), confidence, vibrato_info
"""

import numpy as np
from scipy import signal


def extract_melody(y, sr, hop_length=512):
    """
    Extract fundamental frequency using CREPE or librosa fallback.
    
    Args:
        y: Audio time series
        sr: Sample rate
        hop_length: Hop length in samples
    
    Returns:
        times: Time bins (seconds)
        f0: Fundamental frequency (Hz), NaN for unvoiced
        confidence: Confidence scores for each frame
    """
    try:
        # try to import crepe
        import crepe
        return _extract_with_crepe(y, sr, hop_length)
    except Exception:
        # fallback to librosa's pyin
        return _extract_with_librosa(y, sr, hop_length)


def _extract_with_librosa(y, sr, hop_length):
    """Extract pitch using librosa's pyin algorithm."""
    import librosa
    # librosa.pyin returns f0 in Hz (or NaN), and voiced_prob
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=hop_length
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    return times, f0, voiced_prob


def _extract_with_crepe(y, sr, hop_length):
    """Extract pitch using CREPE model."""
    import crepe
    import numpy as np
    # crepe expects float32 and sample rate
    # crepe.predict takes a numpy array and returns pitch and confidence
    # choose model 'full' or 'tiny' depending on availability (tiny is faster)
    model = "tiny"  # use 'full' if you want higher accuracy and have horsepower
    # crepe wants audio normalized between -1 and 1
    audio = y.astype('float32')
    # crepe.predict returns (time, frequency, confidence, activation)
    time, frequency, confidence, activation = crepe.predict(
        audio, sr, model=model, step_size=hop_length*1000/sr
    )
    # frequency is 0 for unvoiced, convert to np.nan
    f0 = np.where(frequency <= 0, np.nan, frequency)
    return time, f0, confidence


def detect_vibrato(f0, times, fmin=4.0, fmax=8.0, min_duration=0.1):
    """
    Detect vibrato in the pitch contour.
    
    Args:
        f0: Fundamental frequency array
        times: Time array
        fmin: Minimum vibrato frequency (Hz)
        fmax: Maximum vibrato frequency (Hz)
        min_duration: Minimum vibrato duration (seconds)
    
    Returns:
        vibrato_segments: List of (start_time, end_time, rate, extent)
    """
    # Remove NaNs for analysis
    valid_idx = ~np.isnan(f0)
    if np.sum(valid_idx) < 10:
        return []
    
    f0_valid = f0[valid_idx]
    times_valid = times[valid_idx]
    
    # Smooth the contour
    if len(f0_valid) > 5:
        f0_smooth = signal.medfilt(f0_valid, kernel_size=5)
    else:
        f0_smooth = f0_valid
    
    # Compute first derivative (rate of change)
    f0_derivative = np.gradient(f0_smooth)
    
    # Compute FFT of derivative to find dominant vibrato frequency
    sr_frames = 1.0 / np.mean(np.diff(times_valid))  # frame rate
    freqs = np.fft.fftfreq(len(f0_derivative), 1/sr_frames)
    
    # Look only at positive frequencies in vibrato range
    vibrato_mask = (freqs >= fmin) & (freqs <= fmax)
    
    if np.any(vibrato_mask):
        vibrato_freqs = freqs[vibrato_mask]
        power = np.abs(np.fft.fft(f0_derivative)[vibrato_mask])
        if len(power) > 0:
            dominant_freq = vibrato_freqs[np.argmax(power)]
            vibrato_extent = np.std(f0_smooth)
            return [('vibrato', dominant_freq, vibrato_extent)]
    
    return []


def segment_into_notes(f0, times, confidence, threshold_dur=0.05, confidence_thresh=0.5):
    """
    Segment pitch contour into discrete notes.
    
    Args:
        f0: Fundamental frequency array
        times: Time array
        confidence: Confidence scores
        threshold_dur: Minimum note duration (seconds)
        confidence_thresh: Minimum confidence to include frame
    
    Returns:
        notes: List of (start_time, end_time, avg_freq, duration)
    """
    # Filter by confidence
    valid_mask = (confidence >= confidence_thresh) & (~np.isnan(f0))
    
    notes = []
    in_note = False
    note_start = 0
    note_freqs = []
    
    for i in range(len(f0)):
        if valid_mask[i]:
            if not in_note:
                in_note = True
                note_start = i
                note_freqs = [f0[i]]
            else:
                note_freqs.append(f0[i])
        else:
            if in_note:
                # End of note
                duration = times[i-1] - times[note_start]
                if duration >= threshold_dur and len(note_freqs) > 0:
                    avg_freq = np.mean(note_freqs)
                    notes.append((times[note_start], times[i-1], avg_freq, duration))
                in_note = False
                note_freqs = []
    
    # Handle final note
    if in_note and len(note_freqs) > 0:
        duration = times[-1] - times[note_start]
        if duration >= threshold_dur:
            avg_freq = np.mean(note_freqs)
            notes.append((times[note_start], times[-1], avg_freq, duration))
    
    return notes
