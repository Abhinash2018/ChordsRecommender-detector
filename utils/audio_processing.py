"""
Audio processing module for loading and analyzing audio files.
Handles various audio formats and performs spectral analysis.
"""

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
import io


def load_audio(uploaded_file, sr_target=22050):
    """
    Load uploaded audio (file-like) into numpy array mono and sample rate sr_target.
    uploaded_file: a file-like object (BytesIO from Streamlit uploader)
    Returns (y, sr)
    """
    # Read bytes
    audio_bytes = uploaded_file.read()
    # Use soundfile to read bytes buffer
    with io.BytesIO(audio_bytes) as buf:
        # soundfile can detect format
        data, sr = sf.read(buf)
        # If multi-channel, convert to mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        # Resample if necessary
        if sr != sr_target:
            data = librosa.resample(data.astype(float), orig_sr=sr, target_sr=sr_target)
            sr = sr_target
        return data.astype(np.float32), sr


def compute_spectrogram(y, sr, n_fft=2048, hop_length=512):
    """
    Compute Short-Time Fourier Transform spectrogram.
    
    Args:
        y: Audio time series
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Number of samples between frames
    
    Returns:
        S: Complex spectrogram
        freqs: Frequency bins
        times: Time bins
    """
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    frames = np.arange(S.shape[1])
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    
    return S, freqs, times


def compute_mel_spectrogram(y, sr, n_mels=128, n_fft=2048, hop_length=512):
    """
    Compute Mel-scale spectrogram for analysis.
    
    Args:
        y: Audio time series
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Number of samples between frames
    
    Returns:
        S_mel: Mel spectrogram (dB scale)
        times: Time bins
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    
    frames = np.arange(S_db.shape[1])
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    
    return S_db, times


def compute_chroma_features(y, sr, hop_length=512):
    """
    Compute chroma features from audio.
    Useful for key and chord detection.
    
    Args:
        y: Audio time series
        sr: Sample rate
        hop_length: Hop length for STFT
    
    Returns:
        chroma: Chroma features (12, n_frames)
        times: Time bins
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    
    frames = np.arange(chroma.shape[1])
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    
    return chroma, times


def extract_loudness_contour(y, sr, hop_length=512):
    """
    Extract loudness (RMS energy) contour over time.
    
    Args:
        y: Audio time series
        sr: Sample rate
        hop_length: Hop length
    
    Returns:
        loudness: RMS energy per frame
        times: Time bins
    """
    loudness = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    frames = np.arange(len(loudness))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    
    return loudness, times


def smooth_signal(signal_in, window_size=5):
    """
    Apply median smoothing to a signal.
    
    Args:
        signal_in: Input signal
        window_size: Window size for median filter
    
    Returns:
        smoothed: Smoothed signal
    """
    return signal.medfilt(signal_in, kernel_size=window_size)
