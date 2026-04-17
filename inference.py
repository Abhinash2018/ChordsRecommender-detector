"""
Inference script for chord classification and audio analysis.
Can be used to process audio files with the trained model.

Usage:
    python inference.py --audio_file "path/to/audio.wav" --model_path "./chord_classifier_model"
"""

import argparse
import numpy as np
import librosa
from pathlib import Path

from utils.chord_classifier import ChordClassifierModel, extract_mel_spectrogram_features
from utils.audio_processing import (
    load_audio, compute_chroma_features, compute_mel_spectrogram,
    extract_loudness_contour
)
from utils.melody_extraction import extract_melody, segment_into_notes, detect_vibrato
from utils.key_detection import chroma_vector_to_key, estimate_key_from_notes, detect_scale_degrees
from utils.chord_recommender import analyze_mood, recommend_mood_aware_chords, detect_harmonic_patterns
from config import AUDIO_CONFIG, PITCH_CONFIG


def analyze_audio(audio_file, model_path=None, verbose=True):
    """
    Comprehensive audio analysis and chord prediction.
    
    Args:
        audio_file: Path to audio file
        model_path: Path to trained model (optional)
        verbose: Print detailed results
    
    Returns:
        results: Dictionary with analysis results
    """
    results = {}
    
    # Load audio
    print(f"🎧 Loading audio: {audio_file}")
    try:
        y, sr = librosa.load(audio_file, sr=AUDIO_CONFIG["sample_rate"], mono=True)
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return None
    
    results['duration'] = len(y) / sr
    results['sample_rate'] = sr
    
    if verbose:
        print(f"✅ Audio loaded: {results['duration']:.2f}s @ {sr} Hz")
    
    # Extract melody (pitch detection)
    print("\n🎤 Extracting melody...")
    times, f0, confidence = extract_melody(y, sr)
    results['pitch'] = {'times': times, 'f0': f0, 'confidence': confidence}
    
    if len(f0) == 0 or np.all(np.isnan(f0)):
        print("⚠️  No pitched content detected")
        return results
    
    # Key detection
    print("\n🎼 Detecting key...")
    tonic, mode, notes = estimate_key_from_notes(f0, sr)
    results['key'] = {'tonic': tonic, 'mode': mode, 'top_notes': notes}
    
    if verbose:
        print(f"✅ Detected Key: {tonic} {mode.capitalize()}")
    
    # Compute audio features
    print("\n📊 Computing audio features...")
    mel_spec, mel_times = compute_mel_spectrogram(y, sr)
    chroma, chroma_times = compute_chroma_features(y, sr)
    loudness, loud_times = extract_loudness_contour(y, sr)
    
    results['mel_spectrogram'] = mel_spec
    results['chroma'] = chroma
    results['loudness'] = loudness
    
    # Key detection from chroma
    tonic_chroma, mode_chroma, key_conf = chroma_vector_to_key(chroma)
    results['key_confidence'] = key_conf
    
    if verbose:
        print(f"✅ Key confidence: {key_conf:.2%}")
    
    # Mood analysis
    print("\n😊 Analyzing mood...")
    mood, mood_score = analyze_mood(loudness, np.mean(chroma, axis=0), chroma, key_conf)
    results['mood'] = {'type': mood, 'score': mood_score}
    
    if verbose:
        print(f"✅ Detected mood: {mood.capitalize()} (score: {mood_score:.2%})")
    
    # Harmonic pattern detection
    print("\n🔍 Detecting harmonic patterns...")
    chord_changes = detect_harmonic_patterns(chroma, chroma_times)
    results['chord_changes'] = chord_changes
    
    if verbose and chord_changes:
        print(f"✅ Detected {len(chord_changes)} chord changes")
    
    # Vibrato detection
    print("\n🎵 Detecting vibrato...")
    vibrato_info = detect_vibrato(f0, times)
    results['vibrato'] = vibrato_info
    
    if verbose and vibrato_info:
        print(f"✅ Vibrato detected")
    
    # Note segmentation
    print("\n📝 Segmenting notes...")
    note_segments = segment_into_notes(f0, times, confidence)
    results['notes'] = note_segments
    
    if verbose:
        print(f"✅ Detected {len(note_segments)} notes")
    
    # Scale degrees
    scale_degrees = detect_scale_degrees(tonic, mode, notes)
    results['scale_degrees'] = scale_degrees
    
    # Deep learning chord classification
    if model_path:
        print("\n🤖 Running deep learning chord classification...")
        try:
            model = ChordClassifierModel()
            model.load(model_path)
            
            # Pad/truncate mel spectrogram
            from utils.data_utils import FeatureBatcher
            batcher = FeatureBatcher()
            X = batcher.pad_or_truncate(np.expand_dims(mel_spec, axis=0), target_length=40)
            
            # Predict
            predictions = model.predict(X)
            results['chord_predictions'] = predictions
            
            if verbose and predictions:
                print(f"✅ Predicted chord: {predictions[0][0]} ({predictions[0][1]:.2%} confidence)")
        
        except Exception as e:
            print(f"⚠️  Could not load deep learning model: {e}")
    
    return results


def print_analysis_report(results, verbose=True):
    """
    Print comprehensive analysis report.
    
    Args:
        results: Analysis results dictionary
        verbose: Print detailed output
    """
    if results is None:
        print("❌ No results to report")
        return
    
    print("\n" + "=" * 60)
    print("ANALYSIS REPORT")
    print("=" * 60)
    
    # Basic info
    print(f"\n📁 Audio Information:")
    print(f"   Duration: {results.get('duration', 0):.2f}s")
    print(f"   Sample Rate: {results.get('sample_rate', 0)} Hz")
    
    # Key
    key_info = results.get('key', {})
    print(f"\n🎼 Key Detection:")
    print(f"   Key: {key_info.get('tonic', 'N/A')} {key_info.get('mode', 'N/A').capitalize()}")
    print(f"   Confidence: {results.get('key_confidence', 0):.2%}")
    print(f"   Top Notes: {', '.join(key_info.get('top_notes', [])[:5])}")
    
    # Mood
    mood_info = results.get('mood', {})
    print(f"\n😊 Mood Analysis:")
    print(f"   Type: {mood_info.get('type', 'N/A').capitalize()}")
    print(f"   Score: {mood_info.get('score', 0):.2%}")
    
    # Harmonic patterns
    chord_changes = results.get('chord_changes', [])
    print(f"\n🔍 Harmonic Patterns:")
    print(f"   Chord Changes Detected: {len(chord_changes)}")
    for i, (time, strength) in enumerate(chord_changes[:3], 1):
        print(f"   - {i}. {time:.2f}s (strength: {strength:.3f})")
    
    # Notes
    notes = results.get('notes', [])
    print(f"\n📝 Note Segments:")
    print(f"   Total Notes: {len(notes)}")
    if notes:
        print(f"   Avg Duration: {np.mean([n[3] for n in notes]):.3f}s")
    
    # Vibrato
    vibrato = results.get('vibrato', [])
    print(f"\n🎵 Vibrato:")
    if vibrato:
        for vib_type, rate, extent in vibrato:
            print(f"   - Rate: {rate:.2f} Hz")
            print(f"   - Extent: {extent:.2f} cents")
    else:
        print(f"   None detected")
    
    # Deep learning predictions
    predictions = results.get('chord_predictions', [])
    if predictions:
        print(f"\n🤖 Deep Learning Predictions:")
        for pred in predictions[:3]:
            print(f"   - {pred[0]}: {pred[1]:.2%}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze audio and predict chords"
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained chord classifier model"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--no_verbose",
        action="store_true",
        help="Disable verbose output"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    results = analyze_audio(
        args.audio_file,
        model_path=args.model_path,
        verbose=not args.no_verbose
    )
    
    # Print report
    print_analysis_report(results, verbose=not args.no_verbose)
    
    # Save results
    if args.output_file and results:
        import json
        
        # Convert non-serializable types
        results_safe = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_safe[key] = value.tolist()
            elif isinstance(value, dict):
                results_safe[key] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                     for k, v in value.items()}
            elif isinstance(value, list):
                results_safe[key] = [(item.tolist() if isinstance(item, np.ndarray) else item)
                                    for item in value]
            else:
                results_safe[key] = value
        
        with open(args.output_file, 'w') as f:
            json.dump(results_safe, f, indent=2)
        
        print(f"\n✅ Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
