# app.py
"""
AI Audio-to-Chord Recommendation System
Streamlit app: Upload singing audio -> detect melody -> analyze key -> recommend chords
Includes mood-aware suggestions, harmonic pattern detection, and deep learning chord classification.
"""

import streamlit as st
import numpy as np
import io
import matplotlib.pyplot as plt

try:
    from utils.audio_processing import (
        load_audio, compute_mel_spectrogram, compute_chroma_features, 
        extract_loudness_contour
    )
    from utils.melody_extraction import extract_melody, detect_vibrato, segment_into_notes
    from utils.key_detection import (
        estimate_key_from_notes, chroma_vector_to_key, 
        detect_scale_degrees
    )
    from utils.chord_recommender import (
        recommend_progressions, chord_diagram_for, analyze_mood,
        recommend_mood_aware_chords, detect_harmonic_patterns
    )
    MODULES_READY = True
except ImportError as e:
    MODULES_READY = False
    import_error = str(e)

# Page config
st.set_page_config(
    page_title="Sing2Chords - AI Audio-to-Chord System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎵 Sing2Chords")
st.caption("AI-powered chord recommendation from your vocal recordings")

if not MODULES_READY:
    st.error(f"⚠️ Missing dependencies. Please install required packages:\n\n`pip install -r requirements.txt`\n\nError: {import_error}")
    st.stop()


# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    sr = st.slider("Sample Rate (Hz)", 16000, 48000, 22050, 1000)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
    show_advanced = st.checkbox("Show Advanced Analysis", value=False)

# Main content
uploaded = st.file_uploader(
    "Upload a singing file (WAV / MP3 / FLAC / M4A)",
    type=["wav", "mp3", "flac", "m4a"]
)

max_file_mb = 20
if uploaded is not None:
    # Check file size
    uploaded.seek(0, io.SEEK_END)
    size_mb = uploaded.tell() / (1024 * 1024)
    uploaded.seek(0)
    
    if size_mb > max_file_mb:
        st.error(f"❌ File is too big ({size_mb:.1f} MB). Max allowed {max_file_mb} MB.")
    else:
        # Load audio
        with st.spinner("🎧 Loading audio..."):
            y, sr_actual = load_audio(uploaded, sr_target=sr)
        
        st.success(f"✅ Audio loaded — {len(y)/sr_actual:.1f}s at {sr_actual} Hz")
        
        # Create columns for parallel processing
        col1, col2 = st.columns(2)
        
        with col1:
            # Extract melody
            with st.spinner("🎤 Extracting melody (pitch detection)..."):
                times, f0, confidence = extract_melody(y, sr_actual)
        
        with col2:
            # Compute features
            with st.spinner("📊 Computing audio features..."):
                mel_spec, mel_times = compute_mel_spectrogram(y, sr_actual)
                chroma, chroma_times = compute_chroma_features(y, sr_actual)
                loudness, loud_times = extract_loudness_contour(y, sr_actual)
        
        # Validate melody
        if len(f0) == 0 or np.all(np.isnan(f0)):
            st.warning("⚠️ No pitched content detected. Try a clearer vocal take or a different file.")
        else:
            # Main analysis section
            st.markdown("---")
            st.header("📈 Analysis Results")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(
                ["Pitch & Key", "Mood Analysis", "Harmonic Patterns", "Advanced"]
            )
            
            with tab1:
                # Key detection
                tonic, mode, notes = estimate_key_from_notes(f0, sr_actual)
                tonic_chroma, mode_chroma, key_conf = chroma_vector_to_key(chroma)
                
                # Display key results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎼 Detected Key", f"{tonic} {mode.capitalize()}", 
                             f"Confidence: {key_conf:.2%}")
                with col2:
                    st.metric("🎵 Top Notes", len(notes), 
                             f"({', '.join(notes[:3])}...)")
                with col3:
                    st.metric("⏱️ Duration", f"{len(y)/sr_actual:.1f}s")
                
                # Pitch contour plot
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(times, f0, linewidth=2, label="Pitch contour", color='#1f77b4')
                ax.fill_between(times, 0, f0, alpha=0.2, color='#1f77b4')
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Frequency (Hz)")
                ax.set_title("Detected Pitch Contour")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Chroma plot
                fig, ax = plt.subplots(figsize=(12, 4))
                im = ax.imshow(chroma, aspect='auto', origin='lower', cmap='magma')
                ax.set_ylabel("Pitch Class")
                ax.set_xlabel("Time (s)")
                ax.set_title("Chroma Features (Pitch Content)")
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)
                
                # Scale degrees
                scale_degrees = detect_scale_degrees(tonic, mode, f0)
                st.markdown("**Scale Degree Distribution:**")
                st.bar_chart(dict(scale_degrees))
            
            with tab2:
                # Mood analysis
                mood, mood_score = analyze_mood(loudness, np.mean(chroma, axis=0), chroma, key_conf)
                
                st.metric("😊 Detected Mood", mood.capitalize(), f"Score: {mood_score:.2%}")
                
                # Mood-aware chord recommendations
                mood_chords, mood_desc = recommend_mood_aware_chords(tonic, mode, mood)
                st.markdown(f"**Mood-Aware Progression:** {' → '.join(mood_chords)}")
                st.markdown(f"*{mood_desc}*")
                
                # Loudness contour
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(loud_times, loudness, linewidth=2, color='#ff7f0e')
                ax.fill_between(loud_times, loudness, alpha=0.2, color='#ff7f0e')
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Loudness (RMS Energy)")
                ax.set_title("Audio Loudness Envelope")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with tab3:
                # Harmonic patterns and chord changes
                chord_changes = detect_harmonic_patterns(chroma, chroma_times)
                
                if chord_changes:
                    st.markdown("Detected Chord Changes:")
                    for time, strength in chord_changes[:5]:
                        st.markdown(f"  • {time:.2f}s — Strength: {strength:.3f}")
                else:
                    st.info("No significant chord changes detected.")
                
                # Standard progressions
                st.markdown("## Standard Chord Progressions")
                progressions = recommend_progressions(tonic, mode)
                
                for i, p in enumerate(progressions, 1):
                    with st.expander(f"{i}. {p['name']}"):
                        prog_chords = p["chords"]
                        st.markdown(f"**Chord Progression:** {' → '.join(prog_chords)}")
                        
                        # Show chord diagrams
                        cols = st.columns(len(prog_chords))
                        for col, chord in zip(cols, prog_chords):
                            with col:
                                diag = chord_diagram_for(chord)
                                st.code(diag, language="text")
            
            with tab4:
                if show_advanced:
                    # Vibrato detection
                    vibrato_info = detect_vibrato(f0, times)
                    if vibrato_info:
                        st.markdown("**Vibrato Detected:**")
                        for info_type, rate, extent in vibrato_info:
                            st.markdown(f"  • Rate: {rate:.2f} Hz, Extent: {extent:.2f} cents")
                    
                    # Note segmentation
                    note_segments = segment_into_notes(f0, times, confidence)
                    if note_segments:
                        st.markdown(f"**Note Segments: {len(note_segments)} notes detected**")
                        st.dataframe({
                            'Start (s)': [s[0] for s in note_segments],
                            'End (s)': [s[1] for s in note_segments],
                            'Frequency (Hz)': [f"{s[2]:.1f}" for s in note_segments],
                            'Duration (s)': [f"{s[3]:.3f}" for s in note_segments],
                        })
                    
                    # Mel spectrogram
                    fig, ax = plt.subplots(figsize=(12, 6))
                    im = ax.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
                    ax.set_ylabel("Mel Frequency Bin")
                    ax.set_xlabel("Time (s)")
                    ax.set_title("Mel Spectrogram")
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("Enable 'Show Advanced Analysis' in settings to see detailed features.")
            
            # Tips section
            st.markdown("---")
            st.info(
                "💡 Tips for best results:\n"
                "- Use a clear vocal recording without background music\n"
                "- Ensure good audio quality and minimal noise\n"
                "- If suggested chords feel high/low, try using a capo\n"
                "- Transpose by a semitone or two if it doesn't match your key"
            )
