#  Sing2Chords - AI Audio-to-Chord Recommendation System

An intelligent system for analyzing vocal recordings and recommending guitar or piano chords in real-time.

---

## 🎯 Project Overview

**Sing2Chords** is a comprehensive Python application combining deep learning and music theory to analyze uploaded/recorded audio and recommend chord progressions. The system uses:

### Core Technologies
- **Audio Analysis**: librosa, CREPE for pitch detection
- **Deep Learning**: TensorFlow/Keras (CNN-LSTM for chord classification)
- **Feature Extraction**: Spectral analysis, chroma features, mel-spectrograms
- **Music Theory**: music21 library for harmonic analysis
- **UI Framework**: Streamlit for interactive web interface

### Key Features
✅ **Pitch Detection** - CREPE or librosa pYIN for accurate fundamental frequency (F0) extraction  
✅ **Key & Scale Detection** - Krumhansl-Schmuckler algorithm with key profile correlation  
✅ **Chord Progression Detection** - Analyzes harmonic patterns and detects chord changes  
✅ **Mood-Aware Suggestions** - Recommends chords based on audio tone and emotional content  
✅ **Deep Learning Classification** - CNN-LSTM models trained for robust chord recognition  
✅ **Vibrato & Note Segmentation** - Detects articulation, vibrato, and note boundaries  
✅ **Interactive Visualization** - Pitch contours, spectrograms, chroma features, and loudness envelopes

---

## 📂 Project Structure

```
singing-to-chords/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── utils/
    ├── audio_processing.py         # Audio loading, spectral analysis, feature extraction
    ├── melody_extraction.py        # Pitch detection, vibrato detection, note segmentation
    ├── key_detection.py            # Key/scale detection, harmonic pattern analysis
    ├── chord_recommender.py        # Chord progression generation, mood analysis
    └── chord_classifier.py         # Deep learning models for chord classification
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/singing-to-chords.git
   cd singing-to-chords
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

The app will be available at `http://localhost:8501`

---

## 📊 Core Modules

### `audio_processing.py`
Handles all audio loading and feature extraction:
- `load_audio()` - Load and resample audio files
- `compute_spectrogram()` - STFT-based spectral analysis
- `compute_mel_spectrogram()` - Mel-scale frequency conversion
- `compute_chroma_features()` - Pitch class energy vectors
- `extract_loudness_contour()` - RMS energy tracking
- `smooth_signal()` - Noise reduction via median filtering

### `melody_extraction.py`
Extracts melodic information from vocal recordings:
- `extract_melody()` - CREPE or librosa-based F0 estimation
- `detect_vibrato()` - FFT-based vibrato frequency and extent detection
- `segment_into_notes()` - Automatic note boundary detection

### `key_detection.py`
Advanced harmonic analysis and key estimation:
- `chroma_vector_to_key()` - Krumhansl-Schmuckler key profile matching
- `estimate_key_from_notes()` - Key estimation from pitch contour
- `detect_scale_degrees()` - Classifies notes as scale degrees
- Key profile templates for major/minor modes

### `chord_recommender.py`
Generates chord progressions with mood awareness:
- `recommend_progressions()` - Classic progression templates
- `analyze_mood()` - Mood classification from audio features
- `recommend_mood_aware_chords()` - Context-sensitive recommendations
- `detect_harmonic_patterns()` - Identifies chord change points
- `chord_diagram_for()` - Guitar fingering diagrams
- `get_guitar_voicings()` - Alternative chord voicings

### `chord_classifier.py`
Deep learning-based chord recognition:
- `ChordClassifierModel` - CNN-LSTM architecture
  - 3 convolutional blocks with batch normalization
  - LSTM layers for temporal modeling
  - 24 output classes (12 major + 12 minor chords)
- `ChordEmbedding` - Harmonic embedding space
  - Harmonic relationships between chords
  - Chord similarity computation
  - Similar chord discovery

---

## 🎸 Usage Examples

### Basic Usage
1. Open the Streamlit app
2. Upload a vocal recording (WAV, MP3, FLAC, M4A)
3. Adjust confidence threshold if needed
4. View detected key, mood, and recommended chord progressions

### Advanced Analysis
- Enable "Show Advanced Analysis" to see:
  - Vibrato detection and characteristics
  - Note segmentation with precise timing
  - Full mel-spectrogram visualization
  - Scale degree distribution

### Programmatic Usage
```python
from utils.melody_extraction import extract_melody
from utils.key_detection import estimate_key_from_notes
from utils.chord_recommender import recommend_progressions
import librosa

# Load audio
y, sr = librosa.load('singing.mp3')

# Extract melody
times, f0, confidence = extract_melody(y, sr)

# Detect key
tonic, mode, notes = estimate_key_from_notes(f0, sr)

# Get chord recommendations
progressions = recommend_progressions(tonic, mode)
```

---

## 🧠 Deep Learning Model Details

### Architecture: CNN-LSTM
```
Input: Mel-spectrogram (128x40x1) → Conv blocks (32→64→128 filters)
  → MaxPooling (2x2) with Dropout(0.3)
  → Reshape for temporal processing
  → LSTM layers (256→128 units)
  → Dense layers (64→32)
  → Softmax output (24 chord classes)
```

### Training Parameters
- **Loss**: Categorical cross-entropy
- **Optimizer**: Adam (lr=0.001)
- **Regularization**: Dropout, BatchNormalization, L2
- **Callbacks**: EarlyStopping, ReduceLROnPlateau
- **Data normalization**: StandardScaler

### Model Performance
- 24-class chord classification
- Handles major and minor chords
- Confidence scoring for predictions
- Harmonic embedding space for chord relationships

---

## 🎼 Music Theory Background

### Key Detection Algorithm
Uses **Krumhansl-Schmuckler key profiles** - statistical distributions showing the likelihood of each pitch class in major/minor keys. Compares extracted chroma features against these profiles for robust key detection.

### Chord Progression Templates
- **I–V–vi–IV** (pop standard)
- **I–IV–V** (classic trio)
- **vi–IV–I–V** (ballad movement)
- **ii–V–I** (jazz changes)
- **I–vi–IV–V** (1950s progression)

### Mood-Based Recommendations
Analyzes:
- **Brightness**: Spectral centroid (high=bright, low=dark)
- **Energy**: RMS loudness (high=energetic, low=calm)
- **Harmony**: Key detection confidence
Recommends chords matching detected mood

---

## 📈 Feature Extraction Details

### Chroma Features (12 pitch classes)
Represents energy in each semitone, independent of octave. Essential for:
- Key detection
- Chord identification
- Harmonic analysis

### Mel-Spectrogram (128 bands)
Frequency representation matching human auditory perception. Used for:
- Deep learning models
- Spectral visualization
- Audio-visual understanding

### Loudness Contour (RMS energy)
Tracks intensity over time. Used for:
- Mood/energy analysis
- Dynamic expression detection
- Normalization

---

## ⚙️ Configuration & Tips

### For Best Results
- **Audio quality**: 16+ kHz sample rate, minimal background noise
- **Vocal clarity**: Cleaner vocal takes improve pitch detection
- **Duration**: Works best with 30+ seconds of audio
- **Capo adjustment**: Use detected key to choose appropriate capo

### Confidence Thresholds
- **0.6-0.7**: Balanced precision/recall
- **< 0.6**: More permissive, includes uncertain predictions
- **> 0.8**: High confidence, may miss valid chords

### Sample Rate Considerations
- 22050 Hz (default): Good balance of accuracy and speed
- 44100 Hz: Higher frequency resolution for analysis
- 16000 Hz: Faster processing, acceptable quality

---

## 🔬 Technical Highlights

### Pitch Detection
- **CREPE**: Deep learning-based, very accurate (±50 cents)
- **librosa PYIN**: Probabilistic YIN algorithm, robust fallback

### Harmonic Pattern Detection
Uses chroma distance metrics to identify chord changes by computing:
- Frame-by-frame chroma differences
- Peak detection on difference signal
- Temporal localization of chord changes

### Vibrato Detection
FFT analysis on pitch derivatives to find:
- Dominant vibrato frequency (typically 4-8 Hz)
- Vibrato extent (frequency modulation depth)

---

## 📚 Dependencies

See `requirements.txt` for full list:
- **Audio**: librosa, soundfile, crepe
- **ML/DL**: tensorflow, keras, scikit-learn, numpy
- **Visualization**: matplotlib
- **Music**: music21
- **UI**: streamlit
- **Utilities**: scipy, pandas

---

## 🚧 Future Enhancements

- [ ] Piano chord voicing recommendations
- [ ] Real-time processing with audio stream input
- [ ] Model fine-tuning on custom datasets
- [ ] Export chord charts as PDF/image
- [ ] Harmony generation and accompaniment playback
- [ ] Multi-language support for UI
- [ ] Mobile app version

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact & Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: abhighimire111@gmail.com

---

**Built with ❤️ for musicians and developers**

