# Technical Roadmap: Advanced Emotional Prosody & Expressive Speech

## 1. Goals
- Add controllable emotional expression (joy, sadness, anger, fear, neutral) to ChatterBox Turbo.
- Preserve speaker identity/vocal consistency across emotional states.
- Expose fine-grained controls for pitch, tempo, intensity, and style in the Web UI and API.

## 2. Current Baseline
- ChatterboxTurboTTS.generate supports: temperature, repetition_penalty, min_p, top_p, top_k, exaggeration, cfg_weight, norm_loudness.
- Web UI already exposes these plus speed, stability, style, speaker_boost.
- No explicit emotion labels or prosody conditioning.

## 3. Approach

### 3.1 Emotion Conditioning
- Add an emotion embedding layer (learned or CLAP/CLIP-style) injected into the TTS decoder.
- Train on emotion-labeled speech (RAVDESS, CREMA-D, ESD, plus synthetic labels).
- Support multi-label and intensity scaling (0-1).

### 3.2 Prosody Control
- Pitch: F0 contour prediction with user-controllable offset (semitones) and variance.
- Tempo: speaking-rate control via duration predictor scaling (independent of pitch).
- Intensity: energy/loudness modulation and articulation strength.

### 3.3 Vocal Consistency
- Speaker embedding (ECAPA-TDNN or x-vector) extracted from reference audio.
- Emotion applied as residual/style modulation, not speaker replacement.
- Consistency loss: cosine similarity between neutral and emotional speaker embeddings.

### 3.4 Fine-Grained Parameters
- `emotion`: enum [neutral, joy, sadness, anger, fear]
- `emotion_intensity`: 0.0-1.0
- `pitch_shift`: semitones (-6..+6)
- `tempo`: 0.5-2.0
- `intensity`: 0.0-1.0
- `style`: existing style weight

## 4. Implementation Phases

### Phase 1: Data & Baseline (2-3 weeks)
- Collect/label emotion data.
- Add emotion tags to training pipeline.
- Baseline evaluation (MOS, emotion accuracy).

### Phase 2: Model Changes (3-4 weeks)
- Add emotion embedding + prosody predictors.
- Fine-tune decoder with emotion + consistency losses.
- Export updated checkpoint.

### Phase 3: API & UI (1-2 weeks)
- Add parameters to `/generate` and `generate_audio`.
- Add sliders/dropdowns in Web UI.
- Update history metadata to store emotion settings.

### Phase 4: Evaluation & Polish (1-2 weeks)
- ABX tests, emotion recognition, speaker similarity.
- Performance optimization (batching, caching).

## 5. Evaluation Metrics
- Emotion classification accuracy (target >85%).
- Speaker similarity (cosine >0.85 across emotions).
- MOS for naturalness (>4.0).
- Latency impact <20%.

## 6. Risks
- Emotion leakage into speaker identity.
- Over-exaggeration at high intensity.
- Data bias across languages.

## 7. Next Steps
- Prototype emotion embedding on a small dataset.
- Integrate into training script.
- Expose parameters in API/UI behind a feature flag.
'@