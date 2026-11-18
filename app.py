import os
import sys
import uuid
import re
import json as _json
import subprocess
os.environ.setdefault("SAFETENSORS_NO_MMAP", "1")
os.environ.setdefault("COQUI_TOS_AGREED", "1")
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, stream_with_context

BASE = Path(__file__).resolve().parent
CKPT = BASE / "models" / "chatterbox-turbo"
OUT_DIR = BASE / "static" / "outputs"
VOICES_DIR = BASE / "voices"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

import time
from datetime import datetime

FAV_FILE = BASE / "favorites.json"
HIST_FILE = BASE / "history.json"


def load_history():
    if HIST_FILE.exists():
        try:
            return list(_json.loads(HIST_FILE.read_text(encoding="utf-8")))
        except Exception:
            return []
    return []


def save_history(hist):
    HIST_FILE.write_text(_json.dumps(hist, ensure_ascii=False), encoding="utf-8")


def load_favs():
    if FAV_FILE.exists():
        try:
            return list(_json.loads(FAV_FILE.read_text(encoding="utf-8")))
        except Exception:
            return []
    return []


def save_favs(favs):
    FAV_FILE.write_text(_json.dumps(sorted(set(favs)), ensure_ascii=False), encoding="utf-8")

app = Flask(__name__, template_folder=str(BASE / "templates"), static_folder=str(BASE / "static"))
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading ChatterBox Turbo on {device}...")
# Stub perth (audio watermarking) because Windows Application Control blocks
# torchaudio's native extension. Watermarking is optional; this no-op stub
# lets the TTS model load and generate audio without it.
import types as _types
import importlib.machinery as _im
_perth = _types.ModuleType("perth")
class _PerthImplicitWatermarker:
    def apply_watermark(self, wav, sample_rate=None):
        return wav
_perth.PerthImplicitWatermarker = _PerthImplicitWatermarker
sys.modules["perth"] = _perth
_ta = _types.ModuleType("torchaudio")
_ta.__spec__ = _im.ModuleSpec("torchaudio", None)
sys.modules["torchaudio"] = _ta
_ta_t = _types.ModuleType("torchaudio.transforms")
_ta_t.__spec__ = _im.ModuleSpec("torchaudio.transforms", None)
# provide minimal stubs used by encodec/perth/s3tokenizer
class _Resample:
    def __init__(self, src_sr, dst_sr, *a, **k):
        self.src_sr = src_sr; self.dst_sr = dst_sr
    def __call__(self, x):
        if self.src_sr == self.dst_sr:
            return x
        import librosa, torch
        arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        squeeze = (arr.ndim == 1)
        arr = np.atleast_1d(arr).ravel() if arr.ndim <= 2 else arr.reshape(-1)
        # preserve batch: if input was (1, N), process 1D then re-add batch
        was_batched = (torch.is_tensor(x) and x.dim() == 2 and x.shape[0] == 1)
        out = librosa.resample(arr, orig_sr=self.src_sr, target_sr=self.dst_sr)
        t = torch.from_numpy(out).float()
        if was_batched:
            t = t.unsqueeze(0)
        if torch.is_tensor(x):
            t = t.to(x.device)
        return t
    def to(self, *a, **k): return self
_ta_t.Resample = _Resample
_ta_t.Spectrogram = type("Spectrogram", (), {"__init__": lambda self, *a, **k: None, "__call__": lambda self, x: x})
_ta_t.InverseSpectrogram = type("InverseSpectrogram", (), {"__init__": lambda self, *a, **k: None, "__call__": lambda self, x: x})
_ta_t.TimeStretch = type("TimeStretch", (), {"__init__": lambda self, *a, **k: None, "__call__": lambda self, x: x})
_ta.transforms = _ta_t
sys.modules["torchaudio.transforms"] = _ta_t
# Stub torchaudio.compliance.kaldi (used by chatterbox's x-vector speaker
# encoder) because Windows Application Control blocks the native extension.
# Returning a zero fbank makes speaker embedding degenerate but keeps TTS
# working for the default voice; voice cloning may be degraded.
_ta_c = _types.ModuleType("torchaudio.compliance")
_ta_c.__spec__ = _im.ModuleSpec("torchaudio.compliance", None)
_ta_k = _types.ModuleType("torchaudio.compliance.kaldi")
_ta_k.__spec__ = _im.ModuleSpec("torchaudio.compliance.kaldi", None)
def _fake_fbank(waveform, *a, **k):
    n = k.get("num_mel_bins", 80)
    sr = k.get("sample_frequency", 16000)
    if torch.is_tensor(waveform):
        y = waveform.detach().cpu().numpy()
    else:
        y = np.asarray(waveform)
    y = np.atleast_1d(y).astype(np.float32).ravel()
    if y.size < sr * 0.01:
        return torch.zeros(1, n, dtype=torch.float32)
    import librosa
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n, n_fft=512, hop_length=160)
    logmel = np.log(mel + 1e-6).T  # (frames, n_mels)
    if logmel.shape[0] == 0:
        logmel = np.zeros((1, n), dtype=np.float32)
    return torch.from_numpy(logmel).float()
_ta_k.fbank = _fake_fbank
_ta_c.kaldi = _ta_k
sys.modules["torchaudio.compliance"] = _ta_c
sys.modules["torchaudio.compliance.kaldi"] = _ta_k

# Patch CAMPPlus (x-vector speaker encoder) to move input to the model's
# device, since our stubbed Kaldi.fbank returns CPU tensors.
try:
    from chatterbox.models.s3gen.xvector import CAMPPlus as _CAMPPlus, extract_feature as _extract_feature
    def _patched_inference(self, audio_list):
        if torch.is_tensor(audio_list) and audio_list.dim() == 1:
            audio_list = audio_list.unsqueeze(0)
        speech, speech_lengths, speech_times = _extract_feature(audio_list)
        speech = speech.to(next(self.parameters()).device).to(torch.float32)
        return self.forward(speech)
    _CAMPPlus.inference = _patched_inference
except Exception as _e:
    print("CAMPPlus patch failed:", _e)

from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_local(CKPT, device)
print("Model ready.")

# Meta MMS-TTS (Bengali / bn_BD). Fully local via transformers; downloads
# weights once, then runs offline. Wrapped so the app still starts if this
# fails (English-only).
mms_bn = None
mms_bn_sr = 16000
mms_bn_tokenizer = None
try:
    from transformers import VitsTokenizer, VitsModel as _VitsModel
    print("Loading Meta MMS-TTS (bn_BD, Bengali)...")
    _bn_model_id = "facebook/mms-tts-ben"
    mms_bn_tokenizer = VitsTokenizer.from_pretrained(_bn_model_id)
    mms_bn = _VitsModel.from_pretrained(_bn_model_id)
    mms_bn_sr = int(getattr(mms_bn.config, "sampling_rate", 16000))
    print(f"MMS-TTS bn_BD ready (sr={mms_bn_sr}).")
except Exception as _mms_e:
    print("MMS-TTS bn_BD unavailable (Bengali disabled):", _mms_e)
    mms_bn = None
    mms_bn_tokenizer = None

BREAK_RE = re.compile(r'<break\s+time="([\d.]+)s"\s*/>')
SENT_RE = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text):
    text = text.strip()
    if len(text) <= 280:
        return [text]
    parts = SENT_RE.split(text)
    chunks = []
    cur = ""
    for p in parts:
        if not p.strip():
            continue
        if len(cur) + len(p) + 1 > 260 and cur:
            chunks.append(cur.strip())
            cur = p
        else:
            cur = (cur + " " + p).strip() if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks or [text]


def synthesize_text(text, audio_prompt_path, temperature, repetition_penalty, top_p, top_k, norm_loudness, min_p=0.0, exaggeration=0.0, cfg_weight=0.0):
    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        repetition_penalty=repetition_penalty,
        min_p=min_p,
        top_p=top_p,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        top_k=top_k,
        norm_loudness=norm_loudness,
    )
    return wav.squeeze().cpu().numpy().astype(np.float32)


def normalize_bn(text):
    """Normalize Bengali text for clearer TTS pronunciation.
    - NFC unicode normalization
    - strip zero-width / invisible chars
    - standardise Bengali digits and punctuation
    - collapse repeated punctuation/spaces"""
    import unicodedata
    if not text:
        return text
    # NFC normalize (combining marks -> precomposed)
    text = unicodedata.normalize("NFC", text)
    # remove zero-width and invisible characters
    text = "".join(ch for ch in text if ch not in {
        "\u200b", "\u200c", "\u200d", "\u200e", "\u200f",
        "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
        "\u2060", "\ufeff", "\u00ad",
    })
    # convert common Bengali-affected Latin digits to Bengali digits
    text = text.translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"))
    # normalise Bengali punctuation: use Bengali full stop (।) where a sentence ends
    text = re.sub(r"\s*\.\s*", "। ", text)
    # collapse multiple spaces/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def synthesize_mms_bn(text):
    """Synthesize Bengali (bn_BD) with Meta MMS-TTS. Fully local."""
    if mms_bn is None or mms_bn_tokenizer is None:
        raise RuntimeError("MMS-TTS bn_BD model not loaded (Bengali unavailable).")
    text = normalize_bn(text)
    inputs = mms_bn_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = mms_bn(**inputs)
        wav = output.waveform.float().cpu().numpy().reshape(-1)
    # MMS outputs 16kHz; resample to 24kHz to match the rest of the pipeline
    import librosa
    if mms_bn_sr != model.sr:
        wav = librosa.resample(wav, orig_sr=mms_bn_sr, target_sr=model.sr)
    return wav.astype(np.float32)


def generate_audio(text, audio_prompt_path, temperature, repetition_penalty, top_p, top_k, speed, norm_loudness, min_p=0.0, exaggeration=0.0, cfg_weight=0.0, fmt="wav", max_duration=0.0, lang="en"):
    import json as _j
    if lang == "bn":
        yield from _generate_audio_mms_bn(text, audio_prompt_path, speed, fmt, max_duration)
        return
    sr = model.sr
    pos = 0
    seq = []
    for m in BREAK_RE.finditer(text):
        before = text[pos:m.start()].strip()
        if before:
            seq.append(("text", before))
        seq.append(("break", float(m.group(1))))
        pos = m.end()
    if text[pos:].strip():
        seq.append(("text", text[pos:].strip()))

    if not seq:
        seq = [("text", text.strip())]

    # count sentence steps for progress
    total_steps = 0
    for kind, val in seq:
        if kind == "text":
            total_steps += len([s for s in split_sentences(val) if s.strip()])
    if total_steps == 0:
        total_steps = 1

    chunks = []
    step = 0
    for kind, val in seq:
        if kind == "break":
            n = max(1, int(val * sr))
            chunks.append(np.zeros(n, dtype=np.float32))
        else:
            for sent in split_sentences(val):
                if not sent:
                    continue
                arr = synthesize_text(sent, audio_prompt_path, temperature, repetition_penalty, top_p, top_k, norm_loudness, min_p, exaggeration, cfg_weight)
                chunks.append(arr)
                chunks.append(np.zeros(int(0.06 * sr), dtype=np.float32))
                step += 1
                yield {"type": "progress", "progress": step, "total": total_steps}

    if not chunks:
        arr = np.zeros(0, dtype=np.float32)
        sr = model.sr
        out = arr
    else:
        out = np.concatenate(chunks)

    base_duration = len(out) / sr if out.size else 0.0
    required_speed = 1.0
    MAX_SPEED = 2.5
    if max_duration and max_duration > 0 and base_duration > 0:
        required_speed = base_duration / max_duration
        if required_speed > MAX_SPEED:
            yield {"type": "error", "error": f"Cannot fit: requires {required_speed:.2f}x speed (max {MAX_SPEED}x). Shorten text or raise max duration."}
            return
        if required_speed > 1.0:
            speed = max(speed, required_speed)
        speed = min(speed, MAX_SPEED)

    # speed change via time stretch
    if abs(speed - 1.0) > 0.01:
        import librosa
        out = librosa.effects.time_stretch(out, rate=speed).astype(np.float32)
    # peak normalize
    peak = np.max(np.abs(out)) if out.size else 0.0
    if peak > 0:
        target = 10 ** (-1.5 / 20.0)
        out = out * (target / peak)

    # build output filename from first 4-5 words + timestamp
    import unicodedata
    words = (text or "").split()[:5]
    slug = "-".join(words)
    slug = unicodedata.normalize("NFKD", slug).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9_-]", "", slug)
    slug = re.sub(r"-+", "-", slug).strip("-").lower()[:40] or "audio"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{slug}_{ts}"
    # ensure uniqueness
    i = 1
    while (OUT_DIR / f"{stem}.wav").exists() or (OUT_DIR / f"{stem}.mp3").exists():
        stem = f"{slug}_{ts}_{i}"
        i += 1
    out_name = f"{stem}.wav"
    out_path = OUT_DIR / out_name
    sf.write(str(out_path), out.astype(np.float32), sr)

    audio_url = f"/static/outputs/{out_name}"
    if fmt == "mp3":
        mp3_name = out_name.replace(".wav", ".mp3")
        mp3_path = OUT_DIR / mp3_name
        try:
            subprocess.run(["ffmpeg", "-y", "-i", str(out_path), "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                os.remove(out_path)
            except OSError:
                pass
            audio_url = f"/static/outputs/{mp3_name}"
        except Exception:
            pass

    if audio_prompt_path and str(audio_prompt_path).startswith(str(OUT_DIR)):
        try:
            os.remove(audio_prompt_path)
        except OSError:
            pass

    # persist history
    hist_entry = {"id": Path(audio_url).name, "text": text, "audio_url": audio_url, "duration": round(len(out) / sr, 2), "sr": sr, "created": int(time.time()), "format": fmt}
    try:
        hist = load_history()
        hist.insert(0, hist_entry)
        save_history(hist)
    except Exception:
        pass

    yield {"type": "done", "audio_url": audio_url, "duration": round(len(out) / sr, 2), "sr": sr, "history_id": Path(audio_url).name, "required_speed": round(required_speed, 2), "max_duration": max_duration, "speed_applied": round(speed, 2)}


def _generate_audio_mms_bn(text, audio_prompt_path, speed, fmt, max_duration):
    """Bengali (bn_BD) generation via Meta MMS-TTS. Fully local.
    Produces the same SSE event format as the ChatterBox path."""
    import json as _j
    if mms_bn is None or mms_bn_tokenizer is None:
        yield {"type": "error", "error": "MMS-TTS bn_BD model not loaded (Bengali unavailable)."}
        return
    sr = model.sr
    yield {"type": "progress", "progress": 0, "total": 1}

    parts = [p.strip() for p in SENT_RE.split(text) if p.strip()] or [text.strip()]
    total_steps = len(parts)
    chunks = []
    step = 0
    for sent in parts:
        arr = synthesize_mms_bn(sent)
        if arr.size:
            chunks.append(arr)
            chunks.append(np.zeros(int(0.06 * sr), dtype=np.float32))
        step += 1
        yield {"type": "progress", "progress": step, "total": total_steps}

    if not chunks:
        out = np.zeros(0, dtype=np.float32)
    else:
        out = np.concatenate(chunks)

    base_duration = len(out) / sr if out.size else 0.0
    required_speed = 1.0
    MAX_SPEED = 2.5
    if max_duration and max_duration > 0 and base_duration > 0:
        required_speed = base_duration / max_duration
        if required_speed > MAX_SPEED:
            yield {"type": "error", "error": f"Cannot fit: requires {required_speed:.2f}x speed (max {MAX_SPEED}x). Shorten text or raise max duration."}
            return
        if required_speed > 1.0:
            speed = max(speed, required_speed)
        speed = min(speed, MAX_SPEED)

    if abs(speed - 1.0) > 0.01:
        import librosa
        out = librosa.effects.time_stretch(out, rate=speed).astype(np.float32)
    peak = np.max(np.abs(out)) if out.size else 0.0
    if peak > 0:
        target = 10 ** (-1.5 / 20.0)
        out = out * (target / peak)

    import unicodedata
    words = (text or "").split()[:5]
    slug = "-".join(words)
    slug = unicodedata.normalize("NFKD", slug).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9_-]", "", slug)
    slug = re.sub(r"-+", "-", slug).strip("-").lower()[:40] or "audio"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{slug}_{ts}"
    i = 1
    while (OUT_DIR / f"{stem}.wav").exists() or (OUT_DIR / f"{stem}.mp3").exists():
        stem = f"{slug}_{ts}_{i}"
        i += 1
    out_name = f"{stem}.wav"
    out_path = OUT_DIR / out_name
    sf.write(str(out_path), out.astype(np.float32), sr)

    audio_url = f"/static/outputs/{out_name}"
    if fmt == "mp3":
        mp3_name = out_name.replace(".wav", ".mp3")
        mp3_path = OUT_DIR / mp3_name
        try:
            subprocess.run(["ffmpeg", "-y", "-i", str(out_path), "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                os.remove(out_path)
            except OSError:
                pass
            audio_url = f"/static/outputs/{mp3_name}"
        except Exception:
            pass

    if audio_prompt_path and str(audio_prompt_path).startswith(str(OUT_DIR)):
        try:
            os.remove(audio_prompt_path)
        except OSError:
            pass

    hist_entry = {"id": Path(audio_url).name, "text": text, "audio_url": audio_url, "duration": round(len(out) / sr, 2), "sr": sr, "created": int(time.time()), "format": fmt}
    try:
        hist = load_history()
        hist.insert(0, hist_entry)
        save_history(hist)
    except Exception:
        pass

    yield {"type": "done", "audio_url": audio_url, "duration": round(len(out) / sr, 2), "sr": sr, "history_id": Path(audio_url).name, "required_speed": round(required_speed, 2), "max_duration": max_duration, "speed_applied": round(speed, 2)}


def voice_samples(name):
    d = VOICES_DIR / name
    if d.is_dir():
        return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS and p.name != "_merged.wav")
    return sorted(p for p in VOICES_DIR.iterdir() if p.is_file() and p.stem == name and p.suffix.lower() in AUDIO_EXTS)


def merged_ref_path(name):
    """Path to the persistent merged reference for a voice (if it exists)."""
    d = VOICES_DIR / name
    if d.is_dir():
        p = d / "_merged.wav"
        return p if p.exists() else None
    return None


def rebuild_merged(name):
    """Rebuild the persistent merged reference for a voice from all its samples.
    Called automatically when samples are added or removed, so every future
    generation uses the merged voice automatically."""
    samples = voice_samples(name)
    if len(samples) <= 1:
        # single sample: no merge needed; remove any stale merged file
        p = merged_ref_path(name)
        if p:
            try: p.unlink()
            except OSError: pass
        return None
    combined_path = combine_samples(samples)
    if not combined_path:
        return None
    d = VOICES_DIR / name
    if not d.is_dir():
        # flat legacy voice: nothing to merge into (keep combined in OUT_DIR)
        return combined_path
    merged = d / "_merged.wav"
    try:
        os.replace(combined_path, merged)
    except OSError:
        try:
            import shutil
            shutil.copyfile(combined_path, merged)
            os.remove(combined_path)
        except OSError:
            return combined_path
    return str(merged)


def combine_samples(paths):
    arrays = []
    sr = None
    for p in paths:
        data, s = sf.read(str(p), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr is None:
            sr = s
        elif s != sr:
            import librosa
            data = librosa.resample(data, orig_sr=s, target_sr=sr).astype(np.float32)
        arrays.append(data.astype(np.float32))
        arrays.append(np.zeros(int(0.25 * sr), dtype=np.float32))
    if not arrays:
        return None
    out = np.concatenate(arrays)
    out_path = OUT_DIR / f"ref_combined_{uuid.uuid4().hex}.wav"
    sf.write(str(out_path), out.astype(np.float32), sr)
    return str(out_path)


def list_voices():
    out = []
    seen = set()
    for d in sorted(VOICES_DIR.iterdir()):
        if d.is_dir():
            samples = voice_samples(d.name)
            if samples:
                seen.add(d.name)
                out.append({"name": d.name, "samples": len(samples), "sample_urls": [f"/voices/{d.name}/{p.name}" for p in samples], "url": f"/voices/{d.name}/{samples[0].name}"})
    for p in sorted(VOICES_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            if p.stem in seen:
                continue
            # group flat files sharing a stem
            grp = [x for x in VOICES_DIR.iterdir() if x.is_file() and x.stem == p.stem and x.suffix.lower() in AUDIO_EXTS]
            if p.stem in seen:
                continue
            seen.add(p.stem)
            out.append({"name": p.stem, "samples": len(grp), "sample_urls": [f"/voices/{x.name}" for x in grp], "url": f"/voices/{p.name}"})
    return out


@app.route("/")
def index():
    return render_template("index.html", device=device, model_name="ChatterboxTurboTTS", sample_rate=model.sr, voice_count=len(list_voices()))


@app.route("/generate", methods=["POST"])
def generate():
    data = request.form
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Text is required"}), 400

    exaggeration = 0.0
    temperature = float(data.get("temperature", "") or 0)
    repetition_penalty = float(data.get("repetition_penalty", "") or 0)
    min_p = float(data.get("min_p", "") or 0)
    top_p = float(data.get("top_p", "") or 0)
    top_k = int(float(data.get("top_k", "") or 0))
    speed = float(data.get("speed", 1.0))
    if speed <= 0:
        speed = 1.0
    max_duration = float(data.get("max_duration", "") or 0)
    stability = float(data.get("stability", 0.5))
    style = float(data.get("style", 0.0))
    speaker_boost = data.get("speaker_boost", "true").lower() == "true"
    similarity_boost = float(data.get("similarity_boost", 0.5))
    cfg_weight = float(data.get("cfg_weight", "") or 0)
    exaggeration = float(data.get("exaggeration", "") or 0)

    # Derive advanced params if not explicitly provided
    if temperature <= 0:
        temperature = max(0.1, min(1.5, 1.0 - 0.4 * stability + 0.3 * style))
    if repetition_penalty <= 0:
        repetition_penalty = max(1.0, min(2.0, 1.0 + 0.5 * stability))
    if top_p <= 0:
        top_p = 0.95
    if top_k <= 0:
        top_k = 1000

    fmt = (data.get("format") or "wav").lower()
    if fmt not in ("wav", "mp3"):
        fmt = "wav"

    lang = (data.get("lang") or "en").lower()
    if lang not in ("en", "bn"):
        lang = "en"

    audio_prompt_path = None
    if "audio" in request.files and request.files["audio"].filename:
        f = request.files["audio"]
        ext = Path(f.filename).suffix or ".wav"
        tmp = OUT_DIR / f"ref_{uuid.uuid4().hex}{ext}"
        f.save(tmp)
        audio_prompt_path = str(tmp)
    else:
        vn = (data.get("voice_name") or "").strip()
        if vn:
            # Prefer the persistent merged reference (rebuilt automatically
            # whenever samples are added/removed) so every generation uses
            # the full multi-sample voice.
            merged = merged_ref_path(vn)
            if merged:
                audio_prompt_path = str(merged)
            else:
                samples = voice_samples(vn)
                if len(samples) == 1:
                    audio_prompt_path = str(samples[0])
                elif len(samples) > 1:
                    audio_prompt_path = combine_samples(samples)

    def gen():
        import json as _j
        try:
            norm_loudness = speaker_boost
            for event in generate_audio(text, audio_prompt_path, temperature, repetition_penalty, top_p, top_k, speed, norm_loudness, min_p, exaggeration, cfg_weight, fmt, max_duration, lang):
                yield f"data: {_j.dumps(event)}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            if audio_prompt_path and str(audio_prompt_path).startswith(str(OUT_DIR)):
                try:
                    os.remove(audio_prompt_path)
                except OSError:
                    pass
            yield f"data: {_j.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return Response(stream_with_context(gen()), mimetype="text/event-stream")


@app.route("/api/info", methods=["GET"])
def api_info():
    return jsonify({
        "device": device,
        "model": "ChatterboxTurboTTS",
        "sample_rate": model.sr,
        "voices": len(list_voices()),
        "bn_loaded": mms_bn is not None,
        "bn_sr": mms_bn_sr,
        "languages": ["en", "bn"],
    })
@app.route("/api/voices", methods=["GET"])
def api_voices():
    return jsonify({"voices": list_voices()})


@app.route("/api/voices/save", methods=["POST"])
def api_voice_save():
    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name required"}), 400
    files = [f for f in request.files.getlist("audio") if f and f.filename]
    if not files:
        # backward compat single field
        if "audio" in request.files and request.files["audio"].filename:
            files = [request.files["audio"]]
    if not files:
        return jsonify({"error": "Audio file required"}), 400
    safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip().replace(" ", "_")
    if not safe:
        return jsonify({"error": "Invalid name"}), 400
    dest_dir = VOICES_DIR / safe
    dest_dir.mkdir(exist_ok=True)
    # migrate legacy flat files for this voice into the folder
    for p in list(VOICES_DIR.iterdir()):
        if p.is_file() and p.stem == safe and p.suffix.lower() in AUDIO_EXTS:
            try:
                p.replace(dest_dir / p.name)
            except OSError:
                pass
    for f in files:
        ext = Path(f.filename).suffix or ".wav"
        dest = dest_dir / f"{uuid.uuid4().hex[:8]}{ext}"
        f.save(dest)
    # automatically merge all samples into one persistent voice reference
    # so every future generation uses the full multi-sample voice
    try:
        rebuild_merged(safe)
    except Exception as _rm:
        print("rebuild_merged failed:", _rm)
    samples = voice_samples(safe)
    return jsonify({"ok": True, "voice": {"name": safe, "samples": len(samples), "url": f"/voices/{safe}/{samples[0].name}" if samples else "", "merged": bool(merged_ref_path(safe))}})


@app.route("/api/voices/<name>", methods=["DELETE"])
def api_voice_delete(name):
    safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
    d = VOICES_DIR / safe
    if d.is_dir():
        for p in d.iterdir():
            try:
                p.unlink()
            except OSError:
                pass
        try:
            d.rmdir()
        except OSError:
            pass
        return jsonify({"ok": True})
    matches = [p for p in VOICES_DIR.iterdir() if p.is_file() and p.stem == safe]
    if not matches:
        return jsonify({"error": "Not found"}), 404
    for p in matches:
        try:
            p.unlink()
        except OSError:
            pass
    try:
        rebuild_merged(safe)
    except Exception as _rm:
        print("rebuild_merged failed:", _rm)
    return jsonify({"ok": True})


@app.route("/voices/<path:filename>")
def serve_voice(filename):
    return send_from_directory(str(VOICES_DIR), filename)


@app.route("/static/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(str(OUT_DIR), filename)


@app.route("/api/history", methods=["GET"])
def api_history_get():
    hist = load_history()
    by_name = {}
    for h in hist:
        name = Path(h.get("audio_url", "")).name
        if name:
            by_name[name] = h

    files = []
    for p in OUT_DIR.iterdir():
        try:
            if p.is_file():
                files.append(p)
        except OSError:
            pass
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    out = []
    for p in files:
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        if p.name.startswith("ref_"):
            continue
        h = by_name.get(p.name, {})
        try:
            info = sf.info(str(p))
            duration = round(info.duration, 2)
            sr = info.samplerate
            size = p.stat().st_size
            mtime = p.stat().st_mtime
        except Exception:
            duration = h.get("duration", 0)
            sr = h.get("sr", 0)
            size = h.get("size", 0)
            mtime = h.get("created", 0)
        created = h.get("created") or int(mtime)
        out.append({
            "id": p.name,
            "text": h.get("text") or p.stem,
            "audio_url": f"/static/outputs/{p.name}",
            "duration": duration,
            "sr": sr,
            "size": size,
            "format": p.suffix.lstrip(".").lower(),
            "created": created,
            "created_str": datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return jsonify({"history": out})


@app.route("/api/history/<hid>", methods=["DELETE"])
def api_history_delete(hid):
    if "/" in hid or "\\" in hid or ".." in hid:
        return jsonify({"error": "Invalid id"}), 400
    hist = load_history()
    item = next((h for h in hist if h.get("id") == hid), None)
    fname = hid
    if item:
        fname = Path(item.get("audio_url", "")).name or hid
    if "/" in fname or "\\" in fname or ".." in fname:
        return jsonify({"error": "Invalid filename"}), 400
    fpath = OUT_DIR / fname
    try:
        fpath.resolve().relative_to(OUT_DIR.resolve())
    except Exception:
        return jsonify({"error": "Invalid path"}), 400
    if fpath.exists() and fpath.is_file():
        try:
            fpath.unlink()
        except OSError:
            pass
    hist = [h for h in hist if h.get("id") != hid and Path(h.get("audio_url", "")).name != fname]
    save_history(hist)
    return jsonify({"ok": True})


@app.route("/api/voices/<name>/samples/<path:filename>", methods=["DELETE"])
def api_voice_sample_delete(name, filename):
    if "/" in filename or "\\" in filename or ".." in filename:
        return jsonify({"error": "Invalid filename"}), 400
    safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip().replace(" ", "_")
    d = VOICES_DIR / safe
    if d.is_dir():
        p = d / filename
    else:
        p = VOICES_DIR / filename
    # ensure resolved path stays inside VOICES_DIR
    try:
        p.resolve().relative_to(VOICES_DIR.resolve())
    except Exception:
        return jsonify({"error": "Invalid path"}), 400
    if p.exists() and p.is_file():
        try:
            p.unlink()
        except OSError:
            pass
    if d.is_dir():
        try:
            if not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass
    # rebuild the merged reference after any sample change
    try:
        rebuild_merged(safe)
    except Exception as _rm:
        print("rebuild_merged failed:", _rm)
    return jsonify({"ok": True})


@app.route("/api/favorites", methods=["GET"])
def api_favs_get():
    return jsonify({"favorites": load_favs()})


@app.route("/api/favorites", methods=["POST"])
def api_favs_add():
    name = (request.form.get("name") or (request.get_json(silent=True) or {}).get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name required"}), 400
    favs = load_favs()
    if name not in favs:
        favs.append(name)
        save_favs(favs)
    return jsonify({"ok": True, "favorites": favs})


@app.route("/api/favorites/<name>", methods=["DELETE"])
def api_favs_remove(name):
    favs = [f for f in load_favs() if f != name]
    save_favs(favs)
    return jsonify({"ok": True, "favorites": favs})


@app.after_request
def no_cache(response):
    if response.mimetype == "text/html":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
