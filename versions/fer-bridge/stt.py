# stt.py
# Simple local STT using sounddevice + faster-whisper
# No VAD model download needed — uses volume threshold instead

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE      = 16000
SILENCE_THRESHOLD = 0.01   # volume below this = silence
SILENCE_DURATION  = 1.5    # seconds of silence to stop recording
MAX_DURATION      = 10.0   # max recording length
MIN_DURATION      = 0.5    # ignore clips shorter than this

print("Loading Whisper model (first time may take ~30 seconds)...")
_model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper ready.")


def listen_and_transcribe(language=None) -> str:
    """
    Listen from microphone until silence, then transcribe.
    Returns transcribed text, or empty string if nothing detected.
    """
    print("Listening... (speak now)")

    recorded   = []
    is_speaking = False
    silence_sec = 0.0
    total_sec   = 0.0
    chunk_sec   = 0.032  # ~32ms chunks

    chunk_size  = int(SAMPLE_RATE * chunk_sec)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=chunk_size) as stream:
        while total_sec < MAX_DURATION:
            chunk, _ = stream.read(chunk_size)
            audio     = chunk[:, 0]
            volume    = np.sqrt(np.mean(audio ** 2))
            total_sec += chunk_sec

            if volume > SILENCE_THRESHOLD:
                if not is_speaking:
                    print("Speech detected...")
                    is_speaking = True
                    silence_sec = 0.0
                recorded.append(audio)
                silence_sec = 0.0
            else:
                if is_speaking:
                    recorded.append(audio)
                    silence_sec += chunk_sec
                    if silence_sec >= SILENCE_DURATION:
                        break

    if not recorded or total_sec < MIN_DURATION:
        return ""

    audio_np = np.concatenate(recorded)
    segments, _ = _model.transcribe(audio_np, language=language, beam_size=5)
    text = " ".join(s.text for s in segments).strip()
    print(f"Transcribed: {text}")
    return text
