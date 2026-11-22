import os
import time
import wave
import queue
import requests
import numpy as np
import sounddevice as sd
import torch
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Silero VAD configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512

VAD_THRESHOLD = 0.5
SILENCE_LIMIT = 1.5
MAX_RECORD_SECONDS = 10.0
MIN_RECORD_SECONDS = 0.5

# None means use Ubuntu default microphone
INPUT_DEVICE = None

# All recorded wav files will be saved here
RECORDINGS_DIR = "recordings"

print("Loading Silero VAD model...")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
)

audio_queue = queue.Queue()


def save_wav(filename, audio_float32):
    """
    Save float32 audio as 16-bit PCM wav.
    """
    if audio_float32 is None or len(audio_float32) == 0:
        print("No audio data, file not saved.")
        return

    # Make sure the recordings folder exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    audio_float32 = np.clip(audio_float32, -1.0, 1.0)
    audio_int16 = (audio_float32 * 32767).astype(np.int16)

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    print(f"Audio saved to: {os.path.abspath(filename)}")


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}", flush=True)

    audio_queue.put(indata[:, 0].copy())


def listen_until_speech_finished():
    """
    Use Silero VAD to detect real speech.
    Start recording when speech is detected.
    Stop after continuous non-speech.
    """
    print("Listening with Silero VAD... speak when ready.")

    is_recording = False
    silence_time = 0.0
    audio_buffer = []
    record_start_time = None

    # Create recordings folder if it does not exist
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    # Use readable timestamp to avoid overwriting old recordings
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    audio_file = os.path.join(RECORDINGS_DIR, f"driver_vad_input_{timestamp}.wav")

    stream_kwargs = {
        "samplerate": SAMPLE_RATE,
        "channels": CHANNELS,
        "dtype": "float32",
        "blocksize": CHUNK_SIZE,
        "callback": audio_callback,
    }

    if INPUT_DEVICE is not None:
        stream_kwargs["device"] = INPUT_DEVICE

    with sd.InputStream(**stream_kwargs):
        while True:
            chunk = audio_queue.get()
            tensor_chunk = torch.from_numpy(chunk)

            speech_prob = vad_model(tensor_chunk, SAMPLE_RATE).item()

            if speech_prob > VAD_THRESHOLD:
                if not is_recording:
                    print("Speech detected. Recording...")
                    is_recording = True
                    silence_time = 0.0
                    record_start_time = time.time()
                    audio_buffer = []

                silence_time = 0.0
                audio_buffer.append(chunk)

            else:
                if is_recording:
                    silence_time += CHUNK_SIZE / SAMPLE_RATE
                    audio_buffer.append(chunk)

                    elapsed = time.time() - record_start_time

                    if silence_time >= SILENCE_LIMIT and elapsed >= MIN_RECORD_SECONDS:
                        print("Speech finished. Stop recording.")
                        break

                    if elapsed >= MAX_RECORD_SECONDS:
                        print("Max recording time reached. Stop recording.")
                        break

    final_audio = (
        np.concatenate(audio_buffer) if audio_buffer else np.array([], dtype=np.float32)
    )

    save_wav(audio_file, final_audio)
    return audio_file


def speech_to_text_deepgram(audio_file, language=None):
    """
    Online method: Deepgram REST API.

    language=None: automatic language detection
    language="en": force English
    language="zh": force Chinese
    """
    if not DEEPGRAM_API_KEY:
        raise ValueError("Missing DEEPGRAM_API_KEY in .env file")

    url = "https://api.deepgram.com/v1/listen"

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
    }

    params = {
        "model": "nova-3",
        "smart_format": "true",
    }

    if language:
        params["language"] = language

    with open(audio_file, "rb") as f:
        response = requests.post(
            url,
            headers=headers,
            params=params,
            data=f,
            timeout=30,
        )

    if response.status_code != 200:
        raise RuntimeError(f"Deepgram error {response.status_code}: {response.text}")

    data = response.json()
    text = data["results"]["channels"][0]["alternatives"][0]["transcript"]

    return text.strip()


def speech_to_text_local(audio_file, language=None):
    """
    Offline fallback: faster-whisper.
    """
    print("Using local faster-whisper fallback...")

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )

    segments, info = model.transcribe(
        audio_file,
        language=language,
        beam_size=5,
    )

    print("Local detected language:", info.language)

    text = " ".join(segment.text for segment in segments)

    return text.strip()


def get_driver_speech_text_auto(language=None):
    """
    Main function for teammate to call.

    It listens to driver speech,
    uses Silero VAD to detect speech,
    tries Deepgram first,
    and switches to local faster-whisper if Deepgram fails.

    Returns:
        driver_text: str
    """
    audio_file = listen_until_speech_finished()

    try:
        print("Trying Deepgram cloud STT...")
        text = speech_to_text_deepgram(audio_file, language=language)

        if text:
            print("Deepgram transcription succeeded.")
            return text

        print("Deepgram returned empty text. Switching to local fallback.")
        return speech_to_text_local(audio_file, language=language)

    except Exception as e:
        print("Deepgram failed:", e)
        print("Switching to local fallback.")
        return speech_to_text_local(audio_file, language=language)


if __name__ == "__main__":
    while True:
        driver_text = get_driver_speech_text_auto(language=None)

        if driver_text:
            print("Driver said:", driver_text)
        else:
            print("No clear speech detected.")

        print("-" * 50)
        time.sleep(1)
