# /// script
# dependencies = [
#   "sounddevice",
#   "numpy",
#   "scipy",
#   "pynput",
#   "requests",
#   "python-dotenv",
#   "google-genai",
# ]
# ///

"""A command-line tool for recording whisper/low-volume audio with enhancement for Whisper transcription"""

import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import requests
from dotenv import load_dotenv
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T | None]]:
    """
    Generic retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        retryable_exceptions: Exception types that trigger a retry
        on_retry: Callback function on retry, receives (exception, attempt)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T | None]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T | None:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        if on_retry:
                            on_retry(e, attempt)
                        else:
                            logger.warning(f"Error: {e}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Max retries exceeded: {e}")
                        return None
            return None
        return wrapper
    return decorator


# Audio settings optimized for Whisper
SAMPLE_RATE = 16000  # Whisper's preferred sample rate
CHANNELS = 1
DTYPE = np.float32

# Audio enhancement settings
HIGHPASS_FREQ = 80  # Hz, remove low-frequency rumble
TARGET_RMS_DB = -18  # dB, target loudness for normalization
MAX_GAIN = 20.0  # Maximum amplification to prevent noise boost
SOFT_LIMIT_THRESHOLD = 0.7  # Soft clipping threshold
PEAK_NORMALIZE_TARGET = 0.95  # Peak normalization target

# Output directory (default: ~/whisper_recordings)
OUTPUT_DIR = Path.home() / "whisper_recordings"


def enhance_whisper_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Enhance low-volume/whisper audio for better Whisper recognition.

    Pipeline:
    1. High-pass filter - remove low-frequency noise/rumble
    2. Adaptive RMS normalization - boost to target loudness
    3. Soft limiting - prevent clipping while preserving dynamics
    4. Peak normalization - ensure safe output level
    """
    if audio.size == 0:
        return audio

    # 1. High-pass filter to remove rumble
    sos_hp = butter(2, HIGHPASS_FREQ, btype='highpass', fs=sample_rate, output='sos')
    audio = sosfilt(sos_hp, audio).astype(np.float32)

    # 2. Adaptive RMS normalization
    target_rms = 10 ** (TARGET_RMS_DB / 20)
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 1e-6:
        gain = target_rms / current_rms
        # Limit gain to prevent over-amplification of noise
        gain = min(gain, MAX_GAIN)
        audio = audio * gain

    # 3. Soft limiting using smooth curve (better than tanh for speech)
    # This preserves more dynamics than tanh
    threshold = SOFT_LIMIT_THRESHOLD
    audio_abs = np.abs(audio)
    audio_sign = np.sign(audio)

    # Vectorized soft clipping: linear below threshold, compressed above
    mask_above = audio_abs > threshold
    compressed = np.where(
        mask_above,
        threshold + (1 - threshold) * np.tanh((audio_abs - threshold) / (1 - threshold)),
        audio_abs
    )
    audio = audio_sign * compressed

    # 4. Peak normalization
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio * (PEAK_NORMALIZE_TARGET / peak)

    return audio.astype(np.float32)


def transcribe_with_groq(audio_path: Path, api_key: str, language: str | None = None, max_retries: int = 3) -> str | None:
    """
    Call Groq Whisper API to transcribe audio, returns plain text.

    Args:
        audio_path: Path to the audio file
        api_key: Groq API Key
        language: Language code (e.g., "zh", "en"), None for auto-detection
        max_retries: Maximum retry attempts (for 429/5xx errors)

    Returns:
        Transcribed text, or None on failure
    """
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    for attempt in range(max_retries):
        try:
            with open(audio_path, "rb") as f:
                files = {"file": (audio_path.name, f, "audio/wav")}
                data = {
                    "model": "whisper-large-v3-turbo",
                    "response_format": "text",
                }
                if language:
                    data["language"] = language
                response = requests.post(url, headers=headers, files=files, data=data, timeout=60)

            if response.status_code == 200:
                return response.text.strip()
            elif response.status_code == 429 or response.status_code >= 500:
                # Retryable errors
                wait_time = 2 ** attempt
                logger.warning(f"API error {response.status_code}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Transcription failed: HTTP {response.status_code}")
                return None
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None

    logger.error("Max retries exceeded")
    return None


def polish_transcript_with_llm(raw_text: str, api_key: str) -> str | None:
    """
    Polish transcript text using Gemini LLM.

    Args:
        raw_text: Raw transcript text
        api_key: Gemini API Key

    Returns:
        Polished text, or None on failure
    """
    from google import genai

    client = genai.Client(api_key=api_key)

    prompt = """You are a professional text editor. Transform the following speech transcript into well-structured written text.

Rules:
1. Keep the SAME language as the original - do NOT translate
2. Convert spoken language to written language:
   - Remove filler words (e.g., "um", "uh", "like", "you know", or equivalents in other languages)
   - Transform colloquial expressions into formal written style
   - Fix transcription errors (misheard words, typos)
3. Reorganize content logically:
   - Group related information together
   - Separate different topics into paragraphs with blank lines
   - Use numbered lists when content describes steps, features, or multiple points
4. Preserve ALL substantive information - only remove verbal fillers, not actual content
5. Add proper punctuation and spacing
6. Output ONLY the final polished text - no comments or annotations

Original transcript:
{raw_text}

Output the polished text directly."""

    @retry_with_backoff(
        max_retries=3,
        retryable_exceptions=(Exception,),
        on_retry=lambda e, _: logger.warning(f"LLM error: {e}, retrying..."),
    )
    def _call_llm() -> str:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-09-2025",
            contents=[prompt.format(raw_text=raw_text)],
        )
        return response.text

    return _call_llm()


class WhisperTranscriber:
    """Audio recorder and transcriber optimized for whisper/low-volume speech"""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._recording_event = threading.Event()  # Thread-safe recording state
        self.audio_data: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags | None
    ) -> None:
        """Callback function for audio stream"""
        if status:
            logger.warning(f"Audio status: {status}")
        if self._recording_event.is_set():
            with self._lock:
                self.audio_data.append(indata.copy())

    def start_recording(self):
        """Start recording audio"""
        if self._recording_event.is_set():
            return

        self.audio_data = []
        self._recording_event.set()

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._audio_callback,
        )
        self.stream.start()
        print("Recording started... (press SPACE to stop)")

    def stop_recording(self) -> tuple[Path, Path | None, Path | None] | None:
        """Stop recording, save the audio file, and transcribe with Groq API"""
        if not self._recording_event.is_set():
            return None

        self._recording_event.clear()

        if self.stream:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
                self.stream = None

        with self._lock:
            if not self.audio_data:
                logger.warning("No audio data recorded")
                return None

            # Concatenate all audio chunks
            audio = np.concatenate(self.audio_data, axis=0).flatten()

        logger.info(f"Recorded {len(audio) / SAMPLE_RATE:.2f} seconds of audio")

        # Enhance the audio for whisper recognition
        logger.info("Enhancing audio for whisper recognition...")
        enhanced_audio = enhance_whisper_audio(audio)

        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"whisper_recording_{timestamp}"
        wav_path = self.output_dir / f"{base_name}.wav"

        # Convert to int16 for WAV file
        audio_int16 = (enhanced_audio * 32767).astype(np.int16)
        wavfile.write(wav_path, SAMPLE_RATE, audio_int16)

        logger.info(f"[1/3] Audio saved to: {wav_path}")

        # Transcribe with Groq API
        raw_md_path = None
        polished_md_path = None
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            logger.info("Transcribing with Groq Whisper API...")
            language = os.getenv("WHISPER_LANGUAGE")  # None if not set
            transcript = transcribe_with_groq(wav_path, api_key, language=language)
            if transcript:
                # Save raw transcript
                raw_md_path = self.output_dir / f"{base_name}.md"
                raw_md_path.write_text(transcript, encoding="utf-8")
                logger.info(f"[2/3] Raw transcript saved to: {raw_md_path}")

                # LLM polishing
                gemini_key = os.getenv("GEMINI_API_KEY")
                if gemini_key:
                    logger.info("Polishing transcript with LLM...")
                    polished = polish_transcript_with_llm(transcript, gemini_key)
                    if polished:
                        polished_md_path = self.output_dir / f"{base_name}_polished.md"
                        polished_md_path.write_text(polished, encoding="utf-8")
                        logger.info(f"[3/3] Polished transcript saved to: {polished_md_path}")

                        # Display result
                        print("\n" + "=" * 50)
                        print("Polished transcript:")
                        print("=" * 50)
                        print(polished)
                        print("=" * 50)

                        # Copy to clipboard (cross-platform)
                        try:
                            if sys.platform == "darwin":
                                subprocess.run(
                                    ["pbcopy"],
                                    input=polished.encode("utf-8"),
                                    check=True,
                                )
                            elif sys.platform == "win32":
                                subprocess.run(
                                    ["clip"],
                                    input=polished.encode("utf-16"),
                                    check=True,
                                )
                            else:  # Linux/Unix
                                # Try xclip first, then xsel
                                try:
                                    subprocess.run(
                                        ["xclip", "-selection", "clipboard"],
                                        input=polished.encode("utf-8"),
                                        check=True,
                                    )
                                except FileNotFoundError:
                                    subprocess.run(
                                        ["xsel", "--clipboard", "--input"],
                                        input=polished.encode("utf-8"),
                                        check=True,
                                    )
                            logger.info("Copied to clipboard!")
                        except (subprocess.CalledProcessError, FileNotFoundError):
                            logger.warning("Failed to copy to clipboard")
                    else:
                        logger.warning("[3/3] LLM polish failed, skipped")
                else:
                    logger.info("[3/3] GEMINI_API_KEY not set, skipping LLM polish")
            else:
                logger.warning("[2/3] Transcription failed")
                logger.info("[3/3] Skipped (no transcript)")
        else:
            logger.info("[2/3] GROQ_API_KEY not set, skipping transcription")
            logger.info("[3/3] Skipped (no transcript)")

        return (wav_path, raw_md_path, polished_md_path)


def main():
    """Main function with keyboard listener"""
    from pynput import keyboard

    load_dotenv()
    recorder = WhisperTranscriber()

    print("=" * 50)
    print("Whisper Transcriber")
    print("=" * 50)
    print("Controls:")
    print("  SPACE - Start/Stop recording")
    print("  ESC   - Exit program")
    print("=" * 50)
    print("\nReady. Press SPACE to start recording...")

    def on_press(key):
        try:
            if key == keyboard.Key.space:
                if recorder._recording_event.is_set():
                    recorder.stop_recording()
                    print("\nReady. Press SPACE to start recording...")
                else:
                    recorder.start_recording()
            elif key == keyboard.Key.esc:
                if recorder._recording_event.is_set():
                    recorder.stop_recording()
                print("\nExiting...")
                return False  # Stop listener
        except (OSError, IOError) as e:
            logger.error(f"I/O error: {e}")
        except requests.RequestException as e:
            logger.error(f"Network error: {e}")

    # Start keyboard listener
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    print("Goodbye!")


if __name__ == "__main__":
    main()
