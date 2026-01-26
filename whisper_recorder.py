# /// script
# dependencies = [
#   "sounddevice",
#   "numpy",
#   "scipy",
#   "pynput",
#   "requests",
#   "python-dotenv",
# ]
# ///

"""A command-line tool for recording whisper/low-volume audio with enhancement for Whisper transcription"""

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import butter, sosfilt


# Audio settings optimized for Whisper
SAMPLE_RATE = 16000  # Whisper's preferred sample rate
CHANNELS = 1
DTYPE = np.float32

# Output directory
OUTPUT_DIR = Path("output_dir")


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

    # 1. High-pass filter at 80 Hz to remove rumble
    sos_hp = butter(2, 80, btype='highpass', fs=sample_rate, output='sos')
    audio = sosfilt(sos_hp, audio).astype(np.float32)

    # 2. Adaptive RMS normalization
    # Target: -18 dB RMS (less aggressive than -15 dB)
    target_rms = 10 ** (-18 / 20)  # ~0.126
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 1e-6:
        gain = target_rms / current_rms
        # Limit gain to prevent over-amplification of noise
        gain = min(gain, 20.0)
        audio = audio * gain

    # 3. Soft limiting using smooth curve (better than tanh for speech)
    # This preserves more dynamics than tanh
    threshold = 0.7
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

    # 4. Peak normalization to 0.95
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio * (0.95 / peak)

    return audio.astype(np.float32)


def transcribe_with_groq(audio_path: Path, api_key: str, max_retries: int = 3) -> str | None:
    """
    调用 Groq Whisper API 转录音频，返回纯文本。

    Args:
        audio_path: 音频文件路径
        api_key: Groq API Key
        max_retries: 最大重试次数（针对 429/5xx 错误）

    Returns:
        转录文本，失败时返回 None
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
                response = requests.post(url, headers=headers, files=files, data=data, timeout=60)

            if response.status_code == 200:
                return response.text.strip()
            elif response.status_code == 429 or response.status_code >= 500:
                # 可重试的错误
                wait_time = 2 ** attempt
                print(f"API error {response.status_code}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Transcription failed: {response.status_code} - {response.text}")
                return None
        except requests.RequestException as e:
            print(f"Request error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None

    print("Max retries exceeded")
    return None


class WhisperRecorder:
    """Audio recorder optimized for whisper/low-volume speech"""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.is_recording = False
        self.audio_data: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        if self.is_recording:
            with self._lock:
                self.audio_data.append(indata.copy())

    def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            return

        self.audio_data = []
        self.is_recording = True

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._audio_callback,
        )
        self.stream.start()
        print("Recording started... (press SPACE to stop)")

    def stop_recording(self) -> tuple[Path, Path | None] | None:
        """Stop recording, save the audio file, and transcribe with Groq API"""
        if not self.is_recording:
            return None

        self.is_recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        with self._lock:
            if not self.audio_data:
                print("No audio data recorded")
                return None

            # Concatenate all audio chunks
            audio = np.concatenate(self.audio_data, axis=0).flatten()

        print(f"Recorded {len(audio) / SAMPLE_RATE:.2f} seconds of audio")

        # Enhance the audio for whisper recognition
        print("Enhancing audio for whisper recognition...")
        enhanced_audio = enhance_whisper_audio(audio)

        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"whisper_recording_{timestamp}.wav"

        # Convert to int16 for WAV file
        audio_int16 = (enhanced_audio * 32767).astype(np.int16)
        wavfile.write(output_path, SAMPLE_RATE, audio_int16)

        print(f"Saved to: {output_path}")

        # Transcribe with Groq API
        md_path = None
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            print("Transcribing with Groq Whisper API...")
            transcript = transcribe_with_groq(output_path, api_key)
            if transcript:
                md_path = output_path.with_suffix(".md")
                md_path.write_text(transcript, encoding="utf-8")
                print(f"Transcript saved to: {md_path}")
            else:
                print("Transcription failed, only WAV file saved")
        else:
            print("Warning: GROQ_API_KEY not set, skipping transcription")

        return (output_path, md_path)


def main():
    """Main function with keyboard listener"""
    from pynput import keyboard

    load_dotenv()
    recorder = WhisperRecorder()
    running = True

    print("=" * 50)
    print("Whisper Recorder")
    print("=" * 50)
    print("Controls:")
    print("  SPACE - Start/Stop recording")
    print("  ESC   - Exit program")
    print("=" * 50)
    print("\nReady. Press SPACE to start recording...")

    def on_press(key):
        nonlocal running

        try:
            if key == keyboard.Key.space:
                if recorder.is_recording:
                    recorder.stop_recording()
                    print("\nReady. Press SPACE to start recording...")
                else:
                    recorder.start_recording()
            elif key == keyboard.Key.esc:
                if recorder.is_recording:
                    recorder.stop_recording()
                running = False
                print("\nExiting...")
                return False  # Stop listener
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)

    # Start keyboard listener
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    print("Goodbye!")


if __name__ == "__main__":
    main()
