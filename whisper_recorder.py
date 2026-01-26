# /// script
# dependencies = [
#   "sounddevice",
#   "numpy",
#   "scipy",
#   "pynput",
# ]
# ///

"""A command-line tool for recording whisper/low-volume audio with enhancement for Whisper transcription"""

import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
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
    1. High-pass filter to remove low-frequency noise
    2. RMS normalization to boost overall loudness
    3. Soft limiting (tanh) to prevent clipping
    4. Peak normalization to ensure safe output level
    """
    if audio.size == 0:
        return audio

    # 1. High-pass filter at 80 Hz (remove rumble, keep whisper frequencies)
    sos = butter(4, 80, btype='highpass', fs=sample_rate, output='sos')
    audio = sosfilt(sos, audio).astype(np.float32)

    # 2. RMS normalization - boost to target loudness (-15 dB, aggressive for whisper)
    target_rms = 10 ** (-15 / 20)  # ~0.178
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 1e-6:  # Avoid division by zero
        gain = target_rms / current_rms
        audio = audio * gain

    # 3. Soft limiting using tanh to prevent harsh clipping
    # Scale so that values around 1.0 get compressed
    audio = np.tanh(audio)

    # 4. Peak normalization to 0.95 (leave headroom)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio * (0.95 / peak)

    return audio.astype(np.float32)


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

    def stop_recording(self) -> Path | None:
        """Stop recording and save the audio file"""
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
        return output_path


def main():
    """Main function with keyboard listener"""
    from pynput import keyboard

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
