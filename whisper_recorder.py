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
    1. Pre-emphasis - enhance high-frequency consonants (P, T, K, S, etc.)
    2. Noise gate - remove low-energy noise below dynamic threshold
    3. Dynamic compression - boost quiet parts while preserving dynamics
    4. Spectral shaping - enhance 1-4kHz speech-critical frequency band
    5. Peak normalization - ensure safe output level
    """
    if audio.size == 0:
        return audio

    # 1. Pre-emphasis: first-order high-pass to enhance consonants
    # This helps preserve plosives like "P" which are weak in whispers
    pre_emphasis = 0.97
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # 2. Noise gate: suppress frames below dynamic threshold
    frame_size = int(sample_rate * 0.02)  # 20ms frames
    hop_size = frame_size // 2  # 50% overlap
    num_frames = (len(audio) - frame_size) // hop_size + 1

    if num_frames > 0:
        # Calculate frame energies
        frame_energies = np.array([
            np.sqrt(np.mean(audio[i * hop_size:i * hop_size + frame_size] ** 2))
            for i in range(num_frames)
        ])

        # Dynamic threshold: 20th percentile of frame energies
        threshold = np.percentile(frame_energies, 20)

        # Apply soft noise gate with smooth attack/release
        gated_audio = np.zeros_like(audio)
        for i in range(num_frames):
            start = i * hop_size
            end = min(start + frame_size, len(audio))
            energy = frame_energies[i]

            if energy > threshold:
                # Soft gate: gradual transition
                gate_gain = min(1.0, (energy / threshold) ** 0.5)
            else:
                gate_gain = 0.1  # Don't completely silence, keep some context

            # Apply with Hann window for smooth transitions
            window = np.hanning(end - start)
            gated_audio[start:end] += audio[start:end] * gate_gain * window

        # Normalize overlap-add result
        audio = gated_audio / (np.max(np.abs(gated_audio)) + 1e-8)

    # 3. Dynamic compression: logarithmic compression instead of tanh
    # Threshold at -30dB, ratio 4:1
    threshold_db = -30
    ratio = 4.0
    threshold_linear = 10 ** (threshold_db / 20)

    # Convert to dB domain for compression
    audio_abs = np.abs(audio)
    audio_sign = np.sign(audio)

    # Avoid log(0)
    audio_abs = np.maximum(audio_abs, 1e-10)
    audio_db = 20 * np.log10(audio_abs)

    # Apply compression above threshold
    mask = audio_db > threshold_db
    compressed_db = np.where(
        mask,
        threshold_db + (audio_db - threshold_db) / ratio,
        audio_db
    )

    # Convert back to linear
    audio = audio_sign * (10 ** (compressed_db / 20))

    # 4. Spectral shaping: enhance 1-4kHz speech-critical band
    # This frequency range contains most consonant information
    sos = butter(2, [1000, 4000], btype='bandpass', fs=sample_rate, output='sos')
    enhanced_band = sosfilt(sos, audio).astype(np.float32)

    # Mix enhanced band with original (additive enhancement)
    audio = audio + 0.5 * enhanced_band

    # 5. Peak normalization to 0.95 (leave headroom)
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
