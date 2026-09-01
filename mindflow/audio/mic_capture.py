"""
MindFlow — Live Microphone Audio Capture Utility.
Records live audio from the default microphone and saves it as a 16kHz mono WAV file.
"""

import time
from pathlib import Path
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000


def record_mic(duration_seconds: float = 5.0, output_path: str = "temp_mic.wav") -> str:
    """
    Records live speech from default microphone for `duration_seconds`
    and saves to `output_path` at 16kHz mono.
    """
    print(f"\n[MIC RECORDING] Speak now into the microphone ({duration_seconds:.1f} seconds)...")
    
    # Visual countdown
    recording = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    
    for remaining in range(int(duration_seconds), 0, -1):
        print(f"   [RECORDING] {remaining}s remaining...", end="\r", flush=True)
        time.sleep(1.0)
    
    sd.wait()
    print("   [DONE] Recording complete! Processing audio...          \n")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_p), recording, SAMPLE_RATE)
    return str(out_p.resolve())


if __name__ == "__main__":
    import sys
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    saved = record_mic(duration_seconds=duration, output_path="recorded_sample.wav")
    print(f"Saved recording to: {saved}")
