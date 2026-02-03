from __future__ import annotations

import math
import os
import random
import struct
import wave
from pathlib import Path


def write_sine_wav(out_path: Path, *, seconds: float = 1.0, sample_rate: int = 22050) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    freq = random.choice([220.0, 330.0, 440.0])
    amplitude = 0.25
    nframes = int(seconds * sample_rate)

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(nframes):
            t = i / sample_rate
            sample = amplitude * math.sin(2.0 * math.pi * freq * t)
            wf.writeframes(struct.pack("<h", int(sample * 32767)))
