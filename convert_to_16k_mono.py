import os
import subprocess
from pathlib import Path

input_dir = Path("2vec/data2wav_in")       # source folder
output_dir = Path("2vec/data2wav_in_16k")  # output folder
output_dir.mkdir(parents=True, exist_ok=True)

for file_path in input_dir.iterdir():
    if file_path.suffix.lower() in [".wav", ".flac"]:
        out_path = output_dir / (file_path.stem + ".wav")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(file_path),
            "-ac", "1",             # mono
            "-ar", "16000",         # 16 kHz
            str(out_path)
        ]
        subprocess.run(cmd, check=True)

print("✅ Conversion complete.")