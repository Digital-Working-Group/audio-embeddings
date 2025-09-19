import numpy as np
import soundfile as sf

# Generate 1 hour at 16kHz (common for speech models)
duration = 180
sample_rate = 16000  # Better for speech models
noise = np.random.normal(0, 0.1, int(duration * sample_rate)).astype(np.float32)
sf.write('noise_1hour_16k.flac', noise, sample_rate)