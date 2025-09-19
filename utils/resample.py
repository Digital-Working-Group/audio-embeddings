import soundfile
import numpy as np
import os
import tqdm
import torch
import torchaudio.functional as F


def resample(audio_data : np.ndarray, curr_sample_rate, new_samplerate = 16000):
    '''Resample audio data using torch functions'''
    audio_tensor = torch.from_numpy(audio_data).float()

    if audio_tensor.dim() > 1:
        audio_tensor = torch.mean(audio_tensor, dim=1)  # Average channels to convert to mono

    if audio_tensor.dim() == 1:  
        audio_tensor = audio_tensor.unsqueeze(0)
    
    resampled = F.resample(
        waveform=audio_tensor,
        orig_freq=curr_sample_rate,
        new_freq=new_samplerate,
        resampling_method="sinc_interp_hann"
    )
    
    # Convert back to numpy
    return resampled.squeeze(0).numpy()
