# from df import enhance, init_df
# import numpy as np
# from torch import Tensor
# import torch

# class Denoiser():
#     def __init__(self, atten_lim_db):
#         self.model, self.df_state, _ = init_df()
#         self.atten_lim_db = atten_lim_db
#     def denoise_deepfilternet(self,audio_chunk: Tensor | np.ndarray, sample_rate: int = 16000):
#         """
#         Denoise audio using DeepFilterNet.
        
#         Returns denoised audio as tensor
#         """
#         if isinstance(audio_chunk, np.ndarray):
#             audio_chunk = torch.from_numpy(audio_chunk).float()
        
#         # Ensure float32 dtype
#         if audio_chunk.dtype != torch.float32:
#             audio_chunk = audio_chunk.float()
        
#         if audio_chunk.ndim == 1:
#             audio_chunk = audio_chunk.unsqueeze(0)
        
#         enhanced = enhance(self.model, self.df_state, audio_chunk,atten_lim_db=self.atten_lim_db)
#         enhanced = enhanced.squeeze()
        
#         return enhanced.cpu().numpy()
        
