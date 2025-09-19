import psutil
import torch
import os

def get_memory_mb():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_gpu_memory_mb():
    """Get GPU memory usage in MB if available"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def get_available_memory_mb():
    """Get available system memory"""
    memory = psutil.virtual_memory()
    available_mb = memory.available / (1024 * 1024)
    return available_mb