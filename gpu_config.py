# gpu_config.py
import torch
import os
import gc

def setup_gpu_optimization():
    print("🎯 Configuring GPU optimizations...")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("✅ GPU optimization configured.")

def clear_gpu_memory():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("🧹 GPU memory cleared.")