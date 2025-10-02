# check_environment.py
import torch
import sys
import subprocess

def check_system():
    print("=" * 50)
    print("SYSTEM ENVIRONMENT CHECK")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA not available - check your installation")
    print("=" * 50)

def check_installed_packages():
    print("\nESSENTIAL PACKAGES CHECK")
    print("=" * 30)
    packages = ["transformers", "datasets", "accelerate", "pandas", "peft", "bitsandbytes"]
    for package in packages:
        try:
            version = subprocess.check_output([
                sys.executable, "-c", f"import {package}; print({package}.__version__)"
            ]).decode().strip()
            print(f"✅ {package}: {version}")
        except Exception:
            print(f"❌ {package}: NOT INSTALLED or FAILED TO IMPORT")

if __name__ == "__main__":
    check_system()
    check_installed_packages()