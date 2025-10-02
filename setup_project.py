# setup_project.py
import os
import subprocess
import sys

def setup_project_structure():
    print("🏗️ Creating project structure...")
    directories = ["data", "notebooks", "training", "results", "models", "utils"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Project structure created.")

def run_environment_check():
    print("\n🔧 Running environment check...")
    try:
        import check_environment
        check_environment.check_system()
        check_environment.check_installed_packages()
    except Exception as e:
        print(f"❌ An error occurred during environment check: {e}")

def configure_gpu():
    print("\n⚙️ Configuring GPU optimizations...")
    try:
        import gpu_config
        gpu_config.setup_gpu_optimization()
    except Exception as e:
        print(f"⚠️ GPU optimization setup failed: {e}")

if __name__ == "__main__":
    print("🚀 HEALTHCARE SENTIMENT ANALYSIS - PROJECT SETUP 🚀")
    print("=" * 50)
    
    setup_project_structure()
    run_environment_check()
    configure_gpu()
    
    print("\n🎉 SETUP COMPLETED! 🎉")
    print("=" * 50)
    print("Next steps:")
    print("1. Place your data (.xlsx files) in the 'data/' folder.")
    print("2. Place your training scripts (.py) in the 'training/' folder and update data paths.")
    print("3. Start training your models!")