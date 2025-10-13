#!/usr/bin/env python3
"""
Installation script for TinyLlama dynamic text generation
Optimized for laptop performance with lightweight AI model
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install TinyLlama dependencies for dynamic text generation"""
    print("🚀 Installing TinyLlama for Dynamic Text Generation...")
    print("=" * 60)
    
    # Core dependencies for TinyLlama
    packages = [
        "torch==2.0.1",
        "transformers==4.33.2",
        "accelerate==0.24.1",
        "bitsandbytes==0.41.3"
    ]
    
    success_count = 0
    total_packages = len(packages)
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        if install_package(package):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Installation Summary:")
    print(f"✅ Successfully installed: {success_count}/{total_packages} packages")
    
    if success_count == total_packages:
        print("🎉 TinyLlama dependencies installed successfully!")
        print("\n🔧 Next steps:")
        print("1. Run: python tinylama_recommendation_engine.py (to test TinyLlama)")
        print("2. Run: python app.py (to start the AI-powered counseling system)")
        print("3. The system will now generate dynamic, personalized recommendations!")
        print("\n💾 Optional: To avoid runtime downloads, you can download the TinyLlama model locally using the included helper:")
        print("   - python download_tinyllama_local.py --output ml_models/tinyllama")
        print("   Then set the LOCAL_TINYLLAMA_PATH environment variable to point to that folder before running the app.")
        
        print("\n📋 TinyLlama Features:")
        print("- Dynamic text generation (no more templates!)")
        print("- Personalized recommendations based on user input")
        print("- Emotion-aware counseling advice")
        print("- Natural language recommendations")
        print("- CPU-optimized for laptop performance")
        print("- 1.1B parameter model (lightweight)")
        
        print("\n🎯 What This Enables:")
        print("- AI generates unique recommendations for each user")
        print("- References specific user emotions and challenges")
        print("- Provides context-aware counseling advice")
        print("- Adapts tone to user's emotional state")
        print("- No more hardcoded template responses")
        
    else:
        print("⚠️ Some packages failed to install. You can still use the system with template-based recommendations.")
        print("💡 Try installing failed packages manually or check your internet connection.")
    
    print("\n📊 Model Information:")
    print("- Model: TinyLlama-1.1B-Chat-v1.0")
    print("- Size: ~2.2GB (laptop-friendly)")
    print("- Parameters: 1.1 billion")
    print("- Inference: CPU-optimized")
    print("- Generation: Dynamic text creation")
    print("- Fallback: Template-based if model fails")

if __name__ == "__main__":
    main()
