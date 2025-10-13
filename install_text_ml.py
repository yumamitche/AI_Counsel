#!/usr/bin/env python3
"""
Installation script for text-based ML dependencies
Optimized for laptop performance with lightweight models
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
    """Install text-based ML dependencies"""
    print("🚀 Installing Text-Based ML Dependencies...")
    print("=" * 50)
    
    # Core dependencies
    packages = [
        "sentence-transformers==2.2.2",
        "scikit-learn==1.3.0", 
        "torch==2.0.1",
        "transformers==4.33.2"
    ]
    
    success_count = 0
    total_packages = len(packages)
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        if install_package(package):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Installation Summary:")
    print(f"✅ Successfully installed: {success_count}/{total_packages} packages")
    
    if success_count == total_packages:
        print("🎉 All text-based ML dependencies installed successfully!")
        print("\n🔧 Next steps:")
        print("1. Run: python text_ml_engine.py (to test the engine)")
        print("2. Run: python app.py (to start the enhanced counseling system)")
        print("3. The system will now use deep text processing for recommendations!")
    else:
        print("⚠️ Some packages failed to install. You can still use the system with basic text analysis.")
        print("💡 Try installing failed packages manually or check your internet connection.")
    
    print("\n📋 Model Information:")
    print("- Sentence Transformer: all-MiniLM-L6-v2 (22MB, lightweight)")
    print("- Text Classification: TF-IDF + Naive Bayes + Logistic Regression")
    print("- Text Clustering: K-Means on sentence embeddings")
    print("- Emotion Detection: Keyword-based analysis")
    print("- Sentiment Analysis: Rule-based + ML hybrid")

if __name__ == "__main__":
    main()
