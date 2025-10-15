#!/usr/bin/env python3
"""
Lightweight AI Counseling System Runner
Optimized for laptop performance with dynamic ML
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask',
        'numpy',
        'pandas',
        'nltk'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("[OK] All required packages are installed")
    return True

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        print("📥 Downloading NLTK data...")
        
        nltk_data = [
            'punkt',
            'stopwords', 
            'wordnet',
            'vader_lexicon'
        ]
        
        for data in nltk_data:
            try:
                nltk.download(data, quiet=True)
                print(f"   [OK] {data}")
            except:
                print(f"   [WARNING] {data} (optional)")
        
        print("[OK] NLTK data downloaded")
        return True
    except Exception as e:
        print(f"[WARNING] NLTK data download failed: {e}")
        print("   System will use basic text processing")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'ml_models',
        'logs',
        'data'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"[OK] Created directory: {directory}")

def main():
    """Main function to run the lightweight AI counseling system"""
    print("[START] Lightweight AI Counseling System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Download NLTK data
    download_nltk_data()
    
    # Create directories
    create_directories()
    
    print("\n[TARGET] System Features:")
    print("   • Pure Python ML Engine with online learning")
    print("   • No external ML library dependencies")
    print("   • Real-time pattern recognition")
    print("   • Laptop-optimized performance")
    print("   • Automatic model updates")
    
    print("\n🌐 Starting server...")
    print("   Open your browser and go to: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Run the Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
