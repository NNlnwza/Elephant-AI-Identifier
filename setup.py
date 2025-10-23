#!/usr/bin/env python3
"""
Setup script for Elephant AI Identifier
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'models', 'static', 'templates']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Directories created successfully!")

def main():
    """Main setup function"""
    print("🐘 Elephant AI Identifier - Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed!")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    print("🎉 Setup completed successfully!")
    print("🚀 You can now run the application with: python run.py")
    print("=" * 40)

if __name__ == "__main__":
    main()
