#!/usr/bin/env python3
"""
Elephant AI Identifier - Hackathon Edition
Runner script for easy deployment
"""

import os
import sys
import subprocess
import webbrowser
from threading import Timer

def check_dependencies():
    """Check if all required packages are installed"""
    try:
        import flask
        import cv2
        import numpy
        import sklearn
        import mediapipe
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'models', 'static', 'templates']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Directories created successfully!")

def open_browser():
    """Open browser after a short delay"""
    webbrowser.open('http://localhost:5000')

def main():
    """Main function to run the application"""
    print("🐘 Elephant AI Identifier - Hackathon Edition")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Start the application
    print("🚀 Starting Elephant AI Identifier...")
    print("📱 The application will open in your browser automatically")
    print("🌐 Manual access: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    
    # Open browser after 2 seconds
    Timer(2.0, open_browser).start()
    
    # Run Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
