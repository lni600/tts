#!/usr/bin/env python3
"""
Setup script for the Streamlit TTS Chatbot.
"""
import subprocess
import sys
import os


def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("📁 Creating necessary directories...")
    
    directories = [
        ".streamlit",
        "temp_audio",
        "llm",
        "realtime", 
        "utils"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ Created {directory}/")
    
    print("✅ All directories created!")


def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Need Python 3.10+")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Streamlit TTS Chatbot\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    print()
    
    # Install requirements
    if install_requirements():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml")
        print("2. Edit secrets.toml with your API keys")
        print("3. Run: streamlit run app.py")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
