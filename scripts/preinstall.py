"""
Pre-installation script for speech-mcp.
This script runs before package installation to check system dependencies.
"""
import sys
import subprocess
import platform
from pathlib import Path

def check_portaudio():
    """Check if portaudio is installed and provide installation instructions if not."""
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        try:
            subprocess.run(["brew", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("\n⚠️  Homebrew is required to install portaudio. Please install it first:")
            print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
            
        try:
            subprocess.run(["brew", "list", "portaudio"], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            print("\n⚠️  portaudio is required for PyAudio but not installed.")
            print("   Please install it using:")
            print("   brew install portaudio")
            return False
            
    elif system == "linux":
        # Check for common package managers
        if Path("/etc/debian_version").exists():  # Debian/Ubuntu
            print("\n⚠️  Please ensure portaudio19-dev is installed:")
            print("   sudo apt-get install python3-dev portaudio19-dev")
        elif Path("/etc/fedora-release").exists():  # Fedora
            print("\n⚠️  Please ensure portaudio-devel is installed:")
            print("   sudo dnf install python3-devel portaudio-devel")
        else:
            print("\n⚠️  Please ensure portaudio development files are installed")
            print("   Consult your distribution's package manager")
        return False
        
    elif system == "windows":
        # PyAudio wheels are available for Windows, so no additional setup needed
        return True
        
    return False

def main():
    """Main entry point for pre-installation checks."""
    # Only run checks when installing the package
    if len(sys.argv) > 1 and sys.argv[1] == "install":
        if not check_portaudio():
            print("\n❌ Required system dependencies are missing.")
            print("   Please install the required dependencies and try again.")
            print("   After installing dependencies, run:")
            print("   pip install speech-mcp\n")
            sys.exit(1)
        
        print("\n✅ System dependencies check passed.\n")
    
if __name__ == "__main__":
    main()