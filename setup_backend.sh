#!/bin/bash

echo "Setting up Python backend environment..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "Python found!"
python3 --version
echo

# Install dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Failed to install dependencies"
    echo "Try running: pip3 install -r requirements.txt"
    exit 1
fi

echo
echo "Setup complete!"
echo
echo "To start the backend server, run:"
echo "  cd backend"
echo "  python3 app.py"
echo
