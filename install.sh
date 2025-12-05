#!/bin/bash
# PyCAM3D Installation Script
# This script installs PyCAM3D and its dependencies

set -e

echo "=================================="
echo "  PyCAM3D Installation"
echo "=================================="

# Check Python version
python3 --version || { echo "ERROR: Python 3 not found"; exit 1; }

# Install in editable mode with all dependencies
echo ""
echo "Installing dependencies..."
pip install -e . --quiet

# Create symlink in /usr/local/bin (optional)
if [ -w /usr/local/bin ]; then
    echo "Creating symlink in /usr/local/bin..."
    ln -sf "$(pwd)/pycam3d" /usr/local/bin/pycam3d 2>/dev/null || true
fi

echo ""
echo "=================================="
echo "  Installation Complete!"
echo "=================================="
echo ""
echo "Usage:"
echo "  ./pycam3d --help              # Show all commands"
echo "  ./pycam3d 4axis piece.stl     # Generate 4-axis G-code"
echo ""
echo "Or if installed globally:"
echo "  pycam3d --help"
echo ""
