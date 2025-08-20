#!/bin/bash
# Script to build and install the speech-mcp extension locally

set -e  # Exit on error

# Change to the project directory
cd "$(dirname "$0")"

# Build web UI assets (uses prebuilt if Node not available)
if command -v node >/dev/null 2>&1; then
  echo "Building web UI assets..."
  (cd src/speech_mcp/web && npm install && npm run build) || echo "Web build skipped (non-fatal). Using committed assets."
else
  echo "Node.js not found; using committed UI assets."
fi

# Build the wheel
echo "Building wheel..."
python -m build

# Get the wheel file path
WHEEL_FILE=$(ls -t dist/*.whl | head -1)
WHEEL_PATH="$(pwd)/$WHEEL_FILE"

echo "Using wheel file: $WHEEL_PATH"

# Install with UVX
if [ "$1" == "--goose" ]; then
    # Install and start a Goose session
    echo "Installing and starting Goose session with local extension..."
    goose session --with-extension "uvx $WHEEL_PATH"
else
    # Just install with UVX
    echo "Installing with UVX..."
    uvx "$WHEEL_PATH"
fi

echo "Installation complete!"