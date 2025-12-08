#!/bin/bash
# Quick start script for JARVIS

cd "$(dirname "$0")" || exit 1

echo "🛰️  Starting JARVIS Voice Assistant..."
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Are you in the jarvis directory?"
    exit 1
fi

# Run JARVIS
python3 -m jarvis.main "$@"

