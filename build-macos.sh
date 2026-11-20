#!/usr/bin/env bash
# Builds the standalone macOS app and a correctly packed .zip for Releases.
#
#   ./build-macos.sh
#
# Needs: Apple Silicon Mac, Homebrew, `brew install python@3.12 python-tk@3.12 ffmpeg`.
#
# Why this script exists: an unsigned .app zipped with Finder's "Compress"
# and downloaded from GitHub opens with "... is damaged and can't be
# opened". Two things fix that on the build side:
#   1. an ad-hoc signature over the WHOLE bundle (codesign --deep -s -) -
#      py2app signs the individual binaries but leaves the bundle itself
#      unsealed, which Gatekeeper reports as "damaged", not "unidentified";
#   2. zipping with `ditto`, which keeps symlinks and signatures intact and
#      doesn't add the __MACOSX/ junk folder.
# The person downloading it still has to clear the quarantine flag once
# (see README, "macOS"), since there's no paid Apple notarization.
set -euo pipefail
cd "$(dirname "$0")"

APP_NAME="Local Transcription & Subtitle Creator"
PY="${PYTHON:-/opt/homebrew/bin/python3.12}"

[[ "$(uname -m)" == "arm64" ]] || { echo "Build on an Apple Silicon Mac."; exit 1; }
command -v ffmpeg >/dev/null || { echo "Run: brew install ffmpeg"; exit 1; }
"$PY" -c "import tkinter" 2>/dev/null || { echo "Run: brew install python-tk@3.12"; exit 1; }

rm -rf build dist .venv-macos
"$PY" -m venv .venv-macos
source .venv-macos/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -r requirements-optional.txt

python -m pytest -q 2>/dev/null || echo "(pytest not installed - skipping tests)"

python setup.py py2app

APP="dist/$APP_NAME.app"
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP" && echo "Signature OK"

VERSION="$(git describe --tags --always 2>/dev/null || echo dev)"
ZIP="dist/Local-Transcription-Subtitle-Creator-macOS-arm64-${VERSION}.zip"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIP"
echo
echo "Built: $ZIP"
echo "Test it like a user would: unzip a copy elsewhere, then"
echo "  xattr -dr com.apple.quarantine \"/path/to/$APP_NAME.app\"  &&  open it."
