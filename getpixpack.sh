#!/usr/bin/env bash
set -e

URL="https://github.com/SauersML/pixpack/archive/refs/heads/main.zip"
TARGET="pixpack"
PATCHES="patches"
TMPDIR="$(mktemp -d)"
ZIP="$TMPDIR/pixpack-main.zip"

# downloader
if command -v curl >/dev/null 2>&1; then DL=(curl -L -o); elif command -v wget >/dev/null 2>&1; then DL=(wget -O); else echo "need curl or wget"; exit 1; fi

mkdir -p "$PATCHES"

# fetch + unpack
"${DL[@]}" "$ZIP" "$URL"
unzip -q "$ZIP" -d "$TMPDIR"
# detect extracted directory dynamically
UPSTREAM="$(find "$TMPDIR" -mindepth 1 -maxdepth 1 -type d -name 'pixpack-*' | head -n 1)"
[ -z "$UPSTREAM" ] && UPSTREAM="$(find "$TMPDIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
[ -d "$UPSTREAM" ] || { echo "unzipped dir not found"; exit 1; }

# diff current vs upstream (if current exists)
if [ -d "$TARGET" ]; then
  if command -v diff >/dev/null 2>&1; then
    diff -ruN \
      --exclude 'target' \
      --exclude '.git' \
      --exclude 'CACHEDIR.TAG' \
      --exclude 'Cargo.lock' \
      "$UPSTREAM" "$TARGET" > "$PATCHES/pixpack-local-delta.patch" || true
  fi
fi

# replace
rm -rf "$TARGET"
mkdir -p "$TARGET"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "$UPSTREAM"/ "$TARGET"/
else
  cp -a "$UPSTREAM"/. "$TARGET"/
fi

rm -rf "$TMPDIR"
echo "ok: synced upstream into ./$TARGET"
[ -f "$PATCHES/pixpack-local-delta.patch" ] && echo "diff: $PATCHES/pixpack-local-delta.patch"

