#!/usr/bin/env sh
set -eu
STUDIO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec java -jar "$STUDIO_ROOT/server/audio-studio-server.jar" --root "$STUDIO_ROOT/dist" "$@"
