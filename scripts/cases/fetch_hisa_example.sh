#!/usr/bin/env bash
# Export one official HiSA example case at the commit used for all records
# (gitlab.com/hisa/hisa, tag validated-v2512). The export is followed by the
# matching prepare_<case>.sh in this directory, which repeats the example's
# own pre-solver steps (mesh conversion, 0.org -> 0) and records checksums.
#
# Usage:
#   fetch_hisa_example.sh --name NAME --cache DIR --dest DIR
#
#   --name   directory below examples/ (shockTube, forwardStep, flatPlate,
#            rae2822, nacaAirfoil, oneraM6, basicFinner)
#   --cache  blobless clone of the official repository (created if absent)
#   --dest   output directory (must not exist)
#
# Exit: 0 exported, 2 error

set -eu

HISA_URL=https://gitlab.com/hisa/hisa.git
HISA_COMMIT=271b44919d2b107eee05566dd92eb735e9bb08e9

name=""; cache=""; dest=""
while [ $# -gt 0 ]; do
    case "$1" in
        --name) name="$2"; shift 2 ;;
        --cache) cache="$2"; shift 2 ;;
        --dest) dest="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "$name" ] && [ -n "$cache" ] && [ -n "$dest" ] || { sed -n '8,13p' "$0"; exit 2; }
case "$name" in
    */*|.*) echo "ERROR: --name must be a single directory name" >&2; exit 2 ;;
esac
[ -e "$dest" ] && { echo "ERROR: $dest already exists" >&2; exit 2; }

if [ ! -d "$cache/.git" ]; then
    git clone --quiet --filter=blob:none --no-checkout "$HISA_URL" "$cache"
fi
git -C "$cache" cat-file -e "$HISA_COMMIT^{commit}" 2>/dev/null \
    || git -C "$cache" fetch --quiet origin
git -C "$cache" cat-file -e "$HISA_COMMIT:examples/$name" 2>/dev/null \
    || { echo "ERROR: examples/$name not found at $HISA_COMMIT" >&2; exit 2; }

mkdir -p "$dest"
git -C "$cache" archive --format=tar "$HISA_COMMIT" "examples/$name" \
    | tar -x -C "$dest" --strip-components=2
(
    cd "$dest"
    find . -type f ! -name MANIFEST.sha256 -print0 | sort -z \
        | xargs -0 sha256sum > MANIFEST.sha256
)
{
    echo "source_url=$HISA_URL"
    echo "source_commit=$HISA_COMMIT"
    echo "source_path=examples/$name"
    echo "exported_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "files=$(wc -l < "$dest/MANIFEST.sha256")"
    echo "bytes=$(du -sb "$dest" | awk '{print $1}')"
} > "$dest/PROVENANCE"
cat "$dest/PROVENANCE"
