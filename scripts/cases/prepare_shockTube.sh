#!/usr/bin/env bash
# Prepare the official HiSA shockTube example for timed runs.
#
# Reproduces the pre-solver steps of the official examples/shockTube/runSim
# (blockMesh -dict system/blockMeshDict; cp -r 0.org 0; setFields) into a
# separate directory, leaving the exported pristine copy untouched. The solver
# itself is NOT run here; scripts/run/run_hisa_case.sh does that on fresh
# copies of the prepared case.
#
# Usage:
#   prepare_shockTube.sh --src DIR --dest DIR --bashrc FILE
#
#   --src     export from fetch_hisa_example.sh (contains simulation/)
#   --dest    prepared case directory (must not exist; WSL ext4)
#   --bashrc  OpenFOAM etc/bashrc of the build under test

set -eu

src=""; dest=""; bashrc=""
while [ $# -gt 0 ]; do
    case "$1" in
        --src) src="$2"; shift 2 ;;
        --dest) dest="$2"; shift 2 ;;
        --bashrc) bashrc="$2"; shift 2 ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "$src" ] && [ -n "$dest" ] && [ -n "$bashrc" ] || { sed -n '11,17p' "$0"; exit 2; }
[ -f "$src/simulation/system/blockMeshDict" ] || { echo "ERROR: $src is not a shockTube export" >&2; exit 2; }
[ -e "$dest" ] && { echo "ERROR: $dest already exists" >&2; exit 2; }
case "$(df -PT "$(dirname "$(realpath -m "$dest")")" 2>/dev/null | awk 'NR==2 {print $2}')" in
    9p|drvfs|fuseblk|ntfs*) echo "ERROR: $dest is on a Windows filesystem" >&2; exit 2 ;;
esac

# Verify the export against its manifest before using it.
( cd "$src" && sha256sum --quiet -c MANIFEST.sha256 )

mkdir -p "$(dirname "$dest")"
cp -a "$src/simulation" "$dest"
cp "$src/PROVENANCE" "$dest/PROVENANCE.export"

(
    set +u
    # OpenFOAM's etc/bashrc can return non-zero even when it succeeds, so
    # judge success by the tools it puts on PATH instead of its status.
    # shellcheck disable=SC1090
    . "$bashrc" >/dev/null 2>&1 || true
    set -u
    for t in blockMesh setFields checkMesh; do
        command -v "$t" >/dev/null || { echo "ERROR: $t not found after sourcing $bashrc" >&2; exit 3; }
    done
    cd "$dest"
    blockMesh -dict system/blockMeshDict > log.blockMesh 2>&1
    cp -r 0.org 0
    setFields > log.setFields 2>&1
    checkMesh > log.checkMesh 2>&1 || true
)

(
    cd "$dest"
    find constant system 0 -type f -print0 | sort -z | xargs -0 sha256sum > PREPARED.sha256
)
{
    echo "prepared_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "bashrc=$bashrc"
    echo "steps=blockMesh -dict system/blockMeshDict; cp -r 0.org 0; setFields (official runSim pre-solver steps)"
    grep -E '^ *cells:' "$dest/log.checkMesh" | sed 's/^ */checkMesh_/; s/: */=/'
    grep -E 'Mesh OK|Failed' "$dest/log.checkMesh" | sed 's/^ */checkMesh_result=/'
} > "$dest/PREPARED"
cat "$dest/PREPARED"
