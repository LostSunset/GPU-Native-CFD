#!/usr/bin/env bash
# Prepare the official HiSA forwardStep example (Mach 3, 2D Euler, transient).
#
# Same pre-solver steps as the official runSim (OpenCFD branch):
#   blockMesh; for zone 1 and 2: topoSet (topoSetDict.N) + refineMesh
#   -dict system/refineMeshDict.ocfd -overwrite; cp -r 0.org 0
# plus checkMesh. The official run is parallel (decomposePar + mpirun);
# here the case is left serial.
#
# Usage: prepare_forwardStep.sh --src DIR --dest DIR --bashrc FILE

set -eu

src=""; dest=""; bashrc=""
while [ $# -gt 0 ]; do
    case "$1" in
        --src) src="$2"; shift 2 ;;
        --dest) dest="$2"; shift 2 ;;
        --bashrc) bashrc="$2"; shift 2 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "$src" ] && [ -n "$dest" ] && [ -n "$bashrc" ] || { sed -n '11p' "$0"; exit 2; }
[ -e "$dest" ] && { echo "ERROR: $dest already exists" >&2; exit 2; }
case "$(df -PT "$(dirname "$(realpath -m "$dest")")" 2>/dev/null | awk 'NR==2 {print $2}')" in
    9p|drvfs|fuseblk|ntfs*) echo "ERROR: $dest is on a Windows filesystem" >&2; exit 2 ;;
esac
( cd "$src" && sha256sum --quiet -c MANIFEST.sha256 )

mkdir -p "$dest"
cp -a "$src/simulation" "$dest/simulation"
cp "$src/PROVENANCE" "$dest/PROVENANCE.export"

(
    set +u
    # shellcheck disable=SC1090
    . "$bashrc" >/dev/null 2>&1 || true
    set -u
    for t in blockMesh topoSet refineMesh checkMesh; do
        command -v "$t" >/dev/null || { echo "ERROR: $t not found" >&2; exit 3; }
    done
    cd "$dest/simulation"
    blockMesh -dict system/blockMeshDict > log.blockMesh 2>&1
    for z in 1 2; do
        cp "system/topoSetDict.$z" system/topoSetDict
        topoSet > "log.topoSet.$z" 2>&1
        refineMesh -dict system/refineMeshDict.ocfd -overwrite > "log.refineMesh.$z" 2>&1
    done
    rm -rf 0
    cp -r 0.org 0
    checkMesh > log.checkMesh 2>&1 || true
)

(
    cd "$dest/simulation"
    find constant system 0 -type f -print0 | sort -z | xargs -0 sha256sum > ../PREPARED.sha256
)
{
    echo "prepared_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "bashrc=$bashrc"
    echo "steps=blockMesh; 2x (topoSet topoSetDict.N; refineMesh -dict refineMeshDict.ocfd -overwrite); cp -r 0.org 0; serial (official run is decomposed)"
    grep -E '^ *cells:' "$dest/simulation/log.checkMesh" | sed 's/^ */checkMesh_/; s/: */=/'
    grep -E 'Max aspect ratio' "$dest/simulation/log.checkMesh" | sed 's/^ */checkMesh_/'
    grep -E 'Mesh OK|Failed' "$dest/simulation/log.checkMesh" | sed 's/^ */checkMesh_result=/'
} > "$dest/PREPARED"
cat "$dest/PREPARED"
