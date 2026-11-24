#!/usr/bin/env bash
# Prepare the official HiSA nacaAirfoil example (supersonic, Mach 1.78).
#
# Same pre-solver steps as the official runSim:
#   star4ToFoam -scale 1 ../mesh/nacaAirfoil; symmetry -> empty in the
#   boundary file; cp -rf 0.org 0
# plus checkMesh. The simulation directory is copied to --dest.
#
# Usage: prepare_nacaAirfoil.sh --src DIR --dest DIR --bashrc FILE

set -eu

src=""; dest=""; bashrc=""
while [ $# -gt 0 ]; do
    case "$1" in
        --src) src="$2"; shift 2 ;;
        --dest) dest="$2"; shift 2 ;;
        --bashrc) bashrc="$2"; shift 2 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "$src" ] && [ -n "$dest" ] && [ -n "$bashrc" ] || { sed -n '10p' "$0"; exit 2; }
[ -e "$dest" ] && { echo "ERROR: $dest already exists" >&2; exit 2; }
case "$(df -PT "$(dirname "$(realpath -m "$dest")")" 2>/dev/null | awk 'NR==2 {print $2}')" in
    9p|drvfs|fuseblk|ntfs*) echo "ERROR: $dest is on a Windows filesystem" >&2; exit 2 ;;
esac
( cd "$src" && sha256sum --quiet -c MANIFEST.sha256 )

mkdir -p "$dest"
cp -a "$src/mesh" "$dest/mesh"
cp -a "$src/simulation" "$dest/simulation"
cp "$src/PROVENANCE" "$dest/PROVENANCE.export"

(
    set +u
    # shellcheck disable=SC1090
    . "$bashrc" >/dev/null 2>&1 || true
    set -u
    for t in star4ToFoam checkMesh; do
        command -v "$t" >/dev/null || { echo "ERROR: $t not found" >&2; exit 3; }
    done
    cd "$dest/simulation"
    star4ToFoam -scale 1 ../mesh/nacaAirfoil > log.star4ToFoam 2>&1
    sed -i -e 's/symmetry\([)]*;\)/empty\1/' constant/polyMesh/boundary
    rm -rf 0
    cp -rf 0.org 0
    checkMesh > log.checkMesh 2>&1 || true
)

(
    cd "$dest/simulation"
    find constant system 0 -type f -print0 | sort -z | xargs -0 sha256sum > ../PREPARED.sha256
)
{
    echo "prepared_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "bashrc=$bashrc"
    echo "steps=star4ToFoam -scale 1 ../mesh/nacaAirfoil; symmetry -> empty in constant/polyMesh/boundary; cp -rf 0.org 0"
    grep -E '^ *cells:' "$dest/simulation/log.checkMesh" | sed 's/^ */checkMesh_/; s/: */=/'
    grep -E 'Max aspect ratio' "$dest/simulation/log.checkMesh" | sed 's/^ */checkMesh_/'
    grep -E 'Mesh OK|Failed' "$dest/simulation/log.checkMesh" | sed 's/^ */checkMesh_result=/'
} > "$dest/PREPARED"
cat "$dest/PREPARED"
