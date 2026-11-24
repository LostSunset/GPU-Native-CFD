#!/usr/bin/env bash
# Prepare the official HiSA rae2822 example.
#
# Follows the official setupMesh/runSim pre-solver steps for OpenFOAM API
# >= 2106, except the Plot3D -> gmsh conversion (p2d2p3d.py, p3d2gmsh.py):
# the converted mesh/p3dMesh/raetaf.msh ships with the example and is
# verified against the export manifest instead of being regenerated.
#   gmshToFoam; stitchMesh -perfect intPatch1 intPatch2; changeDictionary;
#   transformPoints -scale (0.3048 1 0.3048); renumberMesh; cp -r 0.org 0
# The mesh is copied into simulation/constant (official: symlinks).
#
# Usage: prepare_rae2822.sh --src DIR --dest DIR --bashrc FILE

set -eu

src=""; dest=""; bashrc=""
while [ $# -gt 0 ]; do
    case "$1" in
        --src) src="$2"; shift 2 ;;
        --dest) dest="$2"; shift 2 ;;
        --bashrc) bashrc="$2"; shift 2 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "$src" ] && [ -n "$dest" ] && [ -n "$bashrc" ] || { sed -n '13p' "$0"; exit 2; }
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
    for t in gmshToFoam stitchMesh changeDictionary transformPoints renumberMesh checkMesh; do
        command -v "$t" >/dev/null || { echo "ERROR: $t not found" >&2; exit 3; }
    done
    cd "$dest/mesh"
    gmshToFoam p3dMesh/raetaf.msh > log.gmshToFoam 2>&1
    stitchMesh -perfect -toleranceDict stitchMeshToleranceDict intPatch1 intPatch2 -overwrite > log.stitchMesh 2>&1
    changeDictionary > log.changeDictionary 2>&1
    transformPoints -scale '(0.3048 1 0.3048)' > log.transformPoints 2>&1
    rm -f 0/meshPhi
    renumberMesh -overwrite > log.renumberMesh 2>&1
    mkdir -p "$dest/simulation/constant/polyMesh"
    cp -r constant/polyMesh/. "$dest/simulation/constant/polyMesh/"
    cd "$dest/simulation"
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
    echo "steps=gmshToFoam (shipped raetaf.msh); stitchMesh -perfect; changeDictionary; transformPoints -scale (0.3048 1 0.3048); renumberMesh -overwrite; cp -r 0.org 0"
    grep -E '^ *cells:' "$dest/simulation/log.checkMesh" | sed 's/^ */checkMesh_/; s/: */=/'
    grep -E 'Max aspect ratio' "$dest/simulation/log.checkMesh" | sed 's/^ */checkMesh_/'
    grep -E 'Mesh OK|Failed' "$dest/simulation/log.checkMesh" | sed 's/^ */checkMesh_result=/'
} > "$dest/PREPARED"
cat "$dest/PREPARED"
