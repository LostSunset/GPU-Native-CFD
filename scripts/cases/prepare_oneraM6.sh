#!/usr/bin/env bash
# Prepare the official HiSA oneraM6 example.
#
# Follows the official setupMesh/runSim pre-solver steps for OpenFOAM API
# >= 2106, except the Plot3D -> gmsh conversion (expandP3dfmt.py,
# p3d2gmsh.py): the converted mesh/p3dMesh/m6wing.msh ships with the example
# and is verified against the export manifest instead of being regenerated.
#   gmshToFoam -keepOrientation; six stitchMesh -perfect steps (B1KM/B4KM,
#   B1J1/B4J1, B1IM/B2I1, B2KM/B3KM, B2IM/B3I1, B3IM/B4I1); createPatch;
#   changeDictionary; transformPoints -scale (1.1963 1.1963 1.1963);
#   renumberMesh; cp -r 0.org 0
# The mesh is copied into simulation/constant (official: symlinks).
#
# Usage: prepare_oneraM6.sh --src DIR --dest DIR --bashrc FILE

set -eu

src=""; dest=""; bashrc=""
while [ $# -gt 0 ]; do
    case "$1" in
        --src) src="$2"; shift 2 ;;
        --dest) dest="$2"; shift 2 ;;
        --bashrc) bashrc="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "$src" ] && [ -n "$dest" ] && [ -n "$bashrc" ] || { sed -n '15p' "$0"; exit 2; }
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
    for t in gmshToFoam stitchMesh createPatch changeDictionary transformPoints renumberMesh checkMesh; do
        command -v "$t" >/dev/null || { echo "ERROR: $t not found" >&2; exit 3; }
    done
    cd "$dest/mesh"
    gmshToFoam -keepOrientation p3dMesh/m6wing.msh > log.gmshToFoam 2>&1
    i=1
    for pair in "B1KM B4KM" "B1J1 B4J1" "B1IM B2I1" "B2KM B3KM" "B2IM B3I1" "B3IM B4I1"; do
        # shellcheck disable=SC2086
        stitchMesh -perfect -toleranceDict stitchMeshToleranceDict $pair -overwrite > "log.stitchMesh.$i" 2>&1
        i=$((i + 1))
    done
    rm -f 0/meshPhi
    createPatch -overwrite > log.createPatch 2>&1
    changeDictionary > log.changeDictionary 2>&1
    transformPoints -scale '(1.1963 1.1963 1.1963)' > log.transformPoints 2>&1
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
    echo "steps=gmshToFoam -keepOrientation (shipped m6wing.msh); 6x stitchMesh -perfect; createPatch; changeDictionary; transformPoints -scale (1.1963 1.1963 1.1963); renumberMesh -overwrite; cp -r 0.org 0"
    grep -E '^ *cells:' "$dest/simulation/log.checkMesh" | sed 's/^ */checkMesh_/; s/: */=/'
    grep -E 'Max aspect ratio' "$dest/simulation/log.checkMesh" | sed 's/^ */checkMesh_/'
    grep -E 'Mesh OK|Failed' "$dest/simulation/log.checkMesh" | sed 's/^ */checkMesh_result=/'
} > "$dest/PREPARED"
cat "$dest/PREPARED"
