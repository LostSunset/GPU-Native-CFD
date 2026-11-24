#!/usr/bin/env bash
# Prepare the official HiSA flatPlate example, optionally with a different
# wall-normal grading (aspect-ratio grid family for falsification test F2).
#
# Reproduces the official setupMesh + runSim pre-solver steps (blockMesh in
# mesh/, polyMesh into simulation/constant, cp -r 0.org 0) into a separate
# directory; the mesh is copied instead of symlinked so each variant is
# self-contained. Only the wall-normal simpleGrading value of both blocks
# changes (official: 2000); cell counts stay 20x100 + 60x100.
#
# Usage:
#   prepare_flatPlate.sh --src DIR --dest DIR --bashrc FILE [--wall-grading G]

set -eu

src=""; dest=""; bashrc=""; grading=2000
while [ $# -gt 0 ]; do
    case "$1" in
        --src) src="$2"; shift 2 ;;
        --dest) dest="$2"; shift 2 ;;
        --bashrc) bashrc="$2"; shift 2 ;;
        --wall-grading) grading="$2"; shift 2 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "$src" ] && [ -n "$dest" ] && [ -n "$bashrc" ] || { sed -n '12,13p' "$0"; exit 2; }
case "$grading" in *[!0-9.]*|"") echo "ERROR: --wall-grading must be a number" >&2; exit 2 ;; esac
[ -e "$dest" ] && { echo "ERROR: $dest already exists" >&2; exit 2; }
case "$(df -PT "$(dirname "$(realpath -m "$dest")")" 2>/dev/null | awk 'NR==2 {print $2}')" in
    9p|drvfs|fuseblk|ntfs*) echo "ERROR: $dest is on a Windows filesystem" >&2; exit 2 ;;
esac
( cd "$src" && sha256sum --quiet -c MANIFEST.sha256 )

mkdir -p "$dest"
cp -a "$src/mesh" "$dest/mesh"
cp -a "$src/simulation" "$dest/simulation"
cp "$src/PROVENANCE" "$dest/PROVENANCE.export"

bmd="$dest/mesh/system/blockMeshDict"
n=$(grep -cE 'simpleGrading \([^ ]+ +2000 +1\)' "$bmd" || true)
[ "$n" = "2" ] || { echo "ERROR: expected two 'simpleGrading (x 2000 1)' entries, found $n" >&2; exit 2; }
sed -i -E "s/simpleGrading \(([^ ]+) +2000 +1\)/simpleGrading (\1 $grading 1)/" "$bmd"
diff -u "$src/mesh/system/blockMeshDict" "$bmd" > "$dest/GRADING.diff" || true

(
    set +u
    # shellcheck disable=SC1090
    . "$bashrc" >/dev/null 2>&1 || true
    set -u
    for t in blockMesh checkMesh; do
        command -v "$t" >/dev/null || { echo "ERROR: $t not found" >&2; exit 3; }
    done
    cd "$dest/mesh"
    blockMesh -dict system/blockMeshDict > log.blockMesh 2>&1
    mkdir -p "$dest/simulation/constant/polyMesh"
    cp -r constant/polyMesh/. "$dest/simulation/constant/polyMesh/"
    cd "$dest/simulation"
    rm -rf 0
    cp -r 0.org 0
    checkMesh > log.checkMesh 2>&1 || true
)

# Wall-normal geometry from the blockMesh definition (height H, n cells,
# grading G = last/first): r = G^(1/(n-1)), h1 = H (r-1)/(r^n - 1).
python3 - "$grading" > "$dest/GRID_METRICS" <<'EOF'
import sys
G = float(sys.argv[1]); H = 0.9652; n = 100
r = G ** (1.0 / (n - 1))
h1 = H * (r - 1) / (r ** n - 1) if r != 1 else H / n
# plate block: 60 cells over 0.9144 m with streamwise grading 50
Lx, nx, gx = 0.9144, 60, 50.0
rx = gx ** (1.0 / (nx - 1))
dx1 = Lx * (rx - 1) / (rx ** nx - 1)
dxn = dx1 * rx ** (nx - 1)
print(f"wall_grading={G:g}")
print(f"first_cell_height_m={h1:.6e}")
print(f"wall_ar_leading_edge={dx1 / h1:.4g}")
print(f"wall_ar_max={dxn / h1:.4g}")
EOF
(
    cd "$dest/simulation"
    find constant system 0 -type f -print0 | sort -z | xargs -0 sha256sum > ../PREPARED.sha256
)
{
    echo "prepared_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "bashrc=$bashrc"
    echo "steps=blockMesh (mesh/) ; copy polyMesh ; cp -r 0.org 0 (official setupMesh/runSim pre-solver steps; mesh copied, not symlinked)"
    cat "$dest/GRID_METRICS"
    grep -E '^ *cells:' "$dest/simulation/log.checkMesh" | sed 's/^ */checkMesh_/; s/: */=/'
    grep -E 'Max aspect ratio' "$dest/simulation/log.checkMesh" | sed 's/^ */checkMesh_/'
    grep -E 'Mesh OK|Failed' "$dest/simulation/log.checkMesh" | sed 's/^ */checkMesh_result=/'
} > "$dest/PREPARED"
cat "$dest/PREPARED"
