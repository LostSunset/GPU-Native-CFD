#!/usr/bin/env bash
# Prepare one operating point of the official HiSA basicFinner example.
#
# Reproduces the official setupDirs for one row of the case's 'runs' table
# (seq mach temp press AoA lowMach; settings: inputTotal=0, i.e. static p and
# T, GAMMA=1.4, R=287.058) and the mesh part of runSim:
#   rho = p/(R T); c = sqrt(GAMMA R T); |U| = c*Mach;
#   U = (-|U| cos AoA, 0, -|U| sin AoA)
#   0.org/include/freestreamConditions: U, p, T
#   system/forces, forcesNormAxial: CofR (mrcX mrcY mrcZ), magUInf |U|,
#   rhoInf rho, lRef, Aref; fvSchemes lowMachAusm
#   constant/polyMesh: absolute symlinks to the shipped meshHexpress mesh
#   (official runSim links relatively); cp -r 0.org 0
# The official setupDirs uses Python 2; the same arithmetic is done with awk
# here. The case stays serial (official runs decompose to PROC ranks).
#
# Usage: prepare_basicFinner.sh --src DIR --dest DIR --bashrc FILE --seq N

set -eu

src=""; dest=""; bashrc=""; seq=""
while [ $# -gt 0 ]; do
    case "$1" in
        --src) src="$2"; shift 2 ;;
        --dest) dest="$2"; shift 2 ;;
        --bashrc) bashrc="$2"; shift 2 ;;
        --seq) seq="$2"; shift 2 ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "$src" ] && [ -n "$dest" ] && [ -n "$bashrc" ] && [ -n "$seq" ] || { sed -n '19p' "$0"; exit 2; }
[ -e "$dest" ] && { echo "ERROR: $dest already exists" >&2; exit 2; }
case "$(df -PT "$(dirname "$(realpath -m "$dest")")" 2>/dev/null | awk 'NR==2 {print $2}')" in
    9p|drvfs|fuseblk|ntfs*) echo "ERROR: $dest is on a Windows filesystem" >&2; exit 2 ;;
esac
( cd "$src" && sha256sum --quiet -c MANIFEST.sha256 )

row="$(awk -v s="$seq" '$1 !~ /^#/ && $1 == s' "$src/runs")"
[ -n "$row" ] || { echo "ERROR: run $seq not in $src/runs" >&2; exit 2; }
read -r _ mach temp press aoa lowmach <<< "$row"
getset() { awk -F= -v k="$1" '$1 == k {print $2}' "$src/settings"; }
[ "$(getset inputTotal)" = "0" ] || { echo "ERROR: inputTotal != 0 not supported" >&2; exit 2; }
gamma="$(getset GAMMA)"; rgas="$(getset R)"; pi="$(getset PI)"
mrcX="$(getset mrcX)"; mrcY="$(getset mrcY)"; mrcZ="$(getset mrcZ)"
lref="$(getset REFLENGTH)"; aref="$(getset REFAREA)"

read -r rho velc velcx velcz <<< "$(awk -v g="$gamma" -v r="$rgas" -v t="$temp" -v p="$press" \
    -v m="$mach" -v a="$aoa" -v pi="$pi" 'BEGIN {
        rho = p/(r*t); c = sqrt(g*r*t); v = c*m;
        printf "%.10g %.10g %.10g %.10g\n", rho, v, -v*cos(a/180*pi), -v*sin(a/180*pi) }')"

mkdir -p "$dest"
cp -a "$src/simulation" "$dest/simulation"
cp "$src/PROVENANCE" "$dest/PROVENANCE.export"
mesh="$(realpath "$src/meshHexpress/constant/polyMesh")"

cd "$dest/simulation"
sed -i -E "s/^U +.*/U            ($velcx 0.0 $velcz);/; s/^p +.*/p            $press;/; s/^T +.*/T            $temp;/" \
    0.org/include/freestreamConditions
sed -i -E "s/(CofR *).*/CofR           ($mrcX $mrcY $mrcZ);/" system/forces
sed -i -E "s/(magUInf *).*/magUInf             $velc;/; s/(rhoInf   *).*/rhoInf              $rho;/; s/(CofR *).*/CofR                ($mrcX $mrcY $mrcZ);/; s/(lRef *).*/lRef                $lref;/; s/(Aref *).*/Aref                $aref;/" \
    system/forcesNormAxial
sed -i -E "s/(lowMachAusm  *).*/lowMachAusm          $lowmach; /" system/fvSchemes
mkdir -p constant/polyMesh
for f in boundary faces neighbour owner points; do ln -s "$mesh/$f" "constant/polyMesh/$f"; done
rm -rf 0
cp -r 0.org 0

(
    set +u
    # shellcheck disable=SC1090
    . "$bashrc" >/dev/null 2>&1 || true
    set -u
    command -v checkMesh >/dev/null || { echo "ERROR: checkMesh not found" >&2; exit 3; }
    checkMesh > log.checkMesh 2>&1 || true
)

cd "$dest/simulation"
find system 0 -type f -print0 | sort -z | xargs -0 sha256sum > ../PREPARED.sha256
{
    echo "prepared_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "bashrc=$bashrc"
    echo "run_seq=$seq mach=$mach T=$temp p=$press AoA=$aoa lowMachAusm=$lowmach"
    echo "freestream rho=$rho |U|=$velc U=($velcx 0.0 $velcz)"
    echo "mesh=$mesh (absolute symlinks)"
    grep -E '^ *cells:' log.checkMesh | sed 's/^ */checkMesh_/; s/: */=/'
    grep -E 'Max aspect ratio' log.checkMesh | sed 's/^ */checkMesh_/'
    grep -E 'Mesh OK|Failed' log.checkMesh | sed 's/^ */checkMesh_result=/'
} > "$dest/PREPARED"
cat "$dest/PREPARED"
