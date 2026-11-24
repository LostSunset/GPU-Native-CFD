#!/usr/bin/env bash
# N4a-1 的 L1：牆距離（meshWave）對 OpenFOAM 逐位元比對；N4b：Poisson 的案例
# （不加 --meshwave）量測差異（N-022：OpenFOAM 以 GAMG 解，不逐位元）。
#
# 每個案例在 $RUN_ROOT/LABEL/NAME/case 建工作副本（system、constant、0 複製，
# constant/polyMesh 以符號連結）；--meshwave NAME 把該案例 fvSchemes 的
# wallDist/method 改為 meshWave（原本為 Poisson 的案例）。然後：
#   ref/0.gnc   hisa -postProcess -time 0，function object gncHisaDump
#               （fields ()、writeConstants false、wallDist true）：
#               wallDist::New(mesh).y() 的內部值、邊界值與 nUnset
#   gpu.gnc     gncWallDistL1（Poisson 時加 --poisson case/gncExpanded，fvSchemes
#               以 foamDictionary -expand 展開）
#   compare.txt gncCompare ref/0.gnc gpu.gnc（Poisson 的結果記為 POISSON，看 maxRel）
#
# 用法：l1_walldist_verify.sh --host-env FILE --label LABEL --case NAME=DIR [--case ...]
#                             [--meshwave NAME ...]

set -u

host_env=""; label=""
cases=(); meshwave=" "
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --case) cases+=("$2"); shift 2 ;;
        --meshwave) meshwave="$meshwave$2 "; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
case "$label" in ""|*/*|.*) echo "ERROR: --label NAME required" >&2; exit 2 ;; esac
[ "${#cases[@]}" -gt 0 ] || { echo "ERROR: at least one --case NAME=DIR" >&2; exit 2; }

bin="$BUILD_ROOT/bin"
dumplib="$BUILD_ROOT/lib/libgncHisaDump.so"
for exe in gncWallDistL1 gncCompare; do
    [ -x "$bin/$exe" ] || { echo "ERROR: $bin/$exe missing (run scripts/build.sh)" >&2; exit 2; }
done
[ -f "$dumplib" ] || { echo "ERROR: $dumplib missing" >&2; exit 2; }

out_root="$RUN_ROOT/$label"
[ -e "$out_root" ] && { echo "ERROR: $out_root exists" >&2; exit 2; }
mkdir -p "$out_root"
{
    echo "label=$label"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname) profile=$HOST_PROFILE"
} > "$out_root/RUN_INFO"
cp "$BUILD_ROOT/BUILD_INFO" "$out_root/BUILD_INFO"

foam() {
    (
        set +u
        # shellcheck disable=SC1090
        . "$OPENFOAM_BASHRC" >/dev/null 2>&1 || true
        set -u
        "$@"
    )
}

printf 'case\tnCells\tmethod\tref\tgpu\tcompare\n' > "$out_root/SUMMARY.tsv"
overall=0
for spec in "${cases[@]}"; do
    name="${spec%%=*}"; src="${spec#*=}"
    out="$out_root/$name"
    work="$out/case"
    mkdir -p "$work/constant"
    echo "== $name ($src)"
    cp -r "$src/system" "$src/0" "$work/"
    for f in "$src"/constant/*; do
        b="$(basename "$f")"
        if [ "$b" = polyMesh ]; then
            ln -s "$(realpath "$f")" "$work/constant/polyMesh"
        else
            cp -r "$f" "$work/constant/$b"
        fi
    done
    echo "source=$src" > "$out/SOURCE"
    case "$meshwave" in
        *" $name "*)
            foam foamDictionary -entry wallDist/method -set meshWave "$work/system/fvSchemes" \
                > "$out/log.variant" 2>&1 ;;
    esac
    method="$(foam foamDictionary -entry wallDist/method -value "$work/system/fvSchemes" 2>/dev/null)"

    cat > "$work/system/gncDumpDict" <<EOF
FoamFile { version 2.0; format ascii; class dictionary; object gncDumpDict; }
functions
{
    gncDump
    {
        type            gncHisaDump;
        libs            ("$dumplib");
        outputDir       "$out/ref";
        fields          ();
        writeConstants  false;
        wallDist        true;
    }
}
EOF
    (cd "$work" && foam hisa -postProcess -dict system/gncDumpDict -time 0) > "$out/log.ref" 2>&1
    rc_ref=$?
    [ -f "$out/ref/0.gnc" ] || rc_ref=9
    popt=()
    if [ "$method" = Poisson ]; then
        mkdir -p "$work/gncExpanded"
        foam foamDictionary -expand -precision 17 "$work/system/fvSchemes" \
            > "$work/gncExpanded/fvSchemes" 2> "$out/log.expand"
        popt=(--poisson "$work/gncExpanded")
    fi
    "$bin/gncWallDistL1" --case "$work" --out "$out/gpu.gnc" "${popt[@]}" > "$out/log.gpu" 2>&1
    rc_gpu=$?
    compare="-"
    if [ "$rc_ref" -eq 0 ] && [ "$rc_gpu" -eq 0 ]; then
        "$bin/gncCompare" "$out/ref/0.gnc" "$out/gpu.gnc" > "$out/compare.txt" 2>&1
        case $? in 0) compare="BITWISE" ;; 1) compare="DIFF" ;; *) compare="ERROR" ;; esac
        [ "$method" = Poisson ] && [ "$compare" = DIFF ] && compare="POISSON"
    fi
    ncells="$(sed -n 's/^mesh: nCells \([0-9]*\).*/\1/p' "$out/log.gpu")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$ncells" "$method" "$rc_ref" "$rc_gpu" "$compare" \
        >> "$out_root/SUMMARY.tsv"
    echo "   method=$method ref=$rc_ref gpu=$rc_gpu compare=$compare"
    [ "$compare" = "BITWISE" ] || [ "$compare" = "POISSON" ] || overall=1
done
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_root/RUN_INFO"
echo "overall_exit=$overall" >> "$out_root/RUN_INFO"
column -t -s "$(printf '\t')" "$out_root/SUMMARY.tsv"
exit "$overall"
