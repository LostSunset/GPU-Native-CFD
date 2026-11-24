#!/usr/bin/env bash
# 建置 GPU-Native-CFD：
#   1. CMake：libgnc、gncGeomL0、gncOpsL1、gncCompare（不連結 OpenFOAM，N-004）
#   2. wmake：tools/ref 的參考工具（gncGeomRef、gncOpsRef；連結 OpenFOAM，
#      gncOpsRef 另連結 HiSA；只用於驗證）
# 只編譯與連結，不執行 GPU 程式。建置資訊寫到 $BUILD_ROOT/BUILD_INFO：
# 原始碼雜湊、工具鏈版本、編譯選項與各執行檔的 SHA-256。改了原始碼（含
# 註解）就要重建，再收結果。
#
# 用法：build.sh --host-env FILE [--fmad false|true]

set -u

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"

host_env=""; fmad="false"
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --fmad) fmad="$2"; shift 2 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
case "$fmad" in false) fmad_cmake=OFF ;; true) fmad_cmake=ON ;; *) echo "ERROR: --fmad false|true" >&2; exit 2 ;; esac
[ -x "$NVCC" ] || { echo "ERROR: nvcc not found at $NVCC" >&2; exit 2; }

mkdir -p "$BUILD_ROOT"
cmake_dir="$BUILD_ROOT/cmake"
ref_dir="$BUILD_ROOT/ref"
bin_dir="$BUILD_ROOT/bin"
mkdir -p "$cmake_dir" "$ref_dir" "$bin_dir"

# 在 Linux 上算：Git Bash 的 sha256sum 在檔名前加 '*'（二進位模式），逐檔
# 雜湊相同但總雜湊不同
source_sha256() {
    (cd "$root" && find CMakeLists.txt src tests tools apps -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | awk '{print $1}')
}

# 1. CMake
cmake -S "$root" -B "$cmake_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DGNC_FMAD="$fmad_cmake" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > "$BUILD_ROOT/log.cmake" 2>&1
rc_cmake=$?
rc_build=1
if [ "$rc_cmake" -eq 0 ]; then
    cmake --build "$cmake_dir" -j "${BUILD_JOBS:-8}" > "$BUILD_ROOT/log.build" 2>&1
    rc_build=$?
fi
if [ "$rc_build" -eq 0 ]; then
    for exe in gncGeomL0 gncOpsL1 gncInitL2 gncResL2 gncOuterL2 gncRunL3 gncWallDistL1 gncTurbL2 gncPowCheck gncSumBench gncFieldDiff gncHisa gncRhoSimple gncCompare; do
        cp "$cmake_dir/$exe" "$bin_dir/$exe.new" && mv "$bin_dir/$exe.new" "$bin_dir/$exe"
    done
fi

# 2. wmake（原始碼先複製到建置目錄：wmake 把目的檔寫在原始碼旁）
rc_ref=1
(
    set +u
    # shellcheck disable=SC1090
    . "$OPENFOAM_BASHRC" >/dev/null 2>&1 || true
    set -u
    command -v wmake >/dev/null || { echo "ERROR: wmake not found after sourcing $OPENFOAM_BASHRC" >&2; exit 3; }
    export GNC_SRC="$root/src"
    export GNC_REF_BIN="$bin_dir"
    export GNC_REF_LIB="$BUILD_ROOT/lib"
    mkdir -p "$GNC_REF_LIB"
    # gncOpsRef 連結 HiSA（faceLeastSquares、compact）；HiSA 不在 OpenFOAM 樹內時由主機設定給 HISA_DIR
    export HISA_DIR="${HISA_DIR:-$WM_PROJECT_DIR/src/hisa}"
    [ -f "$HISA_DIR/buildDefs.include" ] || { echo "ERROR: $HISA_DIR/buildDefs.include missing" >&2; exit 3; }
    for tool in "$root"/tools/ref/*/; do
        name="$(basename "$tool")"
        rm -rf "$ref_dir/$name"
        cp -a "$tool" "$ref_dir/$name"
        target=""
        grep -qE '^LIB *=' "$ref_dir/$name/Make/files" && target=libso
        (cd "$ref_dir/$name" && wmake $target . > log.wmake 2>&1) || { grep -m 30 -E 'error' "$ref_dir/$name/log.wmake" >&2; exit 4; }
    done
    {
        echo "openfoam_bashrc=$OPENFOAM_BASHRC"
        echo "openfoam_project=$WM_PROJECT_DIR"
        echo "wm_options=$WM_OPTIONS"
        echo "wmake_cxx=$(wmake -show-cxx 2>/dev/null)"
        echo "wmake_cxxflags=$(wmake -show-cxxflags 2>/dev/null)"
    } > "$BUILD_ROOT/BUILD_INFO.ref.env"
)
rc_ref=$?

{
    echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname) profile=$HOST_PROFILE"
    echo "source_dir=$root"
    [ -f "$root/SYNC_INFO" ] && sed 's/^/sync_/' "$root/SYNC_INFO"
    echo "source_sha256=$(source_sha256)"
    echo "cmake=$(cmake --version | head -1)"
    echo "nvcc=$("$NVCC" --version | tail -1)"
    echo "gxx=$(g++ --version | head -1)"
    echo "cuda_archs=$CUDA_ARCHS"
    echo "gnc_fmad=$fmad_cmake"
    echo "cmake_exit=$rc_cmake build_exit=$rc_build ref_exit=$rc_ref"
    echo "compiler_warnings=$(cat "$BUILD_ROOT/log.build" "$ref_dir"/*/log.wmake 2>/dev/null | grep -E 'warning( #[0-9]+-D)?:' | grep -vc 'Clock skew')"
    [ -f "$BUILD_ROOT/BUILD_INFO.ref.env" ] && sed 's/^/ref_/' "$BUILD_ROOT/BUILD_INFO.ref.env"
    if [ -f "$cmake_dir/compile_commands.json" ]; then
        # geometry.cu 的實際 nvcc 指令（驗證 --fmad 設定）
        echo "geometry_cu_command=$(python3 -c 'import json,sys
for e in json.load(open(sys.argv[1])):
    if e["file"].endswith("geometry.cu"):
        print(e.get("command") or " ".join(e["arguments"]))' "$cmake_dir/compile_commands.json")"
    fi
    for f in "$bin_dir"/* "$BUILD_ROOT"/lib/*; do
        [ -f "$f" ] && echo "sha256 $(basename "$f")=$(sha256sum "$f" | awk '{print $1}')"
    done
} | tee "$BUILD_ROOT/BUILD_INFO"

if [ "$rc_build" -ne 0 ]; then
    grep -m 40 -E 'error|Error' "$BUILD_ROOT/log.cmake" "$BUILD_ROOT/log.build" >&2
    exit 1
fi
[ "$rc_ref" -eq 0 ] || exit 1
