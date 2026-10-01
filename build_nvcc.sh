#!/bin/bash
# 用 nvcc 逐个编译项目里的 .cu, 每个源文件生成一个同名可执行文件
# 用法: ./build_nvcc.sh [输出目录, 默认项目根目录]

set -u

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"        # 项目根目录 = 脚本所在目录
OUT_DIR="${1:-$ROOT_DIR}"                        # 第一个参数指定输出目录, 默认项目根目录
mkdir -p "$OUT_DIR"
ARCH="sm_89"                               # 4090D; 换机器时改这里(Blackwell 用 sm_120)

fail=0
while IFS= read -r cu; do
    name=$(basename "$cu" .cu)
    dir=$(dirname "$cu")
    echo "[build] $cu -> $OUT_DIR/$name"
    if ! nvcc -O3 -std=c++17 -arch="$ARCH" -I"$dir" "$cu" -o "$OUT_DIR/$name"; then
        echo "[FAIL ] $cu"
        fail=$((fail + 1))
    fi
done < <(find "$ROOT_DIR/src" -name "*.cu" | sort)

echo "done. failed: $fail"
exit "$fail"
