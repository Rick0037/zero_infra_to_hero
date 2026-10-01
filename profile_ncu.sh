#!/bin/bash
# 对输出目录里的每个可执行文件跑 ncu 全量 profile, 每个生成 <名字>.ncu-rep
# 用法: ./profile_ncu.sh [可执行文件目录, 默认项目根目录] [报告输出目录, 默认项目根目录/ncu_reports]

set -u

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="${1:-$ROOT_DIR}"
REPORT_DIR="${2:-$ROOT_DIR/ncu_reports}"
NCU_BIN="/usr/local/cuda/bin/ncu"
SET="full"          # full 最全但采集 pass 多耗时久; 想快改成 basic
mkdir -p "$REPORT_DIR"

# 先验证一次 sudo, 避免循环里反复输密码
sudo -v || exit 1

fail=0
for bin in "$BIN_DIR"/*; do
    [ -f "$bin" ] && [ -x "$bin" ] || continue
    name=$(basename "$bin")
    echo "[ncu] $bin -> $REPORT_DIR/$name.ncu-rep"
    if ! sudo "$NCU_BIN" -o "$REPORT_DIR/$name" --set "$SET" --force-overwrite "$bin"; then
        echo "[FAIL ] $bin"
        fail=$((fail + 1))
    fi
done

echo "done. failed: $fail, reports in $REPORT_DIR"
exit "$fail"
