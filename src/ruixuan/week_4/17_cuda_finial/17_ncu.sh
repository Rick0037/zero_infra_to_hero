#!/bin/bash
# 用 ncu 逐个 kernel 采集 17_cuda_finial 的性能报告, 生成 4 个 .ncu-rep 给 GUI 打开
# 用法: ./ncu.sh            (默认二进制 ./output/17_cuda_finial)
#       SUDO= ./ncu.sh      (管理员已放开性能计数器权限时, 免 sudo)
set -e

BIN=${BIN:-./output/17_cuda_finial}
OUTDIR=ncu_reports
mkdir -p "$OUTDIR"

# 预热次数, 与 17_cuda.cuh 里的 WARMUP 保持一致
WARMUP=10
NCU_BIN="/usr/local/cuda/bin/ncu"
SUDO=${SUDO:-sudo}

# kernel 名 -> 报告文件名
declare -A KERNELS=(
    [softmax_baseline]="SoftmaxBaseline"
    [softmax_online]="SoftmaxOnline"
    [quantize]="QuantizePerTokenSymmetric"
    [gemv]="GemvHalf"
)

for name in softmax_baseline softmax_online quantize gemv; do
    k=${KERNELS[$name]}
    echo "=== profiling $k -> $OUTDIR/${name}.ncu-rep ==="
    # skip 计数只针对匹配 -k 的 launch, 所以跳过该 kernel 自己的预热
    $SUDO $NCU_BIN --set full \
        -k "$k" \
        --launch-skip "$WARMUP" --launch-count 1 \
        -o "$OUTDIR/$name" --force-overwrite \
        "$BIN"
done

echo
echo "全部完成, 报告在 $OUTDIR/ 下:"
ls -lh "$OUTDIR"
echo "用 Nsight Compute GUI 打开任意 .ncu-rep (GUI 版本需 >= 本机 ncu 版本)"
