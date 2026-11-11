#!/usr/bin/env bash
# Сборка pminer и пакета HiveOS. Нужны CUDA 12.8+ (sm_120 для RTX 50xx) и Rust (cargo).
# Собирать на Ubuntu 20.04 (glibc 2.31), чтобы бинарник работал на HiveOS.
set -euo pipefail
cd "$(dirname "$0")"
VER=$(cat VERSION)
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
CARGO=${CARGO:-cargo}
PEARL_COMMIT=0dbdb2ffd8b81193a5079bb67a59aecc10ad501c
# официальный код Pearl (zk-pow, pearl-blake3) — для доказательств; в git не хранится
if [ ! -d third_party/pearl/zk-pow ]; then
  mkdir -p third_party/pearl
  git -C third_party/pearl init -q
  git -C third_party/pearl fetch -q --depth 1 https://github.com/pearl-research-labs/pearl "$PEARL_COMMIT"
  git -C third_party/pearl checkout -q FETCH_HEAD
fi
(cd pool && "$CARGO" build --release --quiet)
mkdir -p build/pminer dist
$NVCC -O3 -std=c++17 -DPMINER_VERSION="\"$VER\"" -o build/pminer/pminer src/pminer.cu \
  -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_120,code=compute_120 \
  -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc \
  pool/target/release/libpminer_pool.a -lpthread -ldl -lm -lrt
strip build/pminer/pminer
cp hiveos/*.sh hiveos/h-manifest.conf build/pminer/
sed -i "s/^CUSTOM_VERSION=.*/CUSTOM_VERSION=$VER/" build/pminer/h-manifest.conf
tar -C build -czf "dist/pminer-$VER.tar.gz" --owner=0 --group=0 pminer
echo "dist/pminer-$VER.tar.gz"
