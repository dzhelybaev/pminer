#!/usr/bin/env bash
# Профиль k_gemm под Nsight Compute на риге HiveOS (запуск от root).
#   bash ncu.sh            — все три карты рига (0 = 5060, 1 = 4060, 2 = 2060S)
#   bash ncu.sh 2          — только GPU2
# Майнер останавливается на время замера (miner stop) и запускается обратно (miner start).
# Результат: /root/ncu/gpuN.txt (детали — прислать) и /root/ncu/gpuN.ncu-rep (полный отчёт).
set -u
D=/root/ncu
DEB=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/nsight-compute-2025.2.1_2025.2.1.3-1_amd64.deb
mkdir -p "$D"
NCU=$(ls "$D"/opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
if [ -z "$NCU" ]; then
  echo "скачиваю Nsight Compute (~300 МБ, распакованный ~1 ГБ)…"
  df -h "$D" | tail -1
  wget -q --show-progress -O "$D/ncu.deb" "$DEB" || { echo "не скачалось"; exit 1; }
  dpkg -x "$D/ncu.deb" "$D" && rm -f "$D/ncu.deb"
  NCU=$(ls "$D"/opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
  [ -n "$NCU" ] || { echo "ncu не найден после распаковки"; exit 1; }
fi
BIN=/hive/miners/custom/pminer/pminer
[ -x "$BIN" ] || BIN=$(find /hive/miners/custom -name pminer -type f -perm -u+x | head -1)
[ -x "$BIN" ] || { echo "pminer не найден"; exit 1; }
echo "ncu: $NCU"; echo "pminer: $BIN"

miner stop >/dev/null 2>&1; sleep 3
for g in ${@:-0 1 2}; do
  # параметры — как выбирала автонастройка (v0.19–v0.20), имя ядра — шаблон k_gemm<128,64,0,PIPE>
  case $g in
    0) ARGS="--bm 128 --m 512 --n 4096 --lanes 2 --nopipe"; K=k_gemmILi128ELi64ELi0ELb0E ;;
    1) ARGS="--bm 128 --m 512 --n 4096 --lanes 2"; K=k_gemmILi128ELi64ELi0ELb1E ;;
    *) ARGS="--bm 128 --m 8192 --n 16384 --lanes 2 --nopipe --group 2"; K=k_gemmILi128ELi64ELi0ELb0E ;;
  esac
  echo "== GPU$g: $ARGS"
  rm -f "$D/gpu$g.ncu-rep"
  # пропускаем самотест и диагностику (100 запусков), профилируем один запуск в майнинге
  CUDA_VISIBLE_DEVICES=$g timeout 900 "$NCU" --kernel-name-base mangled --kernel-name "regex:$K" \
    --launch-skip 100 --launch-count 1 --set full --import-source no --kill on \
    -o "$D/gpu$g" "$BIN" $ARGS > "$D/gpu$g.log" 2>&1
  if [ -f "$D/gpu$g.ncu-rep" ]; then
    "$NCU" -i "$D/gpu$g.ncu-rep" --page details > "$D/gpu$g.txt"
    "$NCU" -i "$D/gpu$g.ncu-rep" --page source --csv --print-source sass > "$D/gpu$g.sass.csv" 2>/dev/null
    echo "готово: $D/gpu$g.txt ($(wc -l < "$D/gpu$g.txt") строк)"
  else
    echo "нет отчёта, хвост лога:"; tail -n 15 "$D/gpu$g.log"
  fi
done
miner start >/dev/null 2>&1
echo "майнер запущен"
