# pminer — майнер Pearl (PRL, pearlhash) для NVIDIA RTX

## Проект
Свой GPU-майнер Pearl под RTX 5060 (sm_120), 4060 (sm_89), 3060 (sm_86), 2060 Super (sm_75).
Запуск через HiveOS «Custom miner».

## Текущий этап (v0.1)
Только бенчмарк INT8 tensor cores (mma.sync s8) на всех GPU; хэшрейт = MAC/s,
что соответствует «TH/s» pearlhash (попытка 8×16×4096 = 524 288 MAC). Реального майнинга нет.

## Структура
- `src/pminer.cu` — бинарник: поток на GPU, ядро IMMA, stats.json для HiveOS
- `hiveos/` — h-manifest.conf, h-config.sh, h-run.sh, h-stats.sh
- `build.sh` — сборка бинарника и пакета `dist/pminer-<ver>.tar.gz`
- `dist/` — готовый пакет для HiveOS (коммитится, ссылка raw)
- `ref/pearlhash_ref.py` — эталон pearlhash на numpy (dense, сертификат V3 salted)
- `ref/test_ref.py` + `ref/vectors.json` — сверка эталона с векторами из Rust-кода Pearl
- `ref/gen_vectors.patch` — патч к pearl-research-labs/pearl@0dbdb2f (zk-pow/src/ffi/mine.rs),
  генерация векторов: `cargo test --release --lib pminer_dump_vectors` → /tmp/pearl_vectors.json

## Сборка
CUDA 12.8+ на Ubuntu 20.04 (glibc 2.31 — совместимость с HiveOS):
`NVCC=/usr/local/cuda-12.8/bin/nvcc ./build.sh`

## HiveOS
Flight Sheet → Miner: Custom → Setup:
- Installation URL: `https://github.com/dzhelybaev/pminer/raw/version/vX.Y/dist/pminer-X.Y.tar.gz`
- Hash algorithm: любой; Wallet/Pool: любые (пока не используются)
Лог: `/var/log/miner/pminer/pminer.log`, статистика: `/run/pminer/stats.json`.
Драйвер NVIDIA ≥ 570 (нужен для RTX 50xx и CUDA 12.8).

## Правила
- Каждое изменение — отдельный коммит с подробным описанием на русском (что и почему)
- Секретов в репозитории нет; файлы >50 МБ не коммитить
- Версия: VERSION + ветка `version/vX.Y` на коммите версии, без тегов
- Исправление сборки/запуска/предупреждений — без новой версии, ветку двигать (--force-with-lease)

## Известные проблемы
- HiveOS, похоже, обрезает per-GPU хэшрейт в hs_units "hs" (показывал 1000.0) — отдаём в "mhs"
- pminer v0.1 не подключается к пулу: любые сообщения пула не от него
- Репозиторий публичный: нужен для скачивания пакета HiveOS по raw-ссылке

## Алгоритм pearlhash (проверено эталоном)
- job_key = blake3(header 76 Б | config 52 Б); config задаёт k, r, rows/cols pattern (тайл h×w)
- A (m×k), B^T (n×k) — int8 в [-64,64], майнер выбирает их сам (случайные)
- V3: сиды шума из keyed-blake3 хэшей A и B^T, «посоленных» размерами m, n
- шум низкого ранга r=128: значения [-63,63], A+шум и B+шум помещаются в int8 → обычный IMMA
- на каждые r элементов k: XOR всех int32-аккумуляторов тайла, jackpot[t%16]=rotl13(..)^xor
- rotl распределяется над XOR → каждый поток копит свой jackpot[16], в конце XOR-редукция по warp
- удача: blake3_keyed(a_seed, jackpot) как LE-число <= target(nbits)·h·w·k
- значит хэшрейт = MAC/s, а blake3 почти ничего не стоит — всё решает скорость GEMM

## Бэклог
1. ~~Эталон pearlhash на Python~~ — готов, совпадает с Rust-кодом Pearl
2. CUDA: генерация шума, NoisyGEMM на mma.sync + ldmatrix/cp.async, jackpot, проверка таргета
3. Stratum-клиент пула, доказательства через zk-pow (Rust)
4. Тюнинг под каждую архитектуру (Nsight Compute)
