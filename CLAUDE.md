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
- Репозиторий публичный: нужен для скачивания пакета HiveOS по raw-ссылке

## Бэклог
1. Эталон pearlhash на Python по pearl-research-labs/pearl (noise_gen_reference, blake3, jackpot)
2. CUDA: генерация шума, NoisyGEMM на mma.sync + ldmatrix/cp.async, jackpot, проверка таргета
3. Stratum-клиент пула, доказательства через zk-pow (Rust)
4. Тюнинг под каждую архитектуру (Nsight Compute)
