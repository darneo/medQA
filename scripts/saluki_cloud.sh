#!/usr/bin/env bash
# Saluki 27B на облачной Ubuntu-машине с NVIDIA GPU (лучше 24+ ГБ). Запускать из корня проекта:
#   bash scripts/saluki_cloud.sh                       # 50 строк, ru, кратко
#   LIMIT=200 LANGS="ru kk" bash scripts/saluki_cloud.sh
#   RAG=1 bash scripts/saluki_cloud.sh                 # с подсказками из датасета
#   SETUP_ONLY=1 bash scripts/saluki_cloud.sh          # только подготовка
#   CUDA_VER=13.5 bash scripts/saluki_cloud.sh         # если драйвер поддерживает CUDA 13.5
# Сам скачивает llama.cpp (CUDA) и модель, ставит зависимости и запускает прогон. Аренда GPU платная.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RELEASE="b12212"
CUDA_VER="${CUDA_VER:-12.8}"
LIMIT="${LIMIT:-50}"
LANGS="${LANGS:-ru}"
MAX_TOKENS="${MAX_TOKENS:-384}"
CTX="${CTX:-4096}"
RAG="${RAG:-0}"

command -v nvidia-smi >/dev/null || { echo "Ошибка: нет nvidia-smi — нужна машина с NVIDIA GPU и драйвером CUDA."; exit 1; }
echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
GPU_ARGS=()
if [ "${VRAM_MB:-0}" -ge 22000 ]; then
    GPU_ARGS=(--ngl 99)          # модель целиком в видеопамяти
else
    echo "ВНИМАНИЕ: ${VRAM_MB} МБ VRAM — часть слоёв уйдёт в RAM (медленнее). Лучше 24+ ГБ."
fi

mkdir -p tools/llama_cpp logs
if ! find tools/llama_cpp -maxdepth 3 -type f -name llama-server | grep -q .; then
    echo "== llama.cpp ${RELEASE}, CUDA ${CUDA_VER} =="
    base="https://github.com/ggml-org/llama.cpp/releases/download/${RELEASE}"
    curl -L --fail --retry 3 -o /tmp/llama.tar.gz "${base}/llama-${RELEASE}-bin-ubuntu-cuda-${CUDA_VER}-x64.tar.gz"
    curl -L --fail --retry 3 -o /tmp/cudart.tar.gz "${base}/cudart-llama-${RELEASE}-bin-ubuntu-cuda-${CUDA_VER}-x64.tar.gz"
    tar -xzf /tmp/llama.tar.gz -C tools/llama_cpp
    tar -xzf /tmp/cudart.tar.gz -C tools/llama_cpp
    find tools/llama_cpp -maxdepth 3 -type f -name llama-server -exec chmod +x {} \;
fi
export LD_LIBRARY_PATH="$(pwd)/tools/llama_cpp:${LD_LIBRARY_PATH:-}"

if [ ! -x .venv/bin/python ]; then
    echo "== Python-окружение =="
    python3 -m venv .venv || { echo "Не удалось создать venv: установите python3-venv."; exit 1; }
    .venv/bin/python -m pip install -U pip
    .venv/bin/python -m pip install -r requirements.txt
fi

if [ ! -f models/saluki/Underdog-Saluki-27B-1.0-IQ2-mix.gguf ]; then
    .venv/bin/python -m medqa saluki download
fi

RUN=(.venv/bin/python -u -m medqa saluki run --limit "$LIMIT" --langs $LANGS --max-tokens "$MAX_TOKENS"
     --ctx "$CTX" "${GPU_ARGS[@]}")
[ "$RAG" = "1" ] && RUN+=(--rag)
if [ "${SETUP_ONLY:-0}" = "1" ]; then
    echo "Подготовка завершена. Команда запуска:"; echo "${RUN[*]}"; exit 0
fi
echo "== Прогон: ${LIMIT} строк, языки: ${LANGS}, max_tokens: ${MAX_TOKENS}, RAG: ${RAG} =="
exec "${RUN[@]}"
