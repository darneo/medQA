# Окружение для medbot на Windows с видеокартой NVIDIA (запускать из корня проекта):
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_gpu.ps1
# Ставит uv, Python 3.11, .venv, PyTorch с CUDA 12.8 и зависимости. Нужен интернет, ~10–20 минут.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$uv = "$env:USERPROFILE\.local\bin\uv.exe"
if (-not (Test-Path $uv)) {
    Write-Host "Устанавливаю uv ..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
}
$py = ".venv\Scripts\python.exe"
& $uv venv .venv --python 3.11 --python-preference only-managed
& $uv pip install --python $py -r requirements-gpu.txt
# unsloth может поставить CPU-сборку torch — ставим CUDA-сборку поверх (так было исправлено в первом обучении).
& $uv pip install --python $py "torch==2.11.0+cu128" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch --reinstall-package torchvision
& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA недоступна'; print('Готово, GPU:', torch.cuda.get_device_name(0))"
Write-Host "Окружение готово. Обучение: .\scripts\train_shard.ps1   Чат: .venv\Scripts\python.exe -m medqa medbot chat"
