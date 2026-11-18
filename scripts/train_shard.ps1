# Обучение medbot на одной машине или на своей части данных (шарде):
#   .\scripts\train_shard.ps1                     # всё обучение на одной машине
#   .\scripts\train_shard.ps1 -Shard 1 -Total 3   # вторая из трёх машин
# Повторный запуск продолжает с последнего чекпоинта.
param([int]$Shard = 0, [int]$Total = 1, [string]$Output = "models\medbot_train")
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$env:NO_PROXY = ""                     # значение с "[::1]" ломало скачивание с Hugging Face
$env:HF_HUB_DISABLE_TELEMETRY = "1"
& .venv\Scripts\python.exe -m medqa medbot train --shard $Shard --shards $Total --output $Output
