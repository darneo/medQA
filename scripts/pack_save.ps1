# Упаковать проект для переноса на другой компьютер в medbot_save.zip
# (код, датасет, адаптер, последний чекпоинт обучения). Запускать из любой папки.
#   .\scripts\pack_save.ps1              # без базовой модели
#   .\scripts\pack_save.ps1 -WithModel   # + кэш базовой модели Qwen3-4B (~3.4 ГБ) в model_cache.zip
param([switch]$WithModel)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$items = @("medqa", "scripts", "tests", "README.md", "requirements.txt", "requirements-gpu.txt",
           "data\final\qa_en_ru_kk.parquet", "data\new_questions.json")
if (Test-Path "models\medbot_qwen3_4b\lora_adapter") { $items += "models\medbot_qwen3_4b"; Write-Host "Добавлен адаптер" }
$ckptRoot = "models\medbot_train\checkpoints"
if (Test-Path $ckptRoot) {
    $last = Get-ChildItem $ckptRoot -Directory | Sort-Object { [int]($_.Name -replace 'checkpoint-', '') } | Select-Object -Last 1
    if ($last) { $items += $last.FullName; Write-Host "Добавлен чекпоинт: $($last.Name)" }
}
if (Test-Path medbot_save.zip) { Remove-Item medbot_save.zip }
Compress-Archive -Path $items -DestinationPath medbot_save.zip -CompressionLevel Optimal
if ($WithModel) {
    $cache = "$env:USERPROFILE\.cache\huggingface\hub\models--unsloth--Qwen3-4B-unsloth-bnb-4bit"
    if (Test-Path $cache) {
        Compress-Archive -Path $cache -DestinationPath model_cache.zip -CompressionLevel Optimal
        Write-Host "model_cache.zip готов (~3.4 ГБ)"
    } else { Write-Host "Кэш базовой модели не найден: $cache" }
}
$size = [math]::Round((Get-Item medbot_save.zip).Length / 1MB, 1)
Write-Host "Готово: medbot_save.zip ($size МБ). Скопируйте на флешку или в облако."
