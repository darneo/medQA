# Монитор прогонов harness: что идёт сейчас, прогресс, готовые результаты, видеокарта. Обновляется каждые 5 с.
#   .\scripts\watch_harness.ps1        (выход — Ctrl+C, прогоны не останавливаются)
$root = Split-Path -Parent $PSScriptRoot
$logs = @("$root\logs\harness_run.log", "$root\logs\harness_queue.log")
while ($true) {
    Clear-Host
    Write-Host ("Harness — {0:HH:mm:ss}" -f (Get-Date)) -ForegroundColor Cyan
    foreach ($log in $logs) {
        if (-not (Test-Path $log)) { continue }
        # лог открыт на запись другим процессом — читаем с FileShare.ReadWrite
        $fs = [IO.File]::Open($log, 'Open', 'Read', 'ReadWrite')
        $text = (New-Object IO.StreamReader($fs)).ReadToEnd() -split "[`r`n]+"
        $fs.Close()
        $runs = $text | Where-Object { $_ -match '^=== ' }
        $progress = $text | Where-Object { $_ -match 'requests:\s+\d+%' } | Select-Object -Last 1
        $errors = $text | Where-Object { $_ -match 'FAILED|Traceback|OutOfMemory' }
        Write-Host ""
        Write-Host (Split-Path $log -Leaf) -ForegroundColor Yellow
        $runs | ForEach-Object { Write-Host "  $_" }
        if ($progress) { Write-Host ("  сейчас: " + ($progress -replace '\s+', ' ').Trim()) -ForegroundColor Green }
        $errors | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        $done = $text | Where-Object { $_ -match '^exit|QUEUE DONE' }
        $done | ForEach-Object { Write-Host "  $_" -ForegroundColor Magenta }
    }
    Write-Host ""
    Write-Host "Готовые результаты:" -ForegroundColor Yellow
    Get-ChildItem "$root\results\harness" -Recurse -Filter "results_*.json" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | ForEach-Object {
            $d = Get-Content $_.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
            $tasks = ($d.results.PSObject.Properties.Name) -join ', '
            Write-Host ("  {0:HH:mm}  {1,-12} {2}" -f $_.LastWriteTime, $_.Directory.Parent.Name, $tasks)
        }
    Write-Host ""
    $gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    Write-Host "Видеокарта (загрузка, память): $gpu" -ForegroundColor Cyan
    Start-Sleep -Seconds 5
}
