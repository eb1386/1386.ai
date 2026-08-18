# Watches for the step-280,000 checkpoint, then swaps training onto the anneal
# corpus + LR cooldown (steps 280k -> 326k).
#
# Runs detached so it survives a terminal/session teardown:
#   powershell -File scripts\switch_to_anneal.ps1
#
# Safe to run twice: it exits if the anneal phase has already started.

Set-Location $PSScriptRoot\..
$py   = "C:\Users\Evan Borodow\AppData\Local\Programs\Python\Python310\python.exe"
$ck   = "checkpoints\1.1_v3_step_280000.pt"
$cfg  = "configs/pretrain_1.1_v3_anneal.yaml"
$log  = "logs\anneal.log"

function Write-Log($m) {
  $line = ("[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $m)
  Write-Output $line
  Add-Content -Path "logs\switch_to_anneal.log" -Value $line
}

if (Test-Path $log) { Write-Log "anneal already started ($log exists) - exiting"; exit 0 }

Write-Log "waiting for $ck ..."
$stableSize = -1
while ($true) {
  if (Test-Path $ck) {
    # only act once the file has stopped growing (fully flushed)
    $size = (Get-Item $ck).Length
    if ($size -eq $stableSize -and $size -gt 1GB) { break }
    $stableSize = $size
  }
  Start-Sleep -Seconds 60
}
Write-Log "checkpoint present and stable; stopping stable-phase training"

Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*src.train.train*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 20

$a = @("-u","-m","src.train.train","--config",$cfg,
       "--resume","checkpoints/1.1_v3_step_280000.pt","--log-path","logs/pretrain_v3.jsonl")
$p = Start-Process -FilePath $py -ArgumentList $a -RedirectStandardOutput $log `
     -RedirectStandardError "logs\anneal.err" -PassThru -WindowStyle Hidden
Write-Log ("ANNEAL + COOLDOWN STARTED (280k -> 326k), PID " + $p.Id)
