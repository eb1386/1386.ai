# Waits for the cooldown to produce pretrain_1.1_v3_final.pt, then:
#   1. launches SFT on the GPU (detached)
#   2. launches the BASE-model standard benchmark on the CPU in parallel
# so neither resource sits idle during the handoff.
#
#   powershell -File scripts\handoff_to_sft.ps1
# Safe to run twice: exits if SFT already started.

Set-Location $PSScriptRoot\..
$py    = "C:\Users\Evan Borodow\AppData\Local\Programs\Python\Python310\python.exe"
$final = "checkpoints\pretrain_1.1_v3_final.pt"

function Log($m) {
  $l = ("[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $m)
  Write-Output $l; Add-Content -Path "logs\handoff.log" -Value $l
}

if (Test-Path "logs\finetune_v3.log") { Log "SFT already started - exiting"; exit 0 }

Log "waiting for $final ..."
$stable = -1
while ($true) {
  if (Test-Path $final) {
    $s = (Get-Item $final).Length
    if ($s -eq $stable -and $s -gt 1GB) { break }
    $stable = $s
  }
  Start-Sleep -Seconds 60
}
Log "base model complete and stable"

# make sure the cooldown process has fully exited before claiming the GPU
$tries = 0
while ((Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*src.train.train*' }) -and $tries -lt 20) {
  Start-Sleep -Seconds 15; $tries++
}

# preserve the base model outside any rotation
Copy-Item $final "checkpoints\preserved_base_1.1_v3.pt" -ErrorAction SilentlyContinue
Log "preserved base copy"

# 1. SFT on the GPU
$a = @("-u","-m","src.train.train","--config","configs/finetune_1.1_v3.yaml",
       "--finetune",$final,"--log-path","logs/finetune_v3.jsonl")
$p = Start-Process -FilePath $py -ArgumentList $a -RedirectStandardOutput "logs\finetune_v3.log" `
     -RedirectStandardError "logs\finetune_v3.err" -PassThru -WindowStyle Hidden
Log ("SFT STARTED (14,200 steps), PID " + $p.Id)

# 2. base-model benchmark on the CPU, in parallel (BelowNormal so it cannot starve SFT)
Start-Sleep -Seconds 45
$b = @("-u","scripts/eval_standard.py","--device","cpu","--limit","100",
       "--model",("v3_base_final:" + $final + ":configs/pretrain_1.1_v4.yaml"),
       "--model","old_base_v1:checkpoints/pretrain_1.1_final.pt:configs/pretrain_1.1.yaml",
       "--out","logs/eval_base_final.json")
$q = Start-Process -FilePath $py -ArgumentList $b -RedirectStandardOutput "logs\eval_base_final.log" `
     -RedirectStandardError "logs\eval_base_final.err" -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 3
try { $q.PriorityClass = 'BelowNormal' } catch {}
Log ("base benchmark started on CPU, PID " + $q.Id)
