# Gaming-break auto-resume.
#   1. waits 30 minutes (GPU stays completely free)
#   2. finishes the cooldown from checkpoint 324,000 -> 326,000
#   3. hands off to SFT on the GPU
#   4. runs the base-model benchmark on the CPU in parallel
#
#   powershell -File scripts\resume_after_break.ps1
# Everything runs detached, so it survives a terminal/session teardown.

Set-Location $PSScriptRoot\..
$py = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"

function Log($m) {
  $l = ("[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $m)
  Write-Output $l; Add-Content -Path "logs\break_resume.log" -Value $l
}

Log "gaming break started - GPU free for 30 minutes"
Start-Sleep -Seconds 1800
Log "break over - resuming"

# 1. finish the cooldown (blocking) unless it already completed
if (-not (Test-Path "checkpoints\pretrain_1.1_v3_final.pt")) {
  $ck = Get-ChildItem checkpoints\1.1_v3_step_*.pt |
        Sort-Object { [int]($_.BaseName -replace '.*_','') } | Select-Object -Last 1
  Log ("finishing cooldown from " + $ck.Name)
  & $py -u -m src.train.train --config configs/pretrain_1.1_v3_anneal.yaml `
        --resume ("checkpoints/" + $ck.Name) --log-path logs/pretrain_v3.jsonl `
        *>> logs\anneal.log
  Log "cooldown finished"
} else { Log "cooldown already complete" }

if (-not (Test-Path "checkpoints\pretrain_1.1_v3_final.pt")) {
  Log "ERROR: base model missing after cooldown - stopping"; exit 1
}
Copy-Item "checkpoints\pretrain_1.1_v3_final.pt" "checkpoints\preserved_base_1.1_v3.pt" -ErrorAction SilentlyContinue

# 2. SFT on the GPU
$a = @("-u","-m","src.train.train","--config","configs/finetune_1.1_v3.yaml",
       "--finetune","checkpoints/pretrain_1.1_v3_final.pt","--log-path","logs/finetune_v3.jsonl")
$p = Start-Process -FilePath $py -ArgumentList $a -RedirectStandardOutput "logs\finetune_v3.log" `
     -RedirectStandardError "logs\finetune_v3.err" -PassThru -WindowStyle Hidden
Log ("SFT STARTED (14,200 steps), PID " + $p.Id)

# 3. base benchmark on the CPU, in parallel
Start-Sleep -Seconds 45
$b = @("-u","scripts/eval_standard.py","--device","cpu","--limit","100",
       "--model","v3_base_final:checkpoints/pretrain_1.1_v3_final.pt:configs/pretrain_1.1_v4.yaml",
       "--model","old_base_v1:checkpoints/pretrain_1.1_final.pt:configs/pretrain_1.1.yaml",
       "--out","logs/eval_base_final.json")
$q = Start-Process -FilePath $py -ArgumentList $b -RedirectStandardOutput "logs\eval_base_final.log" `
     -RedirectStandardError "logs\eval_base_final.err" -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 3
try { $q.PriorityClass = 'BelowNormal' } catch {}
Log ("base benchmark started on CPU, PID " + $q.Id)
