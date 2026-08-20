# One-shot recovery: relaunch pretraining DETACHED from the newest checkpoint.
# Uses the NO-COMPILE config on purpose: torch.compile autotunes on the GPU at
# startup and will hang for tens of minutes if the GPU is busy (e.g. a game).
# A recovery path must be robust, not fast.
# Survives terminal/session teardown (the failure that idled the GPU for 9h).
#   powershell -File scripts\resume_training.ps1
Set-Location $PSScriptRoot\..
$py = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like '*src.train.train*' }
if ($running) { Write-Output ("already training, PID " + $running.ProcessId); exit 0 }
$ck = Get-ChildItem checkpoints\1.1_v3_step_*.pt |
      Sort-Object { [int]($_.BaseName -replace '.*_','') } | Select-Object -Last 1
if (-not $ck) { Write-Output "no checkpoint found"; exit 1 }
Write-Output ("resuming from " + $ck.Name)
$a = @("-u","-m","src.train.train","--config","configs/pretrain_1.1_v3_anneal_nocompile.yaml",
       "--resume",("checkpoints/"+$ck.Name),"--log-path","logs/pretrain_v3.jsonl")
$p = Start-Process -FilePath $py -ArgumentList $a -RedirectStandardOutput "logs\pretrain_v4.log" `
     -RedirectStandardError "logs\pretrain_v4.err" -PassThru -WindowStyle Hidden
Write-Output ("relaunched detached, PID " + $p.Id)
