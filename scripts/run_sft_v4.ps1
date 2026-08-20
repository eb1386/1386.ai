# plasma 1.1 sft v4 -- one command, detached, resume-safe.
# does NOT touch the v3 checkpoints (different prefix + final name).
#
#   powershell -File scripts\run_sft_v4.ps1
#
# ~4.5h on the 5080. produces checkpoints/finetune_1.1_v4_final.pt.
# the v3 final stays untouched as the baseline.

Set-Location $PSScriptRoot\..
$py = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"

function Log($m) {
  $l = ("[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $m)
  Write-Output $l; Add-Content -Path "logs\sft_v4.log" -Value $l
}

# preflight: data + base checkpoint must exist
if (-not (Test-Path "data\sft_shards_1.1_v4\meta.yaml")) {
  Log "shards missing - building (cpu, ~15 min)"
  & $py -u scripts\build_sft_v4.py
  if (-not (Test-Path "data\sft_shards_1.1_v4\meta.yaml")) {
    Log "ERROR: shard build failed"; exit 1
  }
}
if (-not (Test-Path "checkpoints\pretrain_1.1_v3_final.pt")) {
  Log "ERROR: base checkpoint missing"; exit 1
}

# set max_steps to ~3 epochs of the actual built set
$meta = Get-Content "data\sft_shards_1.1_v4\meta.yaml" | Select-String "n_train_sequences: (\d+)"
if ($meta) {
  $n = [int]$meta.Matches[0].Groups[1].Value
  $steps = [math]::Round($n / 32 * 3 / 100) * 100
  (Get-Content "configs\finetune_1.1_v4.yaml") -replace "max_steps: \d+", "max_steps: $steps" |
    Set-Content "configs\finetune_1.1_v4.yaml"
  Log ("n_train=$n -> max_steps=$steps (3 epochs)")
}

# resume from the newest v4 checkpoint if one exists
$resume = @()
$ck = Get-ChildItem "checkpoints\1.1_v4_ft_step_*.pt" -ErrorAction SilentlyContinue |
      Sort-Object { [int]($_.BaseName -replace '.*_','') } | Select-Object -Last 1
if ($ck) {
  $resume = @("--resume", "checkpoints/$($ck.Name)")
  Log ("resuming from " + $ck.Name)
}

$a = @("-u","-m","src.train.train","--config","configs/finetune_1.1_v4.yaml",
       "--finetune","checkpoints/pretrain_1.1_v3_final.pt",
       "--log-path","logs/finetune_v4.jsonl") + $resume
$p = Start-Process -FilePath $py -ArgumentList $a `
     -RedirectStandardOutput "logs\finetune_v4.log" `
     -RedirectStandardError "logs\finetune_v4.err" -PassThru -WindowStyle Hidden
Log ("SFT v4 STARTED, PID " + $p.Id + " - watch: Get-Content logs\finetune_v4.log -Wait -Tail 5")
