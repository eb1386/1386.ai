# v4 post-training eval, armed before the run finishes.
# waits for the final checkpoint, then scores v4 against v3:
#   1. multiple-choice suite on gpu (template-free, fair to both)
#   2. 309-prompt battery on cpu, each model under ITS OWN trained template
# detached-safe: survives session teardown.

Set-Location $PSScriptRoot\..
$py = "C:\Users\Evan Borodow\AppData\Local\Programs\Python\Python310\python.exe"
$final = "checkpoints\finetune_1.1_v4_final.pt"

function Log($m) {
  $l = ("[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $m)
  Write-Output $l; Add-Content -Path "logs\v4_eval.log" -Value $l
}

$waited = 0
while (-not (Test-Path $final)) {
  if ($waited -ge 43200) { Log "ERROR: timed out waiting for $final"; exit 1 }
  Start-Sleep -Seconds 120
  $waited += 120
}
Start-Sleep -Seconds 60

# gaming guard: give the user 5 minutes to launch a game, then stay off the
# gpu until the session ends (checked every 2 minutes)
Start-Sleep -Seconds 300
while (Get-Process -Name "Spider-Man2" -ErrorAction SilentlyContinue) {
  Start-Sleep -Seconds 120
}
Log "v4 final checkpoint present, gpu clear - starting eval"

# 1. mc suite on gpu: loglikelihood scoring, no chat template involved.
# score mid checkpoints too - val loss went flat, so the last checkpoint is
# not automatically the best (the v3 lesson)
$models = @("--model", "v4_final:checkpoints/finetune_1.1_v4_final.pt:configs/finetune_1.1_v4.yaml")
foreach ($s in 6000, 8000) {
  if (Test-Path "checkpoints\1.1_v4_ft_step_$s.pt") {
    $models += @("--model", "v4_${s}:checkpoints/1.1_v4_ft_step_$s.pt:configs/finetune_1.1_v4.yaml")
  }
}
$models += @("--model", "v3_final:checkpoints/finetune_1.1_v3_final.pt:configs/finetune_1.1_v3.yaml")
& $py -u scripts/eval_standard.py --device cuda --limit 300 @models `
  --out logs/v4_eval_standard.json *>> logs\v4_eval_standard.log
Log ("mc suite done, exit " + $LASTEXITCODE)

# 2. battery on cpu, two shards, v4 under its trained template
$common = @("-u","scripts/audit/run_battery.py","--subset","full","--conditions","fixed",
            "--template","v4","--checkpoint","checkpoints/finetune_1.1_v4_final.pt",
            "--config","configs/finetune_1.1_v4.yaml","--threads","3")
$p0 = Start-Process -FilePath $py -ArgumentList ($common + @("--shard","0/2","--out","logs/audit/v4_battery.jsonl")) `
      -RedirectStandardOutput "logs\v4_battery_s0.log" -RedirectStandardError "logs\v4_battery_s0.err" -PassThru -WindowStyle Hidden
$p1 = Start-Process -FilePath $py -ArgumentList ($common + @("--shard","1/2","--out","logs/audit/v4_battery_s1.jsonl")) `
      -RedirectStandardOutput "logs\v4_battery_s1.log" -RedirectStandardError "logs\v4_battery_s1.err" -PassThru -WindowStyle Hidden
Log ("battery shards started, PIDs " + $p0.Id + " " + $p1.Id)

# 3. battery is cpu-only, so the gpu is free: bring chat back NOW
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match "serve_cpu|run\.py" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
$srv = Start-Process -FilePath $py -ArgumentList @("-u","run.py") `
       -RedirectStandardOutput "logs\serve_gpu.log" -RedirectStandardError "logs\serve_gpu.err" `
       -PassThru -WindowStyle Hidden
Log ("chat restarted on gpu, PID " + $srv.Id)

Wait-Process -Id $p0.Id, $p1.Id -Timeout 10800 -ErrorAction SilentlyContinue
Log "battery done"
Log "V4 EVAL COMPLETE: logs/v4_eval_standard.json + logs/audit/v4_battery*.jsonl"
