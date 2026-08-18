# Final evaluation for Plasma 1.1 v3.
#
# Waits for SFT to land, then scores every surviving SFT checkpoint against
# both baselines on the same two suites. The point of scoring 10k/12k/14k/final
# rather than just the last one is that val loss already stopped improving at
# 13k, so the last checkpoint is not automatically the best.
#
#   powershell -File scripts\run_final_eval.ps1
# Runs detached, so it survives a terminal teardown.

Set-Location $PSScriptRoot\..
$py = "C:\Users\Evan Borodow\AppData\Local\Programs\Python\Python310\python.exe"
$final = "checkpoints\finetune_1.1_v3_final.pt"

function Log($m) {
  $l = ("[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $m)
  Write-Output $l
  Add-Content -Path "logs\final_eval.log" -Value $l
}

# wait for the run to finish writing its final checkpoint
$waited = 0
while (-not (Test-Path $final)) {
  if ($waited -ge 10800) { Log "ERROR: timed out waiting for $final"; exit 1 }
  Start-Sleep -Seconds 60
  $waited += 60
}
# let the last write settle before anything reads it
Start-Sleep -Seconds 45
Log "final checkpoint present, starting evaluation"

# candidates + baselines, all scored on the same suites
$v3cfg = "configs/finetune_1.1_v3.yaml"
$models = @()
foreach ($s in 10000, 12000, 14000) {
  $p = "checkpoints\1.1_v3_ft_step_$s.pt"
  if (Test-Path $p) { $models += "v3_ft_${s}:checkpoints/1.1_v3_ft_step_$s.pt:$v3cfg" }
  else { Log "note: $p rotated away, skipping" }
}
$models += "v3_ft_final:checkpoints/finetune_1.1_v3_final.pt:$v3cfg"
$models += "plasma_1.0:checkpoints/finetune_1.0_final.pt:configs/finetune_1.0.yaml"
$models += "old_1.1:checkpoints/finetune_1.1_final.pt:configs/finetune_1.1.yaml"
Log ("scoring " + $models.Count + " models")

# 1. multiple-choice suite (HellaSwag / ARC / PIQA / OpenBookQA / BoolQ)
$a = @("-u", "scripts/eval_standard.py", "--device", "cuda", "--limit", "300",
       "--out", "logs/final_eval_standard.json")
foreach ($m in $models) { $a += @("--model", $m) }
Log "standard benchmarks starting"
& $py @a *>> logs\final_eval_standard.log
Log ("standard benchmarks finished, exit " + $LASTEXITCODE)

# 2. 111-item decontaminated generation suite
$b = @("-u", "scripts/benchmark_v2.py", "--device", "cuda",
       "--out", "logs/final_benchmark_v2.json")
foreach ($m in $models) { $b += @("--add", $m) }
Log "generation suite starting"
& $py @b *>> logs\final_benchmark_v2.log
Log ("generation suite finished, exit " + $LASTEXITCODE)

Log "DONE. results: logs\final_eval_standard.json, logs\final_benchmark_v2.json"
