$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Python = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
$Out = Join-Path $Root "logs\rescue_train.out"
$Err = Join-Path $Root "logs\rescue_train.err"

New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null

& $Python -u -m src.train.train `
  --config configs\finetune_1.1_rescue.yaml `
  --finetune checkpoints\pretrain_1.1_final.pt `
  1>> $Out `
  2>> $Err
