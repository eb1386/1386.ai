@echo off
setlocal
cd /d "%~dp0\.."
if not exist logs mkdir logs
echo ==================================================>> logs\rescue_train.out
echo Rescue training launch %date% %time%>> logs\rescue_train.out
echo ==================================================>> logs\rescue_train.out
"%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe" -u -m src.train.train --config configs\finetune_1.1_rescue.yaml --finetune checkpoints\pretrain_1.1_final.pt >> logs\rescue_train.out 2>> logs\rescue_train.err
echo Rescue training exited %date% %time% errorlevel=%errorlevel%>> logs\rescue_train.out
endlocal
