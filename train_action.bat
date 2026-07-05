@echo off
echo === MemBind Training: ACTION genre (factorized) ===
echo D=896 L=24 H=4 r=16, seq_len=128, K=24, factorized+AMP
echo.
cd /d "C:\Users\black\OneDrive\Desktop\fcp"
echo Cleaning Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
echo Starting training...
python train_large.py --genre ACTION --seq_len 128 --batch_size 8 --accum_steps 4 --dct --slide --log_every 50 --ckpt_every 500 --eval_every 250 --patience 5 --factorized
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Training failed with error code %ERRORLEVEL%
    pause
)
pause
