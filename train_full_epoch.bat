@echo off
chcp 65001 >nul
cd /d "C:\Users\black\OneDrive\Desktop\fcp"
echo MemBind — FULL EPOCH: ACTION (100%% coverage)
echo Patience=999999 (no early stop), resume from latest checkpoint
echo.
python train_large.py --genre ACTION --seq_len 128 --dct --slide ^
    --log_every 100 --ckpt_every 1000 --eval_every 1000000 ^
    --patience 999999 --resume
echo.
pause
