@echo off
echo === MemBind Training (default: fib_root spectrum) ===
echo D=896 L=24 H=4 r=16 bind_r=16
echo.
echo Using --train_chunks 20000 for 2GB VRAM
python train_phase2.py ^
    --arch membind ^
    --spectrum fib_root ^
    --spec_lo 0.8 --spec_hi 1.8 --n_modes 8 ^
    --n_layers 24 --bottleneck 896 ^
    --seq_len 128 --batch_size 4 --accum_steps 8 ^
    --train_chunks 20000 ^
    --lr 1e-3 --epochs 5 --log_every 100
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Training failed with error code %ERRORLEVEL%
    pause
)
