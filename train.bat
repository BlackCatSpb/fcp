@echo off
echo === MemBind Training (extra args: %*) ===
echo.
echo Using --train_chunks 20000 (override with --train_chunks N)
python train_phase2.py --arch membind --train_chunks 20000 %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Training failed with error code %ERRORLEVEL%
    pause
)
