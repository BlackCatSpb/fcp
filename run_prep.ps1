cd C:\Users\black\OneDrive\Desktop\fcp
Write-Host "=== CORPUS PREPARATION (two-pass) ===" -ForegroundColor Cyan
Write-Host "Pass 1: count tokens | Pass 2: fill exact array" -ForegroundColor Cyan
Write-Host "No temp files, single final save." -ForegroundColor Cyan
Write-Host ""
python -u prepare_corpus.py 2>&1
Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Read-Host "Press Enter to close"
