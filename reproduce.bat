@echo off
setlocal
REM VERITAS one-command reproduction
REM   reproduce        quick: committed data and models, no downloads
REM   reproduce full   rebuild everything from raw data

if /I "%~1"=="full" goto full
goto bench

:full
echo [1/4] Downloading NASA POWER data...
python -m src.data.power --all || goto fail
echo [1b] Sentinel-2 download, skipped automatically without CDSE credentials...
python -m src.data.sentinel2 --all
echo [2/4] Aligning data...
python -m src.data.align --all || goto fail
echo [3/4] Training the context model...
python -m src.detect.context_model --train || goto fail
echo [4/4] Tuning thresholds...
python -m src.eval.tune || goto fail

:bench
echo Running the benchmark, this takes a few minutes...
python -m src.eval.run_benchmark --sweep || goto fail
echo Comparing with the reference results...
python scripts\check_results.py || goto fail
echo Running the tests...
python -m pytest -q || goto fail
echo.
echo [OK] VERITAS reproduced successfully.
exit /b 0

:fail
echo.
echo [FAIL] Reproduction stopped at the step above.
exit /b 1