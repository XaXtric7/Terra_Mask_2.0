@echo off
echo ========================================
echo U-Net vs STEGO Model Comparison
echo ========================================
echo.

echo Testing the comparison framework...
cd src
python test_comparison.py

echo.
echo If tests pass, you can run the full comparison:
echo.
echo 1. Train STEGO model:
echo    python train_simple_stego.py
echo.
echo 2. Run comparison:
echo    python compare_models.py
echo.
echo 3. Or run everything at once:
echo    python run_comparison.py
echo.

pause
