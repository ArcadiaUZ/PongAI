@echo off
chcp 65001 >nul
echo ============================================
echo    Pong AI - O'qitish (GPU)
echo ============================================
echo.
echo GPU tekshirilmoqda...
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU')"
echo.
echo PARALLEL o'qitish: 8 ta o'yin bir vaqtda GPU da
echo ~100000 epizod (bir necha soat)
echo.
echo Natijalarni kuzating:
echo   - Mukofot oshib borishi kerak
echo   - Epsilon kamayib borishi kerak
echo   - Yutuq foizi oshib borishi kerak
echo.
echo To'xtatish uchun: Ctrl+C
echo.
python train_ai.py
echo.
echo O'qitish tugadi!
pause