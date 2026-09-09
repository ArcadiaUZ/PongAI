@echo off
chcp 65001 >nul
echo ============================================
echo    Pong AI - O'qitish (GPU)
echo ============================================
echo.
echo GPU tekshirilmoqda...
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU')"
echo.
echo 10000 epizod GPU da o'qitiladi (~5-7 daqiqa)
echo.
echo Natijalarni kuzating:
echo   - Mukofot oshib borishi kerak
echo   - Epsilon kamayib borishi kerak
echo.
echo To'xtatish uchun: Ctrl+C
echo.
python train_ai.py
echo.
echo O'qitish tugadi!
pause