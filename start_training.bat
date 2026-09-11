@echo off
chcp 65001 >nul
echo ============================================
echo    Pong AI - O'qitish (GPU 4x Kuchaytirilgan)
echo ============================================
echo.
echo GPU tekshirilmoqda...
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU')"
echo.
echo PARALLEL o'qitish: 32 ta o'yin bir vaqtda GPU da
echo ~500000 epizod (uzoq vaqt)
echo Hidden: 512 ^| Qatlam: 3 ^| Feature: 10
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