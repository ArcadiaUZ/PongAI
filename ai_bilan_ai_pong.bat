@echo off
chcp 65001 >nul
echo ========================================
echo  AI vs AI - YANGI vs ESKI
echo  Ikki neyron tarmog'i vizualizatsiyasi
echo ========================================
echo.

REM Yangi AI modeli
if not exist "pong_ai_model.pt" (
    echo [XATO] Yangi AI modeli topilmadi: pong_ai_model.pt
    echo Avval 'start_training.bat' ni ishga tushiring!
    pause
    exit /b 1
)

REM Eski AI modeli
if not exist "enemy_pong_ai\pong_ai_model.pt" (
    echo [XATO] Eski AI modeli topilmadi: enemy_pong_ai\pong_ai_model.pt
    pause
    exit /b 1
)

echo [OK] Yangi AI: pong_ai_model.pt
echo [OK] Eski AI: enemy_pong_ai\pong_ai_model.pt
echo.
echo O'yin ishga tushirilmoqda...
echo Chap taraf (Yashil) = YANGI AI
echo O'ng taraf (Ko'k)   = ESKI AI
echo.
echo ESC  - Chiqish
echo SPACE - Yangi o'yin (tugaganda)
echo.

python ai_vs_ai.py

pause