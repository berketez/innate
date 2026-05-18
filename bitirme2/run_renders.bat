@echo off
REM ================================================================
REM run_renders.bat - 2D + 3D LES render sırayla
REM Önce 2D (15-20 dk) sonra 3D (60-90 dk)
REM Toplam: ~75-110 dk
REM ================================================================

cd /d C:\Users\berke\Desktop\nsneuron1\bitirme2
set PYTHONPATH=C:\Users\berke\Desktop\nsneuron1
set PYTHONIOENCODING=utf-8
set PY="C:\Users\berke\AppData\Local\Programs\Python\Python311\python.exe"

echo.
echo ================================================================
echo BASLIYOR: 2D + 3D LES Render Pipeline
echo Baslangic: %date% %time%
echo ================================================================
echo.

REM === ADIM 1: 2D RENDER (matplotlib, 15-20 dk) ===
echo [1/2] 2D render basliyor (les_only_30s_4k.mp4) - 30 sn 4K@60fps
echo Beklenen sure: 15-20 dakika
echo ----------------------------------------------------------------
%PY% viz_pipeline\03_render_ansys.py ^
    --les data\sim_states\les_real_60k.npz ^
    --output data\sim_states\les_only_30s_4k.mp4 ^
    --les-duration 30 ^
    --transition-duration 0 ^
    --innate-duration 0 ^
    --fps 60

if errorlevel 1 (
    echo.
    echo HATA: 2D render basarisiz! 3D render ATLANIYOR.
    pause
    exit /b 1
)

echo.
echo [1/2] 2D render TAMAM. Cikti: data\sim_states\les_only_30s_4k.mp4
echo ----------------------------------------------------------------
echo.

REM === ADIM 2: 3D RENDER (PyVista, 60-90 dk) ===
echo [2/2] 3D render basliyor (les_only_60s_4k_3D.mp4) - 60 sn 4K@60fps
echo Beklenen sure: 60-90 dakika
echo ----------------------------------------------------------------
%PY% viz_pipeline\04_render_3d.py ^
    --input data\sim_states\les_real_60k.npz ^
    --output data\sim_states\les_only_60s_4k_3D.mp4 ^
    --label "LES Referansi (Smagorinsky SGS, Re=10K, Cs=0.17)" ^
    --duration 60 ^
    --fps 60 ^
    --rotate ^
    --bitrate 40M

if errorlevel 1 (
    echo.
    echo HATA: 3D render basarisiz!
    pause
    exit /b 1
)

echo.
echo ================================================================
echo TUM RENDER'LAR TAMAMLANDI!
echo Bitis: %date% %time%
echo ================================================================
echo.
echo Ciktilar:
echo   - data\sim_states\les_only_30s_4k.mp4    (2D, ~80-150 MB)
echo   - data\sim_states\les_only_60s_4k_3D.mp4 (3D, ~600-1200 MB)
echo.
echo Mac'e indirmek icin (Mac terminalden):
echo   scp 4090:/Users/berke/Desktop/nsneuron1/bitirme2/data/sim_states/les_only_30s_4k.mp4 ~/Desktop/
echo   scp 4090:/Users/berke/Desktop/nsneuron1/bitirme2/data/sim_states/les_only_60s_4k_3D.mp4 ~/Desktop/
echo.
pause
