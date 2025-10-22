@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title ImDisk RAM Disk Management Tool

echo ========================================
echo      ImDisk RAM Disk Management Tool
echo ========================================
echo.

REM ===============================
REM Check if running as administrator
REM ===============================
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: This script requires administrator privileges.
    echo Please right-click and select "Run as administrator".
    pause
    exit /b 1
)

REM ===============================
REM Check if ImDisk is installed
REM ===============================
where imdisk >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: ImDisk program not found.
    echo Please ensure ImDisk is installed and added to system PATH.
    pause
    exit /b 1
)

REM ===============================
REM Create 27GB RAM disk on Z:
REM ===============================
echo Creating 27GB RAM disk on Z: drive...
imdisk -a -s 27G -m Z: -p "/fs:NTFS /q /y"
if %errorlevel% neq 0 (
    echo Error: Failed to create RAM disk.
    pause
    exit /b 1
)
echo RAM disk created successfully!

REM ===============================
REM Modify Z: drive permissions to allow everyone full access
REM ===============================
echo.
echo Modifying Z: drive permissions...

REM Create a temporary folder to enable inheritance
mkdir "Z:\temp" >nul 2>&1

REM Grant Everyone full control (including inheritance)
icacls "Z:\" /grant Everyone:(OI)(CI)F /T /C >nul 2>&1

REM Remove temporary folder
rd /s /q "Z:\temp" >nul 2>&1

if %errorlevel% neq 0 (
    echo Warning: Permission modification may not have completed fully.
) else (
    echo Z: drive permissions modified successfully.
    echo All users now have full read/write/delete access.
)

REM ===============================
REM User input loop
REM ===============================
echo.
echo ========================================
echo Z: RAM disk has been created successfully.
echo.
echo Press any key to continue using this disk...
echo Or enter Q to delete the disk and exit.
echo ========================================
pause >nul

:input_loop
set /p input="Enter command (Q=delete disk, other=continue): "
if /i "%input%"=="q" (
    goto delete_disk
) else (
    goto input_loop
)

REM ===============================
REM Delete RAM disk (clear contents first)
REM ===============================
:delete_disk
echo.
echo Deleting all contents in Z: drive...
REM Remove everything inside Z: including hidden/system files
del /f /s /q "Z:\*" >nul 2>&1
for /d %%D in (Z:\*) do rd /s /q "%%D"

echo.
echo Unmounting RAM disk Z: drive...
imdisk -d -m Z:
if %errorlevel% neq 0 (
    echo Error: Failed to unmount RAM disk.
    pause
    exit /b 1
)
echo RAM disk deleted successfully!
echo.
pause
exit /b 0
