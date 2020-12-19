@echo off
set PYTHON_ENVS=py35 py36 py37 py38

set CUDA_COMPILER="%CUDA_PATH_V9_2:\=/%/bin/nvcc.exe"

where /q vswhere.exe
if ERRORLEVEL 1 (
    echo INFO: vswhere.exe is not in PATH -- assuming it is in "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
    set VSWHERE="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
) else (
    set VSWHERE=vswhere.exe
)

echo INFO: Finding Visual Studio installation path
for /f "usebackq tokens=*" %%i in (`%VSWHERE% -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set VS_INSTALL_DIR=%%i
)
echo INFO: Found Visual Studio in %VS_INSTALL_DIR%

echo INFO: Finding vcvarsall.bat
if exist "%VS_INSTALL_DIR%\VC\Auxiliary\Build\vcvarsall.bat" (
    set VCVARSALL="%VS_INSTALL_DIR%\VC\Auxiliary\Build\vcvarsall.bat"
) else (
    echo ERROR: Could not find vcvarsall.bat in Visual Studio installation path
    exit /b 1
)
echo INFO: Found vcvarsall.bat as %VCVARSALL%

echo INFO: Set Visual Studio Toolset to 2015
call %VCVARSALL% x64 -vcvars_ver=14.0

echo INFO: Deleting dist files
del /q dist\*

echo INFO: --------------------------------------------
echo INFO: Building
for %%v in (%PYTHON_ENVS%) do (
    rem Build package
    call conda activate %%v
    call python -V
    call python setup.py bdist_wheel -- -DCMAKE_CUDA_COMPILER=%CUDA_COMPILER%

    rem Install and run tests
    call pip install pygorpho --no-index --find-links=dist/
    call pytest
    call pytest --doctest-modules src/pygorpho
    call pip uninstall pygorpho --yes
)

call python setup.py sdist -- -DCMAKE_CUDA_COMPILER=%CUDA_COMPILER%

echo INFO: --------------------------------------------
echo INFO: Uploading
call twine upload dist/*

echo INFO: Done
