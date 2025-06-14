@echo off
setlocal enabledelayedexpansion

REM --- Knowledge Building Pipeline - Sentences Processing ---
REM This script processes markdown files into sentences with embeddings

REM --- Initialize variables ---
set INPUT_DIR=
set OUTPUT_FILE=
set INTERMEDIATE_DIR=
set CLEANUP_INTERMEDIATE=false
set SCRIPT_DIR=%~dp0
set STEPS_DIR=%SCRIPT_DIR%scripts\sentences_pipeline_steps
set VENV_DIR=%SCRIPT_DIR%venv

REM --- Ensure Python virtual environment exists ---
if not exist "%VENV_DIR%\Scripts\python.exe" (
    call :print_status "Creating Python virtual environment..."
    python -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        call :print_error "Failed to create virtual environment."
        exit /b 1
    )
)

REM --- Activate the venv ---
call "%VENV_DIR%\Scripts\activate.bat"
if not defined VIRTUAL_ENV (
    call :print_error "Failed to activate virtual environment."
    exit /b 1
)

REM --- Install dependencies ---
if exist "%SCRIPT_DIR%requirements.txt" (
    call :print_status "Installing dependencies from requirements.txt..."
    pip install -r "%SCRIPT_DIR%requirements.txt"
    if !errorlevel! neq 0 (
        call :print_error "Failed to install Python dependencies."
        exit /b 1
    )
) else (
    call :print_error "requirements.txt not found at: %SCRIPT_DIR%requirements.txt"
    exit /b 1
)

REM --- Load .env variables ---
call :load_env

REM --- Parse arguments ---
:parse_args
if "%~1"=="" goto validate_args
set ARG=%~1
set VALUE=%~2

if /i "%ARG%"=="--input_dir" (
    set "INPUT_DIR=%VALUE%"
    shift & shift & goto parse_args
)
if /i "%ARG%"=="--output_file" (
    set "OUTPUT_FILE=%VALUE%"
    shift & shift & goto parse_args
)
if /i "%ARG%"=="--intermediate_dir" (
    set "INTERMEDIATE_DIR=%VALUE%"
    shift & shift & goto parse_args
)
if /i "%ARG%"=="--help" goto usage

echo ❌ Unknown option: %ARG%
goto usage

:validate_args
if "%INPUT_DIR%"=="" goto missing_args
if "%OUTPUT_FILE%"=="" goto missing_args

if not exist "%INPUT_DIR%" (
    call :print_error "Input directory does not exist: %INPUT_DIR%"
    exit /b 1
)

if "%AZURE_OPENAI_API_KEY%"=="" goto missing_env
if "%AZURE_OPENAI_ENDPOINT%"=="" goto missing_env

REM --- Setup intermediate directory ---
if "%INTERMEDIATE_DIR%"=="" (
    for /f %%i in ('powershell -command "[System.IO.Path]::GetTempPath() + [System.Guid]::NewGuid().ToString()"') do set INTERMEDIATE_DIR=%%i
    mkdir "!INTERMEDIATE_DIR!" 2>nul
    set CLEANUP_INTERMEDIATE=true
    call :print_status "Created temporary intermediate directory: !INTERMEDIATE_DIR!"
) else (
    mkdir "%INTERMEDIATE_DIR%" 2>nul
)

if not exist "%STEPS_DIR%" (
    call :print_error "Steps directory not found: %STEPS_DIR%"
    exit /b 1
)

REM --- Define intermediate outputs ---
set STEP1_OUTPUT=%INTERMEDIATE_DIR%\step1_paragraphs.jsonl
set STEP2_OUTPUT=%INTERMEDIATE_DIR%\step2_coref_resolved.jsonl
set STEP3_OUTPUT=%INTERMEDIATE_DIR%\step3_sentences.jsonl
set STEP4_OUTPUT=%INTERMEDIATE_DIR%\step4_filtered_sentences.jsonl

REM --- Pipeline Execution ---
call :print_status "Starting Knowledge Building Pipeline"
call :print_status "Input directory: %INPUT_DIR%"
call :print_status "Output file: %OUTPUT_FILE%"
call :print_status "Intermediate directory: %INTERMEDIATE_DIR%"
echo.

REM --- Step 1 ---
call :print_status "Step 1/5: Splitting paragraphs from markdown files..."
python "%STEPS_DIR%\split_paragraphs.py" --input_dir "%INPUT_DIR%" --output_file "%STEP1_OUTPUT%"
if !errorlevel! neq 0 (
    call :print_error "Step 1 failed: split_paragraphs.py"
    goto cleanup_and_exit
)
call :print_success "Step 1 completed: Paragraphs extracted"

REM --- Step 2 ---
call :print_status "Step 2/5: Resolving coreferences..."
python "%STEPS_DIR%\resolve_coreferences.py" --input_file "%STEP1_OUTPUT%" --output_file "%STEP2_OUTPUT%"
if !errorlevel! neq 0 (
    call :print_error "Step 2 failed: resolve_coreferences.py"
    goto cleanup_and_exit
)
call :print_success "Step 2 completed: Coreferences resolved"

REM --- Step 3 ---
call :print_status "Step 3/5: Splitting sentences..."
python "%STEPS_DIR%\split_sentences.py" --input_file "%STEP2_OUTPUT%" --output_file "%STEP3_OUTPUT%"
if !errorlevel! neq 0 (
    call :print_error "Step 3 failed: split_sentences.py"
    goto cleanup_and_exit
)
call :print_success "Step 3 completed: Sentences split"

REM --- Step 4 ---
call :print_status "Step 4/5: Filtering image sentences..."
python "%STEPS_DIR%\filter_image_sentences.py" --input_file "%STEP3_OUTPUT%" --output_file "%STEP4_OUTPUT%"
if !errorlevel! neq 0 (
    call :print_error "Step 4 failed: filter_image_sentences.py"
    goto cleanup_and_exit
)
call :print_success "Step 4 completed: Image sentences filtered"

REM --- Step 5 ---
call :print_status "Step 5/5: Adding embeddings..."
python "%STEPS_DIR%\add_embeddings.py" --input_file "%STEP4_OUTPUT%" --output_file "%OUTPUT_FILE%"
if !errorlevel! neq 0 (
    call :print_error "Step 5 failed: add_embeddings.py"
    goto cleanup_and_exit
)
call :print_success "Step 5 completed: Embeddings added"

echo.
call :print_success "Pipeline completed successfully!"
call :print_success "Final output saved to: %OUTPUT_FILE%"
if exist "%OUTPUT_FILE%" (
    for /f %%i in ('find /c /v "" ^< "%OUTPUT_FILE%"') do call :print_status "Total sentences with embeddings: %%i"
)
goto cleanup

:cleanup_and_exit
exit /b 1

:cleanup
if "%CLEANUP_INTERMEDIATE%"=="true" if exist "%INTERMEDIATE_DIR%" (
    call :print_status "Cleaning up temporary directory: %INTERMEDIATE_DIR%"
    rmdir /s /q "%INTERMEDIATE_DIR%" 2>nul
)
exit /b 0

REM --- Helper Functions ---
:load_env
set ENV_FILE=%SCRIPT_DIR%.env
if exist "%ENV_FILE%" (
    call :print_status "Loading environment variables from: %ENV_FILE%"
    for /f "usebackq tokens=1* delims==" %%a in ("%ENV_FILE%") do (
        set "key=%%a"
        set "value=%%b"
        set "!key!=!value!"
    )
)
goto :eof

:print_status
echo [%date% %time%] %~1
goto :eof

:print_success
echo ✅ %~1
goto :eof

:print_error
echo ❌ %~1
goto :eof

:usage
echo Usage: %0 --input_dir ^<markdown_dir^> --output_file ^<final_output.jsonl^> [--intermediate_dir ^<temp_dir^>]
echo.
echo Options:
echo   --input_dir         Directory containing markdown files
echo   --output_file       Final output JSONL file with embeddings
echo   --intermediate_dir  Optional directory for intermediate files
echo   --help              Show this help message
echo.
echo Environment variables required in .env file:
echo   AZURE_OPENAI_API_KEY
echo   AZURE_OPENAI_ENDPOINT
exit /b 1

:missing_args
call :print_error "Missing required arguments"
goto usage

:missing_env
call :print_error "Required environment variables not set:"
call :print_error "  AZURE_OPENAI_API_KEY"
call :print_error "  AZURE_OPENAI_ENDPOINT"
call :print_error ""
call :print_error "You can set them in a .env file or via set commands in this script"
exit /b 1
