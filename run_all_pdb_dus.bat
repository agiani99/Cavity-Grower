@echo off
setlocal

REM Batch-run de_novo_cavity_growth.py over all PDBs in a folder and all DU markers.
REM Outputs are written to .\out_sdf\{PDB}_DU{index}_default.(sdf|csv)

set "DEFAULT_PDB_DIR=..\dogsite_results_450\pdb_with_du"
set "PDB_DIR=%~1"

REM If arg1 is empty or looks like a flag, use default.
if "%PDB_DIR%"=="" goto UseDefault
set "FIRSTCHAR=%PDB_DIR:~0,1%"
if "%FIRSTCHAR%"=="-" goto UseDefault
if "%FIRSTCHAR%"=="/" goto UseDefault

REM Otherwise arg1 is the PDB folder; remove it from %* so remaining args are forwarded.
shift
goto Run

:UseDefault
set "PDB_DIR=%DEFAULT_PDB_DIR%"

:Run

REM Forward any remaining arguments to de_novo_cavity_growth.py.
REM Example:
REM   run_all_pdb_dus.bat . --beam-width 80 --n-steps 40

REM Special-case batch runner options that users may want from the .bat.
REM Currently supported here:
REM   --dry-run   (prints commands, does not execute)
REM   --use-installed-adfr   (auto-detect installed ADFR Suite and forward grower flags)
set "BATCH_ARGS="
set "DENOVO_ARGS="

:ParseArgs
if "%~1"=="" goto ArgsDone
if "%~1"=="--dry-run" (
	set "BATCH_ARGS=%BATCH_ARGS% --dry-run"
) else if "%~1"=="--use-installed-adfr" (
	set "BATCH_ARGS=%BATCH_ARGS% --use-installed-adfr"
) else (
	set "DENOVO_ARGS=%DENOVO_ARGS% %1"
)
shift
goto ParseArgs

:ArgsDone
python "%~dp0batch_run_pdb_dus.py" --pdb-dir "%PDB_DIR%" --out-dir "%~dp0out_sdf" --suffix default %BATCH_ARGS% -- %DENOVO_ARGS%

endlocal
exit /b %ERRORLEVEL%
