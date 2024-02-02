@echo off

:: TfIdf-PolynomialBayesianClassifier_Train.bat 300 SAT-DSAT-Lable-train.xlsx SAT-DSAT-L1

:: 检查是否提供了足够的参数
if not %1.==. goto :check_file
echo Usage: %0 <number_of_times> <file_name>
goto :eof

:check_file
:: 检查文件是否存在
if not exist %2 (
    echo File %2 does not exist. Exiting.
    goto :eof
)

:: 使用 for /l 循环执行指定次数的命令，每轮循环延时1秒
for /l %%i in (1, 1, %1) do (
    python TfIdf-PolynomialBayesianClassifier.py --mode train --infile %2  --model %3 --incols "SAT" "DSAT" --merge but --target L1
    echo python TfIdf-PolynomialBayesianClassifier.py --mode train --infile %2  --model %3 --incols "SAT" "DSAT" --merge but --target L1 - Iteration %%i
    timeout /nobreak /t 1 >nul
)

echo Finished run cmd for %1 times