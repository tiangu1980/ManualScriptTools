:: python TfIdf-PolynomialBayesianClassifier.py --mode train --infile SAT-DSAT-Lable-train.xlsx --model SAT-L5 --incols "SAT" --merge but --target L5

@echo off
setlocal enabledelayedexpansion

REM 获取调用对象脚本的次数
set calls=%1
shift

REM 将第二个参数开始的所有内容保存为变量 mycmd
set "mycmd=python"
:build_mycmd
if not "%2"=="" (
    set "mycmd=!mycmd! %2"
    shift
    goto build_mycmd
)

REM 运行 mycmd 命令 calls 次
for /l %%i in (1,1,%calls%) do (
    echo %%i / %calls% Running： %mycmd%
    !mycmd!
)

endlocal
