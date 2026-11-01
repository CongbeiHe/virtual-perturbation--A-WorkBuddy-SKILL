@echo off
title Geneformer - liver HCC VNN1 overexpression
echo ===============================================
echo  Geneformer VNN1 过表达分析 - 实时输出窗口
echo  日志同时写入 E:\Geneformer\logs\
echo ===============================================
powershell -NoProfile -ExecutionPolicy Bypass -File "E:\Geneformer\scripts\run_liver_vnn1.ps1"
echo.
echo (窗口保持打开，可回看全部输出)
