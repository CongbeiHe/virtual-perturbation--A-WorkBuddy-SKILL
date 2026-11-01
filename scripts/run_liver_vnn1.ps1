# 肝 HCC VNN1 过表达全流程 —— 在独立终端窗口中实时运行
# 日志同时写入 E:\Geneformer\logs\

$py   = "E:\Geneformer\venv\Scripts\python.exe"
$logs = "E:\Geneformer\logs"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:PYTHONUNBUFFERED = "1"
Set-Location E:\Geneformer

function Run-Step($name, $script, $log) {
  Write-Host ""
  Write-Host ("=" * 70) -ForegroundColor Cyan
  Write-Host ("  " + $name) -ForegroundColor Cyan
  Write-Host ("=" * 70) -ForegroundColor Cyan
  & $py -u $script 2>&1 | Tee-Object -FilePath $log
  if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host ("[错误] {0} 失败 (exit={1})，详见 {2}" -f $name, $LASTEXITCODE, $log) -ForegroundColor Red
  } else {
    Write-Host ("[完成] {0}" -f $name) -ForegroundColor Green
  }
}

Run-Step "步骤 1-5  tokenize + VNN1 过表达扰动 + 统计" `
         "E:\Geneformer\scripts\11_liver_vnn1_pipeline.py" `
         "$logs\liver_pipeline.log"

Run-Step "步骤 6  UMAP 位移分析" `
         "E:\Geneformer\scripts\12_umap_vnn1.py" `
         "$logs\liver_umap.log"

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Green
Write-Host "  全部步骤执行完毕，结果见 E:\Geneformer\outputs\liver_isp*" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Green
