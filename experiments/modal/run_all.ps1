# Runs all real experiments sequentially on Modal, appending each RESULT_JSON to results.jsonl.
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
Set-Location "D:\Coding_Workspace\Research_Agent"

$base = "research/focal-loss-cifar10/experiments"
$results = "$base/results.jsonl"
$log = "$base/run_all.log"
$app = "$base/modal/modal_app.py"
Remove-Item $results -ErrorAction SilentlyContinue
"started $(Get-Date -Format o)" | Out-File $log -Encoding utf8

$exps = "exp01","exp02","exp03","exp04","exp05","exp06","exp07"
foreach ($e in $exps) {
  "=== $e starting ===" | Add-Content $log
  $cfg = "@$base/modal/configs/$e.json"
  $out = py -3.13 -m modal run $app --config-json $cfg
  $line = ($out | Select-String "^RESULT_JSON:" | Select-Object -First 1)
  if ($line) {
    ($line.ToString() -replace "^RESULT_JSON:","") | Add-Content $results
    "$e OK: $line" | Add-Content $log
  } else {
    "$e FAILED (no RESULT_JSON). Tail:" | Add-Content $log
    ($out | Select-Object -Last 20) | Add-Content $log
  }
}
"ALL DONE $(Get-Date -Format o)" | Add-Content $log