param([string]$Run)
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$jobProcess=Start-Process python -ArgumentList @('tools/whole/expression_trial.py',$Run) -NoNewWindow -PassThru
$samples=@()
while(-not $jobProcess.HasExited) {
    $all=Get-CimInstance Win32_Process
    $owned=@($jobProcess.Id)
    for($depth=0;$depth -lt 5;$depth++) {
        $owned+=@($all | Where-Object { $_.ParentProcessId -in $owned } | Select-Object -ExpandProperty ProcessId)
        $owned=@($owned | Select-Object -Unique)
    }
    $procs=@(Get-Process -Id $owned -ErrorAction SilentlyContinue)
    $gpu=(& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>$null) -join '|'
    $samples+=@{time=(Get-Date -Format o); pids=$owned; workingSetBytes=($procs | Measure-Object WorkingSet64 -Sum).Sum; cpuSeconds=($procs | Measure-Object CPU -Sum).Sum; gpuWholeDevice=$gpu}
    Start-Sleep -Seconds 3
    $jobProcess.Refresh()
}
$samples | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 "$Run/resources.json"
exit $jobProcess.ExitCode
