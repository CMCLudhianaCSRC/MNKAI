<#
  scripts\download_fine_isic.ps1
  Downloads dermoscopic images per class using ISIC CLI queries.
  - Creates D:\AIFINAL\fine_pool\<class> with images + download.log
  - Probes metadata to show match counts before downloading
  - Optional probe-only mode via -ProbeOnly
#>

param(
  [string]$IsicExe     = '.\isic.exe',
  [string]$OutRoot     = 'D:\AIFINAL\fine_pool',
  [int]   $PerClass    = 4200,
  [switch]$ProbeOnly,
  [int]   $ProbeSample = 50
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not (Test-Path $IsicExe)) {
  throw "Cannot find isic.exe at '$IsicExe'. Put it next to this script or pass -IsicExe."
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

function Quote-IsicArgument {
  param([string]$Value)

  if ($null -eq $Value) { return '""' }
  if ($Value.Length -eq 0) { return '""' }
  if ($Value -notmatch '[\s\"]') { return $Value }

  $sb = [System.Text.StringBuilder]::new()
  [void]$sb.Append('"')

  $i = 0
  while ($i -lt $Value.Length) {
    $bsCount = 0
    while (($i -lt $Value.Length) -and ($Value[$i] -eq '\')) {
      $bsCount++
      $i++
    }

    if ($i -ge $Value.Length) {
      for ($n = 0; $n -lt $bsCount * 2; $n++) { [void]$sb.Append('\') }
      break
    }

    $ch = $Value[$i]
    if ($ch -eq '"') {
      for ($n = 0; $n -lt ($bsCount * 2) + 1; $n++) { [void]$sb.Append('\') }
      [void]$sb.Append('"')
    } else {
      for ($n = 0; $n -lt $bsCount; $n++) { [void]$sb.Append('\') }
      [void]$sb.Append($ch)
    }

    $i++
  }

  [void]$sb.Append('"')
  return $sb.ToString()
}

function Join-ProcessArguments {
  param([string[]]$ArgumentList)

  $escaped = foreach ($arg in $ArgumentList) { Quote-IsicArgument -Value $arg }
  return ($escaped -join ' ')
}

function Run-Isic {
  param([string[]]$Arguments)

  $psi = [System.Diagnostics.ProcessStartInfo]::new()
  $psi.FileName = $IsicExe
  $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.CreateNoWindow = $true

  if ($psi.PSObject.Properties.Name -contains 'ArgumentList') {
    foreach ($item in $Arguments) { [void]$psi.ArgumentList.Add($item) }
  } else {
    $psi.Arguments = Join-ProcessArguments -ArgumentList $Arguments
  }

  $proc = [System.Diagnostics.Process]::Start($psi)
  $stdout = $proc.StandardOutput.ReadToEnd()
  $stderr = $proc.StandardError.ReadToEnd()
  $proc.WaitForExit()

  [pscustomobject]@{
    Code = $proc.ExitCode
    Out  = $stdout
    Err  = $stderr
  }
}

function Probe-Count {
  param(
    [string]$Query,
    [int]$SampleSize
  )

  $tmp = Join-Path $env:TEMP ("isic_probe_{0}.csv" -f ([guid]::NewGuid().ToString('N')))
  if (Test-Path $tmp) { Remove-Item $tmp -Force -ErrorAction SilentlyContinue }

  $args = @('metadata', 'download', '--search', $Query)
  if ($SampleSize -gt 0) { $args += @('--limit', "$SampleSize") }
  $args += @('--outfile', $tmp)

  $result = Run-Isic -Arguments $args
  if ($result.Code -ne 0) {
    throw "isic metadata probe failed (exit $($result.Code)): $($result.Err.Trim())"
  }

  $count = 0
  if (Test-Path $tmp) {
    $lines = (Get-Content $tmp | Measure-Object -Line).Lines
    if ($lines -gt 0) { $count = $lines - 1 }
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
  }

  return $count
}

function Get-ExistingCount {
  param([string]$Dir)

  $metadataPath = Join-Path $Dir 'metadata.csv'
  if (Test-Path $metadataPath) {
    $lineCount = (Get-Content $metadataPath | Measure-Object -Line).Lines
    if ($lineCount -le 0) { return 0 }
    return $lineCount - 1
  }

  $images = Get-ChildItem -Path $Dir -Include *.jpg,*.jpeg,*.png -Recurse -ErrorAction SilentlyContinue
  return ($images | Measure-Object).Count
}

$queries = @{
  mel  = @(
    'diagnosis_3:"Melanoma in situ" AND image_type:dermoscopic',
    'diagnosis_3:"Melanoma Invasive" AND image_type:dermoscopic',
    'diagnosis_3:"Melanoma, NOS" AND image_type:dermoscopic'
  )
  nv   = @(
    'diagnosis_3:Nevus AND image_type:dermoscopic'
  )
  bcc  = @(
    'diagnosis_3:"Basal cell carcinoma" AND image_type:dermoscopic'
  )
  scc  = @(
    'diagnosis_3:"Squamous cell carcinoma" AND image_type:dermoscopic',
    'diagnosis_3:"Intraepidermal carcinoma (Bowen disease)" AND image_type:dermoscopic'
  )
  df   = @(
    'diagnosis_3:Dermatofibroma AND image_type:dermoscopic'
  )
  bkl  = @(
    'diagnosis_3:"Seborrheic keratosis" AND image_type:dermoscopic',
    'diagnosis_3:"Solar lentigo" AND image_type:dermoscopic',
    'diagnosis_3:"Lichen planus-like keratosis" AND image_type:dermoscopic'
  )
  ak   = @(
    'diagnosis_3:"Actinic keratosis" AND image_type:dermoscopic'
  )
  vasc = @(
    'diagnosis_3:"Vascular lesion" AND image_type:dermoscopic',
    'diagnosis_3:Hemangioma AND image_type:dermoscopic',
    'diagnosis_3:Angioma AND image_type:dermoscopic',
    'diagnosis_3:"Pyogenic granuloma" AND image_type:dermoscopic',
    'diagnosis_3:"Venous lake" AND image_type:dermoscopic'
  )
}

foreach ($cls in $queries.Keys) {
  $dir = Join-Path $OutRoot $cls
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $logPath = Join-Path $dir 'download.log'

  $have = Get-ExistingCount -Dir $dir
  Write-Host "==== [$cls] start have=$have / target=$PerClass -> $dir ====" -ForegroundColor Cyan
  Add-Content -Path $logPath -Value ("[{0}] start have={1} target={2}" -f (Get-Date), $have, $PerClass)

  if ($have -ge $PerClass) {
    Write-Host "[$cls] already at/over target, skipping." -ForegroundColor Yellow
    Add-Content -Path $logPath -Value ("[{0}] skip already >= target" -f (Get-Date))
    continue
  }

  foreach ($query in $queries[$cls]) {
    Write-Host "[$cls] probe: $query" -ForegroundColor DarkYellow

    try {
      $probeCount = Probe-Count -Query $query -SampleSize $ProbeSample
      Write-Host "[$cls] probe matched approximately $probeCount (sample of $ProbeSample)" -ForegroundColor Yellow
      Add-Content -Path $logPath -Value ("[{0}] probe ok: {1} -> {2}" -f (Get-Date), $query, $probeCount)
    } catch {
      Write-Warning "[$cls] probe failed: $($_.Exception.Message)"
      Add-Content -Path $logPath -Value ("[{0}] probe FAILED: {1}" -f (Get-Date), $query)
      continue
    }

    if ($ProbeOnly) { continue }

    $remaining = $PerClass - (Get-ExistingCount -Dir $dir)
    if ($remaining -le 0) { break }

    Write-Host "[$cls] download (max $remaining) with query:" -NoNewline
    Write-Host " $query" -ForegroundColor Green
    Add-Content -Path $logPath -Value ("[{0}] download begin: left={1} query={2}" -f (Get-Date), $remaining, $query)

    $downloadArgs = @('image', 'download', '--search', $query, '--limit', "$remaining", $dir)
    $download = Run-Isic -Arguments $downloadArgs

    if ($download.Code -ne 0) {
      Write-Warning "[$cls] download failed (exit $($download.Code)): $($download.Err.Trim())"
      Add-Content -Path $logPath -Value ("[{0}] download FAILED: {1}" -f (Get-Date), $download.Err.Trim())
      continue
    }

    Write-Host "[$cls] download chunk complete." -ForegroundColor Green
    Add-Content -Path $logPath -Value ("[{0}] download OK" -f (Get-Date))

    $have = Get-ExistingCount -Dir $dir
    Write-Host "[$cls] now have=$have / target=$PerClass" -ForegroundColor Cyan

    if ($have -ge $PerClass) { break }
  }

  $finalCount = Get-ExistingCount -Dir $dir
  Write-Host "==== [$cls] finish have=$finalCount / target=$PerClass ====" -ForegroundColor Cyan
  Add-Content -Path $logPath -Value ("[{0}] finish have={1}" -f (Get-Date), $finalCount)
}

Write-Host "All done. See per-class logs under $OutRoot\<class>\download.log" -ForegroundColor Cyan
