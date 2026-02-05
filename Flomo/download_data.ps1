# Flomo animation data download script
# Downloads motion capture datasets for testing/training

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$dataDir = Join-Path $scriptDir "data"

Write-Host "Flomo Animation Data Download" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host ""
Write-Host "WARNING: This will download several GB of animation data." -ForegroundColor Yellow
Write-Host "         This may take a long time depending on your connection." -ForegroundColor Yellow
Write-Host ""

# Create data folder if needed
if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
    Write-Host "Created data folder"
}

# Helper function to download a file with progress bar
function Download-WithProgress {
    param (
        [string]$url,
        [string]$outFile,
        [string]$description
    )

    $uri = [System.Uri]::new($url)
    $request = [System.Net.HttpWebRequest]::Create($uri)
    $request.Timeout = 30000

    $response = $request.GetResponse()
    $totalBytes = $response.ContentLength
    $responseStream = $response.GetResponseStream()

    $fileStream = [System.IO.File]::Create($outFile)
    $buffer = New-Object byte[] 65536
    $bytesRead = 0
    $totalRead = 0
    $lastPercent = -1

    $totalMB = [math]::Round($totalBytes / 1MB, 1)

    while (($bytesRead = $responseStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
        $fileStream.Write($buffer, 0, $bytesRead)
        $totalRead += $bytesRead

        if ($totalBytes -gt 0) {
            $percent = [math]::Floor(($totalRead / $totalBytes) * 100)
            if ($percent -ne $lastPercent) {
                $downloadedMB = [math]::Round($totalRead / 1MB, 1)
                $barWidth = 30
                $filled = [math]::Floor($percent / 100 * $barWidth)
                $empty = $barWidth - $filled
                $bar = ("#" * $filled) + ("-" * $empty)
                Write-Host -NoNewline "`r  [$bar] $percent% ($downloadedMB / $totalMB MB)  "
                $lastPercent = $percent
            }
        }
    }

    Write-Host ""
    $fileStream.Close()
    $responseStream.Close()
    $response.Close()
}

# Helper function to download and extract a zip to a folder
function Download-AnimData {
    param (
        [string]$url,
        [string]$destFolder,
        [string]$description
    )

    if (Test-Path $destFolder) {
        $fileCount = (Get-ChildItem -Path $destFolder -File -Recurse).Count
        if ($fileCount -gt 0) {
            Write-Host "[OK] $description already exists ($fileCount files)" -ForegroundColor Green
            return
        }
    }

    $zipFile = Join-Path $env:TEMP "anim_download_temp.zip"

    Write-Host "[DOWNLOADING] $description" -ForegroundColor Yellow
    Write-Host "  URL: $url"

    Download-WithProgress $url $zipFile $description

    Write-Host "[EXTRACTING] $description" -ForegroundColor Yellow

    # Extract to temp folder first
    $tempExtract = Join-Path $env:TEMP "anim_extract_temp"
    if (Test-Path $tempExtract) {
        Remove-Item -Recurse -Force $tempExtract
    }

    Expand-Archive -Path $zipFile -DestinationPath $tempExtract -Force

    # Check if there's a single subfolder (e.g. bvh.zip contains a "bvh" folder)
    $extractedItems = Get-ChildItem -Path $tempExtract
    if ($extractedItems.Count -eq 1 -and $extractedItems[0].PSIsContainer) {
        # Move contents of the subfolder to dest
        $sourceFolder = $extractedItems[0].FullName
    }
    else {
        $sourceFolder = $tempExtract
    }

    # Create dest folder and move contents
    if (-not (Test-Path $destFolder)) {
        New-Item -ItemType Directory -Path $destFolder -Force | Out-Null
    }

    Get-ChildItem -Path $sourceFolder | Move-Item -Destination $destFolder -Force

    # Cleanup
    Remove-Item -Recurse -Force $tempExtract
    Remove-Item -Force $zipFile

    $fileCount = (Get-ChildItem -Path $destFolder -File -Recurse).Count
    Write-Host "[OK] $description installed ($fileCount files)" -ForegroundColor Green
}

# Dataset definitions
$datasets = @(
    @{
        Name = "lafan"
        BaseUrl = "https://theorangeduck.com/media/uploads/Geno/lafan1-resolved"
    },
    @{
        Name = "motorica"
        BaseUrl = "https://theorangeduck.com/media/uploads/Geno/motorica-retarget"
    },
    @{
        Name = "100style"
        BaseUrl = "https://theorangeduck.com/media/uploads/Geno/100style-retarget"
    }
)

foreach ($dataset in $datasets) {
    $name = $dataset.Name
    $baseUrl = $dataset.BaseUrl

    Write-Host ""
    Write-Host "Processing $name dataset..." -ForegroundColor Cyan

    $bvhFolder = Join-Path $dataDir "$name/bvh"
    $fbxFolder = Join-Path $dataDir "$name/fbx"

    Download-AnimData "$baseUrl/bvh.zip" $bvhFolder "$name BVH"
    Download-AnimData "$baseUrl/fbx.zip" $fbxFolder "$name FBX"
}

Write-Host ""
Write-Host "All animation data downloaded!" -ForegroundColor Green
Write-Host "Data location: $dataDir" -ForegroundColor Cyan
