Add-Type -AssemblyName System.IO.Compression.FileSystem

$SourceDir = "."        # Change to your root search directory if needed
$TargetDir = "C:\extract_here"

# Recursively process all .jar files
Get-ChildItem -Path $SourceDir -Recurse -Include *.jar | ForEach-Object {
    $jarPath = $_.FullName
    $zip = [System.IO.Compression.ZipFile]::OpenRead($jarPath)
    foreach ($entry in $zip.Entries) {
        if ($entry.FullName -match "natives\/(.+)") {
            $insideNativesPath = $Matches[1].Replace('/', '\')
            $destPath = Join-Path $TargetDir $insideNativesPath
            $parentDir = Split-Path $destPath -Parent
            if (-not (Test-Path $parentDir)) { New-Item -ItemType Directory -Path $parentDir | Out-Null }
            if (-not $entry.FullName.EndsWith("/")) {
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true)
                Write-Output "Extracted: $jarPath -> $destPath"
            }
        }
    }
    $zip.Dispose()
}
