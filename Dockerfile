$jarPath = 'C:\path\to\yourfile.jar'
$fileInJar = 'path/in/jar/yourdll.dll'   # Use the exact path as seen in jar listing
$destDir = 'C:\extract\here'

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($jarPath)
$entry = $zip.Entries | Where-Object { $_.FullName -eq $fileInJar }
if ($entry) {
    if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir | Out-Null }
    $destPath = Join-Path $destDir ($entry.Name)
    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true)
    Write-Output "Extracted to $destPath"
} else {
    Write-Warning "File not found in archive."
}
$zip.Dispose()
