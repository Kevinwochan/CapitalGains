Add-Type -AssemblyName System.IO.Compression.FileSystem

function Get-DllFromJar ($jarPath) {
    $zip = [System.IO.Compression.ZipFile]::OpenRead($jarPath)
    $dlls = $zip.Entries | Where-Object { $_.FullName -like "*.dll" }
    foreach ($dll in $dlls) {
        Write-Output "$($jarPath): $($dll.FullName)"
    }
    $zip.Dispose()
}

Get-ChildItem -Path . -Recurse -Include *.jar | ForEach-Object {
    Get-DllFromJar $_.FullName
}
