# Builds gempy_model_builder.zip for "Install from ZIP" in QGIS.
# Wraps everything in a single top-level folder and skips __pycache__ / *.pyc,
# which QGIS rejects.
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem

$base = $PSScriptRoot
$top = "gempy_model_builder"
$zipPath = Join-Path $base "$top.zip"
$items = "__init__.py", "metadata.txt", "LICENSE", "README.md", "USER_GUIDE.txt", "methods.pdf", "test_module"

if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
$zip = [IO.Compression.ZipFile]::Open($zipPath, 'Create')
try {
    foreach ($i in $items) {
        $path = Join-Path $base $i
        if (-not (Test-Path $path)) { continue }
        $files = if (Test-Path $path -PathType Leaf) { Get-Item $path } else {
            Get-ChildItem $path -Recurse -File |
                Where-Object { $_.FullName -notmatch '\\__pycache__\\' -and $_.Extension -notin '.pyc', '.pyo' }
        }
        foreach ($f in $files) {
            $rel = $f.FullName.Substring($base.Length + 1).Replace('\', '/')
            [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $f.FullName, "$top/$rel", 'Optimal')
        }
    }
} finally {
    $zip.Dispose()
}

$names = [IO.Compression.ZipFile]::OpenRead($zipPath).Entries.FullName
Write-Host "$zipPath : $($names.Count) entries, __pycache__ present: $([bool]($names -match '__pycache__'))"
