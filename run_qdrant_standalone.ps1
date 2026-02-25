# Script za pokretanje Qdrant standalone verzije (bez Docker-a)
# Za Windows - preuzima i pokreće Qdrant binary

$qdrantVersion = "v1.12.5"
$qdrantUrl = "https://github.com/qdrant/qdrant/releases/download/$qdrantVersion/qdrant-x86_64-pc-windows-msvc.zip"
$qdrantDir = "qdrant_standalone"
$zipFile = "$qdrantDir\qdrant.zip"

Write-Host "=" * 70
Write-Host "Qdrant Standalone Setup"
Write-Host "=" * 70

# Kreiraj direktorijum
if (-not (Test-Path $qdrantDir)) {
    New-Item -ItemType Directory -Path $qdrantDir | Out-Null
    Write-Host "✓ Kreiran folder: $qdrantDir"
}

# Proveri da li već postoji exe
$exePath = "$qdrantDir\qdrant.exe"
if (Test-Path $exePath) {
    Write-Host "✓ Qdrant već preuzet"
} else {
    Write-Host "Preuzimam Qdrant $qdrantVersion..."
    Write-Host "URL: $qdrantUrl"
    
    try {
        # Proveri proxy
        $proxy = $env:https_proxy
        if ($proxy) {
            Write-Host "Koristim proxy: $proxy"
            Invoke-WebRequest -Uri $qdrantUrl -OutFile $zipFile -Proxy $proxy -UseBasicParsing
        } else {
            Invoke-WebRequest -Uri $qdrantUrl -OutFile $zipFile -UseBasicParsing
        }
        
        Write-Host "✓ Preuzeto"
        
        # Raspakuj
        Write-Host "Raspakujem..."
        Expand-Archive -Path $zipFile -DestinationPath $qdrantDir -Force
        Write-Host "✓ Raspаkovano"
        
        # Obriši zip
        Remove-Item $zipFile
        
    } catch {
        Write-Host "✗ Greška pri preuzimanju: $_"
        Write-Host ""
        Write-Host "ALTERNATIVA: Preuzmi ručno:"
        Write-Host "1. Idi na: https://github.com/qdrant/qdrant/releases/latest"
        Write-Host "2. Preuzmi: qdrant-x86_64-pc-windows-msvc.zip"
        Write-Host "3. Raspakuj u: $qdrantDir\"
        Write-Host "4. Pokreni: .\$qdrantDir\qdrant.exe"
        exit 1
    }
}

# Kreiraj storage folder
$storageDir = "$qdrantDir\storage"
if (-not (Test-Path $storageDir)) {
    New-Item -ItemType Directory -Path $storageDir | Out-Null
}

# Pokreni Qdrant
Write-Host ""
Write-Host "=" * 70
Write-Host "Pokrećem Qdrant..."
Write-Host "=" * 70
Write-Host ""
Write-Host "REST API: http://localhost:6333"
Write-Host "gRPC API: http://localhost:6334"
Write-Host "Dashboard: http://localhost:6333/dashboard"
Write-Host ""
Write-Host "Pritisni Ctrl+C da zaustaviš server"
Write-Host ""

Set-Location $qdrantDir
.\qdrant.exe
