# Quick Setup Script - Српски историчар React Frontend
# This script sets up the React frontend quickly

Write-Host "Setting up React frontend..." -ForegroundColor Cyan

# Navigate to frontend directory
if (-not (Test-Path "frontend")) {
    Write-Host "ERROR: frontend directory not found!" -ForegroundColor Red
    Write-Host "Make sure you're running this from the srpski-istoricar root directory" -ForegroundColor Red
    exit 1
}

Push-Location frontend

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
npm install

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies!" -ForegroundColor Red
    Pop-Location
    exit 1
}

Write-Host "✓ Dependencies installed successfully!" -ForegroundColor Green

Pop-Location

# Copy images
Write-Host ""
Write-Host "Copying images..." -ForegroundColor Yellow
Copy-Item "pictures\hero-bg.jpg" "frontend\public\" -Force -ErrorAction SilentlyContinue
Copy-Item "pictures\SvetiSavaMileseva.jpg" "frontend\public\" -Force -ErrorAction SilentlyContinue
Write-Host "✓ Images copied!" -ForegroundColor Green

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Setup complete! 🎉" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the application:" -ForegroundColor Yellow
Write-Host "  1. In one terminal: python app.py" -ForegroundColor White
Write-Host "  2. In another terminal: cd frontend; npm start" -ForegroundColor White
Write-Host ""
Write-Host "Or use the automated script:" -ForegroundColor Yellow
Write-Host "  .\start.ps1" -ForegroundColor White
Write-Host ""
