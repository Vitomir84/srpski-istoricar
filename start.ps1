# Српски историчар - React Migration
# This script sets up and starts both the Flask backend and React frontend

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   Српски историчар - Setup and Start" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Node.js is installed
Write-Host "Checking prerequisites..." -ForegroundColor Yellow
$nodeVersion = node --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Node.js is not installed!" -ForegroundColor Red
    Write-Host "Please install Node.js from https://nodejs.org/" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Node.js is installed: $nodeVersion" -ForegroundColor Green

# Check if Python is installed
$pythonVersion = python --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python is not installed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Python is installed: $pythonVersion" -ForegroundColor Green
Write-Host ""

# Install frontend dependencies if needed
Write-Host "Setting up React frontend..." -ForegroundColor Yellow
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
    Push-Location frontend
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install Node dependencies!" -ForegroundColor Red
        Pop-Location
        exit 1
    }
    Pop-Location
    Write-Host "✓ Node dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✓ Node dependencies already installed" -ForegroundColor Green
}

# Copy image files to frontend public folder
Write-Host ""
Write-Host "Copying images to frontend..." -ForegroundColor Yellow
if (Test-Path "pictures\hero-bg.jpg") {
    Copy-Item "pictures\hero-bg.jpg" "frontend\public\" -Force
    Write-Host "✓ Copied hero-bg.jpg" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: pictures\hero-bg.jpg not found" -ForegroundColor Yellow
}

if (Test-Path "pictures\SvetiSavaMileseva.jpg") {
    Copy-Item "pictures\SvetiSavaMileseva.jpg" "frontend\public\" -Force
    Write-Host "✓ Copied SvetiSavaMileseva.jpg" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: pictures\SvetiSavaMileseva.jpg not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   Starting Application" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Backend (Flask): http://localhost:5000" -ForegroundColor Cyan
Write-Host "Frontend (React): http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting Flask backend..." -ForegroundColor Yellow
Write-Host "Starting React frontend..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop both servers" -ForegroundColor Yellow
Write-Host ""

# Start Flask backend in background
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    python app.py
}

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# Start React frontend in current terminal
Push-Location frontend
npm start
Pop-Location

# When React stops, stop the backend too
Stop-Job $backendJob
Remove-Job $backendJob

Write-Host ""
Write-Host "Application stopped." -ForegroundColor Yellow
