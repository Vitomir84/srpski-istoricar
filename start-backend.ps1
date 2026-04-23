# Start Backend Only
# This script starts only the FastAPI backend server

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   Starting FastAPI Backend Server" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend will run on: http://localhost:5000" -ForegroundColor Green
Write-Host "API endpoints will be available at: http://localhost:5000/api/*" -ForegroundColor Green
Write-Host "API documentation (Swagger): http://localhost:5000/docs" -ForegroundColor Green
Write-Host "Alternative docs (ReDoc): http://localhost:5000/redoc" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python app.py
