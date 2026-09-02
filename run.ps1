# Enterprise HR AI — Quick Launcher
# Run this from ANY location: .\run.ps1
# Usage:
#   .\run.ps1 dashboard   → starts Streamlit on :8501
#   .\run.ps1 api         → starts FastAPI on :8000
#   .\run.ps1 both        → starts both in separate windows
#   .\run.ps1 setup       → runs setup.py (first-time init)
#   .\run.ps1 notebooks   → opens notebooks folder

$ROOT = "c:\Users\Rahul kumar goyal\OneDrive\Documents\enterprise_hr_ai"

function Start-Dashboard {
    Write-Host "`n🧠 Starting Streamlit Dashboard on http://localhost:8501 ...`n" -ForegroundColor Cyan
    Set-Location $ROOT
    python -m streamlit run app/frontend/streamlit_app.py --server.port 8501
}

function Start-API {
    Write-Host "`n⚡ Starting FastAPI on http://localhost:8000 ...`n" -ForegroundColor Green
    Set-Location $ROOT
    python -m uvicorn app.backend.main:app --reload --port 8000
}

function Start-Both {
    Write-Host "`n🚀 Starting both Dashboard + API in separate windows ...`n" -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ROOT'; python -m streamlit run app/frontend/streamlit_app.py --server.port 8501"
    Start-Sleep 2
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ROOT'; python -m uvicorn app.backend.main:app --reload --port 8000"
    Write-Host "✔ Dashboard: http://localhost:8501" -ForegroundColor Cyan
    Write-Host "✔ API Docs:  http://localhost:8000/docs" -ForegroundColor Green
}

function Run-Setup {
    Write-Host "`n⚙️  Running setup (pipeline + model training) ...`n" -ForegroundColor Magenta
    Set-Location $ROOT
    python setup.py
}

$cmd = $args[0]
switch ($cmd) {
    "dashboard"  { Start-Dashboard }
    "api"        { Start-API }
    "both"       { Start-Both }
    "setup"      { Run-Setup }
    default {
        Write-Host "`n📖 Enterprise HR AI — Launcher" -ForegroundColor Cyan
        Write-Host "Usage:" -ForegroundColor White
        Write-Host "  .\run.ps1 dashboard  → Streamlit Dashboard (:8501)" -ForegroundColor Yellow
        Write-Host "  .\run.ps1 api        → FastAPI Backend (:8000)" -ForegroundColor Yellow
        Write-Host "  .\run.ps1 both       → Both in separate windows" -ForegroundColor Yellow
        Write-Host "  .\run.ps1 setup      → First-time setup`n" -ForegroundColor Yellow
    }
}
