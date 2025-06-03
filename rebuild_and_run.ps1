# Sign Language Extractor - Docker Rebuild and Run Script (PowerShell)
# This script stops, removes, rebuilds, and runs the Docker container

$ErrorActionPreference = "Continue"

Write-Host "Sign Language Extractor - Docker Rebuild Script" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Stop the current container (if running)
Write-Host ""
Write-Host "Step 1: Stopping existing container..." -ForegroundColor Yellow
try {
    $runningContainer = docker ps -q -f name=sign-app-instance 2>$null
    if ($runningContainer) {
        Write-Host "   Stopping sign-app-instance container..." -ForegroundColor White
        docker stop sign-app-instance
        Write-Host "   Container stopped successfully" -ForegroundColor Green
    } else {
        Write-Host "   No running container found with name 'sign-app-instance'" -ForegroundColor Blue
    }
} catch {
    Write-Host "   Error checking running containers: $($_.Exception.Message)" -ForegroundColor Yellow
}

# 2. Remove the stopped container
Write-Host ""
Write-Host "Step 2: Removing existing container..." -ForegroundColor Yellow
try {
    $existingContainer = docker ps -a -q -f name=sign-app-instance 2>$null
    if ($existingContainer) {
        Write-Host "   Removing sign-app-instance container..." -ForegroundColor White
        docker rm sign-app-instance
        Write-Host "   Container removed successfully" -ForegroundColor Green
    } else {
        Write-Host "   No container found with name 'sign-app-instance'" -ForegroundColor Blue
    }
} catch {
    Write-Host "   Error checking existing containers: $($_.Exception.Message)" -ForegroundColor Yellow
}

# 3. Rebuild the image (crucial!)
Write-Host ""
Write-Host "Step 3: Rebuilding Docker image..." -ForegroundColor Yellow
Write-Host "   Building sign-language-app image..." -ForegroundColor White
try {
    docker build -t sign-language-app .
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   Image built successfully" -ForegroundColor Green
    } else {
        Write-Host "   Error building image" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "   Error building image: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# 4. Run the newly built image
Write-Host ""
Write-Host "Step 4: Running new container..." -ForegroundColor Yellow
Write-Host "   Starting sign-app-instance container..." -ForegroundColor White
try {
    $currentPath = (Get-Location).Path
    docker run -d -p 5000:5000 -v "${currentPath}/uploads:/app/uploads" -v "${currentPath}/data:/app/data" --name sign-app-instance sign-language-app
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   Container started successfully" -ForegroundColor Green
    } else {
        Write-Host "   Error starting container" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "   Error starting container: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# 5. Display container status
Write-Host ""
Write-Host "Step 5: Container status..." -ForegroundColor Yellow
try {
    docker ps -f name=sign-app-instance --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}"
} catch {
    Write-Host "   Error getting container status: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "All done! Your Sign Language Extractor is now running." -ForegroundColor Green
Write-Host "Access the application at: http://localhost:5000" -ForegroundColor Cyan
Write-Host "To view logs: docker logs sign-app-instance" -ForegroundColor Gray
Write-Host "To stop: docker stop sign-app-instance" -ForegroundColor Gray 