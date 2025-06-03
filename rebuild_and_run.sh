#!/bin/bash

# Sign Language Extractor - Docker Rebuild and Run Script
# This script stops, removes, rebuilds, and runs the Docker container

set -e  # Exit on any error

echo "🐳 Sign Language Extractor - Docker Rebuild Script"
echo "=================================================="

# 1. Stop the current container (if running)
echo "📋 Step 1: Stopping existing container..."
if docker ps -q -f name=sign-app-instance | grep -q .; then
    echo "   Stopping sign-app-instance container..."
    docker stop sign-app-instance
    echo "   ✅ Container stopped successfully"
else
    echo "   ℹ️  No running container found with name 'sign-app-instance'"
fi

# 2. Remove the stopped container
echo ""
echo "📋 Step 2: Removing existing container..."
if docker ps -a -q -f name=sign-app-instance | grep -q .; then
    echo "   Removing sign-app-instance container..."
    docker rm sign-app-instance
    echo "   ✅ Container removed successfully"
else
    echo "   ℹ️  No container found with name 'sign-app-instance'"
fi

# 3. Rebuild the image (crucial!)
echo ""
echo "📋 Step 3: Rebuilding Docker image..."
echo "   Building sign-language-app image..."
docker build -t sign-language-app .
echo "   ✅ Image built successfully"

# 4. Run the newly built image
echo ""
echo "📋 Step 4: Running new container..."
echo "   Starting sign-app-instance container..."
docker run -d \
    -p 5000:5000 \
    -v "$(pwd)/uploads:/app/uploads" \
    -v "$(pwd)/data:/app/data" \
    --name sign-app-instance \
    sign-language-app

echo "   ✅ Container started successfully"

# 5. Display container status
echo ""
echo "📋 Step 5: Container status..."
docker ps -f name=sign-app-instance --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🎉 All done! Your Sign Language Extractor is now running."
echo "📱 Access the application at: http://localhost:5000"
echo "🔍 To view logs: docker logs sign-app-instance"
echo "🛑 To stop: docker stop sign-app-instance" 