#!/bin/bash

# Source Nix environment
source /home/runner/.bashrc

# Start FastAPI backend
$PYTHONBIN /home/runner/CloudMind-AcademyBackend-1/main.py &

# Wait for FastAPI server to be ready
while ! curl -s http://localhost:8000/health > /dev/null; do
  sleep 1
done

# Run the certificate test
$PYTHONBIN /home/runner/CloudMind-AcademyBackend-1/test_certificates.py

# Keep the script running
wait

# Navigate to frontend directory and start Next.js
cd /home/runner/CloudMind-AcademyBackend-1/frontend && npm run dev &

# Navigate to mobile app directory and start React Native
cd /home/runner/CloudMind-AcademyBackend-1/mobile_app && npm start &

# Wait for all background processes to finish
wait