#!/bin/bash

# Start FastAPI backend
python main.py &

# Wait for FastAPI server to be ready
while ! curl -s http://localhost:8000/health > /dev/null; do
  sleep 1
done

# Run the certificate test
python test_certificates.py

# Keep the script running
wait

# Navigate to frontend directory and start Next.js
cd frontend && npm run dev &

# Navigate to mobile app directory and start React Native
cd ../mobile_app && npm start &

# Wait for all background processes to finish
wait