#!/bin/bash

if command -v python &>/dev/null; then
    nohup python mlops_proj/src/webapp.py &
    HOST=0.0.0.0 PORT=8080 npm run start
elif command -v python3 &>/dev/null; then
    nohup python3 mlops_proj/src/webapp.py
    HOST=0.0.0.0 PORT=8080 npm run start
else
    echo "Error: Python is not installed."
    exit 1
fi
