#!/bin/bash

if command -v python &>/dev/null; then
    nohup python src/webapp.py &
elif command -v python3 &>/dev/null; then
    nohup python3 src/webapp.py &
else
    echo "Python is not installed."
    exit 1
fi

