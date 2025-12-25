#!/bin/bash

# Make this script executable with: chmod +x run.sh

echo "Installing dependencies..."
go mod download

echo "Starting AI Model Interface..."
go run main.go