#!/bin/bash

# Start SSH server in background
/usr/sbin/sshd -D &

# Start the Go application
exec /app/go-ai-gen
