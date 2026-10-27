#!/usr/bin/env bash
# ==============================================================================
# Script: monitor_ra.sh
# Purpose: Real-time telemetry monitor for Cyberpunk 2077 Ray Accelerator activity
# ==============================================================================
LOG_FILE="/tmp/cyberpunk_ra.log"

echo "======================================================================"
echo " [Cyberpunk-RA] Hardware Telemetry Stream"
echo " Monitoring target: $LOG_FILE"
echo " Press Ctrl+C to exit."
echo "======================================================================"

# Wait for log file if game is starting
while [ ! -f "$LOG_FILE" ]; do
    echo -n "."
    sleep 1
done
echo ""
echo "[+] Log file detected! Streaming active GPU telemetry..."
echo "----------------------------------------------------------------------"

tail -n 50 -f "$LOG_FILE"
