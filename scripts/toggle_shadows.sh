#!/usr/bin/env bash
# ==============================================================================
# Script: toggle_shadows.sh
# Purpose: Live toggle for shadow map bypassing in Cyberpunk 2077
# ==============================================================================
TRIGGER_FILE="/tmp/cyberpunk_ra_skip_shadows"

ACTION="${1:-status}"

case "$ACTION" in
    on|enable)
        touch "$TRIGGER_FILE"
        echo "[+] Shadow map bypass: ACTIVATED (bypassing square shadow cascades live)"
        ;;
    off|disable)
        rm -f "$TRIGGER_FILE"
        echo "[-] Shadow map bypass: DEACTIVATED (normal raster shadow rendering restored)"
        ;;
    status)
        if [ -f "$TRIGGER_FILE" ]; then
            echo "[*] Current state: Shadow map bypass is ON (bypassing)"
        else
            echo "[*] Current state: Shadow map bypass is OFF (normal rendering)"
        fi
        ;;
    *)
        echo "Usage: $0 [on|off|status]"
        exit 1
        ;;
esac
