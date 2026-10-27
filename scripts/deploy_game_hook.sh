#!/usr/bin/env bash
# ==============================================================================
# Script: deploy_game_hook.sh
# Purpose: Deploy or restore instrumented VKD3D-Proton Ray Accelerator binaries
# Target: Proton-CachyOS & Cyberpunk 2077 Prefix
# ==============================================================================
set -euo pipefail

VKD3D_BUILD_DIR="${VKD3D_BUILD_DIR:-$HOME/Desktop/vkd3d-proton/build.x64}"
PROTON_VKD3D_DIR="${PROTON_VKD3D_DIR:-$HOME/.local/share/Steam/compatibilitytools.d/Proton-CachyOS Latest/files/lib/wine/vkd3d-proton/x86_64-windows}"
PREFIX_SYSTEM32="${PREFIX_SYSTEM32:-$HOME/wineprefix/drive_c/windows/system32}"

ACTION="${1:-deploy}"

if [ "$ACTION" == "restore" ]; then
    echo "[*] Restoring original VKD3D-Proton libraries..."
    if [ -f "$PROTON_VKD3D_DIR/d3d12core.dll.orig" ]; then
        cp -fv "$PROTON_VKD3D_DIR/d3d12core.dll.orig" "$PROTON_VKD3D_DIR/d3d12core.dll"
        cp -fv "$PROTON_VKD3D_DIR/d3d12.dll.orig" "$PROTON_VKD3D_DIR/d3d12.dll"
        echo "[+] Restored Proton-CachyOS vkd3d-proton binaries."
    fi
    if [ -f "$PREFIX_SYSTEM32/d3d12core.dll.orig" ]; then
        cp -fv "$PREFIX_SYSTEM32/d3d12core.dll.orig" "$PREFIX_SYSTEM32/d3d12core.dll"
        cp -fv "$PREFIX_SYSTEM32/d3d12.dll.orig" "$PREFIX_SYSTEM32/d3d12.dll"
        echo "[+] Restored Cyberpunk wineprefix system32 binaries."
    fi
    echo "[+] Restore complete."
    exit 0
fi

if [ "$ACTION" == "deploy" ]; then
    echo "[*] Deploying custom VKD3D-Proton Ray Accelerator telemetry binaries..."
    
    # 1. Verify build artifacts
    if [ ! -f "$VKD3D_BUILD_DIR/libs/d3d12core/d3d12core.dll" ] || [ ! -f "$VKD3D_BUILD_DIR/libs/d3d12/d3d12.dll" ]; then
        echo "[-] Build artifacts missing in $VKD3D_BUILD_DIR. Please build vkd3d-proton first."
        exit 1
    fi

    # 2. Backup original Proton-CachyOS files if not already backed up
    if [ -d "$PROTON_VKD3D_DIR" ]; then
        if [ ! -f "$PROTON_VKD3D_DIR/d3d12core.dll.orig" ] && [ -f "$PROTON_VKD3D_DIR/d3d12core.dll" ]; then
            cp -v "$PROTON_VKD3D_DIR/d3d12core.dll" "$PROTON_VKD3D_DIR/d3d12core.dll.orig"
            cp -v "$PROTON_VKD3D_DIR/d3d12.dll" "$PROTON_VKD3D_DIR/d3d12.dll.orig"
            echo "[+] Created backup of original Proton-CachyOS binaries."
        fi
        cp -fv "$VKD3D_BUILD_DIR/libs/d3d12core/d3d12core.dll" "$PROTON_VKD3D_DIR/d3d12core.dll"
        cp -fv "$VKD3D_BUILD_DIR/libs/d3d12/d3d12.dll" "$PROTON_VKD3D_DIR/d3d12.dll"
        echo "[+] Installed custom binaries to Proton-CachyOS."
    fi

    # 3. Backup and update Wineprefix system32
    if [ -d "$PREFIX_SYSTEM32" ]; then
        if [ ! -f "$PREFIX_SYSTEM32/d3d12core.dll.orig" ] && [ -f "$PREFIX_SYSTEM32/d3d12core.dll" ]; then
            cp -v "$PREFIX_SYSTEM32/d3d12core.dll" "$PREFIX_SYSTEM32/d3d12core.dll.orig"
            cp -v "$PREFIX_SYSTEM32/d3d12.dll" "$PREFIX_SYSTEM32/d3d12.dll.orig"
            echo "[+] Created backup of wineprefix system32 binaries."
        fi
        cp -fv "$VKD3D_BUILD_DIR/libs/d3d12core/d3d12core.dll" "$PREFIX_SYSTEM32/d3d12core.dll"
        cp -fv "$VKD3D_BUILD_DIR/libs/d3d12/d3d12.dll" "$PREFIX_SYSTEM32/d3d12.dll"
        echo "[+] Installed custom binaries to Cyberpunk wineprefix system32."
    fi

    echo "[+] Deployment successful. Telemetry will write to /tmp/cyberpunk_ra.log when game runs."
    exit 0
fi

echo "Unknown action: $ACTION. Usage: $0 [deploy|restore]"
exit 1
