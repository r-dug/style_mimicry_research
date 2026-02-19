#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# setup_glaze_vm.sh — Set up a local Windows QEMU/KVM VM for Glaze
#
# Glaze has known fp arithmetic issues on certain NVIDIA GPUs (e.g.
# GTX 1660), so this VM runs Glaze on CPU only.
#
# Usage:
#   chmod +x setup_glaze_vm.sh
#   ./setup_glaze_vm.sh          # first-time setup
#   ./setup_glaze_vm.sh start    # start the VM
#   ./setup_glaze_vm.sh stop     # shut down the VM
#   ./setup_glaze_vm.sh status   # check if running
#
# Image transfer:
#   Put input images in  vm/shared/input/
#   Glazed outputs go in vm/shared/output/
#   Inside Windows, access via \\10.0.2.4\qemu in File Explorer
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VM_DIR="${SCRIPT_DIR}/vm"
DISK_IMG="${VM_DIR}/windows.qcow2"
SHARED_DIR="${VM_DIR}/shared"
DISK_SIZE="64G"
RAM="8G"
CPUS="4"
VNC_DISPLAY="0"                     # VNC on localhost:5900

# ── helpers ──────────────────────────────────────────────────────────

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

check_deps() {
    local missing=()
    for cmd in qemu-system-x86_64 qemu-img; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing: ${missing[*]}
  Install with:  sudo apt install qemu-system-x86 qemu-utils"
    fi

    if [[ -e /dev/kvm ]] && [[ -r /dev/kvm ]] && [[ -w /dev/kvm ]]; then
        info "KVM acceleration available."
        KVM_FLAG="-enable-kvm"
    elif [[ -e /dev/kvm ]]; then
        warn "/dev/kvm exists but you lack permissions."
        warn "Fix with: sudo usermod -aG kvm \$USER  (then re-login)"
        KVM_FLAG=""
    else
        warn "No KVM — VM will run in pure emulation (slow but functional)."
        KVM_FLAG=""
    fi
}

# ── ISO handling ─────────────────────────────────────────────────────

find_iso() {
    local iso_dir="${VM_DIR}/iso"
    mkdir -p "$iso_dir"

    WIN_ISO=$(find "$iso_dir" -maxdepth 1 -name "*.iso" ! -name "virtio*" -print -quit 2>/dev/null || true)
    if [[ -n "$WIN_ISO" ]]; then
        info "Found Windows ISO: $(basename "$WIN_ISO")"
        return 0
    fi

    echo ""
    info "No Windows ISO found in: ${iso_dir}/"
    echo ""
    echo "  Download a free evaluation ISO and place it there:"
    echo ""
    echo "  Windows 10 (90-day eval, recommended for VMs):"
    echo "    https://www.microsoft.com/en-us/evalcenter/download-windows-10-enterprise"
    echo ""
    echo "  Windows 11 (90-day eval):"
    echo "    https://www.microsoft.com/en-us/evalcenter/download-windows-11-enterprise"
    echo ""
    echo "  Then re-run: $0"
    echo ""
    return 1
}

# ── VirtIO drivers ───────────────────────────────────────────────────

get_virtio() {
    VIRTIO_ISO="${VM_DIR}/iso/virtio-win.iso"
    if [[ -f "$VIRTIO_ISO" ]]; then
        info "VirtIO drivers present."
        return
    fi

    info "Downloading VirtIO drivers (better disk/network speed)..."
    if wget -q --show-progress -O "$VIRTIO_ISO" \
        "https://fedorapeople.org/groups/virt/virtio-win/direct-downloads/stable-virtio/virtio-win.iso" 2>/dev/null; then
        info "VirtIO drivers downloaded."
    else
        warn "VirtIO download failed — VM will still work with emulated devices."
        VIRTIO_ISO=""
    fi
}

# ── disk image ───────────────────────────────────────────────────────

create_disk() {
    if [[ -f "$DISK_IMG" ]]; then
        info "Disk image exists ($(du -h "$DISK_IMG" | cut -f1) used)."
        return
    fi
    info "Creating ${DISK_SIZE} virtual disk (thin-provisioned)..."
    qemu-img create -f qcow2 "$DISK_IMG" "$DISK_SIZE"
}

# ── start VM ─────────────────────────────────────────────────────────

start_vm() {
    if [[ -f "${VM_DIR}/qemu.pid" ]] && kill -0 "$(cat "${VM_DIR}/qemu.pid")" 2>/dev/null; then
        info "VM is already running (PID: $(cat "${VM_DIR}/qemu.pid"))."
        info "Connect with: vncviewer localhost:5900"
        return
    fi

    mkdir -p "$SHARED_DIR/input" "$SHARED_DIR/output"

    # Build args
    local boot_args=""
    if [[ ! -f "${VM_DIR}/.installed" ]]; then
        find_iso || exit 1
        boot_args="-cdrom ${WIN_ISO} -boot d"
    fi

    local virtio_args=""
    local virtio_path="${VM_DIR}/iso/virtio-win.iso"
    if [[ -f "$virtio_path" ]]; then
        virtio_args="-drive file=${virtio_path},media=cdrom,index=1"
    fi

    # CPU flag: use host passthrough with KVM, otherwise basic qemu64
    local cpu_flag=""
    if [[ -n "${KVM_FLAG:-}" ]]; then
        cpu_flag="-cpu host"
    fi

    info "Starting VM..."
    info "  RAM: ${RAM}  |  CPUs: ${CPUS}  |  KVM: ${KVM_FLAG:-off}"
    info "  VNC: localhost:5900"
    echo ""

    # shellcheck disable=SC2086
    qemu-system-x86_64 \
        ${KVM_FLAG:-} \
        ${cpu_flag} \
        -m "$RAM" \
        -smp "$CPUS" \
        -drive file="$DISK_IMG",format=qcow2,if=virtio \
        ${boot_args} \
        ${virtio_args} \
        -nic user,model=virtio-net-pci,hostfwd=tcp::3389-:3389,smb="$SHARED_DIR" \
        -vga qxl \
        -vnc :${VNC_DISPLAY} \
        -usb -device usb-tablet \
        -daemonize \
        -pidfile "${VM_DIR}/qemu.pid" \
    && info "VM started. PID: $(cat "${VM_DIR}/qemu.pid")" \
    || error "Failed to start VM. Check output above."

    if [[ ! -f "${VM_DIR}/.installed" ]]; then
        echo ""
        info "══════════════════════════════════════════════════════════"
        info " INSTALLATION STEPS:"
        info "  1. Connect:  vncviewer localhost:5900"
        info "  2. Install Windows from the ISO"
        info "  3. Install VirtIO drivers from the second CD drive"
        info "     (Device Manager → update drivers → browse CD)"
        info "  4. Download Glaze from https://glaze.cs.uchicago.edu/"
        info "  5. Access shared folder in Windows Explorer:"
        info "       \\\\10.0.2.4\\qemu"
        info "     - input/  → put images to protect here"
        info "     - output/ → save glazed images here"
        info "  6. Mark installation done:"
        info "       touch ${VM_DIR}/.installed"
        info "══════════════════════════════════════════════════════════"
    else
        echo ""
        info "Connect:  vncviewer localhost:5900"
        info "Shared folder in Windows: \\\\10.0.2.4\\qemu"
    fi
}

# ── stop VM ──────────────────────────────────────────────────────────

stop_vm() {
    local pidfile="${VM_DIR}/qemu.pid"
    if [[ ! -f "$pidfile" ]]; then
        info "VM is not running (no PID file)."
        return
    fi

    local pid
    pid=$(cat "$pidfile")
    if ! kill -0 "$pid" 2>/dev/null; then
        info "VM not running (stale PID file)."
        rm -f "$pidfile"
        return
    fi

    info "Shutting down VM (PID: $pid)..."
    kill "$pid"
    for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done

    if kill -0 "$pid" 2>/dev/null; then
        warn "Graceful shutdown timed out — forcing..."
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
    info "VM stopped."
}

# ── main ─────────────────────────────────────────────────────────────

main() {
    local action="${1:-setup}"

    case "$action" in
        setup)
            info "Setting up Glaze VM..."
            check_deps
            mkdir -p "$VM_DIR/iso"
            find_iso || exit 0   # exits with download instructions
            get_virtio
            create_disk
            echo ""
            info "Setup complete. Run: $0 start"
            ;;
        start)
            check_deps
            start_vm
            ;;
        stop)
            stop_vm
            ;;
        status)
            if [[ -f "${VM_DIR}/qemu.pid" ]] && kill -0 "$(cat "${VM_DIR}/qemu.pid")" 2>/dev/null; then
                info "VM running (PID: $(cat "${VM_DIR}/qemu.pid"))"
                info "Connect: vncviewer localhost:5900"
            else
                info "VM is not running."
            fi
            ;;
        *)
            echo "Usage: $0 {setup|start|stop|status}"
            exit 1
            ;;
    esac
}

main "$@"
