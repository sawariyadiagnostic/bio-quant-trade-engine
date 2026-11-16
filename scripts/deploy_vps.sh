#!/usr/bin/env bash
# ==============================================================================
# RAVANA v4.3 - Autonomous Live Paper Trading Daemon VPS Deployment Script
# Target OS: Ubuntu 22.04 / 24.04 LTS or Debian 12
# ==============================================================================
set -euo pipefail

echo "======================================================================"
echo "  RAVANA v4.3: DEPLOYING AUTONOMOUS PAPER TRADING SYSTEM TO VPS"
echo "======================================================================"

# 1. Detect current directory and executing user
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURRENT_USER="$(id -un)"
CURRENT_GROUP="$(id -gn)"
echo "[INFO] Installation Directory: ${INSTALL_DIR}"
echo "[INFO] Running as User: ${CURRENT_USER} (${CURRENT_GROUP})"

# 2. Update system & install OS packages
echo "[INFO] Updating package indices and installing OS dependencies..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    jq \
    htop \
    tmux \
    logrotate

# 3. Create Virtual Environment
VENV_DIR="${INSTALL_DIR}/venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo "[INFO] Creating Python virtual environment in ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

echo "[INFO] Activating virtual environment & installing Python dependencies..."
"${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/pip" install -r "${INSTALL_DIR}/requirements.txt"

# 4. Create necessary audit directories
mkdir -p "${INSTALL_DIR}/audit"
mkdir -p "${INSTALL_DIR}/data"
chmod -R 775 "${INSTALL_DIR}/audit"

# 5. Check or initialize .env file
ENV_FILE="${INSTALL_DIR}/.env"
if [ ! -f "${ENV_FILE}" ]; then
    echo "[INFO] Generating baseline .env configuration..."
    cat << 'EOF' > "${ENV_FILE}"
# RAVANA v4.3 VPS Configuration
GEMINI_API_KEY=""
SHOONYA_USER_ID=""
SHOONYA_PASSWORD=""
SHOONYA_API_KEY=""
SHOONYA_TOTP_SECRET=""
SHOONYA_VENDOR_CODE=""
SHOONYA_IMEI=""
RAVANA_PAPER_TRADING=true
RAVANA_ENABLE_DECISIONS=true

# Telegram Telemetry & Mobile C2 Bridge
TELEGRAM_BOT_TOKEN=""
TELEGRAM_CHAT_ID=""
EOF
    chmod 600 "${ENV_FILE}"
    echo "[WARNING] Created ${ENV_FILE}. Please edit it with your GEMINI_API_KEY, Shoonya credentials, and Telegram tokens."
fi

# 6. Tailscale Private Mesh VPN Installation & Zero-Exposure Firewall
echo "[INFO] Checking Tailscale installation..."
if ! command -v tailscale &>/dev/null; then
    echo "[INFO] Installing Tailscale Private Mesh VPN..."
    curl -fsSL https://tailscale.com/install.sh | sh
    echo "[INFO] Tailscale installed successfully. Run 'sudo tailscale up' to authenticate."
else
    echo "[INFO] Tailscale is already installed."
fi

echo "[INFO] Configuring UFW firewall for Zero Public Attack Surface..."
sudo apt-get install -y --no-install-recommends ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp comment 'SSH Public until Tailscale connected'
sudo ufw allow in on tailscale0 to any port 22 proto tcp comment 'SSH over Tailscale'
sudo ufw allow in on tailscale0 to any port 7860 proto tcp comment 'ADHD Cockpit over Tailscale'
sudo ufw --force enable
echo "[INFO] Firewall active: Public Port 7860 closed. Accessible strictly over private Tailscale mesh."

# 7. Configure systemd Service
SERVICE_FILE="/etc/systemd/system/ravana.service"
echo "[INFO] Configuring systemd service at ${SERVICE_FILE}..."
sudo bash -c "cat << EOF > ${SERVICE_FILE}
[Unit]
Description=RAVANA v4.3 Autonomous Live Paper Trading Daemon
After=network.target network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=simple
User=${CURRENT_USER}
Group=${CURRENT_GROUP}
WorkingDirectory=${INSTALL_DIR}
ExecStart=${VENV_DIR}/bin/python ${INSTALL_DIR}/scripts/run_live_autonomous_paper_daemon.py
Restart=always
RestartSec=10
TimeoutStopSec=30
WatchdogSec=300

EnvironmentFile=-${ENV_FILE}
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONIOENCODING=utf-8

LimitNOFILE=65535
MemoryHigh=2G
MemoryMax=3G

StandardOutput=append:${INSTALL_DIR}/audit/ravana_live_daemon.log
StandardError=append:${INSTALL_DIR}/audit/ravana_live_daemon.log

[Install]
WantedBy=multi-user.target
EOF"

# 8. Configure Logrotate
LOGROTATE_FILE="/etc/logrotate.d/ravana"
echo "[INFO] Configuring log rotation at ${LOGROTATE_FILE}..."
sudo bash -c "cat << EOF > ${LOGROTATE_FILE}
${INSTALL_DIR}/audit/*.log {
    daily
    rotate 14
    missingok
    notifempty
    compress
    delaycompress
    copytruncate
}
EOF"

# 9. Reload systemd daemon
echo "[INFO] Reloading systemd..."
sudo systemctl daemon-reload
sudo systemctl enable ravana.service

# 10. Create Helper Management Scripts
cat << 'EOF' > "${INSTALL_DIR}/status.sh"
#!/usr/bin/env bash
echo "=== RAVANA DAEMON SYSTEMD STATUS ==="
sudo systemctl status ravana.service --no-pager
echo ""
echo "=== RECENT LIVE DAEMON LOGS ==="
tail -n 25 audit/ravana_live_daemon.log 2>/dev/null || echo "No logs yet."
echo ""
echo "=== RECENT PAPER TRADES ==="
tail -n 15 audit/shoonya_paper_orders.json 2>/dev/null || echo "No paper orders yet."
EOF
chmod +x "${INSTALL_DIR}/status.sh"

cat << 'EOF' > "${INSTALL_DIR}/start.sh"
#!/usr/bin/env bash
sudo systemctl start ravana.service
echo "RAVANA Daemon started. Monitor with: ./status.sh or journalctl -u ravana -f"
EOF
chmod +x "${INSTALL_DIR}/start.sh"

cat << 'EOF' > "${INSTALL_DIR}/stop.sh"
#!/usr/bin/env bash
sudo systemctl stop ravana.service
echo "RAVANA Daemon stopped."
EOF
chmod +x "${INSTALL_DIR}/stop.sh"

echo "======================================================================"
echo "  RAVANA v4.3 VPS DEPLOYMENT COMPLETED SUCCESSFULLY!"
echo "======================================================================"
echo "Next steps:"
echo "1. Verify .env with your credentials:  nano ${ENV_FILE}"
echo "2. Start the daemon:                   ./start.sh (or sudo systemctl start ravana)"
echo "3. Check live status & logs:           ./status.sh"
echo "4. Follow live daemon in real time:    journalctl -u ravana -f"
echo "======================================================================"
