# RAVANA v4.3: Live Autonomous Paper Trading Daemon & VPS Deployment Master Plan

## 1. Executive Summary & Objective

The objective is to establish **100% autonomous, zero-intervention live paper trading** for the entire trading week, starting immediately with today's afternoon session (Monday, 16 November 2026, 12:05 to 15:30 IST) across the 10 Sector BeES ETF universe.

The daemon will run headless on a dedicated VPS (or local background process), stream live 5-minute candles directly from NSE, dispatch real-time decision requests to Google Decisions (`gemini-decision-1`), parse responses with zero latency, execute risk-managed orders via Shoonya Broker Gateway, and manage the full intraday and overnight lifecycle without human intervention.

```mermaid
flowchart TD
    subgraph MarketData [Live Market Ingestion - Every 5m Boundary]
        NSE[NSE Live Market 5m Candles] --> Feeder[Live Candle Ingestion Engine]
        Feeder --> Tech[Technical Vector Engine: VWAP, ATR, OFI, VR20]
    end

    subgraph Signals [Signal Generation & Crown Contention]
        Tech --> Triggers[Quantitative Triggers: ORB, Drift, Sweep, Europe Open, iNAV]
        Triggers --> Contention[UniverseAlphaCrown Contention Sorter]
    end

    subgraph GoogleDecisions [Google Decisions Model Integration]
        Contention --> Client[GeminiDecisionClient (gemini-decision-1)]
        Client --> Veto[DecisionVetoGuard: False Breakout & Dead Money Checks]
        Veto --> ProbCheck{Conviction >= 0.70 & Veto Passed?}
    end

    subgraph Execution [Autonomous Execution & Position Management]
        ProbCheck -- Yes --> Sizing[Geometric Compounding: 0.20% Equity Risk]
        Sizing --> Gateway[ShoonyaBrokerGateway: Paper Market Order]
        Gateway --> Ledger[(ACID SQLite State Ledger: WAL Mode)]
        ProbCheck -- No (Vetoed) --> CashYield[Hold LiquidBeES: 6.5% Yield]
    end

    subgraph Lifecycle [Session Risk & Lifecycle Rules]
        Gateway --> Monitor[Real-Time Position Tracker]
        Monitor --> BR[Breakeven Ratchet at +0.50 ATR]
        Monitor --> ML[11:30 AM Morning Lock: Bank Morning Gains]
        Monitor --> PS[14:45 PM Pre-Dump Shield: Exit Before 15:00 Panic]
        Monitor --> ON[15:15 PM All-Weather CNC Overnight Rotation]
    end
```

---

## 2. Real-Time Candle Flow & Google Decisions Integration

### A. 5-Minute Candle Clock Synchronization
The daemon continuously sleeps until the exact boundary of each 5-minute bar ($T+5$ seconds for exchange candle finalization):
$$\text{Execution Epochs: } 09:20:05, 09:25:05, \dots, 11:30:05, \dots, 13:15:05, \dots, 15:15:05, 15:25:05$$

### B. Decision Request Cadence
To respect Google Decisions rate limits (120 RPM) while ensuring instant execution:
1. **Candidate Pre-Filtering:** Only signals passing strict mathematical filters ($VR_{20} \ge 0.95$, valid VWAP alignment) generate a Google Decisions request (averaging 1 to 3 requests per 5-minute bar).
2. **Instant Asynchronous Evaluation:** The request is sent to `gemini-decision-1` via `GeminiDecisionClient`.
3. **Structured Response Parsing:**
   - `is_false_breakout_trap` (NOUL: Probability $< 0.72$)
   - `is_dead_money_risk` (NOUL: Probability $< 0.75$)
4. **Order Placement:** If approved, an order is dispatched to Shoonya Paper Gateway in under $20\text{ ms}$, logging full transaction metadata into `audit/shoonya_paper_orders.json`.

---

## 3. Session Windows & Automated Lifecycle Rules

| Session / Window | Time Range (IST) | Action & Trigger Universe | Exit Rule |
| :--- | :--- | :--- | :--- |
| **Morning Open** | 09:15 – 09:35 | Data stabilization, gap calculation | No trades |
| **Morning Impulse** | 09:35 – 10:45 | Morning ORB, Sector Drift, Stop Sweep, iNAV | Target $+1.5\text{ ATR}$ or Breakeven ratchet |
| **Morning Lock** | **11:30** | **Mandatory MIS Square-off** | Bank morning alpha; 100% flat cash |
| **Midday Lull** | 11:30 – 13:00 | No entries (idle cash earns 6.5% yield) | Flat book (zero chop risk) |
| **Afternoon European** | 13:15 – 13:45 | European Open Momentum (`ITBEES`, `NIFTYBEES`) | Target $+1.5\text{ ATR}$ or Breakeven ratchet |
| **Pre-Dump Shield** | **14:45** | **Mandatory MIS Square-off** | Shield from 15:00 retail auto-liquidation |
| **Overnight Rotation** | 15:10 – 15:25 | All-Weather Bull Sector / Bear Hedge / LiquidBeES | CNC buy; hold until 09:15 next morning |

---

## 4. Production VPS Deployment Architecture

To run continuously 24/7 on a Linux (Ubuntu 22.04 / 24.04 LTS) or Windows VPS:

### A. Directory Structure on VPS
```text
/opt/trade_clean/               # Root repository directory
├── .env                        # Live credentials (Shoonya + Gemini API Key)
├── venv/                       # Dedicated Python 3.10+ virtual environment
├── ravana/                     # Core sovereign trading package
│   ├── daemon/                 # Autonomous daemon modules
│   ├── decisions/              # Google Decisions client, schema, veto guard
│   └── execution/              # Shoonya adapter & broker gateway
├── audit/                      # ACID logs & ledgers (git-ignored)
│   ├── ravana_state_ledger.db  # SQLite WAL state database
│   ├── shoonya_paper_orders.json
│   └── ravana_live_daemon.log
└── scripts/
    └── run_live_autonomous_paper_daemon.py
```

### B. Linux Systemd Service (`/etc/systemd/system/ravana.service`)
```ini
[Unit]
Description=RAVANA v4.3 Autonomous Live Paper Trading Daemon
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/trade_clean
ExecStart=/opt/trade_clean/venv/bin/python scripts/run_live_autonomous_paper_daemon.py
Restart=always
RestartSec=10
EnvironmentFile=/opt/trade_clean/.env
StandardOutput=append:/opt/trade_clean/audit/ravana_live_daemon.log
StandardError=append:/opt/trade_clean/audit/ravana_live_daemon.log

[Install]
WantedBy=multi-user.target
```

### C. Self-Healing & Reboot Recovery
* **Crash Proofing:** If the VPS reboots or network drops, Systemd automatically restarts the process after 10 seconds.
* **11:32 AM Reboot Protocol:** On boot, `startup_and_reconcile()` queries `audit/ravana_state_ledger.db` (SQLite WAL mode), reconstructs open positions and trailing stops from persistent storage, and resumes execution seamlessly.

---

## 5. Implementation & Execution Steps for Today

1. **Build `scripts/run_live_autonomous_paper_daemon.py`:**
   - Incorporate the live 5-minute timer loop.
   - Synchronize with live NSE market candles.
   - Dispatch live requests to `GeminiDecisionClient`.
   - Manage the afternoon session (13:15 to 14:45) and overnight close (15:15).
2. **Execute Full Test Suite Verification:** Run `python -m unittest discover -s ravana/tests -p "test_*.py"` to ensure 100% green tests before deployment.
3. **Launch the Live Daemon for Today's Market:** Launch as a persistent background daemon for today's afternoon session (12:05 to 15:30 IST) and monitor live execution logs.
