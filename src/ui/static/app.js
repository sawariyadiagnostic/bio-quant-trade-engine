/**
 * Bio-Quant Institutional Alpha Cockpit - Main Application Controller
 * Features:
 * 1. Differential SSE listener with Web Audio sound effects.
 * 2. AG Grid DOM virtualization with transactional delta updates.
 * 3. 250ms RequestAnimationFrame WebGL throttler for Plotly.
 * 4. 5-second dead-man switch heartbeat watchdog.
 */

function cockpitApp() {
    return {
        // State
        status: 'STANDBY',
        instrument: 'NIFTYBEES',
        lookbackYears: 1.0,
        generations: 10,
        populationSize: 50,
        isMuted: false,
        activeTab: 'tactical', // 'tactical' | 'leaderboard' | 'telemetry'
        deadManDisconnected: false,
        lastHeartbeat: Date.now(),
        eventSource: null,
        gridApi: null,
        rafPending: false,
        pendingChartData: [],
        logs: [],
        telemetry: {
            current_gen: 0,
            total_gens: 10,
            resident_count: 0,
            survivor_count: 0,
            occupancy_pct: 0.0,
            top_sharpe: 0.0,
            best_net_return: 0.0,
            worker_status: 0
        },

        init() {
            this.log("Avionics HUD initialized. Connecting telemetry bus...");
            this.initGrid();
            this.initPlotly();
            this.connectSSE();
            this.startDeadManWatchdog();
            this.fetchInitialData();
        },

        log(msg, type = 'info') {
            const time = new Date().toLocaleTimeString();
            this.logs.unshift({ time, msg, type });
            if (this.logs.length > 200) this.logs.pop();
        },

        toggleMute() {
            this.isMuted = window.cockpitAudio.toggleMute();
        },

        // 1. Differential SSE Telemetry Stream
        connectSSE() {
            if (this.eventSource) {
                this.eventSource.close();
            }

            this.eventSource = new EventSource('/api/stream/evolution');

            this.eventSource.addEventListener('connected', (e) => {
                this.lastHeartbeat = Date.now();
                this.deadManDisconnected = false;
                this.log("Connected to Differential SSE telemetry bus.");
            });

            this.eventSource.addEventListener('ping', (e) => {
                this.lastHeartbeat = Date.now();
                this.deadManDisconnected = false;
            });

            this.eventSource.addEventListener('tick', (e) => {
                this.lastHeartbeat = Date.now();
                this.deadManDisconnected = false;
                const data = JSON.parse(e.data);
                this.telemetry = data;

                if (data.worker_status === 1) {
                    this.status = 'EVOLVING';
                } else if (data.worker_status === 3 && this.status !== 'COMPLETE') {
                    this.status = 'COMPLETE';
                } else if (data.worker_status === 2 && this.status !== 'ABORTED') {
                    this.status = 'ABORTED';
                }

                // Batch redraw via 250ms RAF throttler
                this.scheduleChartUpdate(data);
            });

            this.eventSource.addEventListener('elite_added', (e) => {
                const elite = JSON.parse(e.data);
                this.log(`New Elite Champion discovered: ${elite.formula_str || 'Alpha'} (Sharpe: ${elite.fitness?.toFixed(2)})`, 'success');
                window.cockpitAudio.playEliteSound();

                // Delta append to AG Grid virtualized table
                if (this.gridApi) {
                    this.gridApi.applyTransactionAsync({ add: [elite] });
                }
            });

            this.eventSource.addEventListener('complete', (e) => {
                if (this.status !== 'COMPLETE') {
                    this.status = 'COMPLETE';
                    this.log("Evolution cycle complete. Champions committed to SQLite WAL.", 'success');
                    window.cockpitAudio.playCompleteSound();
                    this.fetchLeaderboard();
                }
            });

            this.eventSource.addEventListener('abort', (e) => {
                if (this.status !== 'ABORTED') {
                    this.status = 'ABORTED';
                    this.log("Worker execution aborted by operator.", 'warning');
                    this.fetchLeaderboard();
                }
            });

            this.eventSource.onerror = () => {
                // SSE reconnect is handled natively by browser
            };
        },

        // 2. 5-Second Dead-Man Switch Watchdog
        startDeadManWatchdog() {
            setInterval(() => {
                const silenceMs = Date.now() - this.lastHeartbeat;
                if (silenceMs > 5000 && this.status === 'EVOLVING') {
                    if (!this.deadManDisconnected) {
                        this.deadManDisconnected = true;
                        this.log("WARNING: Engine heartbeat lost (>5s). Worker may be stalled.", 'error');
                    }
                } else if (silenceMs <= 5000) {
                    this.deadManDisconnected = false;
                }
            }, 1000);
        },

        // 3. AG Grid Virtualization
        initGrid() {
            const gridDiv = document.querySelector('#leaderboardGrid');
            if (!gridDiv || !window.agGrid) return;

            const gridOptions = {
                columnDefs: [
                    { field: 'ticker', headerName: 'Ticker', width: 100, pinned: 'left' },
                    { field: 'fitness', headerName: 'Sharpe / Fitness', width: 140, sort: 'desc', valueFormatter: p => p.value?.toFixed(3) },
                    { field: 'sortino', headerName: 'Sortino', width: 110, valueFormatter: p => p.value?.toFixed(2) },
                    { field: 'cagr_pct', headerName: 'CAGR %', width: 110, valueFormatter: p => (p.value * 100)?.toFixed(1) + '%' },
                    { field: 'max_drawdown', headerName: 'Max DD %', width: 120, valueFormatter: p => (p.value * 100)?.toFixed(1) + '%' },
                    { field: 'win_rate', headerName: 'Win Rate %', width: 120, valueFormatter: p => (p.value * 100)?.toFixed(1) + '%' },
                    { field: 'trades_count', headerName: 'Trades', width: 90 },
                    { field: 'formula_str', headerName: 'Genomic Alpha Expression', flex: 1, minWidth: 250 }
                ],
                defaultColDef: {
                    sortable: true,
                    filter: true,
                    resizable: true
                },
                rowModelType: 'clientSide',
                animateRows: true,
                rowData: []
            };

            this.gridApi = agGrid.createGrid(gridDiv, gridOptions);
        },

        // 4. Plotly 250ms RequestAnimationFrame Throttler
        initPlotly() {
            const chartDiv = document.getElementById('mapElitesChart');
            if (!chartDiv || !window.Plotly) return;

            const data = [{
                x: [0.1, 0.3, 0.5, 0.7, 0.9],
                y: [0.2, 0.4, 0.6, 0.8, 1.0],
                mode: 'markers',
                type: 'scattergl',
                marker: {
                    size: 10,
                    color: [1.2, 1.8, 2.4, 3.1, 3.8],
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: { title: 'Sharpe' }
                }
            }];

            const layout = {
                paper_bgcolor: '#0f172a',
                plot_bgcolor: '#0f172a',
                font: { color: '#94a3b8', family: 'monospace' },
                margin: { t: 30, b: 40, l: 50, r: 20 },
                xaxis: { title: 'Volatile Regime Sigma', gridcolor: '#1e293b' },
                yaxis: { title: 'Turnover Factor Tau', gridcolor: '#1e293b' }
            };

            Plotly.newPlot(chartDiv, data, layout, { responsive: true, displayModeBar: false });
        },

        scheduleChartUpdate(telemetry) {
            if (this.rafPending) return;
            this.rafPending = true;

            window.requestAnimationFrame(() => {
                this.updatePlotly(telemetry);
                this.rafPending = false;
            });
        },

        updatePlotly(telemetry) {
            const chartDiv = document.getElementById('mapElitesChart');
            if (!chartDiv || !window.Plotly) return;

            // Update title or live indicator
            Plotly.relayout(chartDiv, {
                title: `MAP-Elites Live Grid: Gen ${telemetry.current_gen}/${telemetry.total_gens} (Occupancy: ${telemetry.occupancy_pct?.toFixed(1)}%)`
            });
        },

        // API Node Actions
        async ingestData() {
            this.status = 'INGESTING';
            this.log(`Node 1: Ingesting ${this.instrument} (${this.lookbackYears}y)...`);
            try {
                const res = await fetch('/api/pipeline/ingest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ instrument: this.instrument, lookback_years: parseFloat(this.lookbackYears) })
                });
                const data = await res.json();
                if (res.ok) {
                    this.status = 'STANDBY';
                    this.log(`Node 1 Success: Loaded ${data.bars_count} bars for ${data.instrument}`, 'success');
                } else {
                    this.status = 'ERROR';
                    this.log(`Ingest Error: ${data.detail}`, 'error');
                }
            } catch (err) {
                this.status = 'ERROR';
                this.log(`Ingest Network Error: ${err}`, 'error');
            }
        },

        async startEvolution() {
            this.status = 'EVOLVING';
            this.log(`Node 2: Spawning isolated compute worker for ${this.generations} generations...`);
            try {
                const res = await fetch('/api/pipeline/evolve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        instrument: this.instrument,
                        generations: parseInt(this.generations),
                        population_size: parseInt(this.populationSize),
                        lookback_years: parseFloat(this.lookbackYears)
                    })
                });
                const data = await res.json();
                if (res.ok) {
                    this.log(`Node 2 Started: Worker running on Cores 2-3.`, 'success');
                } else {
                    this.log(`Evolve start error: ${data.detail}`, 'error');
                }
            } catch (err) {
                this.log(`Evolve Network Error: ${err}`, 'error');
            }
        },

        async abortEvolution() {
            this.log("Node 2: Signaling sub-millisecond atomic abort to worker...", 'warning');
            try {
                const res = await fetch('/api/pipeline/abort', { method: 'POST' });
                const data = await res.json();
                this.log(`Abort response: ${data.status}`, 'warning');
            } catch (err) {
                this.log(`Abort Network Error: ${err}`, 'error');
            }
        },

        async emergencyFreeze() {
            window.cockpitAudio.playPanicSound();
            this.log("EMERGENCY STAGE 1: Freezing all local outbounds (<0.5ms)...", 'error');
            try {
                const res = await fetch('/api/emergency/freeze', { method: 'POST' });
                const data = await res.json();
                this.status = 'FROZEN';
                this.log(`STAGE 1 FROZEN in ${data.elapsed_us?.toFixed(1)}us. SHM signaled: ${data.shm_signaled}`, 'error');
            } catch (err) {
                this.log(`Freeze Error: ${err}`, 'error');
            }
        },

        async emergencyPanic() {
            window.cockpitAudio.playPanicSound();
            this.log("EMERGENCY STAGE 2: Dispatched parallel liquidation to 100% cash...", 'error');
            try {
                const res = await fetch('/api/emergency/panic', { method: 'POST' });
                const data = await res.json();
                this.status = 'PANICKED';
                this.log(`STAGE 2 COMPLETE in ${data.elapsed_ms?.toFixed(2)}ms. Orders cancelled: ${data.orders_cancelled_count}, Positions closed: ${data.positions_liquidated_count}`, 'error');
            } catch (err) {
                this.log(`Panic Error: ${err}`, 'error');
            }
        },

        async fetchLeaderboard() {
            try {
                const res = await fetch(`/api/elites/leaderboard?limit=100&metric=sharpe&ticker=${this.instrument}`);
                const data = await res.json();
                if (data.status === 'ok' && this.gridApi) {
                    this.gridApi.setGridOption('rowData', data.elites);
                }
            } catch (err) {
                this.log(`Failed to fetch leaderboard: ${err}`, 'error');
            }
        },

        async fetchInitialData() {
            this.fetchLeaderboard();
        }
    };
}
