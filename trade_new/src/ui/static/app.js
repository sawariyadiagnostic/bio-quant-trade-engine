/**
 * Bio-Quant Institutional Alpha Cockpit - Main Application Controller
 * Features:
 * 1. 20x20 MAP-Elites WebGL Behavioral Heatmap (400 Niches with Fog-of-War).
 * 2. Dynamic 100+ Asset Universe Selector with automatic leaderboard and grid illumination.
 * 3. Random Seed Governor for speciation and deterministic replay.
 * 4. 3D Phenotype Strategy Manifold (UMAP & HDBSCAN Clones Pruner).
 * 5. 1024D Protein Latent Space Pairwise Cosine Similarity Heatmap.
 * 6. Differential SSE listener with Web Audio sound effects.
 * 7. AG Grid DOM virtualization with transactional delta updates.
 * 8. 5-second dead-man switch heartbeat watchdog.
 */

function cockpitApp() {
    return {
        // Core State
        status: 'STANDBY',
        instrument: 'NIFTYBEES',
        lookbackYears: 1.0,
        generations: 100,
        populationSize: 100,
        randomSeed: null,
        isMuted: false,
        csrfToken: 'bioquant-safe-session',
        activeTab: 'strategy', // 'strategy' | 'tactical' | 'manifold' | 'cosine' | 'leaderboard' | 'telemetry'
        deadManDisconnected: false,
        lastHeartbeat: Date.now(),
        eventSource: null,
        gridApi: null,
        rafPending: false,

        // Strategy Deck Execution State
        strategyCapital: 100000,
        strategyThreshold: 0.40,
        strategyLoading: false,
        strategyRunningType: null,
        strategyResult: null,

        // Compute Target & Remote GPU State
        computeTarget: 'local',
        remoteUrl: localStorage.getItem('bioquant_remote_url') || '',
        remoteGpuStatus: null,

        // Universe Catalog State
        universeCategories: {},
        universeCount: 0,
        residentTickers: [],

        // MAP-Elites Grid State
        gridMetric: 'fitness',
        gridOccupancy: 0.0,
        gridOccupiedCount: 0,
        currentGridData: null,

        // Manifold State
        manifoldMethod: 'auto',
        manifoldClusters: 0,
        manifoldCompression: 0.0,

        // Logs & Telemetry
        logs: [],
        telemetry: {
            current_gen: 0,
            total_gens: 100,
            resident_count: 0,
            survivor_count: 0,
            occupancy_pct: 0.0,
            top_sharpe: 0.0,
            best_net_return: 0.0,
            worker_status: 0
        },

        init() {
            this.log("Avionics HUD initialized. Connecting telemetry bus...");
            this.fetchCsrfToken();
            this.initGrid();
            this.connectSSE();
            this.startDeadManWatchdog();
            this.fetchUniverseAssets();
            this.fetchGridMatrix();
            this.fetchLeaderboard();
        },

        async fetchCsrfToken() {
            try {
                const res = await fetch('/api/csrf-token');
                const data = await res.json();
                if (data && data.token) this.csrfToken = data.token;
            } catch (e) {}
        },

        log(msg, type = 'info') {
            const time = new Date().toLocaleTimeString();
            this.logs.unshift({ time, msg, type });
            if (this.logs.length > 200) this.logs.pop();
        },

        toggleMute() {
            this.isMuted = window.cockpitAudio.toggleMute();
        },

        randomizeSeed() {
            this.randomSeed = Math.floor(Math.random() * 900000) + 100000;
            this.log(`Exploration seed set: ${this.randomSeed}`);
        },

        setGenerations(val) {
            this.generations = Math.max(1, parseInt(val) || 1);
        },

        setPopulation(val) {
            this.populationSize = Math.max(10, parseInt(val) || 10);
        },

        setTab(tab) {
            this.activeTab = tab;
            setTimeout(() => {
                if (tab === 'tactical' && this.currentGridData) {
                    const chartDiv = document.getElementById('mapElitesChart');
                    if (chartDiv && window.Plotly) {
                        Plotly.Plots.resize(chartDiv);
                        this.renderMapElitesHeatmap(this.currentGridData);
                    }
                } else if (tab === 'manifold') {
                    const chartDiv = document.getElementById('manifoldChart3d');
                    if (chartDiv && window.Plotly) Plotly.Plots.resize(chartDiv);
                    this.fetchManifold();
                } else if (tab === 'cosine') {
                    const chartDiv = document.getElementById('cosineHeatmapChart');
                    if (chartDiv && window.Plotly) Plotly.Plots.resize(chartDiv);
                    this.fetchCosine();
                }
            }, 50);
        },

        // Universe Catalog & Asset Selection
        async fetchUniverseAssets() {
            try {
                const res = await fetch('/api/universe/assets');
                const data = await res.json();
                if (data.status === 'ok') {
                    this.universeCategories = data.categories || {};
                    this.universeCount = data.total_count || 0;
                    this.residentTickers = data.resident_tickers || [];
                }
            } catch (err) {
                this.log(`Failed to discover universe assets: ${err}`, 'warning');
            }
        },

        async onAssetChange() {
            this.log(`Switched target asset to ${this.instrument} (${this.lookbackYears}y)`);
            await this.fetchGridMatrix();
            await this.fetchLeaderboard();
            if (this.activeTab === 'manifold') {
                await this.fetchManifold();
            } else if (this.activeTab === 'cosine') {
                await this.fetchCosine();
            }
        },

        // 1. Differential SSE Telemetry Stream
        connectSSE() {
            if (this.eventSource) {
                this.eventSource.close();
            }

            this.eventSource = new EventSource('/api/stream/evolution');

            this.eventSource.addEventListener('connected', () => {
                this.lastHeartbeat = Date.now();
                this.deadManDisconnected = false;
                this.log("Connected to Differential SSE telemetry bus.");
            });

            this.eventSource.addEventListener('ping', () => {
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

                // Batch redraw via 250ms RequestAnimationFrame throttler
                this.scheduleChartUpdate(data);
            });

            this.eventSource.addEventListener('elite_added', (e) => {
                const elite = JSON.parse(e.data);
                this.log(`New Elite Champion discovered: ${elite.formula_str || 'Alpha'} (Sharpe: ${elite.fitness?.toFixed(2)})`, 'success');
                window.cockpitAudio.playEliteSound();

                // Delta upsert to AG Grid virtualized table (ADHD-43)
                if (this.gridApi) {
                    const rowTicker = elite.ticker || this.instrument || 'ASSET';
                    const rowR = elite.niche_row !== undefined ? elite.niche_row : (elite.row !== undefined ? elite.row : 0);
                    const rowC = elite.niche_col !== undefined ? elite.niche_col : (elite.col !== undefined ? elite.col : 0);
                    elite.ticker = rowTicker;
                    elite.niche_row = rowR;
                    elite.niche_col = rowC;
                    const rowId = `${rowTicker}_${rowR}_${rowC}`;

                    const existingNode = this.gridApi.getRowNode ? this.gridApi.getRowNode(rowId) : null;
                    if (existingNode) {
                        this.gridApi.applyTransactionAsync({ update: [elite] });
                    } else {
                        this.gridApi.applyTransactionAsync({ add: [elite] });
                    }
                }

                // Incrementally paint cell in 20x20 Heatmap
                this.patchHeatmapCell(elite);
            });

            this.eventSource.addEventListener('complete', () => {
                if (this.status !== 'COMPLETE') {
                    this.status = 'COMPLETE';
                    this.log("Evolution cycle complete. Champions committed to SQLite WAL.", 'success');
                    window.cockpitAudio.playCompleteSound();
                    this.fetchLeaderboard();
                    this.fetchGridMatrix();
                }
            });

            this.eventSource.addEventListener('abort', () => {
                if (this.status !== 'ABORTED') {
                    this.status = 'ABORTED';
                    this.log("Worker execution aborted by operator.", 'warning');
                    this.fetchLeaderboard();
                    this.fetchGridMatrix();
                }
            });

            this.eventSource.onerror = () => {
                // Native SSE reconnection handled by browser EventSource
            };
        },

        // 2. 5-Second Dead-Man Switch Watchdog
        scheduleChartUpdate(telemetry) {
            if (this.rafPending) return;
            this.rafPending = true;

            window.requestAnimationFrame(() => {
                this.updatePlotlyHeader(telemetry);
                this.rafPending = false;
            });
        },

        updatePlotlyHeader(telemetry) {
            const chartDiv = document.getElementById('mapElitesChart');
            if (!chartDiv || !window.Plotly) return;
            if (this.status === 'EVOLVING') {
                Plotly.relayout(chartDiv, {
                    'title.text': `🗺️ MAP-Elites 400-Niche Space: ${this.instrument} (Live Gen ${telemetry.current_gen}/${telemetry.total_gens} · ${this.gridOccupancy}%)`
                });
            }
        },

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
                    { field: 'fitness', headerName: 'Sharpe / Fitness', width: 140, sort: 'desc', valueFormatter: p => (p.value !== undefined && p.value !== null) ? p.value.toFixed(3) : '-' },
                    { field: 'sortino', headerName: 'Sortino', width: 110, valueFormatter: p => (p.value !== undefined && p.value !== null) ? p.value.toFixed(2) : '-' },
                    { 
                        field: 'cagr_pct', 
                        headerName: 'CAGR %', 
                        width: 110, 
                        valueFormatter: p => {
                            if (p.value === undefined || p.value === null || isNaN(p.value)) return '-';
                            let val = parseFloat(p.value);
                            if (val <= 1.0 && val > 0.0) val = val * 100.0;
                            return val > 10000 ? '>10,000%' : val.toFixed(1) + '%';
                        }
                    },
                    { 
                        field: 'max_drawdown', 
                        headerName: 'Max DD %', 
                        width: 120, 
                        valueFormatter: p => {
                            if (p.value === undefined || p.value === null || isNaN(p.value)) return '-';
                            let val = parseFloat(p.value);
                            if (val <= 1.0 && val > 0.0) val = val * 100.0;
                            return val.toFixed(1) + '%';
                        }
                    },
                    { 
                        field: 'win_rate', 
                        headerName: 'Win Rate %', 
                        width: 120, 
                        valueFormatter: p => {
                            if (p.value === undefined || p.value === null || isNaN(p.value)) return '-';
                            let val = parseFloat(p.value);
                            if (val > 100.0) val = val / 100.0;
                            else if (val <= 1.0 && val > 0.0) val = val * 100.0;
                            return val.toFixed(1) + '%';
                        }
                    },
                    { field: 'trades_count', headerName: 'Trades', width: 90 },
                    { field: 'formula_str', headerName: 'Genomic Alpha Expression', flex: 1, minWidth: 250 }
                ],
                defaultColDef: {
                    sortable: true,
                    filter: true,
                    resizable: true
                },
                getRowId: params => `${params.data.ticker || 'ASSET'}_${params.data.niche_row}_${params.data.niche_col}`,
                rowModelType: 'clientSide',
                animateRows: true,
                rowData: []
            };

            this.gridApi = agGrid.createGrid(gridDiv, gridOptions);
        },

        // 4. MAP-Elites 20x20 Behavioral Space Heatmap
        async fetchGridMatrix() {
            try {
                const lookbackParam = this.lookbackYears ? `${parseInt(this.lookbackYears)}y` : 'all';
                const url = `/api/elites/grid?ticker=${this.instrument}&lookback=${lookbackParam}&metric=${this.gridMetric}`;
                const res = await fetch(url);
                const data = await res.json();
                if (data.status === 'ok') {
                    this.currentGridData = data;
                    this.gridOccupancy = data.occupancy_pct || 0.0;
                    this.gridOccupiedCount = data.occupied_count || 0;
                    this.renderMapElitesHeatmap(data);
                }
            } catch (err) {
                this.log(`Failed to fetch 20x20 grid matrix: ${err}`, 'error');
            }
        },

        renderMapElitesHeatmap(data) {
            const chartDiv = document.getElementById('mapElitesChart');
            if (!chartDiv || !window.Plotly) return;

            const metricLabels = {
                'fitness': 'Sharpe / Fitness',
                'net_return_pct': 'Net Return %',
                'cagr_pct': 'Annual CAGR %',
                'max_drawdown': 'Max Drawdown %',
                'win_rate': 'Win Rate %',
                'trades_count': 'Trades'
            };

            const trace = {
                z: data.z,
                x: data.x_bins,
                y: data.y_bins,
                text: data.hover_text,
                hovertemplate: '%{text}<extra></extra>',
                type: 'heatmap',
                colorscale: this.gridMetric === 'max_drawdown' ? 'Reds' : 'Viridis',
                colorbar: {
                    title: metricLabels[this.gridMetric] || this.gridMetric.toUpperCase(),
                    tickfont: { color: '#94a3b8' },
                    titlefont: { color: '#e2e8f0', size: 11 }
                },
                xgap: 2,
                ygap: 2
            };

            const layout = {
                paper_bgcolor: '#0f172a',
                plot_bgcolor: '#0b0f19',
                font: { color: '#94a3b8', family: 'monospace' },
                margin: { t: 40, b: 50, l: 80, r: 20 },
                xaxis: { title: 'Turnover Factor Tau (Annualized Portfolio Velocity)', gridcolor: '#1e293b' },
                yaxis: { title: 'Volatile Regime Sigma (Annualized Standard Deviation)', gridcolor: '#1e293b' },
                title: {
                    text: `🗺️ MAP-Elites 400-Niche Space: ${this.instrument} (${data.occupied_count}/400 Illuminated · ${data.occupancy_pct}%)`,
                    font: { color: '#38bdf8', size: 13 }
                }
            };

            Plotly.newPlot(chartDiv, [trace], layout, { responsive: true, displayModeBar: false });
        },

        patchHeatmapCell(elite) {
            if (!this.currentGridData || !this.currentGridData.z) return;
            const r = parseInt(elite.niche_row);
            const c = parseInt(elite.niche_col);
            if (r >= 0 && r < 20 && c >= 0 && c < 20) {
                const metricVal = parseFloat(elite[this.gridMetric] ?? elite.fitness ?? 0.0);
                if (this.currentGridData.z[r][c] === null) {
                    this.gridOccupiedCount += 1;
                    this.gridOccupancy = parseFloat(((this.gridOccupiedCount / 400.0) * 100.0).toFixed(1));
                }
                this.currentGridData.z[r][c] = metricVal;
                const fShort = (elite.formula_str || '').slice(0, 50);
                this.currentGridData.hover_text[r][c] = `Niche [${r}, ${c}]<br>Formula: ${fShort}..<br>Sharpe: ${elite.fitness?.toFixed(2)}<br>CAGR: ${elite.cagr_pct?.toFixed(1)}%<br>Trades: ${elite.trades_count}`;

                const chartDiv = document.getElementById('mapElitesChart');
                if (chartDiv && window.Plotly) {
                    Plotly.restyle(chartDiv, {
                        z: [this.currentGridData.z],
                        text: [this.currentGridData.hover_text]
                    });
                }
            }
        },

        // 5. 3D Phenotype Manifold (UMAP & HDBSCAN Clones Pruner)
        async fetchManifold() {
            const chartDiv = document.getElementById('manifoldChart3d');
            if (!chartDiv || !window.Plotly) return;

            try {
                this.log(`Computing 3D Phenotype Manifold for ${this.instrument} (${this.manifoldMethod})...`);
                const url = `/api/elites/manifold?ticker=${this.instrument}&method=${this.manifoldMethod}&limit=50`;
                const res = await fetch(url);
                const data = await res.json();
                if (data.status === 'ok') {
                    this.manifoldClusters = data.clusters_count || 0;
                    this.manifoldCompression = parseFloat(((data.compression_ratio || 0.0) * 100.0).toFixed(1));
                    this.renderManifoldPlot(data);
                } else {
                    this.log(`Manifold note: ${data.message || 'Need more resident champions.'}`, 'warning');
                }
            } catch (err) {
                this.log(`Failed to fetch manifold: ${err}`, 'error');
            }
        },

        renderManifoldPlot(data) {
            const chartDiv = document.getElementById('manifoldChart3d');
            if (!chartDiv || !window.Plotly || !data.points || data.points.length === 0) return;

            const pts = data.points;
            const trace = {
                x: pts.map(p => p.x),
                y: pts.map(p => p.y),
                z: pts.map(p => p.z),
                mode: 'markers',
                type: 'scatter3d',
                marker: {
                    size: pts.map(p => p.is_orthogonal ? 11 : (p.is_survivor ? 8 : 5)),
                    color: pts.map(p => p.cluster),
                    colorscale: 'Turbo',
                    showscale: true,
                    colorbar: {
                        title: { text: 'Cluster ID', font: { color: '#e2e8f0', size: 10 } },
                        tickfont: { color: '#94a3b8' }
                    },
                    opacity: pts.map(p => p.is_survivor ? 0.95 : 0.45)
                },
                text: pts.map(p => {
                    const fStr = (p.formula || '').slice(0, 60);
                    const tag = p.is_orthogonal ? 'Orthogonal Stream' : (p.is_survivor ? 'Occam Survivor' : 'Pruned Clone');
                    return `<b>Cluster ${p.cluster} (${tag})</b><br>` +
                           `Formula: ${fStr}..<br>` +
                           `Sharpe: ${p.fitness?.toFixed(2)} | CAGR: ${p.cagr_pct?.toFixed(1)}% | MaxDD: ${p.max_drawdown?.toFixed(1)}%`;
                }),
                hovertemplate: '%{text}<extra></extra>'
            };

            const layout = {
                paper_bgcolor: '#0f172a',
                font: { color: '#94a3b8', family: 'monospace' },
                scene: {
                    bgcolor: '#0b0f19',
                    xaxis: {
                        title: { text: 'Dim 1 (AST Topology)', font: { color: '#94a3b8' } },
                        backgroundcolor: '#0f172a',
                        gridcolor: '#1e293b',
                        showbackground: true,
                        zerolinecolor: '#334155'
                    },
                    yaxis: {
                        title: { text: 'Dim 2 (Regime Sensitivity)', font: { color: '#94a3b8' } },
                        backgroundcolor: '#0f172a',
                        gridcolor: '#1e293b',
                        showbackground: true,
                        zerolinecolor: '#334155'
                    },
                    zaxis: {
                        title: { text: 'Dim 3 (Execution Trajectory)', font: { color: '#94a3b8' } },
                        backgroundcolor: '#0f172a',
                        gridcolor: '#1e293b',
                        showbackground: true,
                        zerolinecolor: '#334155'
                    },
                    camera: {
                        eye: { x: 1.5, y: 1.5, z: 1.2 }
                    }
                },
                margin: { t: 30, b: 20, l: 20, r: 20 },
                title: {
                    text: `🌌 3D Strategy Manifold & Occam Pruning: ${this.instrument} (${pts.length} Champions · ${data.clusters_count} Clusters · ${this.manifoldCompression}% Clones Pruned)`,
                    font: { color: '#c084fc', size: 13 }
                }
            };

            Plotly.newPlot(chartDiv, [trace], layout, { responsive: true, displayModeBar: false });
        },

        // 6. 1024D Protein Latent Space Cosine Similarity Heatmap
        async fetchCosine() {
            const chartDiv = document.getElementById('cosineHeatmapChart');
            if (!chartDiv || !window.Plotly) return;

            try {
                this.log(`Computing 1024D latent cosine similarity for ${this.instrument}...`);
                const url = `/api/elites/manifold?ticker=${this.instrument}&method=fast&limit=25`;
                const res = await fetch(url);
                const data = await res.json();
                if (data.status === 'ok' && data.cosine_matrix && data.cosine_matrix.length > 0) {
                    this.renderCosineHeatmap(data);
                }
            } catch (err) {
                this.log(`Failed to fetch cosine matrix: ${err}`, 'error');
            }
        },

        renderCosineHeatmap(data) {
            const chartDiv = document.getElementById('cosineHeatmapChart');
            if (!chartDiv || !window.Plotly) return;

            const trace = {
                z: data.cosine_matrix,
                x: data.labels,
                y: data.labels,
                type: 'heatmap',
                colorscale: 'Viridis',
                zmin: 0.0,
                zmax: 1.0,
                colorbar: { title: 'Cosine Sim', tickfont: { color: '#94a3b8' } },
                xgap: 1,
                ygap: 1
            };

            const layout = {
                paper_bgcolor: '#0f172a',
                plot_bgcolor: '#0b0f19',
                font: { color: '#94a3b8', family: 'monospace', size: 10 },
                margin: { t: 40, b: 100, l: 120, r: 20 },
                title: {
                    text: `🧬 1024D Latent Space Diversity: ${this.instrument} (Unit-Norm Pairwise Cosine Matrix)`,
                    font: { color: '#34d399', size: 13 }
                }
            };

            Plotly.newPlot(chartDiv, [trace], layout, { responsive: true, displayModeBar: false });
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

        async testRemoteGpu() {
            if (!this.remoteUrl) {
                this.remoteGpuStatus = { online: false, message: 'Please enter a Cloudflare Tunnel URL.' };
                return;
            }
            localStorage.setItem('bioquant_remote_url', this.remoteUrl);
            this.remoteGpuStatus = { online: false, message: 'Testing remote connection...' };
            try {
                const res = await fetch('/api/remote/health', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: this.remoteUrl })
                });
                const data = await res.json();
                if (data.online) {
                    this.remoteGpuStatus = {
                        online: true,
                        message: `🟢 ${data.device || 'GPU'} (${data.vram_gb || '16'} GB) · ${data.latency_ms}ms`
                    };
                    this.log(`Remote GPU verified: ${data.device} (${data.latency_ms}ms latency)`, 'success');
                } else {
                    this.remoteGpuStatus = { online: false, message: `❌ ${data.error || 'Connection failed'}` };
                    this.log(`Remote GPU check failed: ${data.error}`, 'error');
                }
            } catch (err) {
                this.remoteGpuStatus = { online: false, message: `❌ Failed: ${err}` };
            }
        },

        async startEvolution() {
            if (this.computeTarget === 'remote') {
                if (!this.remoteUrl) {
                    this.log("Please provide a Kaggle Cloudflare Tunnel URL before starting.", "error");
                    this.remoteGpuStatus = { online: false, message: 'Please enter a Cloudflare Tunnel URL.' };
                    return;
                }
                localStorage.setItem('bioquant_remote_url', this.remoteUrl);
            }

            this.status = 'EVOLVING';
            const targetLabel = this.computeTarget === 'remote' ? 'Kaggle Cloud GPU' : 'Local CPU (Cores 2-3)';
            const seedStr = this.randomSeed ? ` (Seed: ${this.randomSeed})` : '';
            this.log(`Node 2: Dispatching compute to ${targetLabel} for ${this.instrument}, ${this.generations} generations${seedStr}...`);
            try {
                const payload = {
                    instrument: this.instrument,
                    generations: parseInt(this.generations),
                    population_size: parseInt(this.populationSize),
                    lookback_years: parseFloat(this.lookbackYears),
                    compute_target: this.computeTarget,
                    remote_url: this.remoteUrl
                };
                if (this.randomSeed !== null && !isNaN(this.randomSeed)) {
                    payload.seed = parseInt(this.randomSeed);
                }
                const res = await fetch('/api/pipeline/evolve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (res.ok) {
                    const modeMsg = data.compute_target === 'remote' ? 'Kaggle GPU Worker via Cloudflare Tunnel' : 'Local Cores 2-3';
                    this.log(`Node 2 Started: Compute running on ${modeMsg} (Seed: ${data.seed ?? 'auto'}).`, 'success');
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
                const res = await fetch('/api/emergency/freeze', {
                    method: 'POST',
                    headers: { 'X-BioQuant-Token': this.csrfToken || 'bioquant-safe-session' }
                });
                const data = await res.json();
                this.status = 'FROZEN';
                this.log(`STAGE 1 FROZEN in ${data.elapsed_us?.toFixed(1)}us. SHM signaled: ${data.shm_signaled}`, 'error');
            } catch (err) {
                this.log(`Freeze Error: ${err}`, 'error');
            }
        },

        async emergencyPanic() {
            // ADHD-44: Two-step confirmation latch protecting against catastrophic misclick
            if (!confirm("⚠️ CONFIRM EMERGENCY PANIC LIQUIDATION?\n\nThis will trigger immediate atomic liquidation across all connected brokers to 100% cash.")) {
                this.log("Panic liquidation aborted by operator.", "info");
                return;
            }
            window.cockpitAudio.playPanicSound();
            this.log("EMERGENCY STAGE 2: Dispatched parallel liquidation to 100% cash...", 'error');
            try {
                const res = await fetch('/api/emergency/panic', {
                    method: 'POST',
                    headers: { 'X-BioQuant-Token': this.csrfToken || 'bioquant-safe-session' }
                });
                const data = await res.json();
                this.status = 'PANICKED';
                this.log(`STAGE 2 COMPLETE in ${data.elapsed_ms?.toFixed(2)}ms. Orders cancelled: ${data.orders_cancelled_count}, Positions closed: ${data.positions_liquidated_count}`, 'error');
            } catch (err) {
                this.log(`Panic Error: ${err}`, 'error');
            }
        },

        async fetchLeaderboard() {
            try {
                const lookbackParam = this.lookbackYears ? `${parseInt(this.lookbackYears)}y` : 'all';
                const res = await fetch(`/api/elites/leaderboard?limit=100&metric=sharpe&ticker=${this.instrument}&lookback=${lookbackParam}`);
                const data = await res.json();
                if (data.status === 'ok' && this.gridApi) {
                    this.gridApi.setGridOption('rowData', data.elites);
                }
            } catch (err) {
                this.log(`Failed to fetch leaderboard: ${err}`, 'error');
            }
        },

        async runBankbeesStrategy() {
            this.strategyLoading = true;
            this.strategyRunningType = 'bankbees';
            if (window.cockpitAudio) window.cockpitAudio.playTickSound();
            this.log(`Initiating BANKBEES overnight alpha scan (Capital: ₹${this.strategyCapital.toLocaleString()})...`, 'info');
            try {
                const res = await fetch('/api/strategy/bankbees/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ capital: this.strategyCapital, lookback_years: 1.0 })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    const isBuy = data.action_bias === 'BUY';
                    const friction = isBuy ? (this.strategyCapital * 0.00012) : 0.0;
                    const allocInr = isBuy ? this.strategyCapital : 0.0;
                    const units = data.benchmark_price > 0 ? Math.floor(allocInr / data.benchmark_price) : 0;
                    this.strategyResult = {
                        mode: 'single', benchmark_price: data.benchmark_price,
                        conviction_score: data.conviction_score, action_bias: data.action_bias,
                        cagr_pct: data.cagr_pct, sortino: data.sortino, max_drawdown: data.max_drawdown,
                        win_rate: data.win_rate, champions_count: data.champions_count,
                        capital: this.strategyCapital, allocated_capital: allocInr,
                        cash_reserve: isBuy ? 0.0 : this.strategyCapital, statutory_friction: friction,
                        allocations: [{
                            ticker: 'BANKBEES', instrument: 'ETF', friction_bps: 1.2,
                            conviction: data.conviction_score, weight_pct: isBuy ? 100.0 : 0.0,
                            allocated_inr: allocInr, units: units, action: data.action_bias
                        }]
                    };
                    if (window.cockpitAudio) window.cockpitAudio.playEliteSound();
                    this.log(`BANKBEES Scan Success: ${data.action_bias} | Conviction: ${(data.conviction_score * 100).toFixed(1)}% | CAGR: ${data.cagr_pct.toFixed(1)}%`, 'success');
                } else {
                    this.log(`BANKBEES strategy run error: ${data.detail || 'Execution failed'}`, 'error');
                }
            } catch (err) {
                this.log(`BANKBEES Network Error: ${err}`, 'error');
                if (window.cockpitAudio) window.cockpitAudio.playPanicSound();
            } finally {
                this.strategyLoading = false;
                this.strategyRunningType = null;
            }
        },

        async runMultiAssetStrategy() {
            this.strategyLoading = true;
            this.strategyRunningType = 'multi';
            if (window.cockpitAudio) window.cockpitAudio.playTickSound();
            this.log(`Initiating Multi-Asset Overnight scan (Capital: ₹${this.strategyCapital.toLocaleString()}, Threshold: ${this.strategyThreshold})...`, 'info');
            try {
                const res = await fetch('/api/strategy/multi_asset/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ capital: this.strategyCapital, threshold: this.strategyThreshold, lookback_years: 2.0 })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    this.strategyResult = {
                        mode: 'multi', capital: data.capital, allocated_capital: data.allocated_capital,
                        cash_reserve: data.cash_reserve, statutory_friction: data.statutory_friction,
                        cagr_pct: data.cagr_pct, sortino: data.sortino, max_drawdown: data.max_drawdown, win_rate: data.win_rate,
                        conviction_score: data.allocations.length > 0 ? (data.allocations.reduce((acc, a) => acc + a.conviction, 0) / data.allocations.length) : 0.0,
                        allocations: data.allocations
                    };
                    if (window.cockpitAudio) window.cockpitAudio.playCompleteSound();
                    const buyCount = data.allocations.filter(a => a.action === 'BUY').length;
                    this.log(`Multi-Asset Portfolio Success: ${buyCount}/${data.allocations.length} assets allocated | Cash: ₹${Math.round(data.cash_reserve).toLocaleString()} | Friction: ₹${data.statutory_friction.toFixed(2)}`, 'success');
                } else {
                    this.log(`Multi-Asset strategy error: ${data.detail || 'Execution failed'}`, 'error');
                }
            } catch (err) {
                this.log(`Multi-Asset Network Error: ${err}`, 'error');
                if (window.cockpitAudio) window.cockpitAudio.playPanicSound();
            } finally {
                this.strategyLoading = false;
                this.strategyRunningType = null;
            }
        }
    };
}
