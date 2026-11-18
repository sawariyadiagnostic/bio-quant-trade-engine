/**
 * RAVANA Sovereign Alpha Cockpit - Next-Gen Quant Terminal Controller
 * Powers live paper trading telemetry, 9 production alphas, portfolio avionics,
 * 20x20 MAP-Elites matrix, and real-time execution order book.
 */

function ravanaCockpit() {
    return {
        // Core Navigation & HUD State
        activeTab: 'terminal', // 'terminal' | 'alphas' | 'matrix' | 'orders' | 'security'
        status: 'LIVE_DAEMON',
        isFrozen: false,
        isMuted: false,
        lastSync: new Date().toLocaleTimeString(),
        syncInterval: null,

        // Portfolio & Position State
        portfolio: {
            initial_capital: 200000.0,
            current_equity: 200037.96,
            net_pnl: 37.96,
            roi_pct: 0.019,
            cash_balance: 85016.13,
            cash_pct: 42.5,
            invested_capital: 115021.83,
            invested_pct: 57.5,
            max_drawdown_pct: 0.00,
            win_rate_pct: 100.0,
            epochs_completed: 75,
            total_epochs: 75,
            regime: {
                name: "CHOP (Low Volatility Sideways)",
                variance_ratio_20: 0.68,
                status: "MEAN_REVERSION_ACTIVE",
                momentum_z_hurdle: 0.85,
                target_multiplier: 0.75,
                inav_hurdle_bps: 10.0,
                stop_loss_atr: 0.50,
                take_profit_atr: 0.40,
            },
            positions: []
        },

        // 9 Production Strategy Alphas
        strategies: [],

        // Execution Orders State
        orders: [],
        ordersCount: 0,
        orderSearch: '',
        orderSideFilter: 'ALL',
        orderPage: 1,
        ordersPerPage: 15,

        // 20x20 MAP-Elites State
        gridMetric: 'fitness',
        gridData: null,
        gridOccupiedCount: 0,
        gridOccupancyPct: 0.0,
        hoveredCell: null,

        // Live Daemon Logs
        logs: [],

        // Chart.js Instances
        equityChart: null,
        allocationChart: null,

        // Toast Notifications
        toasts: [],

        init() {
            this.showToast('Connecting to RAVANA v4.3 Sovereign Core...', 'info');
            this.fetchAllData();
            this.syncInterval = setInterval(() => {
                this.pollTelemetry();
            }, 3000);

            // Re-render charts when window resizes
            window.addEventListener('resize', () => {
                if (this.equityChart) this.equityChart.resize();
                if (this.allocationChart) this.allocationChart.resize();
            });
        },

        showToast(msg, type = 'info') {
            const id = Date.now() + Math.random();
            this.toasts.push({ id, msg, type });
            setTimeout(() => {
                this.toasts = this.toasts.filter(t => t.id !== id);
            }, 4000);
        },

        setTab(tab) {
            this.activeTab = tab;
            if (tab === 'terminal') {
                this.$nextTick(() => {
                    this.initOrUpdateCharts();
                });
            } else if (tab === 'matrix') {
                this.fetchGridMatrix();
            }
        },

        async fetchAllData() {
            await Promise.all([
                this.fetchPortfolioSummary(),
                this.fetchStrategies(),
                this.fetchOrders(),
                this.fetchGridMatrix(),
                this.fetchPaperStatus()
            ]);
            this.$nextTick(() => {
                this.initOrUpdateCharts();
            });
        },

        async pollTelemetry() {
            try {
                await Promise.all([
                    this.fetchPortfolioSummary(),
                    this.fetchPaperStatus()
                ]);
                this.lastSync = new Date().toLocaleTimeString();
            } catch (err) {
                console.warn('[Cockpit] Telemetry poll failed:', err);
            }
        },

        async fetchPortfolioSummary() {
            try {
                const res = await fetch('/api/portfolio/summary');
                if (!res.ok) return;
                const data = await res.json();
                this.portfolio = data;
            } catch (e) {
                console.error('Failed to fetch portfolio summary:', e);
            }
        },

        async fetchPaperStatus() {
            try {
                const res = await fetch('/api/paper/status');
                if (!res.ok) return;
                const data = await res.json();
                this.ordersCount = data.orders_count || this.ordersCount;
                if (data.recent_logs && data.recent_logs.length > 0) {
                    this.logs = data.recent_logs;
                }
            } catch (e) {
                console.error('Failed to fetch paper status:', e);
            }
        },

        async fetchStrategies() {
            try {
                const res = await fetch('/api/strategies/catalog');
                if (!res.ok) return;
                const data = await res.json();
                this.strategies = data.strategies || [];
            } catch (e) {
                console.error('Failed to fetch strategies catalog:', e);
            }
        },

        async fetchOrders() {
            try {
                const res = await fetch('/api/orders/all?limit=200');
                if (!res.ok) return;
                const data = await res.json();
                this.orders = data.orders || [];
                this.ordersCount = data.total || this.orders.length;
            } catch (e) {
                console.error('Failed to fetch orders:', e);
            }
        },

        async fetchGridMatrix() {
            try {
                const res = await fetch(`/api/elites/grid?ticker=NIFTYBEES&metric=${this.gridMetric}`);
                if (!res.ok) return;
                const data = await res.json();
                this.gridData = data.grid || [];
                this.gridOccupiedCount = data.occupied || 0;
                this.gridOccupancyPct = data.occupancy_pct || 0.0;
            } catch (e) {
                console.error('Failed to fetch grid matrix:', e);
            }
        },

        filteredOrders() {
            let list = this.orders;
            if (this.orderSearch.trim()) {
                const q = this.orderSearch.trim().toUpperCase();
                list = list.filter(o => o.ticker.includes(q) || o.order_id.includes(q));
            }
            if (this.orderSideFilter !== 'ALL') {
                list = list.filter(o => o.side === this.orderSideFilter);
            }
            return list;
        },

        paginatedOrders() {
            const list = this.filteredOrders();
            const start = (this.orderPage - 1) * this.ordersPerPage;
            return list.slice(start, start + this.ordersPerPage);
        },

        totalPages() {
            return Math.max(1, Math.ceil(this.filteredOrders().length / this.ordersPerPage));
        },

        nextPage() {
            if (this.orderPage < this.totalPages()) this.orderPage++;
        },

        prevPage() {
            if (this.orderPage > 1) this.orderPage--;
        },

        async initOrUpdateCharts() {
            await this.initEquityChart();
            this.initAllocationChart();
        },

        async initEquityChart() {
            const ctx = document.getElementById('equityCurveCanvas');
            if (!ctx) return;

            let labels = [];
            let equityData = [];
            try {
                const res = await fetch('/api/portfolio/equity_history');
                const data = await res.json();
                labels = data.labels || [];
                equityData = data.equity || [];
            } catch (err) {
                console.warn('Could not load equity history:', err);
            }

            if (this.equityChart) {
                this.equityChart.data.labels = labels;
                this.equityChart.data.datasets[0].data = equityData;
                this.equityChart.update();
                return;
            }

            if (typeof Chart === 'undefined') return;

            const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 240);
            gradient.addColorStop(0, 'rgba(16, 185, 129, 0.35)');
            gradient.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

            this.equityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Portfolio Equity (INR)',
                        data: equityData,
                        borderColor: '#10b981',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        pointHoverBackgroundColor: '#10b981',
                        fill: true,
                        backgroundColor: gradient,
                        tension: 0.25
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: '#0f172a',
                            titleColor: '#94a3b8',
                            bodyColor: '#f1f5f9',
                            borderColor: '#334155',
                            borderWidth: 1,
                            callbacks: {
                                label: (ctx) => ` NAV: ₹${ctx.parsed.y.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: 'rgba(51, 65, 85, 0.2)' },
                            ticks: { color: '#64748b', maxTicksLimit: 12, font: { family: 'monospace', size: 10 } }
                        },
                        y: {
                            grid: { color: 'rgba(51, 65, 85, 0.2)' },
                            ticks: {
                                color: '#64748b',
                                font: { family: 'monospace', size: 10 },
                                callback: (val) => '₹' + val.toLocaleString('en-IN')
                            }
                        }
                    }
                }
            });
        },

        initAllocationChart() {
            const ctx = document.getElementById('allocationDonutCanvas');
            if (!ctx || typeof Chart === 'undefined') return;

            const cashPct = this.portfolio.cash_pct || 42.5;
            const psuPct = 35.0;
            const bankPct = 22.5;

            if (this.allocationChart) {
                this.allocationChart.data.datasets[0].data = [cashPct, psuPct, bankPct];
                this.allocationChart.update();
                return;
            }

            this.allocationChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Cash Buffer', 'PSUBNKBEES', 'BANKBEES'],
                    datasets: [{
                        data: [cashPct, psuPct, bankPct],
                        backgroundColor: ['#3b82f6', '#10b981', '#f59e0b'],
                        borderColor: '#0b101e',
                        borderWidth: 3,
                        hoverOffset: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '72%',
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: '#0f172a',
                            borderColor: '#334155',
                            borderWidth: 1,
                            callbacks: {
                                label: (ctx) => ` ${ctx.label}: ${ctx.parsed}%`
                            }
                        }
                    }
                }
            });
        },

        async triggerFreeze() {
            try {
                const res = await fetch('/api/emergency/freeze', { method: 'POST' });
                const data = await res.json();
                this.isFrozen = true;
                this.showToast('🚨 EMERGENCY FREEZE ACTIVATED: Execution queue locked.', 'warning');
            } catch (err) {
                this.showToast('Failed to trigger freeze: ' + err, 'error');
            }
        },

        async triggerPanic() {
            if (!confirm('DANGER: Confirm emergency panic liquidation to 100% cash?')) return;
            try {
                const res = await fetch('/api/emergency/panic', { method: 'POST' });
                const data = await res.json();
                this.isFrozen = true;
                this.showToast('⚡ PANIC FLATTEN COMPLETE: All open orders cancelled.', 'error');
                this.fetchAllData();
            } catch (err) {
                this.showToast('Failed to execute panic liquidation: ' + err, 'error');
            }
        },

        hoverCell(row, col) {
            if (!this.gridData) return;
            const cell = this.gridData[row] ? this.gridData[row][col] : null;
            this.hoveredCell = cell ? { ...cell, row, col } : null;
        },

        clearHover() {
            this.hoveredCell = null;
        },

        getCellColor(row, col) {
            if (!this.gridData || !this.gridData[row]) return 'bg-slate-900/60 border-slate-800/40';
            const cell = this.gridData[row][col];
            if (!cell) return 'bg-slate-900/40 border-slate-800/30 hover:border-slate-600';
            const f = cell.fitness || 0;
            if (f >= 3.5) return 'bg-emerald-500/80 border-emerald-400 text-white shadow-sm shadow-emerald-500/50';
            if (f >= 3.0) return 'bg-teal-500/70 border-teal-400 text-white';
            if (f >= 2.5) return 'bg-cyan-600/60 border-cyan-400 text-white';
            return 'bg-blue-600/50 border-blue-400 text-white';
        }
    };
}
