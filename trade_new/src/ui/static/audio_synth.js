/**
 * Zero-Dependency Pure Web Audio API Sound Synthesizer.
 * Synthesizes audio harmonics dynamically in memory without external audio files.
 * Hardened with:
 * - User gesture audio arming gate (ADHD-40).
 * - Master DynamicsCompressor / Limiter to prevent clipping and ear fatigue (ADHD-41).
 * - Automatic oscillator and gain node cleanup on audio termination (ADHD-41).
 */
class CockpitAudioSynthesizer {
    constructor() {
        this.ctx = null;
        this.limiter = null;
        this.isMuted = false;
        this.isArmed = false;

        // Auto-arm on first user gesture anywhere on page
        const armHandler = () => {
            this.armAudio();
            ['click', 'keydown', 'pointerdown'].forEach(ev => {
                window.removeEventListener(ev, armHandler);
            });
        };
        ['click', 'keydown', 'pointerdown'].forEach(ev => {
            window.addEventListener(ev, armHandler, { once: true, passive: true });
        });
    }

    _initContext() {
        if (!this.ctx) {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (AudioContextClass) {
                this.ctx = new AudioContextClass();
                this._setupMasterLimiter();
            }
        }
    }

    _setupMasterLimiter() {
        if (this.ctx && !this.limiter) {
            // Master brickwall limiter: threshold -12dB, high ratio, fast attack
            this.limiter = this.ctx.createDynamicsCompressor();
            this.limiter.threshold.setValueAtTime(-12, this.ctx.currentTime);
            this.limiter.knee.setValueAtTime(30, this.ctx.currentTime);
            this.limiter.ratio.setValueAtTime(12, this.ctx.currentTime);
            this.limiter.attack.setValueAtTime(0.003, this.ctx.currentTime);
            this.limiter.release.setValueAtTime(0.25, this.ctx.currentTime);
            this.limiter.connect(this.ctx.destination);
        }
    }

    armAudio() {
        this._initContext();
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume().then(() => {
                this.isArmed = true;
                this._updateHUDState();
            }).catch(() => {});
        } else if (this.ctx && this.ctx.state === 'running') {
            this.isArmed = true;
            this._updateHUDState();
        }
    }

    _updateHUDState() {
        const pill = document.getElementById('audio-armed-pill');
        if (pill) {
            if (this.isArmed && !this.isMuted) {
                pill.textContent = '🔊 AUDIO ARMED';
                pill.className = 'text-[11px] px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800';
            } else if (this.isMuted) {
                pill.textContent = '🔇 MUTED';
                pill.className = 'text-[11px] px-2 py-0.5 rounded bg-slate-900 text-slate-500 border border-slate-700';
            } else {
                pill.textContent = '🔈 CLICK TO ARM';
                pill.className = 'text-[11px] px-2 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-800 cursor-pointer animate-pulse';
            }
        }
    }

    toggleMute() {
        this.isMuted = !this.isMuted;
        if (!this.isMuted && !this.isArmed) {
            this.armAudio();
        }
        this._updateHUDState();
        return this.isMuted;
    }

    playTickSound() {
        if (this.isMuted) return;
        this._initContext();
        if (!this.ctx || this.ctx.state !== 'running' || !this.limiter) return;

        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(140, this.ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(40, this.ctx.currentTime + 0.04);

        gain.gain.setValueAtTime(0.08, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.04);

        osc.connect(gain);
        gain.connect(this.limiter);

        osc.onended = () => {
            try {
                osc.disconnect();
                gain.disconnect();
            } catch (e) {}
        };

        osc.start();
        osc.stop(this.ctx.currentTime + 0.04);
    }

    playEliteSound() {
        if (this.isMuted) return;
        this._initContext();
        if (!this.ctx || this.ctx.state !== 'running' || !this.limiter) return;

        // Crystal arpeggio: C5 (523Hz), E5 (659Hz), G5 (784Hz), C6 (1046Hz)
        const notes = [523.25, 659.25, 783.99, 1046.50];
        const now = this.ctx.currentTime;

        notes.forEach((freq, idx) => {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(freq, now + idx * 0.06);

            gain.gain.setValueAtTime(0.001, now + idx * 0.06);
            gain.gain.linearRampToValueAtTime(0.12, now + idx * 0.06 + 0.02);
            gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.06 + 0.25);

            osc.connect(gain);
            gain.connect(this.limiter);

            osc.onended = () => {
                try {
                    osc.disconnect();
                    gain.disconnect();
                } catch (e) {}
            };

            osc.start(now + idx * 0.06);
            osc.stop(now + idx * 0.06 + 0.25);
        });
    }

    playPanicSound() {
        if (this.isMuted) return;
        this._initContext();
        if (!this.ctx || this.ctx.state !== 'running' || !this.limiter) return;

        const now = this.ctx.currentTime;
        const freqs = [880, 440, 880, 440];
        freqs.forEach((freq, idx) => {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(freq, now + idx * 0.12);

            gain.gain.setValueAtTime(0.2, now + idx * 0.12);
            gain.gain.exponentialRampToValueAtTime(0.01, now + idx * 0.12 + 0.11);

            osc.connect(gain);
            gain.connect(this.limiter);

            osc.onended = () => {
                try {
                    osc.disconnect();
                    gain.disconnect();
                } catch (e) {}
            };

            osc.start(now + idx * 0.12);
            osc.stop(now + idx * 0.12 + 0.12);
        });
    }

    playCompleteSound() {
        if (this.isMuted) return;
        this._initContext();
        if (!this.ctx || this.ctx.state !== 'running' || !this.limiter) return;

        const now = this.ctx.currentTime;
        const notes = [440, 554.37, 659.25, 880]; // A Major
        notes.forEach((freq, idx) => {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, now + idx * 0.08);

            gain.gain.setValueAtTime(0.15, now + idx * 0.08);
            gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.08 + 0.4);

            osc.connect(gain);
            gain.connect(this.limiter);

            osc.onended = () => {
                try {
                    osc.disconnect();
                    gain.disconnect();
                } catch (e) {}
            };

            osc.start(now + idx * 0.08);
            osc.stop(now + idx * 0.08 + 0.4);
        });
    }
}

window.cockpitAudio = new CockpitAudioSynthesizer();
