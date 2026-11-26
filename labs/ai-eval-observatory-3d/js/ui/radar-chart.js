/**
 * High-DPI Canvas 2D multi-axis AI Evaluation Radar Chart
 * with smooth exponential animation between evaluation states.
 */
export class EvalRadarChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.axes = [
      { key: 'faithfulness', label: 'FAITHFUL' },
      { key: 'relevance',    label: 'RELEVANCE' },
      { key: 'ndcgAtK',      label: 'NDCG@K' },
      { key: 'safetyMargin', label: 'SAFETY' },
      { key: 'efficiency',   label: 'KL-STABLE' }
    ];

    this.current = [0.5, 0.5, 0.5, 0.5, 0.5];
    this.target  = [0.5, 0.5, 0.5, 0.5, 0.5];
    this.accentHex = '#38bdf8';

    this._resize();
    window.addEventListener('resize', () => this._resize());
  }

  _resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(220, rect.width || 256);
    const h = Math.max(140, rect.height || 154);
    this.width = w;
    this.height = h;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  setMetrics(metrics, verdictClass = 'pass') {
    this.target = [
      metrics.faithfulness,
      metrics.relevance,
      metrics.ndcgAtK,
      metrics.safetyMargin,
      Math.max(0.08, 1.0 - metrics.klDivergence)
    ];

    if (verdictClass === 'fail') this.accentHex = '#f43f5e';
    else if (verdictClass === 'warn') this.accentHex = '#f59e0b';
    else if (verdictClass === 'blocked') this.accentHex = '#38bdf8';
    else this.accentHex = '#10b981';
  }

  render(dt) {
    const blend = Math.min(1, dt * 8.5);
    for (let i = 0; i < this.current.length; i++) {
      this.current[i] += (this.target[i] - this.current[i]) * blend;
    }

    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    ctx.clearRect(0, 0, w, h);

    const cx = w * 0.5;
    const cy = h * 0.53;
    const radius = Math.min(w * 0.31, h * 0.37);
    const n = this.axes.length;

    // Concentric reference pentagons (25%, 50%, 75%, 100%)
    ctx.lineWidth = 1;
    for (let ring = 1; ring <= 4; ring++) {
      const r = (radius * ring) / 4;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const angle = -Math.PI / 2 + (i * Math.PI * 2) / n;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = ring === 4 ? 'rgba(148, 163, 184, 0.24)' : 'rgba(148, 163, 184, 0.11)';
      ctx.stroke();
    }

    // Radial spokes & crisp axis labels
    ctx.font = '600 9px "JetBrains Mono", monospace';
    ctx.textBaseline = 'middle';

    for (let i = 0; i < n; i++) {
      const angle = -Math.PI / 2 + (i * Math.PI * 2) / n;
      const ex = cx + Math.cos(angle) * radius;
      const ey = cy + Math.sin(angle) * radius;

      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ex, ey);
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.14)';
      ctx.stroke();

      const lx = cx + Math.cos(angle) * (radius + 14);
      const ly = cy + Math.sin(angle) * (radius + 12);
      ctx.fillStyle = '#94a3b8';
      ctx.textAlign = Math.cos(angle) > 0.25 ? 'left' : Math.cos(angle) < -0.25 ? 'right' : 'center';
      ctx.fillText(this.axes[i].label, lx, ly);
    }

    // Data polygon
    ctx.beginPath();
    const pts = [];
    for (let i = 0; i < n; i++) {
      const angle = -Math.PI / 2 + (i * Math.PI * 2) / n;
      const val = Math.max(0.06, Math.min(1, this.current[i]));
      const px = cx + Math.cos(angle) * radius * val;
      const py = cy + Math.sin(angle) * radius * val;
      pts.push([px, py]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();

    ctx.save();
    ctx.fillStyle = this._hexToRgba(this.accentHex, 0.22);
    ctx.fill();
    ctx.strokeStyle = this.accentHex;
    ctx.lineWidth = 1.8;
    ctx.shadowColor = this.accentHex;
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.restore();

    // Glowing vertices
    for (const [px, py] of pts) {
      ctx.beginPath();
      ctx.arc(px, py, 2.8, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = this.accentHex;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  _hexToRgba(hex, alpha) {
    const clean = hex.replace('#', '');
    const r = parseInt(clean.slice(0, 2), 16);
    const g = parseInt(clean.slice(2, 4), 16);
    const b = parseInt(clean.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
}
