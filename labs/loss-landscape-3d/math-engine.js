/**
 * ThinkModule — Module 03: Exact Analytical Loss Manifold & Optimizer Engine
 * Closed-form Loss L(w1, w2), analytical Gradients ∇L, exact 2×2 Hessians H,
 * closed-form Spectral Eigendecomposition, and smooth deterministic trajectories
 * with Cosine LR annealing & mathematical convergence locking (zero vibration).
 */

export const DOMAIN = 2.0;
export const MAX_STEPS = 180;

export const OPTIMIZERS = {
  sgd: {
    id: 'sgd',
    name: 'SGD',
    color: '#fb7185',
    hex: 0xfb7185,
    rule: 'θₜ = θₜ₋₁ − ηₜ∇L',
    desc: 'Takes pure downhill steps proportional to local slope. Honest and simple, but stalls on flat saddles and zig-zags in steep canyons.'
  },
  momentum: {
    id: 'momentum',
    name: 'Momentum',
    color: '#fbbf24',
    hex: 0xfbbf24,
    rule: 'vₜ = β₁vₜ₋₁ + ∇L; Δθ = −ηₜvₜ',
    desc: 'Acts like a heavy steel ball accumulating velocity. Dampens canyon zig-zags and carries enough inertia to jump over shallow barriers.'
  },
  adamw: {
    id: 'adamw',
    name: 'AdamW',
    color: '#34d399',
    hex: 0x34d399,
    rule: 'Adaptive m̂ₜ / (√v̂ₜ + ε) + λθ',
    desc: 'Rescales each coordinate by its recent gradient variance. Slices straight through ill-conditioned canyons and escapes saddle points rapidly.'
  }
};

export const SCENARIOS = {
  saddle: {
    id: 'saddle',
    title: 'Saddle Point Escape',
    story: 'Near a saddle point, one direction curves up while the other curves down. Watch SGD slow to a crawl near the ridge while AdamW adapts per-axis and drops straight into the valley.',
    formula: 'L(w₁,w₂) = 0.34w₁² + 0.20(w₂² − 1.35)² + 0.03w₁w₂ + 0.04',
    start: [-1.52, 0.03],
    baseLr: { sgd: 0.085, momentum: 0.065, adamw: 0.065 },
    beta1: 0.88,
    beta2: 0.99,
    wd: 0.004,
    criticalPoints: [
      { label: 'Saddle', type: 'saddle', w1: 0, w2: 0 },
      { label: 'Global Min', type: 'min', w1: -0.051, w2: 1.162 },
      { label: 'Global Min', type: 'min', w1: 0.051, w2: -1.162 }
    ],
    eval(w1, w2) {
      const q = w2 * w2 - 1.35;
      const loss = 0.34 * w1 * w1 + 0.20 * q * q + 0.03 * w1 * w2 + 0.04;
      const g1 = 0.68 * w1 + 0.03 * w2;
      const g2 = 0.80 * w2 * q + 0.03 * w1;
      return {
        loss,
        g1,
        g2,
        h11: 0.68,
        h12: 0.03,
        h22: 0.80 * (3 * w2 * w2 - 1.35)
      };
    }
  },

  ravine: {
    id: 'ravine',
    title: 'Steep Canyon (κ = 30.8)',
    story: 'A classic ill-conditioned valley: 30× steeper across w₂ than along w₁. SGD wastes energy bouncing wall-to-wall, while Momentum and AdamW glide smoothly down the floor.',
    formula: 'L(w₁,w₂) = 0.06w₁² + 1.85w₂² + 0.02  (κ = λ₁ / λ₂ = 30.8)',
    start: [-1.72, 0.84],
    baseLr: { sgd: 0.44, momentum: 0.14, adamw: 0.065 },
    beta1: 0.82,
    beta2: 0.99,
    wd: 0.002,
    criticalPoints: [
      { label: 'Global Min', type: 'min', w1: 0, w2: 0 }
    ],
    eval(w1, w2) {
      return {
        loss: 0.06 * w1 * w1 + 1.85 * w2 * w2 + 0.02,
        g1: 0.12 * w1,
        g2: 3.70 * w2,
        h11: 0.12,
        h12: 0.0,
        h22: 3.70
      };
    }
  },

  localmin: {
    id: 'localmin',
    title: 'Local Minimum & Barrier Jump',
    story: 'Non-convex neural networks contain shallow traps. SGD gets stuck permanently in the local bowl, whereas Momentum builds enough inertia to clear the ridge into the deep global minimum.',
    formula: 'L(w₁,w₂) = 0.15‖w‖² − 1.05G_loc − 1.85G_glob + 0.25G_ridge + 1.85',
    start: [-1.78, -0.44],
    baseLr: { sgd: 0.065, momentum: 0.068, adamw: 0.072 },
    beta1: 0.90,
    beta2: 0.99,
    wd: 0.002,
    criticalPoints: [
      { label: 'Local Trap', type: 'local', w1: -0.78, w2: -0.17 },
      { label: 'Global Min', type: 'min', w1: 0.93, w2: 0.21 }
    ],
    eval(w1, w2) {
      const blobs = [
        { a: -1.05, cx: -0.82, cy: -0.18, s: 0.48 },
        { a: -1.85, cx: 0.95, cy: 0.22, s: 0.76 },
        { a: 0.25, cx: 0.0, cy: 0.0, s: 0.34 }
      ];
      let loss = 0.15 * (w1 * w1 + w2 * w2) + 1.85;
      let g1 = 0.30 * w1;
      let g2 = 0.30 * w2;
      let h11 = 0.30;
      let h12 = 0.0;
      let h22 = 0.30;

      for (const b of blobs) {
        const dx = w1 - b.cx;
        const dy = w2 - b.cy;
        const g = b.a * Math.exp(-(dx * dx + dy * dy) / b.s);
        const invS = 1 / b.s;
        const invS2 = invS * invS;
        loss += g;
        g1 += (-2 * dx * invS) * g;
        g2 += (-2 * dy * invS) * g;
        h11 += (4 * dx * dx * invS2 - 2 * invS) * g;
        h22 += (4 * dy * dy * invS2 - 2 * invS) * g;
        h12 += (4 * dx * dy * invS2) * g;
      }
      return { loss: Math.max(0.01, loss), g1, g2, h11, h12, h22 };
    }
  }
};

/**
 * Closed-form eigendecomposition of symmetric 2×2 Hessian H.
 */
export function decomposeHessian(h11, h12, h22) {
  const trace = h11 + h22;
  const det = h11 * h22 - h12 * h12;
  const disc = Math.sqrt(Math.max(0, (h11 - h22) * (h11 - h22) + 4 * h12 * h12));
  const l1 = 0.5 * (trace + disc);
  const l2 = 0.5 * (trace - disc);

  let e1, e2;
  if (Math.abs(h12) > 1e-10) {
    const vx = l1 - h22;
    const vy = h12;
    const norm = Math.hypot(vx, vy) || 1;
    e1 = [vx / norm, vy / norm];
    e2 = [-e1[1], e1[0]];
  } else if (h11 >= h22) {
    e1 = [1, 0];
    e2 = [0, 1];
  } else {
    e1 = [0, 1];
    e2 = [1, 0];
  }

  const a1 = Math.abs(l1);
  const a2 = Math.abs(l2);
  const kappa = Math.max(a1, a2) / Math.max(Math.min(a1, a2), 1e-5);

  let regime = 'Flat Region';
  let regimeClass = 'saddle';
  const eps = 1e-3;
  if (l1 > eps && l2 > eps) {
    regime = 'Convex Valley';
    regimeClass = 'convex';
  } else if (l1 < -eps && l2 < -eps) {
    regime = 'Concave Peak';
    regimeClass = 'concave';
  } else if (l1 > eps && l2 < -eps) {
    regime = 'Saddle Point';
    regimeClass = 'saddle';
  }

  return { trace, det, l1, l2, e1, e2, kappa, regime, regimeClass };
}

/**
 * Smooth sculpted height mapping (monotonic, preserves exact sign of slopes).
 */
export function lossToHeight(loss) {
  const safe = Math.max(0, loss);
  return 0.24 + 1.38 * Math.log(1 + safe * 1.12);
}

export function dLossToHeight(loss) {
  const safe = Math.max(0, loss);
  return (1.38 * 1.12) / (1 + safe * 1.12);
}

/**
 * Deterministic pre-computed optimization engine.
 * Guarantees silky-smooth interpolation and complete calm settling with zero jitter.
 */
export class LossEngine {
  constructor() {
    this.scenarioId = 'saddle';
    this.lrScale = 1.0;
    this.focusedId = 'adamw';
    this.history = {};
    this.recompute();
  }

  get scenario() {
    return SCENARIOS[this.scenarioId];
  }

  evaluate(w1, w2) {
    const raw = this.scenario.eval(w1, w2);
    const spec = decomposeHessian(raw.h11, raw.h12, raw.h22);
    return { ...raw, ...spec, gradNorm: Math.hypot(raw.g1, raw.g2) };
  }

  setScenario(id) {
    if (!SCENARIOS[id]) return;
    this.scenarioId = id;
    this.recompute();
  }

  setLrScale(scale) {
    this.lrScale = scale;
    this.recompute();
  }

  recompute() {
    const sc = this.scenario;
    const initEval = this.evaluate(sc.start[0], sc.start[1]);
    const eps = 1e-8;
    this.history = {};

    for (const id of Object.keys(OPTIMIZERS)) {
      const baseEta = sc.baseLr[id] * this.lrScale;
      let w1 = sc.start[0];
      let w2 = sc.start[1];
      let m1 = 0, m2 = 0, v1 = 0, v2 = 0;
      let converged = false;
      let ev = initEval;

      const frames = [{
        w1, w2, dw1: 0, dw2: 0, mHat1: 0, mHat2: 0, vHat1: 0, vHat2: 0,
        eta: baseEta, converged: false, status: 'Start', eval: ev
      }];

      for (let t = 1; t <= MAX_STEPS; t++) {
        // Smooth cosine annealing toward 2% of baseEta so spheres settle cleanly with zero vibration
        const p = t / MAX_STEPS;
        const eta = baseEta * (0.02 + 0.98 * 0.5 * (1 + Math.cos(Math.PI * p)));

        if (converged) {
          const prev = frames[t - 1];
          frames.push({ ...prev, dw1: 0, dw2: 0, eta, converged: true, status: 'Converged' });
          continue;
        }

        const { g1, g2 } = ev;
        let dw1 = 0, dw2 = 0;
        let mHat1 = 0, mHat2 = 0, vHat1 = 0, vHat2 = 0;

        if (id === 'sgd') {
          dw1 = -eta * g1;
          dw2 = -eta * g2;
        } else if (id === 'momentum') {
          m1 = sc.beta1 * m1 + g1;
          m2 = sc.beta1 * m2 + g2;
          mHat1 = m1;
          mHat2 = m2;
          dw1 = -eta * m1;
          dw2 = -eta * m2;
        } else {
          m1 = sc.beta1 * m1 + (1 - sc.beta1) * g1;
          m2 = sc.beta1 * m2 + (1 - sc.beta1) * g2;
          v1 = sc.beta2 * v1 + (1 - sc.beta2) * g1 * g1;
          v2 = sc.beta2 * v2 + (1 - sc.beta2) * g2 * g2;
          const bc1 = 1 - Math.pow(sc.beta1, t);
          const bc2 = 1 - Math.pow(sc.beta2, t);
          mHat1 = m1 / bc1;
          mHat2 = m2 / bc2;
          vHat1 = v1 / bc2;
          vHat2 = v2 / bc2;
          dw1 = -eta * (mHat1 / (Math.sqrt(vHat1) + eps) + sc.wd * w1);
          dw2 = -eta * (mHat2 / (Math.sqrt(vHat2) + eps) + sc.wd * w2);
        }

        // Clamp safely inside sculpted slab domain
        const nw1 = Math.max(-DOMAIN + 0.06, Math.min(DOMAIN - 0.06, w1 + dw1));
        const nw2 = Math.max(-DOMAIN + 0.06, Math.min(DOMAIN - 0.06, w2 + dw2));
        dw1 = nw1 - w1;
        dw2 = nw2 - w2;
        w1 = nw1;
        w2 = nw2;
        ev = this.evaluate(w1, w2);

        const stepLen = Math.hypot(dw1, dw2);
        if (ev.gradNorm < 0.006 && stepLen < 0.004 && ev.l1 > 0 && ev.l2 > 0) {
          converged = true;
        }

        const status = converged
          ? 'Converged'
          : (ev.l2 < -0.05 && stepLen < 0.008)
            ? 'Stalled'
            : 'Descending';

        frames.push({ w1, w2, dw1, dw2, mHat1, mHat2, vHat1, vHat2, eta, converged, status, eval: ev });
      }

      this.history[id] = frames;
    }
  }

  /**
   * Smoothly interpolated state at continuous step u ∈ [0, MAX_STEPS].
   */
  sample(id, continuousStep) {
    const arr = this.history[id];
    const clamped = Math.max(0, Math.min(MAX_STEPS, continuousStep));
    const i0 = Math.floor(clamped);
    const i1 = Math.min(MAX_STEPS, i0 + 1);
    const alpha = clamped - i0;
    const f0 = arr[i0];
    const f1 = arr[i1];
    const w1 = f0.w1 + (f1.w1 - f0.w1) * alpha;
    const w2 = f0.w2 + (f1.w2 - f0.w2) * alpha;
    return {
      ...f0,
      w1,
      w2,
      loss: this.scenario.eval(w1, w2).loss
    };
  }

  getLeader(stepIndex) {
    const idx = Math.max(0, Math.min(MAX_STEPS, Math.round(stepIndex)));
    let bestId = 'adamw';
    let bestLoss = Infinity;
    for (const id of Object.keys(OPTIMIZERS)) {
      const L = this.history[id][idx].eval.loss;
      if (L < bestLoss) {
        bestLoss = L;
        bestId = id;
      }
    }
    return { id: bestId, loss: bestLoss };
  }
}