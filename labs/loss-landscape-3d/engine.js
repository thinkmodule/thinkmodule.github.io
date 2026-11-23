/**
 * ThinkModule — Module 03: Exact Analytical Optimization & Differential Geometry Engine
 * Provides closed-form Loss L(w1, w2), exact Gradients ∇L, exact 2×2 Hessians H,
 * closed-form Spectral Eigendecomposition, and 4 exact Optimizer integrators.
 */

export const DOMAIN = 2.2;
export const MAX_STEPS = 240;

export const OPTIMIZER_META = {
  sgd: {
    id: 'sgd',
    name: 'SGD',
    color: '#fb7185',
    hex: 0xfb7185,
    rule: 'θₜ = θₜ₋₁ − η·∇L'
  },
  momentum: {
    id: 'momentum',
    name: 'Momentum',
    color: '#fbbf24',
    hex: 0xfbbf24,
    rule: 'vₜ = β₁vₜ₋₁ + ∇L; Δθ = −ηvₜ'
  },
  rmsprop: {
    id: 'rmsprop',
    name: 'RMSProp',
    color: '#a78bfa',
    hex: 0xa78bfa,
    rule: 'v̂ₜ bias-corr. RMS; Δθ = −η·∇L/√v̂ₜ'
  },
  adamw: {
    id: 'adamw',
    name: 'AdamW',
    color: '#34d399',
    hex: 0x34d399,
    rule: 'Δθ = −η(m̂ₜ/(√v̂ₜ+ε) + λθₜ₋₁)'
  }
};

export const SCENARIOS = {
  saddle: {
    id: 'saddle',
    title: 'Saddle Escape Manifold',
    formula: 'L(w₁,w₂) = 0.35w₁² + 0.18(w₂² − 1.44)² + 0.04w₁w₂ + 0.035',
    defaults: { start: [1.65, 0.04], lr: 0.08, beta1: 0.90, beta2: 0.990, wd: 0.01 },
    eval(w1, w2) {
      const q = w2 * w2 - 1.44;
      const loss = 0.35 * w1 * w1 + 0.18 * q * q + 0.04 * w1 * w2 + 0.035;
      const g1 = 0.70 * w1 + 0.04 * w2;
      const g2 = 0.72 * w2 * q + 0.04 * w1;
      const h11 = 0.70;
      const h12 = 0.04;
      const h22 = 0.72 * (3 * w2 * w2 - 1.44);
      return { loss, g1, g2, h11, h12, h22 };
    }
  },

  ravine: {
    id: 'ravine',
    title: 'Ill-Conditioned Ravine (κ = 32.5)',
    formula: 'L(w₁,w₂) = 0.06w₁² + 1.95w₂² + 0.01  (push η > 0.51 for SGD divergence)',
    defaults: { start: [-1.85, 0.92], lr: 0.42, beta1: 0.85, beta2: 0.990, wd: 0.005 },
    eval(w1, w2) {
      const loss = 0.06 * w1 * w1 + 1.95 * w2 * w2 + 0.01;
      const g1 = 0.12 * w1;
      const g2 = 3.90 * w2;
      return { loss, g1, g2, h11: 0.12, h12: 0.0, h22: 3.90 };
    }
  },

  rosenbrock: {
    id: 'rosenbrock',
    title: 'Rosenbrock Curved Canyon',
    formula: 'L(w₁,w₂) = 0.06(1 − w₁)² + 0.65(w₂ − w₁² + 0.4)² + 0.01',
    defaults: { start: [-1.25, 1.15], lr: 0.09, beta1: 0.90, beta2: 0.990, wd: 0.005 },
    eval(w1, w2) {
      const u = w2 - w1 * w1 + 0.4;
      const loss = 0.06 * (1 - w1) * (1 - w1) + 0.65 * u * u + 0.01;
      const g1 = 0.12 * (w1 - 1) - 2.60 * w1 * u;
      const g2 = 1.30 * u;
      const h11 = 0.12 - 2.60 * u + 5.20 * w1 * w1;
      const h12 = -2.60 * w1;
      const h22 = 1.30;
      return { loss, g1, g2, h11, h12, h22 };
    }
  },

  localmin: {
    id: 'localmin',
    title: 'Non-Convex Barrier & Local Min Trap',
    formula: 'L(w₁,w₂) = 0.16‖w‖² − 1.15G_loc − 1.85G_glob + 0.30G_ridge + 1.82',
    defaults: { start: [-1.92, -0.58], lr: 0.085, beta1: 0.92, beta2: 0.990, wd: 0.005 },
    eval(w1, w2) {
      const blobs = [
        { a: -1.15, cx: -0.78, cy: -0.22, s: 0.52 },
        { a: -1.85, cx: 0.96, cy: 0.28, s: 0.78 },
        { a: 0.30, cx: 0.0, cy: 0.0, s: 0.36 }
      ];
      let loss = 0.16 * (w1 * w1 + w2 * w2) + 1.82;
      let g1 = 0.32 * w1;
      let g2 = 0.32 * w2;
      let h11 = 0.32;
      let h12 = 0.0;
      let h22 = 0.32;

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
      return { loss: Math.max(0.005, loss), g1, g2, h11, h12, h22 };
    }
  }
};

/**
 * Exact closed-form eigendecomposition of real symmetric 2×2 Hessian H.
 */
export function decomposeHessian(h11, h12, h22) {
  const trace = h11 + h22;
  const det = h11 * h22 - h12 * h12;
  const disc = Math.sqrt(Math.max(0, (h11 - h22) * (h11 - h22) + 4 * h12 * h12));
  const l1 = 0.5 * (trace + disc);
  const l2 = 0.5 * (trace - disc);

  let e1;
  let e2;
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

  let regime = 'Flat / Degenerate';
  let regimeClass = 'saddle';
  const eps = 1e-3;
  if (l1 > eps && l2 > eps) {
    regime = 'Convex Bowl';
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
 * Monotonic smooth height transform for crisp 3D display (preserves exact surface slope).
 */
export function lossToY(loss) {
  const safe = Math.max(0, loss);
  return 0.58 + 2.05 * Math.log(1 + safe * 1.15);
}

/**
 * Exact derivative dY/dL for mathematically tangent 3D vector arrows.
 */
export function dLossToY(loss) {
  const safe = Math.max(0, loss);
  return (2.05 * 1.15) / (1 + safe * 1.15);
}

/**
 * Complete Multi-Optimizer Simulation Engine.
 */
export class LossEngine {
  constructor() {
    this.scenarioId = 'saddle';
    this.params = { lr: 0.08, beta1: 0.90, beta2: 0.990, wd: 0.01 };
    this.start = [1.65, 0.04];
    this.stepCount = 0;
    this.active = { sgd: true, momentum: true, rmsprop: true, adamw: true };
    this.inspected = 'adamw';
    this.optimizers = {};
    this.loadScenario('saddle');
  }

  get scenario() {
    return SCENARIOS[this.scenarioId];
  }

  evaluate(w1, w2) {
    const res = this.scenario.eval(w1, w2);
    const spectral = decomposeHessian(res.h11, res.h12, res.h22);
    const gradNorm = Math.hypot(res.g1, res.g2);
    return { ...res, ...spectral, gradNorm };
  }

  loadScenario(id) {
    if (!SCENARIOS[id]) return;
    this.scenarioId = id;
    const d = SCENARIOS[id].defaults;
    this.start = [...d.start];
    this.params = { lr: d.lr, beta1: d.beta1, beta2: d.beta2, wd: d.wd };
    this.reset();
  }

  setStart(w1, w2) {
    const c = (v) => Math.max(-DOMAIN + 0.05, Math.min(DOMAIN - 0.05, v));
    this.start = [c(w1), c(w2)];
    this.reset();
  }

  reset() {
    this.stepCount = 0;
    const initEval = this.evaluate(this.start[0], this.start[1]);
    this.optimizers = {};
    for (const id of Object.keys(OPTIMIZER_META)) {
      this.optimizers[id] = {
        id,
        w1: this.start[0],
        w2: this.start[1],
        m1: 0,
        m2: 0,
        v1: 0,
        v2: 0,
        mHat1: 0,
        mHat2: 0,
        vHat1: 0,
        vHat2: 0,
        dw1: 0,
        dw2: 0,
        t: 0,
        diverged: false,
        eval: initEval,
        path: [{ w1: this.start[0], w2: this.start[1], loss: initEval.loss }],
        losses: [initEval.loss]
      };
    }
  }

  step() {
    if (this.stepCount >= MAX_STEPS) return false;
    this.stepCount += 1;

    const { lr, beta1, beta2, wd } = this.params;
    const eps = 1e-8;

    for (const id of Object.keys(this.optimizers)) {
      const opt = this.optimizers[id];
      if (!this.active[id] || opt.diverged) continue;

      opt.t += 1;
      const { g1, g2 } = opt.eval;
      let dw1 = 0;
      let dw2 = 0;

      if (id === 'sgd') {
        dw1 = -lr * g1;
        dw2 = -lr * g2;
      } else if (id === 'momentum') {
        opt.m1 = beta1 * opt.m1 + g1;
        opt.m2 = beta1 * opt.m2 + g2;
        opt.mHat1 = opt.m1;
        opt.mHat2 = opt.m2;
        dw1 = -lr * opt.m1;
        dw2 = -lr * opt.m2;
      } else if (id === 'rmsprop') {
        opt.v1 = beta2 * opt.v1 + (1 - beta2) * g1 * g1;
        opt.v2 = beta2 * opt.v2 + (1 - beta2) * g2 * g2;
        const bc2 = 1 - Math.pow(beta2, opt.t);
        opt.vHat1 = opt.v1 / bc2;
        opt.vHat2 = opt.v2 / bc2;
        dw1 = (-lr * g1) / (Math.sqrt(opt.vHat1) + eps);
        dw2 = (-lr * g2) / (Math.sqrt(opt.vHat2) + eps);
      } else if (id === 'adamw') {
        opt.m1 = beta1 * opt.m1 + (1 - beta1) * g1;
        opt.m2 = beta1 * opt.m2 + (1 - beta1) * g2;
        opt.v1 = beta2 * opt.v1 + (1 - beta2) * g1 * g1;
        opt.v2 = beta2 * opt.v2 + (1 - beta2) * g2 * g2;
        const bc1 = 1 - Math.pow(beta1, opt.t);
        const bc2 = 1 - Math.pow(beta2, opt.t);
        opt.mHat1 = opt.m1 / bc1;
        opt.mHat2 = opt.m2 / bc2;
        opt.vHat1 = opt.v1 / bc2;
        opt.vHat2 = opt.v2 / bc2;
        dw1 = -lr * (opt.mHat1 / (Math.sqrt(opt.vHat1) + eps) + wd * opt.w1);
        dw2 = -lr * (opt.mHat2 / (Math.sqrt(opt.vHat2) + eps) + wd * opt.w2);
      }

      const nw1 = opt.w1 + dw1;
      const nw2 = opt.w2 + dw2;
      opt.dw1 = dw1;
      opt.dw2 = dw2;

      if (!Number.isFinite(nw1) || !Number.isFinite(nw2) || Math.abs(nw1) > DOMAIN + 0.35 || Math.abs(nw2) > DOMAIN + 0.35) {
        opt.diverged = true;
        opt.w1 = Math.max(-DOMAIN, Math.min(DOMAIN, Number.isFinite(nw1) ? nw1 : opt.w1));
        opt.w2 = Math.max(-DOMAIN, Math.min(DOMAIN, Number.isFinite(nw2) ? nw2 : opt.w2));
        opt.eval = this.evaluate(opt.w1, opt.w2);
        opt.path.push({ w1: opt.w1, w2: opt.w2, loss: opt.eval.loss });
        opt.losses.push(opt.eval.loss);
        continue;
      }

      opt.w1 = nw1;
      opt.w2 = nw2;
      opt.eval = this.evaluate(opt.w1, opt.w2);
      opt.path.push({ w1: opt.w1, w2: opt.w2, loss: opt.eval.loss });
      opt.losses.push(opt.eval.loss);
    }

    return true;
  }

  getLeader() {
    let best = null;
    for (const id of Object.keys(this.optimizers)) {
      if (!this.active[id]) continue;
      const o = this.optimizers[id];
      if (o.diverged) continue;
      if (!best || o.eval.loss < best.eval.loss) best = o;
    }
    return best;
  }
}