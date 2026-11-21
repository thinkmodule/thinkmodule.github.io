/**
 * ThinkModule Lab — 3D Neural Loss Landscape, Hessian & Optimizer Engine (v1.0)
 * Exact analytical surfaces L(w1, w2), first-order gradients, 2x2 Hessian matrices,
 * closed-form spectral eigendecomposition, and authentic SGD / Momentum / RMSProp / AdamW steppers.
 */
(function (global) {
  "use strict";

  const DOMAIN_MIN = -2.2;
  const DOMAIN_MAX = 2.2;
  const EPS = 1e-8;

  // Smooth monotonic visual elevation compression for sculpted 3D walls + exact derivative for tangent arrows
  function visualHeight(L) {
    const safeL = Math.max(0, Number.isFinite(L) ? L : 12);
    return 0.78 * Math.log1p(1.35 * safeL);
  }

  function visualHeightDeriv(L) {
    const safeL = Math.max(0, Number.isFinite(L) ? L : 12);
    return (0.78 * 1.35) / (1 + 1.35 * safeL);
  }

  // Exact analytical landscapes: each returns { L, gx, gy, hxx, hxy, hyy }
  const LANDSCAPES = [
    {
      id: "saddle",
      name: "Monkey / Classic Saddle Valley",
      shortName: "Saddle Point",
      formula: "L = 0.36x² - 0.36y² + 0.09(x⁴+y⁴) + 0.04xy + 0.40",
      defaultStart: [1.55, 0.05],
      globalMin: [0.04, -1.41],
      eval(x, y) {
        const x2 = x * x, y2 = y * y;
        const L = 0.36 * x2 - 0.36 * y2 + 0.09 * (x2 * x2 + y2 * y2) + 0.04 * x * y + 0.40;
        const gx = 0.72 * x + 0.36 * x * x2 + 0.04 * y;
        const gy = -0.72 * y + 0.36 * y * y2 + 0.04 * x;
        const hxx = 0.72 + 1.08 * x2;
        const hyy = -0.72 + 1.08 * y2;
        const hxy = 0.04;
        return { L: Math.max(0.001, L), gx, gy, hxx, hxy, hyy };
      }
    },
    {
      id: "ravine",
      name: "Ill-Conditioned Ravine (κ = 32.5)",
      shortName: "Narrow Ravine",
      formula: "L = 0.08u² + 2.60v² + 0.02 (rotated 24°)",
      defaultStart: [-1.75, 0.95],
      globalMin: [0.0, 0.0],
      eval(x, y) {
        const theta = (24 * Math.PI) / 180;
        const c = Math.cos(theta), s = Math.sin(theta);
        const a = 0.08, b = 2.60;
        const u = x * c + y * s;
        const v = -x * s + y * c;
        const L = a * u * u + b * v * v + 0.02;
        const lu = 2 * a * u;
        const lv = 2 * b * v;
        const gx = lu * c - lv * s;
        const gy = lu * s + lv * c;
        const hxx = 2 * a * c * c + 2 * b * s * s;
        const hyy = 2 * a * s * s + 2 * b * c * c;
        const hxy = 2 * (a - b) * c * s;
        return { L, gx, gy, hxx, hxy, hyy };
      }
    },
    {
      id: "rosenbrock",
      name: "Rosenbrock Banana Canyon",
      shortName: "Rosenbrock",
      formula: "L = 0.25(1 - x)² + 1.5(y - 0.55x² + 0.5)² + 0.01",
      defaultStart: [-1.50, 0.95],
      globalMin: [1.0, 0.05],
      eval(x, y) {
        const g = y - 0.55 * x * x + 0.5;
        const L = 0.25 * (1 - x) * (1 - x) + 1.5 * g * g + 0.01;
        const gx = 0.5 * (x - 1) - 3.3 * x * g;
        const gy = 3.0 * g;
        const hxx = 0.5 - 3.3 * g + 3.63 * x * x;
        const hxy = -3.3 * x;
        const hyy = 3.0;
        return { L, gx, gy, hxx, hxy, hyy };
      }
    },
    {
      id: "local_trap",
      name: "Non-Convex Local Trap vs. Global Basin",
      shortName: "Local Min Trap",
      formula: "L = 0.14(x²+y²) - 0.98·G_global - 0.60·G_local + 1.05",
      defaultStart: [-1.55, -1.15],
      globalMin: [1.02, 0.82],
      eval(x, y) {
        const baseL = 0.14 * (x * x + y * y) + 1.05;
        let L = baseL;
        let gx = 0.28 * x;
        let gy = 0.28 * y;
        let hxx = 0.28;
        let hyy = 0.28;
        let hxy = 0.0;

        // Two Gaussian wells: deeper global basin + shallower deceptive trap
        const wells = [
          { cx: 1.05, cy: 0.85, A: 0.98, s: 0.68 },
          { cx: -0.88, cy: -0.68, A: 0.60, s: 0.48 }
        ];
        for (let i = 0; i < wells.length; i++) {
          const w = wells[i];
          const dx = x - w.cx;
          const dy = y - w.cy;
          const E = w.A * Math.exp(-(dx * dx + dy * dy) / w.s);
          L -= E;
          gx += E * ((2 * dx) / w.s);
          gy += E * ((2 * dy) / w.s);
          hxx += E * (2 / w.s - (4 * dx * dx) / (w.s * w.s));
          hyy += E * (2 / w.s - (4 * dy * dy) / (w.s * w.s));
          hxy += E * ((-4 * dx * dy) / (w.s * w.s));
        }
        return { L: Math.max(0.01, L), gx, gy, hxx, hxy, hyy };
      }
    }
  ];

  const PRESETS = {
    saddle: {
      landscapeId: "saddle",
      start: [1.55, 0.05],
      lr: 0.08,
      beta1: 0.90,
      beta2: 0.990,
      wd: 0.005,
      inspect: "adamw"
    },
    ravine: {
      landscapeId: "ravine",
      start: [-1.75, 0.95],
      lr: 0.34,
      beta1: 0.85,
      beta2: 0.990,
      wd: 0.005,
      inspect: "sgd"
    },
    rosenbrock: {
      landscapeId: "rosenbrock",
      start: [-1.50, 0.95],
      lr: 0.06,
      beta1: 0.90,
      beta2: 0.990,
      wd: 0.005,
      inspect: "adamw"
    },
    local_trap: {
      landscapeId: "local_trap",
      start: [-1.55, -1.15],
      lr: 0.14,
      beta1: 0.93,
      beta2: 0.990,
      wd: 0.005,
      inspect: "momentum"
    }
  };

  const OPTIMIZER_META = {
    sgd:      { id: "sgd",      name: "SGD",      color: "#f87171" },
    momentum: { id: "momentum", name: "Momentum", color: "#fbbf24" },
    rmsprop:  { id: "rmsprop",  name: "RMSProp",  color: "#a78bfa" },
    adamw:    { id: "adamw",    name: "AdamW",    color: "#34d399" }
  };

  function getLandscape(id) {
    return LANDSCAPES.find(l => l.id === id) || LANDSCAPES[0];
  }

  // Exact closed-form 2x2 symmetric eigendecomposition
  function eigen2x2(hxx, hxy, hyy) {
    const tr = hxx + hyy;
    const diff = hxx - hyy;
    const disc = Math.sqrt(diff * diff + 4 * hxy * hxy);
    const l1 = 0.5 * (tr + disc);
    const l2 = 0.5 * (tr - disc);

    let e1x, e1y;
    if (Math.abs(hxy) > 1e-9) {
      const vx = l1 - hyy;
      const vy = hxy;
      const norm = Math.hypot(vx, vy) || 1;
      e1x = vx / norm;
      e1y = vy / norm;
    } else if (hxx >= hyy) {
      e1x = 1; e1y = 0;
    } else {
      e1x = 0; e1y = 1;
    }
    const e2x = -e1y;
    const e2y = e1x;

    const abs1 = Math.abs(l1);
    const abs2 = Math.abs(l2);
    const kappa = Math.max(abs1, abs2) / Math.max(Math.min(abs1, abs2), 1e-3);

    let regime = "Convex Bowl";
    let regimeClass = "convex";
    if (l1 > 0.03 && l2 < -0.03) {
      regime = "Saddle Point";
      regimeClass = "saddle";
    } else if (l1 < -0.03 && l2 < -0.03) {
      regime = "Concave Peak";
      regimeClass = "danger";
    } else if (kappa > 12 && l1 > 0 && l2 > 0) {
      regime = "Steep Ravine";
      regimeClass = "saddle";
    }

    return {
      l1, l2,
      e1: [e1x, e1y],
      e2: [e2x, e2y],
      kappa,
      det: hxx * hyy - hxy * hxy,
      trace: tr,
      regime,
      regimeClass
    };
  }

  function createOptimizerState(id, startX, startY, landscape) {
    const ev = landscape.eval(startX, startY);
    const eig = eigen2x2(ev.hxx, ev.hxy, ev.hyy);
    return {
      id,
      name: OPTIMIZER_META[id].name,
      color: OPTIMIZER_META[id].color,
      x: startX,
      y: startY,
      prevX: startX,
      prevY: startY,
      t: 0,
      mx: 0, my: 0,
      vx: 0, vy: 0,
      mHatX: 0, mHatY: 0,
      vHatX: 0, vHatY: 0,
      dx: 0, dy: 0,
      L: ev.L,
      gx: ev.gx,
      gy: ev.gy,
      gradNorm: Math.hypot(ev.gx, ev.gy),
      hess: { hxx: ev.hxx, hxy: ev.hxy, hyy: ev.hyy },
      eig,
      diverged: false,
      trail: [{ x: startX, y: startY, L: ev.L }],
      lossHistory: [ev.L]
    };
  }

  function stepOptimizer(opt, landscape, params) {
    if (opt.diverged) {
      opt.lossHistory.push(opt.lossHistory[opt.lossHistory.length - 1]);
      return;
    }

    const { lr, beta1, beta2, wd } = params;
    const ev = landscape.eval(opt.x, opt.y);
    opt.prevX = opt.x;
    opt.prevY = opt.y;
    opt.t += 1;

    const gx = ev.gx;
    const gy = ev.gy;
    let dx = 0, dy = 0;

    if (opt.id === "sgd") {
      dx = -lr * (gx + wd * opt.x);
      dy = -lr * (gy + wd * opt.y);
      opt.mx = gx; opt.my = gy;
      opt.vx = 1; opt.vy = 1;
    } else if (opt.id === "momentum") {
      opt.mx = beta1 * opt.mx + gx + wd * opt.x;
      opt.my = beta1 * opt.my + gy + wd * opt.y;
      dx = -lr * opt.mx;
      dy = -lr * opt.my;
      opt.mHatX = opt.mx; opt.mHatY = opt.my;
    } else if (opt.id === "rmsprop") {
      opt.vx = beta2 * opt.vx + (1 - beta2) * gx * gx;
      opt.vy = beta2 * opt.vy + (1 - beta2) * gy * gy;
      dx = -lr * (gx / (Math.sqrt(opt.vx) + EPS) + wd * opt.x);
      dy = -lr * (gy / (Math.sqrt(opt.vy) + EPS) + wd * opt.y);
      opt.vHatX = opt.vx; opt.vHatY = opt.vy;
    } else if (opt.id === "adamw") {
      opt.mx = beta1 * opt.mx + (1 - beta1) * gx;
      opt.my = beta1 * opt.my + (1 - beta1) * gy;
      opt.vx = beta2 * opt.vx + (1 - beta2) * gx * gx;
      opt.vy = beta2 * opt.vy + (1 - beta2) * gy * gy;
      const bc1 = 1 - Math.pow(beta1, opt.t);
      const bc2 = 1 - Math.pow(beta2, opt.t);
      opt.mHatX = opt.mx / Math.max(bc1, 1e-8);
      opt.mHatY = opt.my / Math.max(bc1, 1e-8);
      opt.vHatX = opt.vx / Math.max(bc2, 1e-8);
      opt.vHatY = opt.vy / Math.max(bc2, 1e-8);
      dx = -lr * (opt.mHatX / (Math.sqrt(opt.vHatX) + EPS)) - lr * wd * opt.x;
      dy = -lr * (opt.mHatY / (Math.sqrt(opt.vHatY) + EPS)) - lr * wd * opt.y;
    }

    const nx = opt.x + dx;
    const ny = opt.y + dy;
    opt.dx = dx;
    opt.dy = dy;

    if (!Number.isFinite(nx) || !Number.isFinite(ny) || Math.abs(nx) > 2.55 || Math.abs(ny) > 2.55) {
      opt.x = Math.max(DOMAIN_MIN, Math.min(DOMAIN_MAX, Number.isFinite(nx) ? nx : opt.x));
      opt.y = Math.max(DOMAIN_MIN, Math.min(DOMAIN_MAX, Number.isFinite(ny) ? ny : opt.y));
      opt.diverged = true;
      const failEv = landscape.eval(opt.x, opt.y);
      opt.L = failEv.L;
      opt.gx = failEv.gx;
      opt.gy = failEv.gy;
      opt.gradNorm = Math.hypot(failEv.gx, failEv.gy);
      opt.hess = { hxx: failEv.hxx, hxy: failEv.hxy, hyy: failEv.hyy };
      opt.eig = eigen2x2(failEv.hxx, failEv.hxy, failEv.hyy);
      opt.trail.push({ x: opt.x, y: opt.y, L: opt.L });
      opt.lossHistory.push(opt.L);
      return;
    }

    opt.x = nx;
    opt.y = ny;
    const nextEv = landscape.eval(opt.x, opt.y);
    opt.L = nextEv.L;
    opt.gx = nextEv.gx;
    opt.gy = nextEv.gy;
    opt.gradNorm = Math.hypot(nextEv.gx, nextEv.gy);
    opt.hess = { hxx: nextEv.hxx, hxy: nextEv.hxy, hyy: nextEv.hyy };
    opt.eig = eigen2x2(nextEv.hxx, nextEv.hxy, nextEv.hyy);

    opt.trail.push({ x: opt.x, y: opt.y, L: opt.L });
    if (opt.trail.length > 260) opt.trail.shift();
    opt.lossHistory.push(opt.L);
    if (opt.lossHistory.length > 260) opt.lossHistory.shift();
  }

  function buildDiagnosis(inspected, allOpts, landscape, params) {
    const eig = inspected.eig;
    if (inspected.diverged) {
      const limit = 2 / Math.max(Math.abs(eig.l1), 0.1);
      return {
        severity: "danger",
        title: `${inspected.name} DIVERGED — Learning Rate Exceeded Curvature Limit`,
        body: `Step size η = ${params.lr.toFixed(3)} overshot the Hessian stability bound (2/λ_max ≈ ${limit.toFixed(2)}), launching weights out of the basin. Lower η or switch to AdamW/RMSProp whose √v_t term dampens steep gradients automatically.`
      };
    }

    if (eig.regime === "Saddle Point") {
      return {
        severity: "warn",
        title: `Saddle Point Detected (λ₁ = +${eig.l1.toFixed(2)}, λ₂ = ${eig.l2.toFixed(2)})`,
        body: `Opposite Hessian eigenvalue signs prove indefinite curvature: the surface curves UP along green e₁ and DOWN along red e₂. Vanilla SGD slows dramatically when ‖∇L‖ is small, while AdamW rescales the tiny escape gradient via √v̂_t to drop cleanly into the valley.`
      };
    }

    if (eig.kappa > 12) {
      return {
        severity: "warn",
        title: `Ill-Conditioned Canyon (Condition Number κ = ${eig.kappa.toFixed(1)})`,
        body: `Curvature is ${eig.kappa.toFixed(1)}× steeper across the valley wall than along the valley floor. Notice how SGD zig-zags violently across the steep axis while adaptive optimizers (RMSProp / AdamW) equalize per-axis updates to glide straight toward the minimum.`
      };
    }

    if (inspected.gradNorm < 0.035 && inspected.t > 12) {
      const best = allOpts.filter(o => !o.diverged).sort((a, b) => a.L - b.L)[0];
      if (best && inspected.L > best.L + 0.12) {
        return {
          severity: "warn",
          title: `Trapped in Shallow Local Minimum (L = ${inspected.L.toFixed(3)} vs. ${best.name} ${best.L.toFixed(3)})`,
          body: `${inspected.name} converged where ∇L ≈ 0 inside a suboptimal basin. Higher Momentum (β₁ ≈ 0.93) or a wider initial step carries enough kinetic energy to clear the ridge into the deeper global basin.`
        };
      }
      return {
        severity: "good",
        title: `Converged to Minimum (${inspected.name}: L = ${inspected.L.toFixed(4)}, ‖∇L‖ = ${inspected.gradNorm.toFixed(4)})`,
        body: `Both Hessian eigenvalues are positive (λ₁ = ${eig.l1.toFixed(2)}, λ₂ = ${eig.l2.toFixed(2)}), mathematically certifying a stable local bowl. Click anywhere on the 2D Contour Map below to test a new starting coordinate.`
      };
    }

    return {
      severity: "info",
      title: `Active Descent on ${landscape.shortName} — Step ${inspected.t} (‖∇L‖ = ${inspected.gradNorm.toFixed(3)})`,
      body: `Tracking ${inspected.name} at θ = (${inspected.x.toFixed(2)}, ${inspected.y.toFixed(2)}). Inspect the 3D cyan gradient arrow (−∇L) and the 5-stage execution circuit below to watch how Backprop derivatives transform into parameter updates.`
    };
  }

  global.LossEngine = {
    DOMAIN_MIN,
    DOMAIN_MAX,
    LANDSCAPES,
    PRESETS,
    OPTIMIZER_META,
    visualHeight,
    visualHeightDeriv,
    getLandscape,
    eigen2x2,
    createOptimizerState,
    stepOptimizer,
    buildDiagnosis
  };
})(window);