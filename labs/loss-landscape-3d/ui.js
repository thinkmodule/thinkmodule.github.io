/**
 * ThinkModule Lab — 3D Loss Landscape, Hessian & Optimizer Studio Renderer (v1.0)
 * Custom zero-dependency 60 FPS 3D shaded mesh renderer, Painter's algorithm,
 * tangent-plane 3D gradient & Hessian eigenvectors, live 5-stage Backprop circuit,
 * clickable 2D contour heatmap, and log-scale convergence telemetry.
 */
(function () {
  "use strict";

  const Engine = window.LossEngine;
  if (!Engine) {
    console.error("LossEngine not loaded.");
    return;
  }

  const $ = id => document.getElementById(id);

  const state = {
    landscapeId: Engine.PRESETS.saddle.landscapeId,
    activePreset: "saddle",
    start: [...Engine.PRESETS.saddle.start],
    params: {
      lr: Engine.PRESETS.saddle.lr,
      beta1: Engine.PRESETS.saddle.beta1,
      beta2: Engine.PRESETS.saddle.beta2,
      wd: Engine.PRESETS.saddle.wd
    },
    inspectId: Engine.PRESETS.saddle.inspect,
    enabled: { sgd: true, momentum: true, rmsprop: true, adamw: true },
    optimizers: {},
    stepCount: 0,
    maxSteps: 220,

    // Playback & Animation
    playing: true,
    speedMult: 1.0,
    stepAccum: 0,
    animTime: 0,
    lastTs: 0,
    showVectors: true,
    autoOrbit: false,

    // 3D Camera
    cam: {
      yaw: -0.68,
      pitch: 0.62,
      zoom: 1.04,
      dragging: false,
      lastX: 0,
      lastY: 0
    },

    // Precomputed surface mesh & contour bitmap cache
    meshGrid: null,
    contourBitmap: null
  };

  const dom = {
    presetGrid: $("presetGrid"),
    landscapeSelect: $("landscapeSelect"),
    inspectSelect: $("inspectSelect"),
    optToggleGrid: $("optToggleGrid"),
    lrSlider: $("lrSlider"),
    lrVal: $("lrVal"),
    beta1Slider: $("beta1Slider"),
    beta1Val: $("beta1Val"),
    beta2Slider: $("beta2Slider"),
    beta2Val: $("beta2Val"),
    wdSlider: $("wdSlider"),
    wdVal: $("wdVal"),
    pytorchPreview: $("pytorchPreview"),
    heroBadges: $("heroBadges"),

    playPauseBtn: $("playPauseBtn"),
    stepOnceBtn: $("stepOnceBtn"),
    resetRunBtn: $("resetRunBtn"),
    speedCalmBtn: $("speedCalmBtn"),
    speedNormalBtn: $("speedNormalBtn"),
    speedFastBtn: $("speedFastBtn"),
    vectorsBtn: $("vectorsBtn"),
    autoOrbitBtn: $("autoOrbitBtn"),
    resetCamBtn: $("resetCamBtn"),

    canvas3d: $("canvas3d"),
    canvasFlow: $("canvasFlow"),
    canvasContour: $("canvasContour"),
    canvasLoss: $("canvasLoss"),

    kpiStep: $("kpiStep"),
    kpiLoss: $("kpiLoss"),
    kpiLossSub: $("kpiLossSub"),
    kpiGrad: $("kpiGrad"),
    kpiKappa: $("kpiKappa"),
    kpiCurvature: $("kpiCurvature"),
    kpiCurvSub: $("kpiCurvSub"),
    kpiWinner: $("kpiWinner"),
    kpiWinnerSub: $("kpiWinnerSub"),

    diagnosisBox: $("diagnosisBox"),
    diagnosisTitle: $("diagnosisTitle"),
    diagnosisBody: $("diagnosisBody"),

    flowSubLabel: $("flowSubLabel"),
    contourHint: $("contourHint"),
    contourCoords: $("contourCoords"),
    lossMeta: $("lossMeta"),

    matrixInspectorBadge: $("matrixInspectorBadge"),
    matTheta: $("matTheta"),
    matHess: $("matHess"),
    matEig: $("matEig"),
    matUpdate: $("matUpdate")
  };

  // ============================================================================
  // HELPERS & COLORMAPS
  // ============================================================================

  function setupCanvas(canvas, cssHeight) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(Math.floor(rect.width), 280);
    const height = cssHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function roundRect(ctx, x, y, w, h, r) {
    const rad = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rad, y);
    ctx.arcTo(x + w, y, x + w, y + h, rad);
    ctx.arcTo(x + w, y + h, x, y + h, rad);
    ctx.arcTo(x, y + h, x, y, rad);
    ctx.arcTo(x, y, x + w, y, rad);
    ctx.closePath();
  }

  // Scientific Viridis-Cyan-Amber palette for height t in [0, 1]
  function heightColorRGB(t) {
    const u = Math.max(0, Math.min(1, t));
    const stops = [
      [0.00, [14, 38, 88]],
      [0.24, [15, 118, 152]],
      [0.52, [40, 178, 150]],
      [0.78, [232, 168, 56]],
      [1.00, [238, 86, 104]]
    ];
    for (let i = 0; i < stops.length - 1; i++) {
      const [t0, c0] = stops[i];
      const [t1, c1] = stops[i + 1];
      if (u <= t1) {
        const f = (u - t0) / Math.max(t1 - t0, 1e-6);
        return [
          Math.round(c0[0] + (c1[0] - c0[0]) * f),
          Math.round(c0[1] + (c1[1] - c0[1]) * f),
          Math.round(c0[2] + (c1[2] - c0[2]) * f)
        ];
      }
    }
    return [238, 86, 104];
  }

  function fmtSigned(v, dec) {
    const d = dec === undefined ? 3 : dec;
    if (!Number.isFinite(v)) return "NaN";
    return (v >= 0 ? "+" : "") + v.toFixed(d);
  }

  // ============================================================================
  // SURFACE MESH & CONTOUR CACHE
  // ============================================================================

  function rebuildSurfaceCaches() {
    const land = Engine.getLandscape(state.landscapeId);
    const N = 34;
    const min = Engine.DOMAIN_MIN;
    const max = Engine.DOMAIN_MAX;
    const verts = [];
    let zMin = Infinity, zMax = -Infinity;

    for (let j = 0; j <= N; j++) {
      const row = [];
      const y = min + (j / N) * (max - min);
      for (let i = 0; i <= N; i++) {
        const x = min + (i / N) * (max - min);
        const ev = land.eval(x, y);
        const z = Engine.visualHeight(ev.L);
        const df = Engine.visualHeightDeriv(ev.L);
        // Exact surface normal for directional 3D lighting
        const nx = -df * ev.gx;
        const ny = -df * ev.gy;
        const nz = 1.0;
        const nLen = Math.hypot(nx, ny, nz) || 1;
        if (z < zMin) zMin = z;
        if (z > zMax) zMax = z;
        row.push({ x, y, z, L: ev.L, nx: nx / nLen, ny: ny / nLen, nz: nz / nLen });
      }
      verts.push(row);
    }
    state.meshGrid = { N, verts, zMin, zMax: Math.max(zMax, zMin + 0.25) };

    // Build crisp 180x180 offscreen contour heatmap canvas
    const size = 180;
    const off = document.createElement("canvas");
    off.width = size;
    off.height = size;
    const octx = off.getContext("2d");
    const img = octx.createImageData(size, size);
    const data = img.data;

    for (let py = 0; py < size; py++) {
      const y = max - (py / (size - 1)) * (max - min);
      for (let px = 0; px < size; px++) {
        const x = min + (px / (size - 1)) * (max - min);
        const ev = land.eval(x, y);
        const z = Engine.visualHeight(ev.L);
        const t = (z - zMin) / Math.max(zMax - zMin, 1e-4);
        const rgb = heightColorRGB(t);

        // Crisp elevation contour isolines
        const iso = Math.abs(((t * 14) % 1) - 0.5);
        const lineBoost = iso < 0.045 ? 0.32 : 0;

        const idx = (py * size + px) * 4;
        data[idx] = Math.min(255, Math.round(rgb[0] * 0.62 + lineBoost * 165));
        data[idx + 1] = Math.min(255, Math.round(rgb[1] * 0.62 + lineBoost * 195));
        data[idx + 2] = Math.min(255, Math.round(rgb[2] * 0.66 + lineBoost * 225));
        data[idx + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    state.contourBitmap = off;
  }

  // ============================================================================
  // SIMULATION CONTROL
  // ============================================================================

  function resetSimulation() {
    const land = Engine.getLandscape(state.landscapeId);
    state.stepCount = 0;
    state.stepAccum = 0;
    state.optimizers = {};
    ["sgd", "momentum", "rmsprop", "adamw"].forEach(id => {
      state.optimizers[id] = Engine.createOptimizerState(id, state.start[0], state.start[1], land);
    });
    updateUIReadouts();
    drawContourMap();
    drawLossChart();
  }

  function advanceOneStep() {
    if (state.stepCount >= state.maxSteps) return;
    const land = Engine.getLandscape(state.landscapeId);
    state.stepCount += 1;
    ["sgd", "momentum", "rmsprop", "adamw"].forEach(id => {
      if (state.enabled[id]) {
        Engine.stepOptimizer(state.optimizers[id], land, state.params);
      }
    });
    updateUIReadouts();
    drawContourMap();
    drawLossChart();
  }

  function syncSidebar() {
    dom.landscapeSelect.value = state.landscapeId;
    dom.inspectSelect.value = state.inspectId;
    dom.lrSlider.value = String(state.params.lr);
    dom.lrVal.textContent = state.params.lr.toFixed(3);
    dom.lrVal.className = "field-val" + (state.params.lr >= 0.30 ? " warn" : "");

    dom.beta1Slider.value = String(state.params.beta1);
    dom.beta1Val.textContent = state.params.beta1.toFixed(2);
    dom.beta2Slider.value = String(state.params.beta2);
    dom.beta2Val.textContent = state.params.beta2.toFixed(3);
    dom.wdSlider.value = String(state.params.wd);
    dom.wdVal.textContent = state.params.wd.toFixed(3);

    dom.presetGrid.querySelectorAll(".preset-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.preset === state.activePreset);
    });
    dom.optToggleGrid.querySelectorAll(".opt-chip").forEach(b => {
      b.classList.toggle("active", !!state.enabled[b.dataset.opt]);
    });

    const optClass = state.inspectId === "adamw" ? "AdamW"
      : state.inspectId === "rmsprop" ? "RMSprop"
      : "SGD";
    const extraArg = state.inspectId === "momentum"
      ? `, momentum=<span class="num">${state.params.beta1.toFixed(2)}</span>`
      : state.inspectId === "adamw"
        ? `, betas=(<span class="num">${state.params.beta1.toFixed(2)}</span>,<span class="num">${state.params.beta2.toFixed(3)}</span>)`
        : state.inspectId === "rmsprop"
          ? `, alpha=<span class="num">${state.params.beta2.toFixed(3)}</span>`
          : "";
    dom.pytorchPreview.innerHTML =
`<span class="cm"># Looks innocent in Python:</span>
opt = torch.optim.<span class="fn">${optClass}</span>(
  params, lr=<span class="num">${state.params.lr.toFixed(3)}</span>${extraArg}
)
loss.<span class="fn">backward</span>()   <span class="cm"># computes ∇L</span>
opt.<span class="fn">step</span>()        <span class="cm"># updates θ</span>`;
  }

  function updateUIReadouts() {
    const land = Engine.getLandscape(state.landscapeId);
    const ins = state.optimizers[state.inspectId] || state.optimizers.adamw;
    if (!ins) return;

    const activeList = ["sgd", "momentum", "rmsprop", "adamw"]
      .filter(id => state.enabled[id])
      .map(id => state.optimizers[id]);

    const sortedValid = activeList.filter(o => !o.diverged).sort((a, b) => a.L - b.L);
    const winner = sortedValid[0] || ins;

    // Hero badges
    const badges = [];
    badges.push(`<span class="badge">${land.shortName}</span>`);
    badges.push(`<span class="badge ${ins.eig.regimeClass}">${ins.eig.regime} (&lambda;&sub1;=${fmtSigned(ins.eig.l1, 2)}, &lambda;&sub2;=${fmtSigned(ins.eig.l2, 2)})</span>`);
    const anyDiv = activeList.some(o => o.diverged);
    if (anyDiv) {
      const divNames = activeList.filter(o => o.diverged).map(o => o.name).join(", ");
      badges.push(`<span class="badge danger">⚠ Diverged: ${divNames}</span>`);
    } else {
      badges.push(`<span class="badge">Leader: ${winner.name} (${winner.L.toFixed(4)})</span>`);
    }
    dom.heroBadges.innerHTML = badges.join("");

    // KPIs
    dom.kpiStep.textContent = `${state.stepCount} / ${state.maxSteps}`;
    dom.kpiLoss.textContent = ins.diverged ? "EXPLODED" : ins.L.toFixed(4);
    dom.kpiLoss.className = "kpi-val " + (ins.diverged ? "red" : ins.L < 0.15 ? "green" : "cyan");
    dom.kpiLossSub.textContent = `${ins.name} at (${ins.x.toFixed(2)}, ${ins.y.toFixed(2)})`;

    dom.kpiGrad.textContent = ins.diverged ? "∞" : ins.gradNorm.toFixed(4);
    dom.kpiGrad.className = "kpi-val " + (ins.diverged ? "red" : "violet");

    dom.kpiKappa.textContent = ins.eig.kappa > 999 ? ">999" : ins.eig.kappa.toFixed(1);
    dom.kpiKappa.className = "kpi-val " + (ins.eig.kappa > 12 ? "amber" : "cyan");

    dom.kpiCurvature.textContent = ins.eig.regime;
    dom.kpiCurvature.className = "kpi-val " + (ins.eig.regimeClass === "convex" ? "green" : ins.eig.regimeClass === "danger" ? "red" : "amber");
    dom.kpiCurvSub.textContent = `det(H) = ${fmtSigned(ins.eig.det, 2)}`;

    dom.kpiWinner.textContent = winner.name;
    dom.kpiWinnerSub.textContent = winner.diverged ? "All diverged" : `L = ${winner.L.toFixed(4)}`;

    // Diagnosis
    const diag = Engine.buildDiagnosis(ins, activeList, land, state.params);
    dom.diagnosisBox.className = `diagnosis ${diag.severity}`;
    dom.diagnosisTitle.textContent = diag.title;
    dom.diagnosisBody.textContent = diag.body;

    // Labels & Live Matrices
    dom.flowSubLabel.textContent = `Tracing ${ins.name} (Step ${ins.t}): θ → Forward/Hessian → Backprop ∇L → Moment State → θ_next`;
    dom.matrixInspectorBadge.textContent = `Active: ${ins.name} (Step ${ins.t})`;
    dom.contourCoords.textContent = `Start: (${state.start[0].toFixed(2)}, ${state.start[1].toFixed(2)})`;
    dom.lossMeta.textContent = `Best: ${winner.name} (${winner.L.toFixed(4)})`;

    dom.matTheta.textContent =
`θ_t = [ ${fmtSigned(ins.x, 4)} ]   (w₁)
      [ ${fmtSigned(ins.y, 4)} ]   (w₂)

∇L  = [ ${fmtSigned(ins.gx, 4)} ]   (∂L/∂w₁)
      [ ${fmtSigned(ins.gy, 4)} ]   (∂L/∂w₂)`;

    dom.matHess.textContent =
`H = [ ${fmtSigned(ins.hess.hxx, 3)}   ${fmtSigned(ins.hess.hxy, 3)} ]
    [ ${fmtSigned(ins.hess.hxy, 3)}   ${fmtSigned(ins.hess.hyy, 3)} ]

tr(H)  = ${fmtSigned(ins.eig.trace, 3)}
det(H) = ${fmtSigned(ins.eig.det, 3)}`;

    dom.matEig.textContent =
`λ₁ = ${fmtSigned(ins.eig.l1, 3)}  e₁ = [${fmtSigned(ins.eig.e1[0], 2)}, ${fmtSigned(ins.eig.e1[1], 2)}]
λ₂ = ${fmtSigned(ins.eig.l2, 3)}  e₂ = [${fmtSigned(ins.eig.e2[0], 2)}, ${fmtSigned(ins.eig.e2[1], 2)}]

κ = |λ_max / λ_min| = ${ins.eig.kappa > 999 ? ">999" : ins.eig.kappa.toFixed(2)}
Regime: ${ins.eig.regime}`;

    const stateStr = ins.id === "adamw"
      ? `m̂_t = [${fmtSigned(ins.mHatX, 3)}, ${fmtSigned(ins.mHatY, 3)}]\nv̂_t = [${ins.vHatX.toFixed(4)}, ${ins.vHatY.toFixed(4)}]`
      : ins.id === "momentum"
        ? `v_t = [${fmtSigned(ins.mx, 3)}, ${fmtSigned(ins.my, 3)}]\nβ₁  = ${state.params.beta1.toFixed(2)}`
        : ins.id === "rmsprop"
          ? `s_t = [${ins.vx.toFixed(4)}, ${ins.vy.toFixed(4)}]\nβ₂  = ${state.params.beta2.toFixed(3)}`
          : `No memory state (stateless)\nη   = ${state.params.lr.toFixed(3)}`;

    dom.matUpdate.textContent =
`${stateStr}

Δθ  = [ ${fmtSigned(ins.dx, 4)} ]
      [ ${fmtSigned(ins.dy, 4)} ]`;
  }

  // ============================================================================
  // CANVAS 1: INTERACTIVE 3D SHADED SURFACE + PAINTER'S ALGORITHM
  // ============================================================================

  function project3D(wx, wy, wz, width, height) {
    const yaw = state.cam.yaw;
    const pitch = state.cam.pitch;
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);

    // Center Z vertically around mid-mountain
    const zc = wz - 0.85;

    // Rotate around Z (yaw), then tilt by pitch
    const rx = wx * cy - wy * sy;
    const ry = wx * sy + wy * cy;

    const camY = ry * cp - zc * sp;
    const camZ = ry * sp + zc * cp;

    const fov = Math.min(width, height) * 0.98 * state.cam.zoom;
    const dist = 6.4;
    const depth = dist - camY;
    const scale = fov / Math.max(depth, 0.8);

    return {
      sx: width * 0.5 + rx * scale,
      sy: height * 0.55 - camZ * scale,
      depth,
      scale
    };
  }

  function draw3DScene() {
    const grid = state.meshGrid;
    if (!grid) return;

    const { ctx, width, height } = setupCanvas(dom.canvas3d, 440);
    ctx.clearRect(0, 0, width, height);

    // Subtle cosmic backdrop radial
    const bgGrad = ctx.createRadialGradient(width * 0.5, height * 0.45, 30, width * 0.5, height * 0.5, width * 0.65);
    bgGrad.addColorStop(0, "rgba(22, 40, 76, 0.36)");
    bgGrad.addColorStop(1, "rgba(5, 10, 20, 0)");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    // 1. Draw 3D Base Floor Axes (w1, w2, L)
    const min = Engine.DOMAIN_MIN, max = Engine.DOMAIN_MAX;
    const baseZ = -0.05;
    const c00 = project3D(min, min, baseZ, width, height);
    const c10 = project3D(max, min, baseZ, width, height);
    const c11 = project3D(max, max, baseZ, width, height);
    const c01 = project3D(min, max, baseZ, width, height);
    const zTop = project3D(min, min, 2.35, width, height);

    ctx.strokeStyle = "rgba(56, 189, 248, 0.22)";
    ctx.lineWidth = 1.1;
    ctx.beginPath();
    ctx.moveTo(c00.sx, c00.sy); ctx.lineTo(c10.sx, c10.sy);
    ctx.lineTo(c11.sx, c11.sy); ctx.lineTo(c01.sx, c01.sy);
    ctx.closePath();
    ctx.stroke();

    // Vertical Loss Z-axis
    ctx.strokeStyle = "rgba(167, 139, 250, 0.32)";
    ctx.beginPath();
    ctx.moveTo(c00.sx, c00.sy);
    ctx.lineTo(zTop.sx, zTop.sy);
    ctx.stroke();

    ctx.font = "600 10.5px ui-monospace, Menlo, monospace";
    ctx.fillStyle = "#94a3b8";
    ctx.textAlign = "center";
    ctx.fillText("w₁ (+2.2)", c10.sx, c10.sy + 15);
    ctx.fillText("w₂ (+2.2)", c01.sx, c01.sy + 15);
    ctx.fillStyle = "#a78bfa";
    ctx.fillText("Loss L(θ)", zTop.sx, zTop.sy - 8);

    // 2. Project all mesh vertices once
    const N = grid.N;
    const proj = new Array(N + 1);
    for (let j = 0; j <= N; j++) {
      proj[j] = new Array(N + 1);
      for (let i = 0; i <= N; i++) {
        const v = grid.verts[j][i];
        proj[j][i] = project3D(v.x, v.y, v.z, width, height);
      }
    }

    // 3. Build quads and sort back-to-front (Painter's Algorithm)
    const light = { x: 0.45, y: -0.45, z: 0.77 };
    const quads = [];
    const zSpan = Math.max(grid.zMax - grid.zMin, 0.1);

    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const p0 = proj[j][i];
        const p1 = proj[j][i + 1];
        const p2 = proj[j + 1][i + 1];
        const p3 = proj[j + 1][i];
        const avgDepth = (p0.depth + p1.depth + p2.depth + p3.depth) * 0.25;

        const v0 = grid.verts[j][i];
        const v1 = grid.verts[j][i + 1];
        const v2 = grid.verts[j + 1][i + 1];
        const v3 = grid.verts[j + 1][i];
        const avgZ = (v0.z + v1.z + v2.z + v3.z) * 0.25;

        const nx = (v0.nx + v1.nx + v2.nx + v3.nx) * 0.25;
        const ny = (v0.ny + v1.ny + v2.ny + v3.ny) * 0.25;
        const nz = (v0.nz + v1.nz + v2.nz + v3.nz) * 0.25;

        const ndotl = Math.max(0, nx * light.x + ny * light.y + nz * light.z);
        const shade = 0.44 + 0.62 * ndotl;
        const t = (avgZ - grid.zMin) / zSpan;

        quads.push({ p0, p1, p2, p3, avgDepth, t, shade });
      }
    }
    quads.sort((a, b) => b.avgDepth - a.avgDepth);

    // Draw shaded 3D surface quads
    for (let k = 0; k < quads.length; k++) {
      const q = quads[k];
      const rgb = heightColorRGB(q.t);
      const r = Math.min(255, Math.round(rgb[0] * q.shade));
      const g = Math.min(255, Math.round(rgb[1] * q.shade));
      const b = Math.min(255, Math.round(rgb[2] * q.shade));

      ctx.beginPath();
      ctx.moveTo(q.p0.sx, q.p0.sy);
      ctx.lineTo(q.p1.sx, q.p1.sy);
      ctx.lineTo(q.p2.sx, q.p2.sy);
      ctx.lineTo(q.p3.sx, q.p3.sy);
      ctx.closePath();

      ctx.fillStyle = `rgba(${r},${g},${b},0.84)`;
      ctx.fill();
      ctx.strokeStyle = `rgba(10, 20, 38, 0.42)`;
      ctx.lineWidth = 0.65;
      ctx.stroke();
    }

    // 4. Global Minimum Beacon Star on Surface
    const land = Engine.getLandscape(state.landscapeId);
    if (land.globalMin) {
      const [gx, gy] = land.globalMin;
      const gev = land.eval(gx, gy);
      const gp = project3D(gx, gy, Engine.visualHeight(gev.L) + 0.03, width, height);
      const pulse = 5 + 2.5 * Math.sin(state.animTime * 3);
      ctx.strokeStyle = "rgba(56, 189, 248, 0.75)";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(gp.sx, gp.sy, pulse, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = "#e0f2fe";
      ctx.font = "700 9.5px ui-monospace, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.fillText("★ Global Min", gp.sx, gp.sy - 11);
    }

    // 5. Optimizer Glowing 3D Trails
    const activeIds = ["sgd", "momentum", "rmsprop", "adamw"].filter(id => state.enabled[id]);
    activeIds.forEach(id => {
      const opt = state.optimizers[id];
      if (!opt || opt.trail.length < 2) return;
      ctx.save();
      ctx.strokeStyle = opt.color;
      ctx.lineWidth = id === state.inspectId ? 3.0 : 2.1;
      ctx.shadowColor = opt.color;
      ctx.shadowBlur = id === state.inspectId ? 10 : 5;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      for (let i = 0; i < opt.trail.length; i++) {
        const pt = opt.trail[i];
        const p = project3D(pt.x, pt.y, Engine.visualHeight(pt.L) + 0.035, width, height);
        if (i === 0) ctx.moveTo(p.sx, p.sy);
        else ctx.lineTo(p.sx, p.sy);
      }
      ctx.stroke();
      ctx.restore();
    });

    // 6. 3D Tangent-Plane Gradient (-∇L) & Hessian Eigenvectors at Inspected Optimizer
    const ins = state.optimizers[state.inspectId];
    if (state.showVectors && ins && !ins.diverged && state.enabled[state.inspectId]) {
      const zBase = Engine.visualHeight(ins.L) + 0.04;
      const df = Engine.visualHeightDeriv(ins.L);
      const pStart = project3D(ins.x, ins.y, zBase, width, height);

      // Helper to project a 2D direction vector onto the 3D tangent plane
      const draw3DArrow = (vx, vy, scaleLen, color, label, bothWays) => {
        const dz = df * (ins.gx * vx + ins.gy * vy);
        const len3d = Math.hypot(vx, vy, dz) || 1;
        const ux = (vx / len3d) * scaleLen;
        const uy = (vy / len3d) * scaleLen;
        const uz = (dz / len3d) * scaleLen;

        const pEnd = project3D(ins.x + ux, ins.y + uy, zBase + uz, width, height);
        const pNeg = bothWays ? project3D(ins.x - ux, ins.y - uy, zBase - uz, width, height) : pStart;

        ctx.save();
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = bothWays ? 2.0 : 2.8;
        ctx.shadowColor = color;
        ctx.shadowBlur = 8;

        ctx.beginPath();
        ctx.moveTo(pNeg.sx, pNeg.sy);
        ctx.lineTo(pEnd.sx, pEnd.sy);
        ctx.stroke();

        // Arrowhead
        const ang = Math.atan2(pEnd.sy - pStart.sy, pEnd.sx - pStart.sx);
        const head = 8;
        ctx.beginPath();
        ctx.moveTo(pEnd.sx, pEnd.sy);
        ctx.lineTo(pEnd.sx - head * Math.cos(ang - 0.42), pEnd.sy - head * Math.sin(ang - 0.42));
        ctx.lineTo(pEnd.sx - head * Math.cos(ang + 0.42), pEnd.sy - head * Math.sin(ang + 0.42));
        ctx.closePath();
        ctx.fill();

        ctx.shadowBlur = 0;
        ctx.font = "700 10px ui-monospace, Menlo, monospace";
        ctx.textAlign = "left";
        ctx.fillText(label, pEnd.sx + 6, pEnd.sy + 3);
        ctx.restore();
      };

      // Hessian principal axes e1, e2 (green = positive curvature, coral = negative curvature)
      const c1 = ins.eig.l1 >= 0 ? "#34d399" : "#f87171";
      const c2 = ins.eig.l2 >= 0 ? "#34d399" : "#f87171";
      draw3DArrow(ins.eig.e1[0], ins.eig.e1[1], 0.48, c1, `e₁(${fmtSigned(ins.eig.l1, 1)})`, true);
      draw3DArrow(ins.eig.e2[0], ins.eig.e2[1], 0.42, c2, `e₂(${fmtSigned(ins.eig.l2, 1)})`, true);

      // Negative Gradient Arrow (-∇L)
      if (ins.gradNorm > 0.02) {
        const gLen = Math.min(0.75, 0.25 + ins.gradNorm * 0.22);
        draw3DArrow(-ins.gx / ins.gradNorm, -ins.gy / ins.gradNorm, gLen, "#38bdf8", "−∇L", false);
      }
    }

    // 7. Depth-Sorted 3D Optimizer Spheres + Labels
    const spheres = activeIds.map(id => {
      const opt = state.optimizers[id];
      const z = Engine.visualHeight(opt.L) + 0.05;
      const p = project3D(opt.x, opt.y, z, width, height);
      return { opt, p };
    }).sort((a, b) => b.p.depth - a.p.depth);

    spheres.forEach(({ opt, p }) => {
      const isIns = opt.id === state.inspectId;
      const rad = isIns ? 8.5 : 6.8;

      // Halo ring for inspected optimizer
      if (isIns) {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, rad + 4, 0, Math.PI * 2);
        ctx.stroke();
      }

      // 3D glossy sphere shading
      const sg = ctx.createRadialGradient(p.sx - rad * 0.3, p.sy - rad * 0.35, 1, p.sx, p.sy, rad);
      sg.addColorStop(0, "#ffffff");
      sg.addColorStop(0.35, opt.color);
      sg.addColorStop(1, "#0b1222");

      ctx.fillStyle = sg;
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, rad, 0, Math.PI * 2);
      ctx.fill();

      // Label pill
      const tag = opt.diverged ? `${opt.name} ✕` : opt.name;
      ctx.font = `${isIns ? "700" : "600"} 10px ui-monospace, Menlo, monospace`;
      const tw = ctx.measureText(tag).width + 10;
      const lx = p.sx - tw / 2;
      const ly = p.sy - rad - 21;
      roundRect(ctx, lx, ly, tw, 15, 4);
      ctx.fillStyle = "rgba(5, 10, 20, 0.82)";
      ctx.fill();
      ctx.strokeStyle = opt.color;
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = "#f8fafc";
      ctx.textAlign = "center";
      ctx.fillText(tag, p.sx, ly + 11);
    });
  }

  // ============================================================================
  // CANVAS 2: LIVE 5-STAGE "FROM-HERE-TO-THERE" MATHEMATICAL PIPELINE
  // ============================================================================

  function drawMathFlow() {
    const ins = state.optimizers[state.inspectId];
    if (!ins) return;

    const { ctx, width, height } = setupCanvas(dom.canvasFlow, 215);
    ctx.clearRect(0, 0, width, height);

    const padX = 16;
    const gap = Math.max(18, Math.min(34, width * 0.028));
    const boxW = (width - padX * 2 - gap * 4) / 5;
    const boxY = 26;
    const boxH = 128;

    const stageStateLine = ins.id === "adamw"
      ? [`m̂: ${fmtSigned(ins.mHatX, 2)}, ${fmtSigned(ins.mHatY, 2)}`, `√v̂: ${Math.sqrt(ins.vHatX).toFixed(2)}, ${Math.sqrt(ins.vHatY).toFixed(2)}`]
      : ins.id === "momentum"
        ? [`v_t: ${fmtSigned(ins.mx, 2)}, ${fmtSigned(ins.my, 2)}`, `β₁ = ${state.params.beta1.toFixed(2)}`]
        : ins.id === "rmsprop"
          ? [`√s: ${Math.sqrt(ins.vx).toFixed(2)}, ${Math.sqrt(ins.vy).toFixed(2)}`, `β₂ = ${state.params.beta2.toFixed(3)}`]
          : [`Stateless SGD`, `η = ${state.params.lr.toFixed(3)}`];

    const stages = [
      {
        num: "01",
        title: "WEIGHTS θ_t",
        color: "#38bdf8",
        lines: [`w₁ = ${fmtSigned(ins.prevX, 3)}`, `w₂ = ${fmtSigned(ins.prevY, 3)}`, `Step t = ${ins.t}`]
      },
      {
        num: "02",
        title: "FORWARD & HESS",
        color: "#a78bfa",
        lines: [`L = ${ins.L.toFixed(4)}`, `λ₁ = ${fmtSigned(ins.eig.l1, 2)}`, `λ₂ = ${fmtSigned(ins.eig.l2, 2)}`]
      },
      {
        num: "03",
        title: "BACKPROP ∇L",
        color: "#fbbf24",
        lines: [`∂w₁ = ${fmtSigned(ins.gx, 3)}`, `∂w₂ = ${fmtSigned(ins.gy, 3)}`, `‖∇L‖ = ${ins.gradNorm.toFixed(3)}`]
      },
      {
        num: "04",
        title: `${ins.name.toUpperCase()} STATE`,
        color: ins.color,
        lines: [stageStateLine[0], stageStateLine[1], `WD λ = ${state.params.wd.toFixed(3)}`]
      },
      {
        num: "05",
        title: "UPDATE θ_t+1",
        color: "#34d399",
        lines: [`Δw₁ = ${fmtSigned(ins.dx, 3)}`, `Δw₂ = ${fmtSigned(ins.dy, 3)}`, `→ (${ins.x.toFixed(2)}, ${ins.y.toFixed(2)})`]
      }
    ];

    // Draw connectors + animated calm pulses between boxes
    const midY = boxY + boxH / 2;
    for (let i = 0; i < 4; i++) {
      const x1 = padX + i * (boxW + gap) + boxW;
      const x2 = x1 + gap;
      ctx.strokeStyle = "rgba(56, 189, 248, 0.38)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x1, midY);
      ctx.lineTo(x2, midY);
      ctx.stroke();

      // Arrowhead
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.moveTo(x2, midY);
      ctx.lineTo(x2 - 6, midY - 4);
      ctx.lineTo(x2 - 6, midY + 4);
      ctx.closePath();
      ctx.fill();

      // Animated data pulse traveling cleanly left -> right
      const pPhase = (state.animTime * 0.65 - i * 0.22) % 1;
      const tNorm = pPhase < 0 ? pPhase + 1 : pPhase;
      const px = x1 + tNorm * (gap - 6);
      ctx.fillStyle = stages[i].color;
      ctx.shadowColor = stages[i].color;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(px, midY, 3.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Feedback epoch loop arc along bottom (Stage 5 -> Stage 1)
    const loopStartX = padX + 4 * (boxW + gap) + boxW * 0.5;
    const loopEndX = padX + boxW * 0.5;
    const loopY = boxY + boxH + 30;
    ctx.strokeStyle = "rgba(52, 211, 153, 0.35)";
    ctx.lineWidth = 1.6;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(loopStartX, boxY + boxH);
    ctx.lineTo(loopStartX, loopY);
    ctx.lineTo(loopEndX, loopY);
    ctx.lineTo(loopEndX, boxY + boxH + 4);
    ctx.stroke();
    ctx.setLineDash([]);

    // Moving feedback packet along return line
    const retT = (state.animTime * 0.32) % 1;
    const retX = loopStartX - retT * (loopStartX - loopEndX);
    ctx.fillStyle = "#34d399";
    ctx.beginPath();
    ctx.arc(retX, loopY, 3.2, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#7f92b0";
    ctx.font = "600 10px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.fillText("Feedback Loop: Updated weights θ_{t+1} feed into next Forward Pass", width / 2, loopY + 18);

    // Draw the 5 Stage Cards
    stages.forEach((st, i) => {
      const bx = padX + i * (boxW + gap);
      roundRect(ctx, bx, boxY, boxW, boxH, 9);
      ctx.fillStyle = "#0b1426";
      ctx.fill();
      ctx.strokeStyle = st.color;
      ctx.lineWidth = 1.3;
      ctx.stroke();

      // Header bar
      ctx.fillStyle = st.color;
      ctx.font = "700 10px ui-monospace, Menlo, monospace";
      ctx.textAlign = "left";
      ctx.fillText(`${st.num} • ${st.title}`, bx + 10, boxY + 21);

      ctx.strokeStyle = "rgba(30, 47, 82, 0.85)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(bx + 8, boxY + 30);
      ctx.lineTo(bx + boxW - 8, boxY + 30);
      ctx.stroke();

      ctx.fillStyle = "#e2e8f0";
      ctx.font = "11px ui-monospace, Menlo, monospace";
      st.lines.forEach((ln, li) => {
        ctx.fillStyle = li === 2 ? "#94a3b8" : "#e6eefc";
        ctx.fillText(ln, bx + 10, boxY + 54 + li * 24);
      });
    });
  }

  // ============================================================================
  // CANVAS 3: CLICKABLE 2D CONTOUR MAP + VECTOR FIELD
  // ============================================================================

  function drawContourMap() {
    if (!state.contourBitmap) return;
    const { ctx, width, height } = setupCanvas(dom.canvasContour, 330);
    ctx.clearRect(0, 0, width, height);

    const pad = { top: 18, right: 20, bottom: 34, left: 42 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const min = Engine.DOMAIN_MIN, max = Engine.DOMAIN_MAX;

    const xOf = wx => pad.left + ((wx - min) / (max - min)) * plotW;
    const yOf = wy => pad.top + ((max - wy) / (max - min)) * plotH;

    // Draw cached contour heatmap
    ctx.drawImage(state.contourBitmap, pad.left, pad.top, plotW, plotH);

    // Subtle negative-gradient vector field arrows
    const land = Engine.getLandscape(state.landscapeId);
    ctx.strokeStyle = "rgba(226, 232, 240, 0.25)";
    ctx.lineWidth = 1;
    const step = 0.55;
    for (let y = -1.65; y <= 1.66; y += step) {
      for (let x = -1.65; x <= 1.66; x += step) {
        const ev = land.eval(x, y);
        const gn = Math.hypot(ev.gx, ev.gy);
        if (gn < 0.05) continue;
        const len = Math.min(13, 5 + gn * 2.5);
        const sx = xOf(x), sy = yOf(y);
        const ux = (-ev.gx / gn) * len;
        const uy = (ev.gy / gn) * len; // Canvas Y inverted
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(sx + ux, sy + uy);
        ctx.stroke();
      }
    }

    // Start cross marker
    const stX = xOf(state.start[0]), stY = yOf(state.start[1]);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(stX - 6, stY); ctx.lineTo(stX + 6, stY);
    ctx.moveTo(stX, stY - 6); ctx.lineTo(stX, stY + 6);
    ctx.stroke();

    // Optimizer paths & heads
    ["sgd", "momentum", "rmsprop", "adamw"].forEach(id => {
      if (!state.enabled[id]) return;
      const opt = state.optimizers[id];
      if (!opt) return;
      ctx.strokeStyle = opt.color;
      ctx.lineWidth = id === state.inspectId ? 2.6 : 1.9;
      ctx.beginPath();
      opt.trail.forEach((pt, idx) => {
        const px = xOf(pt.x), py = yOf(pt.y);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();

      const hx = xOf(opt.x), hy = yOf(opt.y);
      ctx.fillStyle = opt.color;
      ctx.beginPath();
      ctx.arc(hx, hy, id === state.inspectId ? 5.5 : 4.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#050a14";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    });

    // Border & Axis ticks
    ctx.strokeStyle = "#2c4373";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.left, pad.top, plotW, plotH);

    ctx.fillStyle = "#8ea2c4";
    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    [-2, -1, 0, 1, 2].forEach(t => {
      ctx.fillText(String(t), xOf(t), pad.top + plotH + 15);
    });
    ctx.fillText("Weight w₁", pad.left + plotW / 2, height - 4);

    ctx.textAlign = "right";
    [-2, -1, 0, 1, 2].forEach(t => {
      ctx.fillText(String(t), pad.left - 7, yOf(t) + 3);
    });
  }

  // ============================================================================
  // CANVAS 4: LOG-SCALE LOSS CONVERGENCE CHART
  // ============================================================================

  function drawLossChart() {
    const { ctx, width, height } = setupCanvas(dom.canvasLoss, 330);
    ctx.clearRect(0, 0, width, height);

    const pad = { top: 28, right: 20, bottom: 38, left: 56 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const activeOpts = ["sgd", "momentum", "rmsprop", "adamw"]
      .filter(id => state.enabled[id])
      .map(id => state.optimizers[id])
      .filter(Boolean);

    let minL = 0.01, maxL = 4.0;
    activeOpts.forEach(o => {
      o.lossHistory.forEach(v => {
        if (Number.isFinite(v) && v > 0) {
          if (v < minL) minL = Math.max(0.005, v * 0.8);
          if (v > maxL) maxL = Math.min(60, v * 1.2);
        }
      });
    });

    const logMin = Math.log10(minL);
    const logMax = Math.log10(maxL);
    const xSpan = Math.max(30, state.stepCount, 40);

    const xOf = step => pad.left + (step / xSpan) * plotW;
    const yOf = val => {
      const clamped = Math.max(minL, Math.min(maxL, Number.isFinite(val) ? val : maxL));
      return pad.top + plotH * (1 - (Math.log10(clamped) - logMin) / Math.max(logMax - logMin, 0.2));
    };

    // Grid
    ctx.strokeStyle = "rgba(30, 47, 82, 0.65)";
    ctx.lineWidth = 1;
    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.fillStyle = "#7f92b0";

    const yLevels = [0.01, 0.05, 0.1, 0.5, 1, 5, 20].filter(v => v >= minL * 0.9 && v <= maxL * 1.05);
    yLevels.forEach(v => {
      const y = yOf(v);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(v < 1 ? v.toFixed(2) : String(v), pad.left - 7, y + 3);
    });

    for (let g = 0; g <= 4; g++) {
      const s = Math.round((xSpan * g) / 4);
      const x = xOf(s);
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
      ctx.textAlign = "center";
      ctx.fillText(String(s), x, pad.top + plotH + 15);
    }

    // Curves
    activeOpts.forEach(opt => {
      if (opt.lossHistory.length < 2) return;
      ctx.strokeStyle = opt.color;
      ctx.lineWidth = opt.id === state.inspectId ? 2.8 : 2.0;
      ctx.beginPath();
      opt.lossHistory.forEach((v, idx) => {
        const px = xOf(idx), py = yOf(v);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    });

    // Legend top
    let legX = pad.left;
    activeOpts.forEach(opt => {
      ctx.fillStyle = opt.color;
      ctx.fillRect(legX, 10, 10, 4);
      ctx.fillStyle = "#cbd5e1";
      ctx.textAlign = "left";
      const label = opt.diverged ? `${opt.name} (DIV)` : `${opt.name} ${opt.L.toFixed(2)}`;
      ctx.fillText(label, legX + 14, 15);
      legX += ctx.measureText(label).width + 26;
    });

    // Axis titles
    ctx.fillStyle = "#94a3b8";
    ctx.textAlign = "center";
    ctx.fillText("Training Iteration (Step t)", pad.left + plotW / 2, height - 6);
  }

  // ============================================================================
  // MAIN 60 FPS ANIMATION LOOP
  // ============================================================================

  function tick(ts) {
    if (!state.lastTs) state.lastTs = ts;
    const dt = Math.min((ts - state.lastTs) / 1000, 0.05);
    state.lastTs = ts;
    state.animTime += dt;

    if (state.autoOrbit && !state.cam.dragging) {
      state.cam.yaw += dt * 0.28;
    }

    if (state.playing && state.stepCount < state.maxSteps) {
      // Calm default: ~4.5 optimization steps/sec so human eyes can follow every bounce
      const stepsPerSec = 4.5 * state.speedMult;
      state.stepAccum += dt * stepsPerSec;
      while (state.stepAccum >= 1 && state.stepCount < state.maxSteps) {
        state.stepAccum -= 1;
        advanceOneStep();
      }
    }

    draw3DScene();
    drawMathFlow();
    requestAnimationFrame(tick);
  }

  // ============================================================================
  // EVENT BINDINGS & INIT
  // ============================================================================

  function applyPreset(key) {
    const p = Engine.PRESETS[key];
    if (!p) return;
    state.activePreset = key;
    state.landscapeId = p.landscapeId;
    state.start = [...p.start];
    state.params = { lr: p.lr, beta1: p.beta1, beta2: p.beta2, wd: p.wd };
    state.inspectId = p.inspect;
    state.enabled = { sgd: true, momentum: true, rmsprop: true, adamw: true };
    state.playing = true;
    dom.playPauseBtn.innerHTML = "&#9646;&#9646; Pause";
    syncSidebar();
    rebuildSurfaceCaches();
    resetSimulation();
  }

  function bindEvents() {
    dom.presetGrid.addEventListener("click", evt => {
      const btn = evt.target.closest(".preset-btn");
      if (btn) applyPreset(btn.dataset.preset);
    });

    dom.landscapeSelect.addEventListener("change", () => {
      state.activePreset = null;
      state.landscapeId = dom.landscapeSelect.value;
      const land = Engine.getLandscape(state.landscapeId);
      state.start = [...land.defaultStart];
      syncSidebar();
      rebuildSurfaceCaches();
      resetSimulation();
    });

    dom.inspectSelect.addEventListener("change", () => {
      state.inspectId = dom.inspectSelect.value;
      state.enabled[state.inspectId] = true;
      syncSidebar();
      updateUIReadouts();
      drawContourMap();
      drawLossChart();
    });

    dom.optToggleGrid.addEventListener("click", evt => {
      const btn = evt.target.closest(".opt-chip");
      if (!btn) return;
      const id = btn.dataset.opt;
      const activeCount = Object.values(state.enabled).filter(Boolean).length;
      if (state.enabled[id] && activeCount <= 1) return; // keep at least 1 active
      state.enabled[id] = !state.enabled[id];
      if (!state.enabled[state.inspectId]) {
        state.inspectId = Object.keys(state.enabled).find(k => state.enabled[k]);
      }
      syncSidebar();
      updateUIReadouts();
      drawContourMap();
      drawLossChart();
    });

    const bindSlider = (el, key) => {
      el.addEventListener("input", () => {
        state.activePreset = null;
        state.params[key] = parseFloat(el.value);
        syncSidebar();
        resetSimulation();
      });
    };
    bindSlider(dom.lrSlider, "lr");
    bindSlider(dom.beta1Slider, "beta1");
    bindSlider(dom.beta2Slider, "beta2");
    bindSlider(dom.wdSlider, "wd");

    // Playback toolbar
    dom.playPauseBtn.addEventListener("click", () => {
      state.playing = !state.playing;
      dom.playPauseBtn.innerHTML = state.playing ? "&#9646;&#9646; Pause" : "&#9654; Play";
    });

    dom.stepOnceBtn.addEventListener("click", () => {
      state.playing = false;
      dom.playPauseBtn.innerHTML = "&#9654; Play";
      advanceOneStep();
    });

    dom.resetRunBtn.addEventListener("click", () => {
      resetSimulation();
    });

    [dom.speedCalmBtn, dom.speedNormalBtn, dom.speedFastBtn].forEach(btn => {
      btn.addEventListener("click", () => {
        state.speedMult = parseFloat(btn.dataset.speed);
        [dom.speedCalmBtn, dom.speedNormalBtn, dom.speedFastBtn].forEach(b => b.classList.toggle("active", b === btn));
      });
    });

    dom.vectorsBtn.addEventListener("click", () => {
      state.showVectors = !state.showVectors;
      dom.vectorsBtn.classList.toggle("active", state.showVectors);
      dom.vectorsBtn.textContent = state.showVectors ? "3D Vectors: ON" : "3D Vectors: OFF";
    });

    dom.autoOrbitBtn.addEventListener("click", () => {
      state.autoOrbit = !state.autoOrbit;
      dom.autoOrbitBtn.classList.toggle("active", state.autoOrbit);
    });

    dom.resetCamBtn.addEventListener("click", () => {
      state.cam.yaw = -0.68;
      state.cam.pitch = 0.62;
      state.cam.zoom = 1.04;
    });

    // 3D Orbit Pointer Drag & Wheel Zoom
    dom.canvas3d.addEventListener("pointerdown", evt => {
      state.cam.dragging = true;
      state.cam.lastX = evt.clientX;
      state.cam.lastY = evt.clientY;
      dom.canvas3d.setPointerCapture(evt.pointerId);
    });

    dom.canvas3d.addEventListener("pointermove", evt => {
      if (!state.cam.dragging) return;
      const dx = evt.clientX - state.cam.lastX;
      const dy = evt.clientY - state.cam.lastY;
      state.cam.lastX = evt.clientX;
      state.cam.lastY = evt.clientY;
      state.cam.yaw += dx * 0.0085;
      state.cam.pitch = Math.max(0.18, Math.min(1.42, state.cam.pitch + dy * 0.0075));
    });

    const endDrag = () => { state.cam.dragging = false; };
    dom.canvas3d.addEventListener("pointerup", endDrag);
    dom.canvas3d.addEventListener("pointercancel", endDrag);

    dom.canvas3d.addEventListener("wheel", evt => {
      evt.preventDefault();
      const factor = evt.deltaY > 0 ? 0.93 : 1.07;
      state.cam.zoom = Math.max(0.65, Math.min(1.85, state.cam.zoom * factor));
    }, { passive: false });

    // Clickable 2D Contour Map -> Drop New Start Point
    dom.canvasContour.addEventListener("click", evt => {
      const rect = dom.canvasContour.getBoundingClientRect();
      const pad = { top: 18, right: 20, bottom: 34, left: 42 };
      const plotW = rect.width - pad.left - pad.right;
      const plotH = 330 - pad.top - pad.bottom;
      const fx = (evt.clientX - rect.left - pad.left) / plotW;
      const fy = (evt.clientY - rect.top - pad.top) / plotH;
      if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return;

      const min = Engine.DOMAIN_MIN, max = Engine.DOMAIN_MAX;
      const wx = min + fx * (max - min);
      const wy = max - fy * (max - min);
      state.activePreset = null;
      state.start = [wx, wy];
      syncSidebar();
      resetSimulation();
    });

    let resizeTimer = null;
    window.addEventListener("resize", () => {
      if (resizeTimer) cancelAnimationFrame(resizeTimer);
      resizeTimer = requestAnimationFrame(() => {
        drawContourMap();
        drawLossChart();
      });
    });
  }

  function init() {
    dom.landscapeSelect.innerHTML = Engine.LANDSCAPES.map(
      l => `<option value="${l.id}">${l.name}</option>`
    ).join("");
    syncSidebar();
    rebuildSurfaceCaches();
    bindEvents();
    resetSimulation();
    requestAnimationFrame(tick);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();