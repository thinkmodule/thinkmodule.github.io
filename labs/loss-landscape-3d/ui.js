/**
 * ThinkModule — Module 03: Full-Viewport Three.js WebGL 3D Renderer & HUD Controller
 * Renders custom GLSL iso-contour loss manifold, 3D coordinate cage, floor contour projection,
 * tangent gradient/Hessian eigenvector arrows, local 2nd-order Taylor bowl, and live math HUD.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  DOMAIN,
  MAX_STEPS,
  OPTIMIZER_META,
  LossEngine,
  lossToY,
  dLossToY
} from './engine.js';

const WORLD_SCALE = 2.15;
const HALF_WORLD = DOMAIN * WORLD_SCALE;
const GRID_SEG = 156;
const TAYLOR_SEG = 18;
const TAYLOR_RADIUS = 0.42;

const toWorldX = (w1) => w1 * WORLD_SCALE;
const toWorldZ = (w2) => -w2 * WORLD_SCALE;
const fromWorldX = (x) => x / WORLD_SCALE;
const fromWorldZ = (z) => -z / WORLD_SCALE;

// Perceptual Turbo-Inspired Scientific Colormap (used identically in GLSL & 2D Contour)
function colormapRGB(t) {
  const x = Math.max(0, Math.min(1, t));
  const stops = [
    [0.00, 12, 34, 78],
    [0.22, 16, 114, 148],
    [0.45, 22, 178, 138],
    [0.68, 228, 178, 42],
    [0.86, 236, 98, 54],
    [1.00, 250, 218, 170]
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i];
    const b = stops[i + 1];
    if (x >= a[0] && x <= b[0]) {
      const f = (x - a[0]) / (b[0] - a[0]);
      return [
        Math.round(a[1] + (b[1] - a[1]) * f),
        Math.round(a[2] + (b[2] - a[2]) * f),
        Math.round(a[3] + (b[3] - a[3]) * f)
      ];
    }
  }
  return [250, 218, 170];
}

// ============================================================================
// 1. THREE.JS SCENE SETUP
// ============================================================================
const engine = new LossEngine();
const viewport = document.getElementById('viewport');
const labelLayer = document.getElementById('labelLayer');

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x04070d, 0.024);

const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 120);
const DEFAULT_CAM = new THREE.Vector3(7.6, 6.9, 8.8);
const DEFAULT_TARGET = new THREE.Vector3(0, 1.55, 0);
camera.position.copy(DEFAULT_CAM);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance', alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(DEFAULT_TARGET);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.maxPolarAngle = Math.PI * 0.485;
controls.minDistance = 4.2;
controls.maxDistance = 24;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.45;

// Studio Lights
scene.add(new THREE.AmbientLight(0x9bb8e8, 0.55));
const keyLight = new THREE.DirectionalLight(0xeef6ff, 1.45);
keyLight.position.set(8, 14, 9);
scene.add(keyLight);

const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.85);
rimLight.position.set(-9, 7, -8);
scene.add(rimLight);

// Subtle Infinite Studio Grid
const studioGrid = new THREE.GridHelper(34, 34, 0x162640, 0x0b1322);
studioGrid.position.y = -0.02;
scene.add(studioGrid);

// ============================================================================
// 2. CUSTOM GLSL 3D LOSS SURFACE WITH CRISP ISO-CONTOURS
// ============================================================================
const surfaceUniforms = {
  uMinL: { value: 0.0 },
  uMaxL: { value: 3.5 },
  uWireframe: { value: 0.0 }
};

const surfaceVertexShader = /* glsl */ `
  attribute float aLoss;
  varying float vLoss;
  varying vec3 vWorldPos;
  varying vec3 vNormalDir;
  varying vec2 vMathCoord;

  void main() {
    vLoss = aLoss;
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorldPos = wp.xyz;
    vNormalDir = normalize(mat3(modelMatrix) * normal);
    vMathCoord = vec2(position.x / ${WORLD_SCALE.toFixed(2)}, -position.z / ${WORLD_SCALE.toFixed(2)});
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

const surfaceFragmentShader = /* glsl */ `
  uniform float uMinL;
  uniform float uMaxL;
  varying float vLoss;
  varying vec3 vWorldPos;
  varying vec3 vNormalDir;
  varying vec2 vMathCoord;

  vec3 palette(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.047, 0.133, 0.306);
    vec3 c1 = vec3(0.063, 0.447, 0.580);
    vec3 c2 = vec3(0.086, 0.698, 0.541);
    vec3 c3 = vec3(0.894, 0.698, 0.165);
    vec3 c4 = vec3(0.925, 0.384, 0.212);
    vec3 c5 = vec3(0.980, 0.855, 0.667);

    if (t < 0.22) return mix(c0, c1, t / 0.22);
    if (t < 0.45) return mix(c1, c2, (t - 0.22) / 0.23);
    if (t < 0.68) return mix(c2, c3, (t - 0.45) / 0.23);
    if (t < 0.86) return mix(c3, c4, (t - 0.68) / 0.18);
    return mix(c4, c5, (t - 0.86) / 0.14);
  }

  void main() {
    float normL = clamp((log(1.0 + vLoss * 1.6) - log(1.0 + uMinL * 1.6)) /
                        max(0.001, log(1.0 + uMaxL * 1.6) - log(1.0 + uMinL * 1.6)), 0.0, 1.0);
    vec3 baseCol = palette(normL);

    vec3 N = normalize(vNormalDir);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(cameraPosition - vWorldPos);
    vec3 L1 = normalize(vec3(7.0, 14.0, 8.0));
    vec3 L2 = normalize(vec3(-8.0, 6.0, -7.0));

    float diff1 = max(dot(N, L1), 0.0);
    float diff2 = max(dot(N, L2), 0.0) * 0.35;
    vec3 H = normalize(L1 + V);
    float spec = pow(max(dot(N, H), 0.0), 48.0) * 0.32;
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.26;

    // Anti-aliased iso-contour lines in log-loss space
    float isoCoord = log(1.0 + vLoss * 2.2) * 7.5;
    float isoDist = abs(fract(isoCoord - 0.5) - 0.5) / max(fwidth(isoCoord), 0.0005);
    float isoLine = 1.0 - smoothstep(0.65, 1.55, isoDist);

    // Subtle w1/w2 coordinate grid every 0.5 units
    vec2 gCoord = vMathCoord * 2.0;
    vec2 gDist = abs(fract(gCoord - 0.5) - 0.5) / max(fwidth(gCoord), vec2(0.0005));
    float gridLine = 1.0 - smoothstep(0.55, 1.4, min(gDist.x, gDist.y));

    vec3 col = baseCol * (0.36 + 0.68 * diff1 + diff2) + vec3(spec) + vec3(0.22, 0.74, 0.97) * fresnel;
    col = mix(col, vec3(0.95, 0.98, 1.0), isoLine * 0.32);
    col = mix(col, vec3(0.75, 0.88, 1.0), gridLine * 0.12);

    // Soft outer rim highlight
    float edge = max(abs(vMathCoord.x), abs(vMathCoord.y));
    float borderGlow = smoothstep(${(DOMAIN - 0.04).toFixed(2)}, ${DOMAIN.toFixed(2)}, edge);
    col = mix(col, vec3(0.38, 0.78, 0.98), borderGlow * 0.45);

    gl_FragColor = vec4(col, 0.93);
  }
`;

const surfaceGeom = new THREE.PlaneGeometry(HALF_WORLD * 2, HALF_WORLD * 2, GRID_SEG, GRID_SEG);
surfaceGeom.rotateX(-Math.PI / 2);
const surfaceMat = new THREE.ShaderMaterial({
  uniforms: surfaceUniforms,
  vertexShader: surfaceVertexShader,
  fragmentShader: surfaceFragmentShader,
  side: THREE.DoubleSide,
  transparent: true
});
const surfaceMesh = new THREE.Mesh(surfaceGeom, surfaceMat);
scene.add(surfaceMesh);

// ============================================================================
// 3. FLOOR 2D CONTOUR PROJECTION + 3D COORDINATE CAGE
// ============================================================================
const floorCanvas = document.createElement('canvas');
floorCanvas.width = 512;
floorCanvas.height = 512;
const floorCtx = floorCanvas.getContext('2d');
const floorTex = new THREE.CanvasTexture(floorCanvas);
floorTex.anisotropy = 8;

const floorMesh = new THREE.Mesh(
  new THREE.PlaneGeometry(HALF_WORLD * 2, HALF_WORLD * 2),
  new THREE.MeshBasicMaterial({ map: floorTex, transparent: true, opacity: 0.72, depthWrite: false })
);
floorMesh.rotation.x = -Math.PI / 2;
floorMesh.position.y = 0.006;
scene.add(floorMesh);

// Coordinate Cage Lines
const cageGroup = new THREE.Group();
scene.add(cageGroup);
const axisLabels3D = [];

function buildCoordinateCage() {
  const matFrame = new THREE.LineBasicMaterial({ color: 0x2b4368, transparent: true, opacity: 0.7 });
  const matAxis = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.85 });

  const hTop = 5.4;
  const pts = [
    [-HALF_WORLD, 0, HALF_WORLD], [HALF_WORLD, 0, HALF_WORLD],
    [HALF_WORLD, 0, HALF_WORLD], [HALF_WORLD, 0, -HALF_WORLD],
    [HALF_WORLD, 0, -HALF_WORLD], [-HALF_WORLD, 0, -HALF_WORLD],
    [-HALF_WORLD, 0, -HALF_WORLD], [-HALF_WORLD, 0, HALF_WORLD],
    [-HALF_WORLD, 0, -HALF_WORLD], [-HALF_WORLD, hTop, -HALF_WORLD]
  ];
  const g = new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(...p)));
  cageGroup.add(new THREE.LineSegments(g, matFrame));

  // Highlight primary axes (w1 front edge, w2 right edge, L vertical back-left)
  const w1Axis = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-HALF_WORLD, 0.015, HALF_WORLD),
    new THREE.Vector3(HALF_WORLD + 0.35, 0.015, HALF_WORLD)
  ]);
  const w2Axis = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(HALF_WORLD, 0.015, HALF_WORLD),
    new THREE.Vector3(HALF_WORLD, 0.015, -HALF_WORLD - 0.35)
  ]);
  const lAxis = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-HALF_WORLD, 0, -HALF_WORLD),
    new THREE.Vector3(-HALF_WORLD, hTop + 0.35, -HALF_WORLD)
  ]);
  cageGroup.add(new THREE.Line(w1Axis, matAxis));
  cageGroup.add(new THREE.Line(w2Axis, matAxis));
  cageGroup.add(new THREE.Line(lAxis, matAxis));

  const addLabel = (text, pos, isTick = false) => {
    const el = document.createElement('div');
    el.className = isTick ? 'axis-tag tick' : 'axis-tag';
    el.textContent = text;
    labelLayer.appendChild(el);
    axisLabels3D.push({ el, pos: new THREE.Vector3(...pos) });
  };

  addLabel('w₁ (Weight 1) →', [0, 0.02, HALF_WORLD + 0.65], false);
  addLabel('w₂ (Weight 2) →', [HALF_WORLD + 0.78, 0.02, 0], false);
  addLabel('Loss L(w₁, w₂) ↑', [-HALF_WORLD - 0.25, hTop + 0.55, -HALF_WORLD], false);

  for (const t of [-2, -1, 0, 1, 2]) {
    const sign = t > 0 ? `+${t}` : `${t}`;
    addLabel(sign, [toWorldX(t), 0.02, HALF_WORLD + 0.28], true);
    addLabel(sign, [HALF_WORLD + 0.3, 0.02, toWorldZ(t)], true);
  }
  for (const lv of [0, 0.5, 1.5, 4.0]) {
    addLabel(lv.toFixed(1), [-HALF_WORLD - 0.32, lossToY(lv), -HALF_WORLD], true);
  }
}
buildCoordinateCage();

// ============================================================================
// 4. OPTIMIZER SPHERES, 3D TRAILS, FLOOR SHADOWS & VECTOR ARROWS
// ============================================================================
const optVisuals = {};
const optIds = Object.keys(OPTIMIZER_META);

for (const id of optIds) {
  const meta = OPTIMIZER_META[id];

  // Glowing PBR Sphere
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.145, 28, 28),
    new THREE.MeshStandardMaterial({
      color: meta.hex,
      emissive: meta.hex,
      emissiveIntensity: 0.55,
      roughness: 0.22,
      metalness: 0.18
    })
  );
  scene.add(sphere);

  // Halo Ring around inspected optimizer
  const halo = new THREE.Mesh(
    new THREE.RingGeometry(0.21, 0.26, 36),
    new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.85 })
  );
  halo.rotation.x = -Math.PI / 2;
  halo.visible = false;
  scene.add(halo);

  // Pre-allocated 3D Trajectory Line
  const maxPts = MAX_STEPS + 4;
  const trailPos = new Float32Array(maxPts * 3);
  const trailGeom = new THREE.BufferGeometry();
  trailGeom.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
  trailGeom.setDrawRange(0, 0);
  const trailLine = new THREE.Line(
    trailGeom,
    new THREE.LineBasicMaterial({ color: meta.hex, transparent: true, opacity: 0.95 })
  );
  scene.add(trailLine);

  // Floor Projection Trail
  const floorPos = new Float32Array(maxPts * 3);
  const floorGeom = new THREE.BufferGeometry();
  floorGeom.setAttribute('position', new THREE.BufferAttribute(floorPos, 3));
  floorGeom.setDrawRange(0, 0);
  const floorLine = new THREE.Line(
    floorGeom,
    new THREE.LineBasicMaterial({ color: meta.hex, transparent: true, opacity: 0.42 })
  );
  scene.add(floorLine);

  // Vertical coordinate drop-line to floor
  const dropGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const dropLine = new THREE.Line(
    dropGeom,
    new THREE.LineBasicMaterial({ color: meta.hex, transparent: true, opacity: 0.35 })
  );
  scene.add(dropLine);

  // Projected HTML Label Pill
  const pill = document.createElement('div');
  pill.className = 'opt-pill';
  pill.innerHTML = `<span class="swatch" style="background:${meta.color}"></span><span>${meta.name}</span>`;
  labelLayer.appendChild(pill);

  optVisuals[id] = { sphere, halo, trailGeom, trailPos, trailLine, floorGeom, floorPos, floorLine, dropGeom, dropLine, pill };
}

// 3D Differential Geometry Arrows (Negative Gradient + Hessian Eigenvectors e1, e2)
const vectorGroup = new THREE.Group();
scene.add(vectorGroup);

const gradArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0x38bdf8, 0.28, 0.16);
const eig1ArrowP = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0x34d399, 0.22, 0.13);
const eig1ArrowN = new THREE.ArrowHelper(new THREE.Vector3(-1, 0, 0), new THREE.Vector3(), 1, 0x34d399, 0.22, 0.13);
const eig2ArrowP = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 1, 0xfb7185, 0.22, 0.13);
const eig2ArrowN = new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), new THREE.Vector3(), 1, 0xfb7185, 0.22, 0.13);
vectorGroup.add(gradArrow, eig1ArrowP, eig1ArrowN, eig2ArrowP, eig2ArrowN);

// Local 2nd-Order Taylor Paraboloid Patch around Inspected Optimizer
const taylorGeom = new THREE.PlaneGeometry(TAYLOR_RADIUS * 2 * WORLD_SCALE, TAYLOR_RADIUS * 2 * WORLD_SCALE, TAYLOR_SEG, TAYLOR_SEG);
taylorGeom.rotateX(-Math.PI / 2);
const taylorMesh = new THREE.Mesh(
  taylorGeom,
  new THREE.MeshBasicMaterial({ color: 0x7dd3fc, wireframe: true, transparent: true, opacity: 0.32 })
);
scene.add(taylorMesh);

// ============================================================================
// 5. SURFACE & FLOOR CONTOUR REBUILD
// ============================================================================
let lossRange = { min: 0, max: 3.5 };

function rebuildSurface() {
  const pos = surfaceGeom.attributes.position;
  const count = pos.count;
  const losses = new Float32Array(count);

  let minL = Infinity;
  let maxL = -Infinity;

  for (let i = 0; i < count; i++) {
    const w1 = fromWorldX(pos.getX(i));
    const w2 = fromWorldZ(pos.getZ(i));
    const res = engine.scenario.eval(w1, w2);
    losses[i] = res.loss;
    if (res.loss < minL) minL = res.loss;
    if (res.loss > maxL) maxL = res.loss;
    pos.setY(i, lossToY(res.loss));
  }

  lossRange = { min: minL, max: maxL };
  surfaceUniforms.uMinL.value = minL;
  surfaceUniforms.uMaxL.value = maxL;
  surfaceGeom.setAttribute('aLoss', new THREE.BufferAttribute(losses, 1));
  pos.needsUpdate = true;
  surfaceGeom.computeVertexNormals();

  renderFloorContourTexture();
}

function renderContourToCanvas(ctx, width, height, drawAxes = false) {
  const img = ctx.createImageData(width, height);
  const data = img.data;
  const logMin = Math.log(1 + lossRange.min * 1.6);
  const logSpan = Math.max(0.001, Math.log(1 + lossRange.max * 1.6) - logMin);

  for (let py = 0; py < height; py++) {
    const w2 = DOMAIN - (py / (height - 1)) * (2 * DOMAIN);
    for (let px = 0; px < width; px++) {
      const w1 = -DOMAIN + (px / (width - 1)) * (2 * DOMAIN);
      const L = engine.scenario.eval(w1, w2).loss;
      const t = (Math.log(1 + L * 1.6) - logMin) / logSpan;
      const [r, g, b] = colormapRGB(t);

      // Subtle topographic contour rings
      const iso = Math.log(1 + L * 2.2) * 7.5;
      const frac = Math.abs((iso % 1 + 1) % 1 - 0.5);
      const boost = frac < 0.045 ? 1.24 : 0.72;

      const idx = (py * width + px) * 4;
      data[idx] = Math.min(255, r * boost);
      data[idx + 1] = Math.min(255, g * boost);
      data[idx + 2] = Math.min(255, b * boost);
      data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  if (drawAxes) {
    ctx.strokeStyle = 'rgba(255,255,255,0.16)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
  }
}

function renderFloorContourTexture() {
  renderContourToCanvas(floorCtx, floorCanvas.width, floorCanvas.height, true);
  floorTex.needsUpdate = true;
  renderContourTabBase();
}

// ============================================================================
// 6. SYNC 3D VISUALS & DIFFERENTIAL GEOMETRY
// ============================================================================
let showVectors = true;
let showTaylor = true;

function update3DOptimizers() {
  for (const id of optIds) {
    const opt = engine.optimizers[id];
    const vis = optVisuals[id];
    const isVisible = engine.active[id];

    vis.sphere.visible = isVisible;
    vis.trailLine.visible = isVisible;
    vis.floorLine.visible = isVisible;
    vis.dropLine.visible = isVisible;
    vis.halo.visible = isVisible && id === engine.inspected;
    vis.pill.style.display = isVisible ? 'flex' : 'none';

    if (!isVisible) continue;

    const x = toWorldX(opt.w1);
    const z = toWorldZ(opt.w2);
    const y = lossToY(opt.eval.loss) + 0.14;

    vis.sphere.position.set(x, y, z);
    vis.halo.position.set(x, y - 0.06, z);

    // Update vertical drop-line to floor
    const dAttr = vis.dropGeom.attributes.position;
    dAttr.setXYZ(0, x, y - 0.12, z);
    dAttr.setXYZ(1, x, 0.02, z);
    dAttr.needsUpdate = true;

    // Update 3D & floor trajectory lines
    const len = opt.path.length;
    for (let i = 0; i < len; i++) {
      const p = opt.path[i];
      const px = toWorldX(p.w1);
      const pz = toWorldZ(p.w2);
      const py = lossToY(p.loss) + 0.045;
      vis.trailPos[i * 3] = px;
      vis.trailPos[i * 3 + 1] = py;
      vis.trailPos[i * 3 + 2] = pz;

      vis.floorPos[i * 3] = px;
      vis.floorPos[i * 3 + 1] = 0.024;
      vis.floorPos[i * 3 + 2] = pz;
    }
    vis.trailGeom.setDrawRange(0, len);
    vis.trailGeom.attributes.position.needsUpdate = true;
    vis.floorGeom.setDrawRange(0, len);
    vis.floorGeom.attributes.position.needsUpdate = true;

    vis.pill.classList.toggle('inspected', id === engine.inspected);
    vis.pill.classList.toggle('diverged', opt.diverged);
    const labelText = opt.diverged ? `${OPTIMIZER_META[id].name} (DIV)` : OPTIMIZER_META[id].name;
    vis.pill.lastElementChild.textContent = labelText;
  }

  updateGeometryOverlays();
}

function updateGeometryOverlays() {
  const opt = engine.optimizers[engine.inspected];
  const canShow = engine.active[engine.inspected] && !opt.diverged;

  vectorGroup.visible = showVectors && canShow;
  taylorMesh.visible = showTaylor && canShow;
  if (!canShow) return;

  const { w1, w2, eval: ev } = opt;
  const origin = new THREE.Vector3(toWorldX(w1), lossToY(ev.loss) + 0.16, toWorldZ(w2));
  const dydL = dLossToY(ev.loss);

  // Tangent 3D direction helper for any 2D vector (dw1, dw2)
  const toTangentDir = (dw1, dw2) => {
    const dL = ev.g1 * dw1 + ev.g2 * dw2;
    return new THREE.Vector3(dw1 * WORLD_SCALE, dydL * dL, -dw2 * WORLD_SCALE).normalize();
  };

  // 1. Negative Gradient Arrow (-∇L)
  if (ev.gradNorm > 1e-5) {
    gradArrow.visible = true;
    gradArrow.position.copy(origin);
    gradArrow.setDirection(toTangentDir(-ev.g1, -ev.g2));
    const gLen = Math.min(2.1, Math.max(0.55, 0.45 + Math.log1p(ev.gradNorm) * 0.85));
    gradArrow.setLength(gLen, 0.26, 0.14);
  } else {
    gradArrow.visible = false;
  }

  // 2. Principal Hessian Eigenvectors (e1, e2 — bidirectional along curvature axes)
  const setEigPair = (arrowP, arrowN, eVec, lambda) => {
    const col = lambda >= 0 ? 0x34d399 : 0xfb7185;
    const len = Math.min(1.55, Math.max(0.55, 0.45 + Math.log1p(Math.abs(lambda)) * 0.55));
    arrowP.position.copy(origin);
    arrowN.position.copy(origin);
    arrowP.setDirection(toTangentDir(eVec[0], eVec[1]));
    arrowN.setDirection(toTangentDir(-eVec[0], -eVec[1]));
    arrowP.setColor(col);
    arrowN.setColor(col);
    arrowP.setLength(len, 0.19, 0.10);
    arrowN.setLength(len, 0.19, 0.10);
  };
  setEigPair(eig1ArrowP, eig1ArrowN, ev.e1, ev.l1);
  setEigPair(eig2ArrowP, eig2ArrowN, ev.e2, ev.l2);

  // 3. Local 2nd-Order Taylor Quadratic Approximation Patch
  if (showTaylor) {
    const pos = taylorGeom.attributes.position;
    let idx = 0;
    for (let j = 0; j <= TAYLOR_SEG; j++) {
      const dw2 = TAYLOR_RADIUS - (j / TAYLOR_SEG) * (2 * TAYLOR_RADIUS);
      for (let i = 0; i <= TAYLOR_SEG; i++) {
        const dw1 = -TAYLOR_RADIUS + (i / TAYLOR_SEG) * (2 * TAYLOR_RADIUS);
        const qLoss =
          ev.loss +
          ev.g1 * dw1 +
          ev.g2 * dw2 +
          0.5 * (ev.h11 * dw1 * dw1 + 2 * ev.h12 * dw1 * dw2 + ev.h22 * dw2 * dw2);
        pos.setXYZ(idx, toWorldX(w1 + dw1), lossToY(Math.max(0, qLoss)) + 0.08, toWorldZ(w2 + dw2));
        idx++;
      }
    }
    pos.needsUpdate = true;
  }
}

// ============================================================================
// 7. 2D CHARTS (LOG10 LOSS CURVE & INTERACTIVE 2D CONTOUR)
// ============================================================================
const lossCanvas = document.getElementById('lossCanvas');
const lossCtx = lossCanvas.getContext('2d');
const contourCanvas = document.getElementById('contourCanvas');
const contourCtx = contourCanvas.getContext('2d');
const contourBaseCanvas = document.createElement('canvas');
contourBaseCanvas.width = contourCanvas.width;
contourBaseCanvas.height = contourCanvas.height;
const contourBaseCtx = contourBaseCanvas.getContext('2d');

function renderContourTabBase() {
  renderContourToCanvas(contourBaseCtx, contourBaseCanvas.width, contourBaseCanvas.height, true);
}

function drawContourTab() {
  const W = contourCanvas.width;
  const H = contourCanvas.height;
  contourCtx.drawImage(contourBaseCanvas, 0, 0);

  const toPx = (w1) => ((w1 + DOMAIN) / (2 * DOMAIN)) * W;
  const toPy = (w2) => ((DOMAIN - w2) / (2 * DOMAIN)) * H;

  // Start crosshair
  const sx = toPx(engine.start[0]);
  const sy = toPy(engine.start[1]);
  contourCtx.strokeStyle = 'rgba(255,255,255,0.75)';
  contourCtx.lineWidth = 1.5;
  contourCtx.beginPath();
  contourCtx.arc(sx, sy, 7, 0, Math.PI * 2);
  contourCtx.stroke();

  // Trajectories
  for (const id of optIds) {
    if (!engine.active[id]) continue;
    const opt = engine.optimizers[id];
    const meta = OPTIMIZER_META[id];
    contourCtx.strokeStyle = meta.color;
    contourCtx.lineWidth = id === engine.inspected ? 3.2 : 2.1;
    contourCtx.beginPath();
    opt.path.forEach((p, i) => {
      const px = toPx(p.w1);
      const py = toPy(p.w2);
      if (i === 0) contourCtx.moveTo(px, py);
      else contourCtx.lineTo(px, py);
    });
    contourCtx.stroke();

    const last = opt.path[opt.path.length - 1];
    contourCtx.fillStyle = meta.color;
    contourCtx.beginPath();
    contourCtx.arc(toPx(last.w1), toPy(last.w2), id === engine.inspected ? 6.5 : 4.5, 0, Math.PI * 2);
    contourCtx.fill();
    contourCtx.strokeStyle = '#04070d';
    contourCtx.lineWidth = 1.5;
    contourCtx.stroke();
  }
}

function drawLossChart() {
  const W = lossCanvas.width;
  const H = lossCanvas.height;
  const pad = { l: 64, r: 18, t: 22, b: 42 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  lossCtx.fillStyle = '#03060c';
  lossCtx.fillRect(0, 0, W, H);

  let minLog = Infinity;
  let maxLog = -Infinity;
  for (const id of optIds) {
    if (!engine.active[id]) continue;
    for (const l of engine.optimizers[id].losses) {
      const lg = Math.log10(Math.max(1e-4, l));
      if (lg < minLog) minLog = lg;
      if (lg > maxLog) maxLog = lg;
    }
  }
  if (!Number.isFinite(minLog)) { minLog = -2; maxLog = 1; }
  minLog = Math.floor(minLog * 2) / 2 - 0.15;
  maxLog = Math.ceil(maxLog * 2) / 2 + 0.15;

  // Grid lines & labels
  lossCtx.strokeStyle = 'rgba(255,255,255,0.08)';
  lossCtx.lineWidth = 1;
  lossCtx.fillStyle = '#7e90ad';
  lossCtx.font = '18px ui-monospace, monospace';

  for (let i = 0; i <= 4; i++) {
    const f = i / 4;
    const y = pad.t + f * ph;
    const val = Math.pow(10, maxLog - f * (maxLog - minLog));
    lossCtx.beginPath();
    lossCtx.moveTo(pad.l, y);
    lossCtx.lineTo(W - pad.r, y);
    lossCtx.stroke();
    lossCtx.textAlign = 'right';
    lossCtx.fillText(val >= 1 ? val.toFixed(1) : val.toFixed(3), pad.l - 8, y + 6);
  }

  const xSpan = Math.max(20, engine.stepCount);
  for (let i = 0; i <= 4; i++) {
    const x = pad.l + (i / 4) * pw;
    lossCtx.textAlign = 'center';
    lossCtx.fillText(Math.round((i / 4) * xSpan), x, H - 14);
  }

  // Curves
  for (const id of optIds) {
    if (!engine.active[id]) continue;
    const opt = engine.optimizers[id];
    const meta = OPTIMIZER_META[id];
    lossCtx.strokeStyle = meta.color;
    lossCtx.lineWidth = id === engine.inspected ? 3.4 : 2.2;
    lossCtx.beginPath();
    opt.losses.forEach((l, idx) => {
      const x = pad.l + (idx / xSpan) * pw;
      const lg = Math.log10(Math.max(1e-4, l));
      const y = pad.t + ((maxLog - lg) / (maxLog - minLog)) * ph;
      if (idx === 0) lossCtx.moveTo(x, y);
      else lossCtx.lineTo(x, y);
    });
    lossCtx.stroke();
  }
}

// ============================================================================
// 8. LIVE MATH INSPECTOR HUD RENDERING
// ============================================================================
const fmt = (n, d = 4) => (n >= 0 ? '+' : '−') + Math.abs(n).toFixed(d);

function updateMathHUD() {
  const opt = engine.optimizers[engine.inspected];
  const ev = opt.eval;

  document.getElementById('scenarioTitle').textContent = engine.scenario.title;
  document.getElementById('surfaceFormula').textContent = engine.scenario.formula;
  document.getElementById('stepCounter').textContent = `Step ${engine.stepCount} / ${MAX_STEPS}`;
  document.getElementById('inspNameBadge').textContent = OPTIMIZER_META[engine.inspected].name;
  document.getElementById('mathRuleHint').textContent = OPTIMIZER_META[engine.inspected].rule;
  document.getElementById('startCoordText').textContent = `Start: (${fmt(engine.start[0], 2)}, ${fmt(engine.start[1], 2)})`;

  const reg = document.getElementById('regimeBadge');
  reg.textContent = `${ev.regime.toUpperCase()} (λ₁=${fmt(ev.l1, 2)}, λ₂=${fmt(ev.l2, 2)})`;
  reg.className = `regime-chip ${ev.regimeClass}`;

  document.getElementById('mathGrad').innerHTML =
    `θₜ = [ ${fmt(opt.w1)} ]  (w₁)   L(θ) = <span class="pos">${ev.loss.toFixed(4)}</span>\n` +
    `     [ ${fmt(opt.w2)} ]  (w₂)\n` +
    `∇L = [ ${fmt(ev.g1)} ]  (∂L/∂w₁)\n` +
    `     [ ${fmt(ev.g2)} ]  (∂L/∂w₂)  ‖∇L‖ = ${ev.gradNorm.toFixed(4)}`;

  const detCls = ev.det > 0 ? 'pos' : 'neg';
  document.getElementById('mathHess').innerHTML =
    `H  = [ ${fmt(ev.h11, 3)}   ${fmt(ev.h12, 3)} ]\n` +
    `     [ ${fmt(ev.h12, 3)}   ${fmt(ev.h22, 3)} ]\n` +
    `tr(H) = ${fmt(ev.trace, 3)}    det(H) = <span class="${detCls}">${fmt(ev.det, 3)}</span>`;

  const l1Cls = ev.l1 >= 0 ? 'pos' : 'neg';
  const l2Cls = ev.l2 >= 0 ? 'pos' : 'neg';
  const kStr = ev.kappa > 999 ? '>999' : ev.kappa.toFixed(2);
  document.getElementById('mathEigen').innerHTML =
    `λ₁ = <span class="${l1Cls}">${fmt(ev.l1, 3)}</span>   e₁ = [${fmt(ev.e1[0], 2)}, ${fmt(ev.e1[1], 2)}]\n` +
    `λ₂ = <span class="${l2Cls}">${fmt(ev.l2, 3)}</span>   e₂ = [${fmt(ev.e2[0], 2)}, ${fmt(ev.e2[1], 2)}]\n` +
    `κ  = |λ_max/λ_min| = <span class="warn">${kStr}</span>  (${ev.regime})`;

  let stateText = '';
  if (opt.id === 'sgd') {
    stateText = `State: Stateless (first-order gradient)\nΔθ = [ ${fmt(opt.dw1)}, ${fmt(opt.dw2)} ]`;
  } else if (opt.id === 'momentum') {
    stateText = `vₜ = [ ${fmt(opt.mHat1, 3)}, ${fmt(opt.mHat2, 3)} ]\nΔθ = [ ${fmt(opt.dw1)}, ${fmt(opt.dw2)} ]`;
  } else if (opt.id === 'rmsprop') {
    stateText = `v̂ₜ = [ ${opt.vHat1.toFixed(4)}, ${opt.vHat2.toFixed(4)} ]\nΔθ = [ ${fmt(opt.dw1)}, ${fmt(opt.dw2)} ]`;
  } else {
    stateText =
      `m̂ₜ = [ ${fmt(opt.mHat1, 3)}, ${fmt(opt.mHat2, 3)} ]\n` +
      `v̂ₜ = [ ${opt.vHat1.toFixed(4)}, ${opt.vHat2.toFixed(4)} ]\n` +
      `Δθ = [ ${fmt(opt.dw1)}, ${fmt(opt.dw2)} ]`;
  }
  if (opt.diverged) stateText += `\n<span class="neg">STATUS: DIVERGED (reduce η)</span>`;
  document.getElementById('mathState').innerHTML = stateText;

  // Leaderboard & Leader badge
  const leader = engine.getLeader();
  document.getElementById('leaderText').textContent = leader
    ? `Leader: ${OPTIMIZER_META[leader.id].name} (${leader.eval.loss.toFixed(4)})`
    : 'Leader: —';

  const lb = document.getElementById('leaderboard');
  lb.innerHTML = optIds
    .map((id) => {
      const o = engine.optimizers[id];
      const meta = OPTIMIZER_META[id];
      const val = !engine.active[id] ? 'OFF' : o.diverged ? 'DIV' : o.eval.loss.toFixed(4);
      return `<div class="lb-item"><span class="lb-name"><i class="lb-dot" style="background:${meta.color}"></i>${meta.name}</span><span>${val}</span></div>`;
    })
    .join('');

  drawLossChart();
  drawContourTab();
}

// ============================================================================
// 9. PROJECTED 3D LABELS WITH AUTOMATIC DE-OVERLAPPING
// ============================================================================
const projVec = new THREE.Vector3();

function projectLabels() {
  const W = window.innerWidth;
  const H = window.innerHeight;

  // Axis & tick labels
  for (const item of axisLabels3D) {
    projVec.copy(item.pos).project(camera);
    if (projVec.z > 1) {
      item.el.style.opacity = '0';
    } else {
      item.el.style.opacity = '1';
      item.el.style.left = `${(projVec.x * 0.5 + 0.5) * W}px`;
      item.el.style.top = `${(-projVec.y * 0.5 + 0.5) * H}px`;
    }
  }

  // Optimizer pills with vertical anti-overlap repulsion
  const activePills = [];
  for (const id of optIds) {
    if (!engine.active[id]) continue;
    const vis = optVisuals[id];
    projVec.copy(vis.sphere.position);
    projVec.y += 0.28;
    projVec.project(camera);
    if (projVec.z > 1) {
      vis.pill.style.opacity = '0';
      continue;
    }
    vis.pill.style.opacity = '1';
    activePills.push({
      el: vis.pill,
      x: (projVec.x * 0.5 + 0.5) * W,
      y: (-projVec.y * 0.5 + 0.5) * H
    });
  }

  activePills.sort((a, b) => a.y - b.y);
  for (let pass = 0; pass < 4; pass++) {
    for (let i = 0; i < activePills.length; i++) {
      for (let j = i + 1; j < activePills.length; j++) {
        const dx = Math.abs(activePills[i].x - activePills[j].x);
        const dy = activePills[j].y - activePills[i].y;
        if (dx < 82 && dy < 24) {
          const push = (24 - dy) * 0.55;
          activePills[i].y -= push;
          activePills[j].y += push;
        }
      }
    }
  }

  for (const p of activePills) {
    p.el.style.left = `${p.x}px`;
    p.el.style.top = `${p.y}px`;
  }
}

// ============================================================================
// 10. UI WIRING & INTERACTIONS
// ============================================================================
let isPlaying = true;
const speeds = [
  { label: '0.5×', interval: 180 },
  { label: '1×', interval: 75 },
  { label: '2×', interval: 28 }
];
let speedIdx = 1;
let lastStepTime = 0;

function syncSlidersFromEngine() {
  const { lr, beta1, beta2, wd } = engine.params;
  document.getElementById('lrSlider').value = lr;
  document.getElementById('lrVal').textContent = lr.toFixed(3);
  document.getElementById('b1Slider').value = beta1;
  document.getElementById('b1Val').textContent = beta1.toFixed(2);
  document.getElementById('b2Slider').value = beta2;
  document.getElementById('b2Val').textContent = beta2.toFixed(3);
  document.getElementById('wdSlider').value = wd;
  document.getElementById('wdVal').textContent = wd.toFixed(3);
}

function buildOptimizerGrid() {
  const grid = document.getElementById('optGrid');
  grid.innerHTML = '';
  for (const id of optIds) {
    const meta = OPTIMIZER_META[id];
    const btn = document.createElement('button');
    btn.className = `opt-toggle ${engine.active[id] ? 'active' : ''} ${engine.inspected === id ? 'inspected' : ''}`;
    btn.innerHTML = `
      <span class="left"><i class="dot" style="background:${meta.color}"></i>${meta.name}</span>
      <span class="eye" title="Inspect vectors & matrices">&#9673;</span>
    `;
    btn.addEventListener('click', (e) => {
      if (e.target.classList.contains('eye')) {
        engine.inspected = id;
        engine.active[id] = true;
      } else {
        engine.active[id] = !engine.active[id];
        if (!engine.active[engine.inspected]) {
          const next = optIds.find((k) => engine.active[k]);
          if (next) engine.inspected = next;
        }
      }
      buildOptimizerGrid();
      update3DOptimizers();
      updateMathHUD();
    });
    grid.appendChild(btn);
  }
}

// Scenario Buttons
document.querySelectorAll('.scen-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.scen-btn').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    engine.loadScenario(btn.dataset.scen);
    syncSlidersFromEngine();
    rebuildSurface();
    update3DOptimizers();
    updateMathHUD();
    isPlaying = true;
    document.getElementById('playBtn').textContent = 'Pause';
  });
});

// Playback Controls
document.getElementById('playBtn').addEventListener('click', (e) => {
  isPlaying = !isPlaying;
  e.target.textContent = isPlaying ? 'Pause' : 'Play';
});
document.getElementById('stepBtn').addEventListener('click', () => {
  isPlaying = false;
  document.getElementById('playBtn').textContent = 'Play';
  engine.step();
  update3DOptimizers();
  updateMathHUD();
});
document.getElementById('restartBtn').addEventListener('click', () => {
  engine.reset();
  update3DOptimizers();
  updateMathHUD();
  isPlaying = true;
  document.getElementById('playBtn').textContent = 'Pause';
});
document.getElementById('speedBtn').addEventListener('click', (e) => {
  speedIdx = (speedIdx + 1) % speeds.length;
  e.target.textContent = speeds[speedIdx].label;
});

// Hyperparameter Sliders
const bindSlider = (id, valId, key, digits) => {
  document.getElementById(id).addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    engine.params[key] = v;
    document.getElementById(valId).textContent = v.toFixed(digits);
  });
};
bindSlider('lrSlider', 'lrVal', 'lr', 3);
bindSlider('b1Slider', 'b1Val', 'beta1', 2);
bindSlider('b2Slider', 'b2Val', 'beta2', 3);
bindSlider('wdSlider', 'wdVal', 'wd', 3);

// Top Action Buttons
document.getElementById('vectorsBtn').addEventListener('click', (e) => {
  showVectors = !showVectors;
  e.target.classList.toggle('active', showVectors);
  updateGeometryOverlays();
});
document.getElementById('taylorBtn').addEventListener('click', (e) => {
  showTaylor = !showTaylor;
  e.target.classList.toggle('active', showTaylor);
  updateGeometryOverlays();
});
document.getElementById('wireBtn').addEventListener('click', (e) => {
  surfaceMat.wireframe = !surfaceMat.wireframe;
  e.target.classList.toggle('active', surfaceMat.wireframe);
});
document.getElementById('orbitBtn').addEventListener('click', (e) => {
  controls.autoRotate = !controls.autoRotate;
  e.target.classList.toggle('active', controls.autoRotate);
});
document.getElementById('resetCamBtn').addEventListener('click', () => {
  camera.position.copy(DEFAULT_CAM);
  controls.target.copy(DEFAULT_TARGET);
  controls.update();
});

// Collapsible Side Panels
const bindCollapse = (cardId, btnId) => {
  const card = document.getElementById(cardId);
  const btn = document.getElementById(btnId);
  btn.addEventListener('click', () => {
    const collapsed = card.classList.toggle('collapsed');
    btn.setAttribute('aria-expanded', String(!collapsed));
  });
};
bindCollapse('controlsCard', 'toggleControlsBtn');
bindCollapse('inspectorCard', 'toggleInspectorBtn');

// Right Inspector Tabs
document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
    drawLossChart();
    drawContourTab();
  });
});

// Interactive 2D Contour Click -> Drop Start Point
contourCanvas.addEventListener('click', (e) => {
  const rect = contourCanvas.getBoundingClientRect();
  const u = (e.clientX - rect.left) / rect.width;
  const v = (e.clientY - rect.top) / rect.height;
  const w1 = -DOMAIN + u * (2 * DOMAIN);
  const w2 = DOMAIN - v * (2 * DOMAIN);
  engine.setStart(w1, w2);
  update3DOptimizers();
  updateMathHUD();
  isPlaying = true;
  document.getElementById('playBtn').textContent = 'Pause';
});

// Direct 3D Surface Raycasting Click -> Drop Start Point
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let downPos = { x: 0, y: 0 };

renderer.domElement.addEventListener('pointerdown', (e) => {
  downPos = { x: e.clientX, y: e.clientY };
});
renderer.domElement.addEventListener('pointerup', (e) => {
  if (Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y) > 5) return;
  pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
  pointer.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects([surfaceMesh, floorMesh]);
  if (hits.length > 0) {
    const pt = hits[0].point;
    engine.setStart(fromWorldX(pt.x), fromWorldZ(pt.z));
    update3DOptimizers();
    updateMathHUD();
    isPlaying = true;
    document.getElementById('playBtn').textContent = 'Pause';
  }
});

// Keyboard Shortcuts
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  if (k === ' ') { e.preventDefault(); document.getElementById('playBtn').click(); }
  if (k === 'v') document.getElementById('vectorsBtn').click();
  if (k === 'q') document.getElementById('taylorBtn').click();
  if (k === 'r') document.getElementById('resetCamBtn').click();
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ============================================================================
// 11. BOOTSTRAP & RENDER LOOP
// ============================================================================
syncSlidersFromEngine();
buildOptimizerGrid();
rebuildSurface();
update3DOptimizers();
updateMathHUD();

function animate(now) {
  requestAnimationFrame(animate);

  if (isPlaying && now - lastStepTime >= speeds[speedIdx].interval) {
    lastStepTime = now;
    if (engine.step()) {
      update3DOptimizers();
      updateMathHUD();
    } else {
      isPlaying = false;
      document.getElementById('playBtn').textContent = 'Play';
    }
  }

  controls.update();
  renderer.render(scene, camera);
  projectLabels();
}
requestAnimationFrame(animate);