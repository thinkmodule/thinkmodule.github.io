/**
 * ThinkModule — Module 03: Tactile PBR 3D Sculpture Renderer
 * Uses Three.js RoomEnvironment studio IBL, soft shadow mapping,
 * a CNC-machined anodized plinth, high-res procedural contour textures,
 * solid sculpted terrain block, glowing 3D TubeGeometry light trails,
 * metallic PBR spheres, and smooth damped camera transitions.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import {
  DOMAIN,
  MAX_STEPS,
  OPTIMIZERS,
  lossToHeight,
  dLossToHeight
} from './math-engine.js';

const WORLD_SCALE = 1.62;
const HALF_W = DOMAIN * WORLD_SCALE;
const GRID_N = 144;
const TUBE_SEGS = 220;
const TUBE_RADIAL = 8;

const toX = (w1) => w1 * WORLD_SCALE;
const toZ = (w2) => -w2 * WORLD_SCALE;

// Refined scientific architectural palette (Deep Navy -> Teal -> Cyan -> Warm Sand -> Terracotta)
function samplePalette(t) {
  const x = Math.max(0, Math.min(1, t));
  const stops = [
    [0.00, 10, 28, 58],
    [0.24, 15, 86, 118],
    [0.48, 22, 148, 134],
    [0.72, 206, 154, 62],
    [0.90, 214, 92, 58],
    [1.00, 238, 178, 132]
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
  return [238, 178, 132];
}

export class LossStudioScene {
  constructor(container) {
    this.container = container;
    this.showVectors = true;
    this.showMarkers = true;
    this.cameraGoal = null;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x04070d, 0.022);

    this.camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.1, 100);
    this.presets = {
      iso: { pos: new THREE.Vector3(6.6, 5.6, 7.6), target: new THREE.Vector3(0, 0.9, 0) },
      top: { pos: new THREE.Vector3(0.01, 10.6, 0.9), target: new THREE.Vector3(0, 0.5, 0) },
      low: { pos: new THREE.Vector3(5.8, 2.55, 6.4), target: new THREE.Vector3(0, 0.95, 0) }
    };
    this.camera.position.copy(this.presets.iso.pos);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance', alpha: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    // Physical Studio IBL Reflections ("zum Anfassen nah")
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.65;
    pmrem.dispose();

    // Rock-solid OrbitControls (No Panning -> sculpture never escapes center)
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(this.presets.iso.target);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.065;
    this.controls.enablePan = false;
    this.controls.minPolarAngle = 0.12;
    this.controls.maxPolarAngle = Math.PI * 0.455;
    this.controls.minDistance = 5.2;
    this.controls.maxDistance = 17.5;
    this.controls.autoRotate = false;
    this.controls.autoRotateSpeed = 0.38;

    this.controls.addEventListener('start', () => {
      this.cameraGoal = null;
    });

    this._setupLights();
    this._setupPedestal();
    this._setupTerrain();
    this._setupOptimizers();
    this._setupVectors();

    this.markersGroup = new THREE.Group();
    this.scene.add(this.markersGroup);

    window.addEventListener('resize', () => this._onResize());
  }

  _setupLights() {
    this.scene.add(new THREE.AmbientLight(0x9cb8e4, 0.35));

    const key = new THREE.DirectionalLight(0xf6faff, 1.85);
    key.position.set(7.5, 13.5, 8.5);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 2;
    key.shadow.camera.far = 28;
    const d = 5.5;
    key.shadow.camera.left = -d;
    key.shadow.camera.right = d;
    key.shadow.camera.top = d;
    key.shadow.camera.bottom = -d;
    key.shadow.bias = -0.0005;
    this.scene.add(key);

    const rim = new THREE.DirectionalLight(0x38bdf8, 0.9);
    rim.position.set(-8.5, 6.5, -7.5);
    this.scene.add(rim);

    const warmFill = new THREE.PointLight(0xfbbf24, 0.55, 18);
    warmFill.position.set(5, 3.5, -6);
    this.scene.add(warmFill);
  }

  _setupPedestal() {
    // Subtle dark architectural floor grid far below
    const grid = new THREE.GridHelper(32, 32, 0x16243d, 0x0b1220);
    grid.position.y = -0.32;
    this.scene.add(grid);

    // CNC-machined dark anodized aluminum base plinth
    const plinthSize = HALF_W * 2 + 0.55;
    const plinth = new THREE.Mesh(
      new THREE.BoxGeometry(plinthSize, 0.28, plinthSize),
      new THREE.MeshStandardMaterial({
        color: 0x0c1322,
        metalness: 0.72,
        roughness: 0.32
      })
    );
    plinth.position.y = -0.15;
    plinth.receiveShadow = true;
    this.scene.add(plinth);

    // Fine luminous cyan trim frame around the plinth rim
    const trimSize = HALF_W * 2 + 0.04;
    const trimPts = [
      [-trimSize / 2, 0.008, trimSize / 2],
      [trimSize / 2, 0.008, trimSize / 2],
      [trimSize / 2, 0.008, -trimSize / 2],
      [-trimSize / 2, 0.008, -trimSize / 2],
      [-trimSize / 2, 0.008, trimSize / 2]
    ].map((p) => new THREE.Vector3(...p));
    const trimLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(trimPts),
      new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.55 })
    );
    this.scene.add(trimLine);

    // Subtle metallic engraved axis markers on the plinth edge (w1, w2)
    const makeBadge = (text) => {
      const c = document.createElement('canvas');
      c.width = 256;
      c.height = 64;
      const ctx = c.getContext('2d');
      ctx.fillStyle = 'rgba(0,0,0,0)';
      ctx.fillRect(0, 0, 256, 64);
      ctx.fillStyle = '#94b8dc';
      ctx.font = '600 28px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, 128, 34);
      const tex = new THREE.CanvasTexture(c);
      tex.anisotropy = 8;
      const m = new THREE.Mesh(
        new THREE.PlaneGeometry(1.15, 0.28),
        new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false })
      );
      m.rotation.x = -Math.PI / 2;
      return m;
    };

    const w1Badge = makeBadge('w₁  (−2 .. +2)');
    w1Badge.position.set(0, 0.006, HALF_W + 0.16);
    this.scene.add(w1Badge);

    const w2Badge = makeBadge('w₂  (−2 .. +2)');
    w2Badge.position.set(HALF_W + 0.16, 0.006, 0);
    w2Badge.rotation.z = Math.PI / 2;
    this.scene.add(w2Badge);
  }

  _setupTerrain() {
    // Sculpted Top Surface Geometry with exact UV mapping
    const vertsPerRow = GRID_N + 1;
    const positions = new Float32Array(vertsPerRow * vertsPerRow * 3);
    const uvs = new Float32Array(vertsPerRow * vertsPerRow * 2);
    const indices = [];

    for (let j = 0; j <= GRID_N; j++) {
      const v = 1 - j / GRID_N;
      const w2 = DOMAIN - (j / GRID_N) * 2 * DOMAIN;
      for (let i = 0; i <= GRID_N; i++) {
        const u = i / GRID_N;
        const w1 = -DOMAIN + u * 2 * DOMAIN;
        const idx = j * vertsPerRow + i;
        positions[idx * 3] = toX(w1);
        positions[idx * 3 + 1] = 0.25;
        positions[idx * 3 + 2] = toZ(w2);
        uvs[idx * 2] = u;
        uvs[idx * 2 + 1] = v;
      }
    }
    for (let j = 0; j < GRID_N; j++) {
      for (let i = 0; i < GRID_N; i++) {
        const a = j * vertsPerRow + i;
        const b = (j + 1) * vertsPerRow + i;
        const c = j * vertsPerRow + (i + 1);
        const d = (j + 1) * vertsPerRow + (i + 1);
        indices.push(a, b, c, b, d, c);
      }
    }

    this.terrainGeom = new THREE.BufferGeometry();
    this.terrainGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.terrainGeom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    this.terrainGeom.setIndex(indices);

    // High-Res 1024x1024 Baked Contour Albedo Texture
    this.texCanvas = document.createElement('canvas');
    this.texCanvas.width = 1024;
    this.texCanvas.height = 1024;
    this.texCtx = this.texCanvas.getContext('2d');
    this.terrainTex = new THREE.CanvasTexture(this.texCanvas);
    this.terrainTex.colorSpace = THREE.SRGBColorSpace;
    this.terrainTex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();

    // Tactile Clearcoated Ceramic / Satin Metal PBR Material
    this.terrainMat = new THREE.MeshPhysicalMaterial({
      map: this.terrainTex,
      roughness: 0.34,
      metalness: 0.14,
      clearcoat: 0.42,
      clearcoatRoughness: 0.22
    });

    this.terrainMesh = new THREE.Mesh(this.terrainGeom, this.terrainMat);
    this.terrainMesh.receiveShadow = true;
    this.terrainMesh.castShadow = true;
    this.scene.add(this.terrainMesh);

    // Solid Machined Obsidian Side Walls (connects surface rim down to plinth)
    this.skirtGroup = new THREE.Group();
    this.skirtMat = new THREE.MeshStandardMaterial({
      color: 0x0b1220,
      metalness: 0.58,
      roughness: 0.38,
      side: THREE.DoubleSide
    });
    this.scene.add(this.skirtGroup);
  }

  _setupOptimizers() {
    this.optMeshes = {};
    this.tubeGroup = new THREE.Group();
    this.scene.add(this.tubeGroup);

    for (const id of Object.keys(OPTIMIZERS)) {
      const meta = OPTIMIZERS[id];

      // Polished Metallic-Ceramic Sphere
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.125, 36, 36),
        new THREE.MeshPhysicalMaterial({
          color: meta.hex,
          emissive: meta.hex,
          emissiveIntensity: 0.38,
          metalness: 0.45,
          roughness: 0.14,
          clearcoat: 0.85,
          clearcoatRoughness: 0.08
        })
      );
      sphere.castShadow = true;
      this.scene.add(sphere);

      // Subtle glowing focus ring floating cleanly around the sphere
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(0.175, 0.215, 48),
        new THREE.MeshBasicMaterial({
          color: 0xffffff,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.85
        })
      );
      ring.rotation.x = -Math.PI / 2;
      this.scene.add(ring);

      this.optMeshes[id] = {
        sphere,
        ring,
        tubeMesh: null,
        cumLen: [],
        totalLen: 0
      };
    }
  }

  _setupVectors() {
    this.vectorGroup = new THREE.Group();
    this.scene.add(this.vectorGroup);

    this.gradArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0x38bdf8, 0.2, 0.11);
    this.eig1P = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0x34d399, 0.14, 0.08);
    this.eig1N = new THREE.ArrowHelper(new THREE.Vector3(-1, 0, 0), new THREE.Vector3(), 1, 0x34d399, 0.14, 0.08);
    this.eig2P = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 1, 0xfb7185, 0.14, 0.08);
    this.eig2N = new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), new THREE.Vector3(), 1, 0xfb7185, 0.14, 0.08);
    this.vectorGroup.add(this.gradArrow, this.eig1P, this.eig1N, this.eig2P, this.eig2N);
  }

  /**
   * Rebuilds sculpted terrain, high-res contour texture, machined side walls, and minima pins.
   */
  rebuildScenario(engine) {
    const sc = engine.scenario;
    const pos = this.terrainGeom.attributes.position;
    let minL = Infinity;
    let maxL = -Infinity;

    for (let i = 0; i < pos.count; i++) {
      const w1 = pos.getX(i) / WORLD_SCALE;
      const w2 = -pos.getZ(i) / WORLD_SCALE;
      const L = sc.eval(w1, w2).loss;
      if (L < minL) minL = L;
      if (L > maxL) maxL = L;
      pos.setY(i, lossToHeight(L));
    }
    pos.needsUpdate = true;
    this.terrainGeom.computeVertexNormals();

    this._bakeContourTexture(sc, minL, maxL);
    this._rebuildSkirts(sc);
    this._rebuildMarkers(sc);
    this.rebuildTrails(engine);
  }

  _bakeContourTexture(sc, minL, maxL) {
    const W = this.texCanvas.width;
    const H = this.texCanvas.height;
    const img = this.texCtx.createImageData(W, H);
    const data = img.data;
    const logMin = Math.log(1 + minL * 1.6);
    const logSpan = Math.max(0.001, Math.log(1 + maxL * 1.6) - logMin);

    for (let py = 0; py < H; py++) {
      const w2 = DOMAIN - (py / (H - 1)) * 2 * DOMAIN;
      for (let px = 0; px < W; px++) {
        const w1 = -DOMAIN + (px / (W - 1)) * 2 * DOMAIN;
        const L = sc.eval(w1, w2).loss;
        const t = (Math.log(1 + L * 1.6) - logMin) / logSpan;
        let [r, g, b] = samplePalette(t);

        // Laser-etched topographic contour lines
        const iso = Math.log(1 + L * 2.4) * 7.8;
        const dist = Math.abs((iso % 1 + 1) % 1 - 0.5);
        if (dist < 0.022) {
          const blend = 1 - dist / 0.022;
          r = r + (245 - r) * blend * 0.42;
          g = g + (250 - g) * blend * 0.42;
          b = b + (255 - b) * blend * 0.42;
        }

        // Ultra-subtle architectural coordinate hairlines at integer values
        const gx = Math.abs(((w1 + 4) % 1) - 0.5);
        const gy = Math.abs(((w2 + 4) % 1) - 0.5);
        if (gx > 0.494 || gy > 0.494) {
          r = r + (220 - r) * 0.14;
          g = g + (235 - g) * 0.14;
          b = b + (255 - b) * 0.14;
        }

        const idx = (py * W + px) * 4;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = 255;
      }
    }
    this.texCtx.putImageData(img, 0, 0);
    this.terrainTex.needsUpdate = true;
  }

  _rebuildSkirts(sc) {
    while (this.skirtGroup.children.length) {
      const m = this.skirtGroup.children.pop();
      m.geometry.dispose();
    }
    const buildWall = (sampleEdge) => {
      const verts = [];
      for (let i = 0; i < GRID_N; i++) {
        const a = sampleEdge(i / GRID_N);
        const b = sampleEdge((i + 1) / GRID_N);
        const ya = lossToHeight(sc.eval(a[0], a[1]).loss);
        const yb = lossToHeight(sc.eval(b[0], b[1]).loss);
        const ax = toX(a[0]), az = toZ(a[1]);
        const bx = toX(b[0]), bz = toZ(b[1]);
        verts.push(ax, 0, az, bx, 0, bz, bx, yb, bz);
        verts.push(ax, 0, az, bx, yb, bz, ax, ya, az);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
      g.computeVertexNormals();
      const mesh = new THREE.Mesh(g, this.skirtMat);
      mesh.receiveShadow = true;
      this.skirtGroup.add(mesh);
    };
    buildWall((t) => [-DOMAIN + 2 * DOMAIN * t, -DOMAIN]);
    buildWall((t) => [-DOMAIN + 2 * DOMAIN * t, DOMAIN]);
    buildWall((t) => [-DOMAIN, -DOMAIN + 2 * DOMAIN * t]);
    buildWall((t) => [DOMAIN, -DOMAIN + 2 * DOMAIN * t]);
  }

  _rebuildMarkers(sc) {
    while (this.markersGroup.children.length) {
      const c = this.markersGroup.children.pop();
      c.geometry.dispose();
      c.material.dispose();
    }
    for (const cp of sc.criticalPoints) {
      const y = lossToHeight(sc.eval(cp.w1, cp.w2).loss) + 0.035;
      const x = toX(cp.w1);
      const z = toZ(cp.w2);
      const color = cp.type === 'min' ? 0x34d399 : cp.type === 'local' ? 0xfbbf24 : 0x38bdf8;

      const ring = new THREE.Mesh(
        new THREE.RingGeometry(0.11, 0.16, 40),
        new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity: 0.9 })
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(x, y, z);
      this.markersGroup.add(ring);

      const pin = new THREE.Mesh(
        new THREE.CylinderGeometry(0.012, 0.012, 0.42, 12),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.65 })
      );
      pin.position.set(x, y + 0.21, z);
      this.markersGroup.add(pin);
    }
    this.markersGroup.visible = this.showMarkers;
  }

  /**
   * Builds smooth, arc-length-parameterized 3D glass-light TubeGeometry trails once per run.
   */
  rebuildTrails(engine) {
    while (this.tubeGroup.children.length) {
      const m = this.tubeGroup.children.pop();
      m.geometry.dispose();
      m.material.dispose();
    }

    for (const id of Object.keys(OPTIMIZERS)) {
      const hist = engine.history[id];
      const meta = OPTIMIZERS[id];
      const rawPts = [];
      const cumLen = [0];

      for (let i = 0; i < hist.length; i++) {
        const f = hist[i];
        const pt = new THREE.Vector3(toX(f.w1), lossToHeight(f.eval.loss) + 0.045, toZ(f.w2));
        if (i > 0) {
          const prev = new THREE.Vector3(
            toX(hist[i - 1].w1),
            lossToHeight(hist[i - 1].eval.loss) + 0.045,
            toZ(hist[i - 1].w2)
          );
          cumLen.push(cumLen[i - 1] + pt.distanceTo(prev));
        }
        if (rawPts.length === 0 || pt.distanceTo(rawPts[rawPts.length - 1]) > 0.008) {
          rawPts.push(pt);
        }
      }

      const totalLen = cumLen[cumLen.length - 1];
      this.optMeshes[id].cumLen = cumLen;
      this.optMeshes[id].totalLen = totalLen;

      if (rawPts.length >= 2) {
        const baseCurve = new THREE.CatmullRomCurve3(rawPts, false, 'centripetal', 0.3);
        const spaced = baseCurve.getSpacedPoints(TUBE_SEGS);
        const uniformCurve = new THREE.CatmullRomCurve3(spaced);
        const geom = new THREE.TubeGeometry(uniformCurve, TUBE_SEGS, 0.028, TUBE_RADIAL, false);
        geom.setDrawRange(0, 0);

        const mat = new THREE.MeshStandardMaterial({
          color: meta.hex,
          emissive: meta.hex,
          emissiveIntensity: 0.65,
          roughness: 0.25,
          metalness: 0.2
        });
        const mesh = new THREE.Mesh(geom, mat);
        this.tubeGroup.add(mesh);
        this.optMeshes[id].tubeMesh = mesh;
      } else {
        this.optMeshes[id].tubeMesh = null;
      }
    }
  }

  /**
   * Updates sphere positions, glowing tubes, and tangent geometry vectors at 60 FPS.
   */
  syncStep(engine, continuousStep, pulseTime) {
    const i0 = Math.floor(continuousStep);
    const i1 = Math.min(MAX_STEPS, i0 + 1);
    const alpha = continuousStep - i0;

    for (const id of Object.keys(OPTIMIZERS)) {
      const sample = engine.sample(id, continuousStep);
      const vis = this.optMeshes[id];
      const isFocused = id === engine.focusedId;

      const x = toX(sample.w1);
      const z = toZ(sample.w2);
      const y = lossToHeight(sample.loss) + 0.125;

      vis.sphere.position.set(x, y, z);
      vis.sphere.scale.setScalar(isFocused ? 1.14 : 0.92);

      vis.ring.visible = isFocused;
      if (isFocused) {
        vis.ring.position.set(x, y - 0.06, z);
        const s = 1 + 0.08 * Math.sin(pulseTime * 0.004);
        vis.ring.scale.setScalar(s);
      }

      if (vis.tubeMesh && vis.totalLen > 1e-4) {
        const lenAtStep = vis.cumLen[i0] + (vis.cumLen[i1] - vis.cumLen[i0]) * alpha;
        const frac = Math.max(0, Math.min(1, lenAtStep / vis.totalLen));
        const segCount = Math.floor(frac * TUBE_SEGS);
        vis.tubeMesh.geometry.setDrawRange(0, segCount * TUBE_RADIAL * 6);
        vis.tubeMesh.material.emissiveIntensity = isFocused ? 0.88 : 0.42;
      }
    }

    // Update 3D Gradient & Hessian Eigenvectors on focused optimizer
    this.vectorGroup.visible = this.showVectors;
    if (this.showVectors) {
      const focused = engine.sample(engine.focusedId, continuousStep);
      const ev = engine.evaluate(focused.w1, focused.w2);
      const origin = new THREE.Vector3(toX(focused.w1), lossToHeight(ev.loss) + 0.14, toZ(focused.w2));
      const dydL = dLossToHeight(ev.loss);

      const tangent = (dw1, dw2) => {
        const dL = ev.g1 * dw1 + ev.g2 * dw2;
        return new THREE.Vector3(dw1 * WORLD_SCALE, dydL * dL, -dw2 * WORLD_SCALE).normalize();
      };

      if (ev.gradNorm > 0.015) {
        this.gradArrow.visible = true;
        this.gradArrow.position.copy(origin);
        this.gradArrow.setDirection(tangent(-ev.g1, -ev.g2));
        const gLen = Math.min(1.25, Math.max(0.38, 0.34 + Math.log1p(ev.gradNorm) * 0.55));
        this.gradArrow.setLength(gLen, 0.18, 0.10);
      } else {
        this.gradArrow.visible = false;
      }

      const setEig = (aP, aN, vec, lambda) => {
        const col = lambda >= 0 ? 0x34d399 : 0xfb7185;
        const len = Math.min(0.95, Math.max(0.36, 0.34 + Math.log1p(Math.abs(lambda)) * 0.32));
        aP.position.copy(origin);
        aN.position.copy(origin);
        aP.setDirection(tangent(vec[0], vec[1]));
        aN.setDirection(tangent(-vec[0], -vec[1]));
        aP.setColor(col);
        aN.setColor(col);
        aP.setLength(len, 0.13, 0.07);
        aN.setLength(len, 0.13, 0.07);
      };
      setEig(this.eig1P, this.eig1N, ev.e1, ev.l1);
      setEig(this.eig2P, this.eig2N, ev.e2, ev.l2);
    }
  }

  flyToPreset(name) {
    const p = this.presets[name];
    if (!p) return;
    this.cameraGoal = { pos: p.pos.clone(), target: p.target.clone() };
  }

  setAutoOrbit(enabled) {
    this.controls.autoRotate = enabled;
  }

  setVectorsVisible(v) {
    this.showVectors = v;
    this.vectorGroup.visible = v;
  }

  setMarkersVisible(v) {
    this.showMarkers = v;
    this.markersGroup.visible = v;
  }

  render() {
    if (this.cameraGoal) {
      this.camera.position.lerp(this.cameraGoal.pos, 0.08);
      this.controls.target.lerp(this.cameraGoal.target, 0.08);
      if (this.camera.position.distanceTo(this.cameraGoal.pos) < 0.02) {
        this.cameraGoal = null;
      }
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  _onResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }
}