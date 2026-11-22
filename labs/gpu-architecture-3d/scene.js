/**
 * ThinkModule Lab — Photorealistic WebGL / Three.js PBR AI-GPU Architecture Scene
 * Features:
 *   - Studio IBL reflections via RoomEnvironment + PMREMGenerator + 2K soft shadows
 *   - Selective UnrealBloomPass + ACESFilmicToneMapping (OutputPass)
 *   - 6 precision-modeled exploded hardware layers with InstancedMesh micro-components
 *   - 128 dynamically firing Tensor Core tiles + 6x 8-Hi HBM3e stacks + 3D optical data pulses
 *   - Live FLIR Thermal IR visualization mode & damped camera layer-focus transitions
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import {
  createPcbTextures,
  createInterposerTexture,
  createDieLithographyTexture,
  createBrushedMetalMap
} from "./textures.js";

// FLIR thermal colormap: t in [0..1] -> RGB
function flirColor(t, targetColor) {
  const u = Math.max(0, Math.min(1, t));
  const stops = [
    [0.00, [0.07, 0.06, 0.28]],
    [0.28, [0.36, 0.08, 0.54]],
    [0.54, [0.85, 0.16, 0.22]],
    [0.78, [1.45, 0.65, 0.10]],
    [1.00, [2.10, 1.85, 0.95]]
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (u <= t1) {
      const f = (u - t0) / Math.max(t1 - t0, 1e-5);
      targetColor.setRGB(
        c0[0] + (c1[0] - c0[0]) * f,
        c0[1] + (c1[1] - c0[1]) * f,
        c0[2] + (c1[2] - c0[2]) * f
      );
      return targetColor;
    }
  }
  return targetColor.setRGB(2.0, 1.8, 0.9);
}

export class GpuScene3D {
  constructor(container) {
    this.container = container;
    this.explodeTarget = 0.55;
    this.explodeCurrent = 0.55;
    this.thermalMode = false;
    this.autoOrbit = true;
    this.focusedLayer = "all";

    // Camera animation target
    this.camLerpActive = false;
    this.camGoalPos = new THREE.Vector3(6.8, 4.9, 7.6);
    this.camGoalTarget = new THREE.Vector3(0, 0.45, 0);

    this.layers = [];
    this.dataPackets = [];
    this.heatParticles = [];
    this.dummy = new THREE.Object3D();
    this.tmpColor = new THREE.Color();

    this._initRenderer();
    this._initLightingAndStage();
    this._buildHardwareLayers();
    this._buildAlignmentGuides();
    this._buildDataPulses();
    this._buildHeatParticles();
    this._bindEvents();
  }

  _initRenderer() {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x04070d, 0.032);

    this.camera = new THREE.PerspectiveCamera(38, w / h, 0.1, 80);
    this.camera.position.copy(this.camGoalPos);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "high-performance"
    });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;
    this.container.appendChild(this.renderer.domElement);

    // Studio IBL reflections so copper, aluminum, gold, and silicon gleam realistically
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.85;
    pmrem.dispose();

    // Post-processing with subtle selective Bloom (threshold 0.85 keeps metals razor-sharp)
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(w, h), 0.36, 0.42, 0.85);
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(new OutputPass());

    // Damped full-360 OrbitControls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.minDistance = 2.6;
    this.controls.maxDistance = 17.5;
    this.controls.maxPolarAngle = Math.PI * 0.96;
    this.controls.target.copy(this.camGoalTarget);

    this.controls.addEventListener("start", () => {
      this.camLerpActive = false;
    });
  }

  _initLightingAndStage() {
    // Key Studio Directional Light
    const key = new THREE.DirectionalLight(0xf4f8ff, 2.2);
    key.position.set(6.5, 11.0, 7.5);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 1;
    key.shadow.camera.far = 30;
    const d = 6.5;
    key.shadow.camera.left = -d;
    key.shadow.camera.right = d;
    key.shadow.camera.top = d;
    key.shadow.camera.bottom = -d;
    key.shadow.bias = -0.0004;
    this.scene.add(key);

    // Cool Cyan Rim Light
    const rim = new THREE.DirectionalLight(0x38bdf8, 1.05);
    rim.position.set(-7.5, 4.0, -6.5);
    this.scene.add(rim);

    // Subtle Warm Copper Fill Light
    const warmFill = new THREE.PointLight(0xff9e5e, 1.2, 16);
    warmFill.position.set(4.0, -1.5, -4.5);
    this.scene.add(warmFill);

    // Minimalist Studio Floor Shadow Catcher & Subtle Architectural Grid
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(42, 42),
      new THREE.MeshStandardMaterial({
        color: 0x050912,
        roughness: 0.72,
        metalness: 0.18
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -2.25;
    floor.receiveShadow = true;
    this.scene.add(floor);

    const grid = new THREE.GridHelper(24, 36, 0x162640, 0x0b1424);
    grid.position.y = -2.24;
    this.scene.add(grid);

    this.chipRoot = new THREE.Group();
    this.scene.add(this.chipRoot);
  }

  _buildHardwareLayers() {
    const pcbTex = createPcbTextures();
    const interposerMap = createInterposerTexture();
    const dieMap = createDieLithographyTexture();
    const copperMap = createBrushedMetalMap("#c87d4e", 0.16);
    const aluMap = createBrushedMetalMap("#242c38", 0.14);

    // Reusable high-craft PBR materials
    this.mats = {
      gold: new THREE.MeshStandardMaterial({ color: 0xd9a84e, metalness: 0.95, roughness: 0.22 }),
      steel: new THREE.MeshStandardMaterial({ color: 0xb8c4d4, metalness: 0.92, roughness: 0.24 }),
      choke: new THREE.MeshStandardMaterial({ color: 0x262d38, metalness: 0.72, roughness: 0.34 }),
      smdTan: new THREE.MeshStandardMaterial({ color: 0x9b744c, metalness: 0.25, roughness: 0.45 }),
      substrate: new THREE.MeshStandardMaterial({ color: 0x0c242b, metalness: 0.35, roughness: 0.42 }),
      copper: new THREE.MeshStandardMaterial({
        map: copperMap,
        metalness: 0.93,
        roughness: 0.26,
        emissive: new THREE.Color(0x000000),
        emissiveIntensity: 0
      }),
      fins: new THREE.MeshStandardMaterial({
        map: aluMap,
        metalness: 0.85,
        roughness: 0.34,
        emissive: new THREE.Color(0x000000),
        emissiveIntensity: 0
      }),
      shroud: new THREE.MeshPhysicalMaterial({
        color: 0x151c27,
        metalness: 0.78,
        roughness: 0.28,
        clearcoat: 0.35,
        clearcoatRoughness: 0.25
      }),
      die: new THREE.MeshPhysicalMaterial({
        map: dieMap,
        metalness: 0.55,
        roughness: 0.14,
        clearcoat: 1.0,
        clearcoatRoughness: 0.08,
        iridescence: 0.65,
        iridescenceIOR: 1.5,
        emissive: new THREE.Color(0x000000),
        emissiveIntensity: 0
      }),
      hbmWafer: new THREE.MeshPhysicalMaterial({
        color: 0x1a2638,
        metalness: 0.68,
        roughness: 0.22,
        clearcoat: 0.7,
        iridescence: 0.45,
        emissive: new THREE.Color(0x000000),
        emissiveIntensity: 0
      }),
      interposer: new THREE.MeshPhysicalMaterial({
        map: interposerMap,
        metalness: 0.5,
        roughness: 0.16,
        clearcoat: 0.95,
        iridescence: 0.55
      })
    };

    const addLayer = (baseY, explodeY) => {
      const g = new THREE.Group();
      g.userData = { baseY, explodeY };
      this.chipRoot.add(g);
      this.layers.push(g);
      return g;
    };

    // =========================================================================
    // LAYER 0: SXM CARRIER PCB & 24-PHASE VRM POWER DELIVERY
    // =========================================================================
    const L0 = addLayer(0.0, -1.65);

    const pcbMesh = new THREE.Mesh(
      new THREE.BoxGeometry(5.6, 0.14, 4.6),
      new THREE.MeshStandardMaterial({
        map: pcbTex.map,
        bumpMap: pcbTex.bumpMap,
        bumpScale: 0.012,
        roughness: 0.52,
        metalness: 0.28
      })
    );
    pcbMesh.castShadow = true;
    pcbMesh.receiveShadow = true;
    L0.add(pcbMesh);

    // 4 Corner Gold Mounting Bosses & Screws
    const bossGeo = new THREE.CylinderGeometry(0.16, 0.18, 0.18, 20);
    const screwGeo = new THREE.CylinderGeometry(0.09, 0.09, 0.22, 6);
    [[-2.45, -1.95], [2.45, -1.95], [-2.45, 1.95], [2.45, 1.95]].forEach(([x, z]) => {
      const boss = new THREE.Mesh(bossGeo, this.mats.gold);
      boss.position.set(x, 0.03, z);
      boss.castShadow = true;
      L0.add(boss);
      const scr = new THREE.Mesh(screwGeo, this.mats.steel);
      scr.position.set(x, 0.06, z);
      L0.add(scr);
    });

    // 24-Phase VRM Chokes (Left & Right power banks: 2 rows x 12)
    const chokeGeo = new THREE.BoxGeometry(0.28, 0.16, 0.28);
    const chokeInst = new THREE.InstancedMesh(chokeGeo, this.mats.choke, 24);
    chokeInst.castShadow = true;
    let ci = 0;
    [-2.25, 2.25].forEach(sideX => {
      for (let r = 0; r < 12; r++) {
        const z = -1.65 + r * 0.30;
        this.dummy.position.set(sideX, 0.15, z);
        this.dummy.rotation.set(0, 0, 0);
        this.dummy.scale.set(1, 1, 1);
        this.dummy.updateMatrix();
        chokeInst.setMatrixAt(ci++, this.dummy.matrix);
      }
    });
    chokeInst.instanceMatrix.needsUpdate = true;
    L0.add(chokeInst);

    // High-density Instanced Ceramic MLCC Capacitors (96 micro-SMDs)
    const smdGeo = new THREE.BoxGeometry(0.08, 0.045, 0.05);
    const smdInst = new THREE.InstancedMesh(smdGeo, this.mats.smdTan, 96);
    let si = 0;
    [-1.92, 1.92].forEach(x => {
      for (let r = 0; r < 24; r++) {
        this.dummy.position.set(x, 0.09, -1.72 + r * 0.15);
        this.dummy.updateMatrix();
        smdInst.setMatrixAt(si++, this.dummy.matrix);
      }
    });
    [-1.88, 1.88].forEach(z => {
      for (let c = 0; c < 24; c++) {
        this.dummy.position.set(-1.62 + c * 0.14, 0.09, z);
        this.dummy.updateMatrix();
        smdInst.setMatrixAt(si++, this.dummy.matrix);
      }
    });
    smdInst.instanceMatrix.needsUpdate = true;
    L0.add(smdInst);

    // NVLink Gold Mezzanine Connector Bars (Front & Back edges)
    const nvBarGeo = new THREE.BoxGeometry(2.6, 0.09, 0.16);
    [-2.14, 2.14].forEach(z => {
      const bar = new THREE.Mesh(nvBarGeo, this.mats.gold);
      bar.position.set(0, 0.09, z);
      L0.add(bar);
    });

    // =========================================================================
    // LAYER 1: ORGANIC PACKAGE SUBSTRATE & BGA SOLDER BALL ARRAY
    // =========================================================================
    const L1 = addLayer(0.18, -0.88);

    // 14x12 Eutectic BGA Solder Sphere Grid (168 spheres)
    const bgaGeo = new THREE.SphereGeometry(0.042, 10, 10);
    const bgaInst = new THREE.InstancedMesh(bgaGeo, this.mats.steel, 168);
    let bi = 0;
    for (let ix = 0; ix < 14; ix++) {
      for (let iz = 0; iz < 12; iz++) {
        const x = -1.62 + ix * 0.25;
        const z = -1.32 + iz * 0.24;
        this.dummy.position.set(x, -0.035, z);
        this.dummy.updateMatrix();
        bgaInst.setMatrixAt(bi++, this.dummy.matrix);
      }
    }
    bgaInst.instanceMatrix.needsUpdate = true;
    L1.add(bgaInst);

    // ABF Multi-Layer Package Substrate
    const subMesh = new THREE.Mesh(new THREE.BoxGeometry(3.85, 0.11, 3.25), this.mats.substrate);
    subMesh.position.y = 0.05;
    subMesh.castShadow = true;
    subMesh.receiveShadow = true;
    L1.add(subMesh);

    // Gold perimeter stiffening frame & decoupling micro-caps on substrate
    const frameGeo = new THREE.BoxGeometry(3.9, 0.03, 3.3);
    const frameRim = new THREE.Mesh(frameGeo, this.mats.gold);
    frameRim.position.y = 0.005;
    L1.add(frameRim);

    // =========================================================================
    // LAYER 2: CoWoS-S SILICON INTERPOSER (2.5D BRIDGE)
    // =========================================================================
    const L2 = addLayer(0.34, -0.10);

    const interposerMesh = new THREE.Mesh(new THREE.BoxGeometry(3.55, 0.055, 2.85), this.mats.interposer);
    interposerMesh.castShadow = true;
    interposerMesh.receiveShadow = true;
    L2.add(interposerMesh);

    // Micro-bump silver edge trim
    const edgeWire = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(3.56, 0.056, 2.86)),
      new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.35 })
    );
    L2.add(edgeWire);

    // =========================================================================
    // LAYER 3: 132-SM COMPUTE DIE + 6× 8-HI HBM3e MEMORY STACKS
    // =========================================================================
    const L3 = addLayer(0.41, 0.65);

    // Central Monolithic Silicon Compute Die
    this.computeDieMesh = new THREE.Mesh(new THREE.BoxGeometry(1.78, 0.11, 2.28), this.mats.die);
    this.computeDieMesh.position.y = 0.055;
    this.computeDieMesh.castShadow = true;
    this.computeDieMesh.receiveShadow = true;
    L3.add(this.computeDieMesh);

    // 128 Individual Firing Tensor Core / SM Tiles (Instanced HDR Material for Selective Bloom)
    const smGeo = new THREE.BoxGeometry(0.155, 0.016, 0.105);
    this.smMat = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
    this.smInst = new THREE.InstancedMesh(smGeo, this.smMat, 128);
    this.smMeta = [];

    let smIdx = 0;
    const colOffsets = [-0.73, -0.54, -0.35, -0.16, 0.16, 0.35, 0.54, 0.73];
    for (let r = 0; r < 16; r++) {
      const z = -1.02 + r * 0.136;
      for (let c = 0; c < 8; c++) {
        const x = colOffsets[c];
        this.dummy.position.set(x, 0.116, z);
        this.dummy.rotation.set(0, 0, 0);
        this.dummy.scale.set(1, 1, 1);
        this.dummy.updateMatrix();
        this.smInst.setMatrixAt(smIdx, this.dummy.matrix);
        const distCenter = Math.hypot(x / 0.85, z / 1.15);
        this.smMeta.push({
          x, z, distCenter,
          phase: (r * 0.41 + c * 0.73) % (Math.PI * 2),
          order: ((r * 7 + c * 13) % 128) / 128
        });
        this.smInst.setColorAt(smIdx, new THREE.Color(0.08, 0.14, 0.22));
        smIdx++;
      }
    }
    this.smInst.instanceMatrix.needsUpdate = true;
    this.smInst.instanceColor.needsUpdate = true;
    L3.add(this.smInst);

    // 6× 8-Hi HBM3e Memory Stacks (3 left, 3 right — each built from 8 vertical silicon wafers!)
    this.hbmPositions = [
      [-1.34, -0.76], [-1.34, 0.0], [-1.34, 0.76],
      [ 1.34, -0.76], [ 1.34, 0.0], [ 1.34, 0.76]
    ];
    const waferGeo = new THREE.BoxGeometry(0.56, 0.018, 0.58);
    const tsvGlowGeo = new THREE.BoxGeometry(0.52, 0.008, 0.54);
    this.tsvMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.55, 0.38, 1.45), toneMapped: false });

    const waferInst = new THREE.InstancedMesh(waferGeo, this.mats.hbmWafer, 6 * 8);
    waferInst.castShadow = true;
    waferInst.receiveShadow = true;
    const tsvInst = new THREE.InstancedMesh(tsvGlowGeo, this.tsvMat, 6 * 7);

    let wi = 0, ti = 0;
    this.hbmPositions.forEach(([hx, hz]) => {
      for (let s = 0; s < 8; s++) {
        this.dummy.position.set(hx, 0.012 + s * 0.029, hz);
        this.dummy.updateMatrix();
        waferInst.setMatrixAt(wi++, this.dummy.matrix);
        if (s < 7) {
          this.dummy.position.set(hx, 0.026 + s * 0.029, hz);
          this.dummy.updateMatrix();
          tsvInst.setMatrixAt(ti++, this.dummy.matrix);
        }
      }
    });
    waferInst.instanceMatrix.needsUpdate = true;
    tsvInst.instanceMatrix.needsUpdate = true;
    L3.add(waferInst);
    L3.add(tsvInst);

    // =========================================================================
    // LAYER 4: NICKEL-PLATED COPPER VAPOR CHAMBER & COLDPLATE
    // =========================================================================
    const L4 = addLayer(0.72, 1.65);

    // Main brushed copper vapor chamber slab
    const vcSlab = new THREE.Mesh(new THREE.BoxGeometry(4.65, 0.16, 3.85), this.mats.copper);
    vcSlab.castShadow = true;
    vcSlab.receiveShadow = true;
    L4.add(vcSlab);

    // Precision underside die-contact coldplate pedestal
    const pedestal = new THREE.Mesh(new THREE.BoxGeometry(3.48, 0.08, 2.42), this.mats.copper);
    pedestal.position.y = -0.10;
    pedestal.castShadow = true;
    L4.add(pedestal);

    // Recessed twin copper heatpipe ribs on top
    const pipeGeo = new THREE.CylinderGeometry(0.06, 0.06, 4.35, 16);
    pipeGeo.rotateZ(Math.PI / 2);
    [-1.15, -0.38, 0.38, 1.15].forEach(z => {
      const pipe = new THREE.Mesh(pipeGeo, this.mats.copper);
      pipe.position.set(0, 0.08, z);
      L4.add(pipe);
    });

    // 4 Spring-loaded Stainless Tension Screws
    const tScrewGeo = new THREE.CylinderGeometry(0.11, 0.11, 0.24, 16);
    [[-2.05, -1.62], [2.05, -1.62], [-2.05, 1.62], [2.05, 1.62]].forEach(([x, z]) => {
      const ts = new THREE.Mesh(tScrewGeo, this.mats.steel);
      ts.position.set(x, 0.06, z);
      ts.castShadow = true;
      L4.add(ts);
    });

    // =========================================================================
    // LAYER 5: SKIVED ANODIZED FIN ARRAY & ARCHITECTURAL CNC SHROUD
    // =========================================================================
    const L5 = addLayer(1.06, 2.72);

    // 36 Skived Brushed Aluminum Cooling Fins
    const finCount = 36;
    const finGeo = new THREE.BoxGeometry(0.032, 0.44, 3.62);
    const finInst = new THREE.InstancedMesh(finGeo, this.mats.fins, finCount);
    finInst.castShadow = true;
    finInst.receiveShadow = true;
    for (let i = 0; i < finCount; i++) {
      const x = -2.18 + (i / (finCount - 1)) * 4.36;
      this.dummy.position.set(x, 0.16, 0);
      this.dummy.updateMatrix();
      finInst.setMatrixAt(i, this.dummy.matrix);
    }
    finInst.instanceMatrix.needsUpdate = true;
    L5.add(finInst);

    // Minimalist Open-Window CNC Shroud Perimeter (lets you see cooling fins even when closed!)
    const railLongGeo = new THREE.BoxGeometry(4.82, 0.48, 0.24);
    [-1.86, 1.86].forEach(z => {
      const rail = new THREE.Mesh(railLongGeo, this.mats.shroud);
      rail.position.set(0, 0.17, z);
      rail.castShadow = true;
      L5.add(rail);
    });

    const railShortGeo = new THREE.BoxGeometry(0.32, 0.48, 3.48);
    [-2.25, 2.25].forEach(x => {
      const rail = new THREE.Mesh(railShortGeo, this.mats.shroud);
      rail.position.set(x, 0.17, 0);
      rail.castShadow = true;
      L5.add(rail);
    });

    // Sleek central titanium bridge across top of shroud
    const bridge = new THREE.Mesh(new THREE.BoxGeometry(4.84, 0.07, 0.68), this.mats.shroud);
    bridge.position.set(0, 0.41, 0);
    bridge.castShadow = true;
    L5.add(bridge);

    // Front Status LED Strip (changes color dynamically with GPU regime)
    this.ledMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.3, 1.5, 2.2), toneMapped: false });
    const ledStrip = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.035, 0.04), this.ledMat);
    ledStrip.position.set(0, 0.32, 1.98);
    L5.add(ledStrip);
  }

  _buildAlignmentGuides() {
    // 4 architectural CAD vertical alignment hairlines that appear subtly when exploded
    this.guideMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.18
    });
    const pts = [];
    [[-2.05, -1.62], [2.05, -1.62], [-2.05, 1.62], [2.05, 1.62]].forEach(([x, z]) => {
      pts.push(new THREE.Vector3(x, -1.65, z), new THREE.Vector3(x, 3.15, z));
    });
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    this.guides = new THREE.LineSegments(geo, this.guideMat);
    this.chipRoot.add(this.guides);
  }

  _buildDataPulses() {
    // 144 glowing optical data packets streaming between 6 HBM3e stacks and central Compute Die
    const count = 144;
    const pGeo = new THREE.BoxGeometry(0.11, 0.016, 0.022);
    this.pulseMat = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
    this.pulseInst = new THREE.InstancedMesh(pGeo, this.pulseMat, count);

    for (let i = 0; i < count; i++) {
      const stackIdx = i % 6;
      const [hx, hz] = this.hbmPositions[stackIdx];
      const side = hx < 0 ? -1 : 1;
      const laneOff = ((Math.floor(i / 6) % 12) - 5.5) * 0.042;
      this.dataPackets.push({
        side,
        startX: side * 1.07,
        endX: side * 0.22,
        z: hz + laneOff,
        t: (i / count + Math.random() * 0.2) % 1,
        speed: 0.35 + (i % 5) * 0.07,
        isKv: i % 3 === 0
      });
    }
    // Attach directly to Layer 3 so packets move cleanly with the Die & HBM3e assembly!
    this.layers[3].add(this.pulseInst);
  }

  _buildHeatParticles() {
    // 90 subtle convective heat micro-particles rising through the vapor chamber & cooling fins
    const count = 90;
    const hGeo = new THREE.SphereGeometry(0.028, 6, 6);
    this.heatMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.72,
      toneMapped: false
    });
    this.heatInst = new THREE.InstancedMesh(hGeo, this.heatMat, count);

    for (let i = 0; i < count; i++) {
      this.heatParticles.push({
        x: (Math.random() - 0.5) * 2.1,
        z: (Math.random() - 0.5) * 2.2,
        t: Math.random(),
        speed: 0.16 + Math.random() * 0.18,
        swirl: Math.random() * Math.PI * 2
      });
    }
    this.chipRoot.add(this.heatInst);
  }

  setExplode(norm01) {
    this.explodeTarget = Math.max(0, Math.min(1, norm01));
  }

  setThermalMode(enabled) {
    this.thermalMode = !!enabled;
  }

  setAutoOrbit(enabled) {
    this.autoOrbit = !!enabled;
  }

  focusLayer(layerKey) {
    this.focusedLayer = layerKey;
    if (layerKey === "all") {
      this.camGoalPos.set(6.8, 4.9, 7.6);
      this.camGoalTarget.set(0, 0.45, 0);
      this.camLerpActive = true;
      return;
    }

    // Ensure exploded view is open enough to clearly inspect inner layers
    const idx = parseInt(layerKey, 10);
    if (idx < 5 && this.explodeTarget < 0.55) {
      this.explodeTarget = 0.72;
    }
    const L = this.layers[idx];
    if (!L) return;

    const targetY = THREE.MathUtils.lerp(L.userData.baseY, L.userData.explodeY, this.explodeTarget);
    const dist = idx === 3 ? 4.4 : 5.4;
    this.camGoalTarget.set(0, targetY, 0);
    this.camGoalPos.set(dist * 0.82, targetY + 2.05, dist * 0.88);
    this.camLerpActive = true;
  }

  resetCamera() {
    this.focusLayer("all");
  }

  update(dt, elapsed, phys) {
    // 1. Smoothly interpolate Exploded View height of all 6 layers
    this.explodeCurrent += (this.explodeTarget - this.explodeCurrent) * Math.min(1, dt * 6.5);
    for (let i = 0; i < this.layers.length; i++) {
      const L = this.layers[i];
      L.position.y = THREE.MathUtils.lerp(L.userData.baseY, L.userData.explodeY, this.explodeCurrent);
    }
    this.guideMat.opacity = Math.max(0, (this.explodeCurrent - 0.08) * 0.24);

    // 2. Smooth camera fly-to or studio auto-orbit
    if (this.camLerpActive) {
      this.camera.position.lerp(this.camGoalPos, Math.min(1, dt * 4.5));
      this.controls.target.lerp(this.camGoalTarget, Math.min(1, dt * 4.5));
      if (this.camera.position.distanceTo(this.camGoalPos) < 0.03) {
        this.camLerpActive = false;
      }
    } else if (this.autoOrbit) {
      this.chipRoot.rotation.y += dt * 0.14;
    }

    // 3. Update 128 Tensor Core / SM micro-tiles (MFU + Thermal IR physics)
    const activeThreshold = phys.mfu;
    for (let i = 0; i < 128; i++) {
      const m = this.smMeta[i];
      const wave = 0.65 + 0.35 * Math.sin(elapsed * 3.8 + m.phase);

      if (this.thermalMode) {
        const localHeat = Math.max(0.12, Math.min(1, phys.heatNorm * (1.15 - m.distCenter * 0.35)));
        flirColor(localHeat, this.tmpColor);
      } else if (m.order < activeThreshold) {
        if (phys.throttlePct > 4) {
          // Overheating Tensor Cores glow intense amber/crimson
          this.tmpColor.setRGB(1.9 * wave, 0.52 * wave, 0.18 * wave);
        } else if (phys.mfu > 0.5) {
          // Saturated Prefill Compute: crisp HDR emerald-cyan
          this.tmpColor.setRGB(0.22 * wave, 1.55 * wave, 1.35 * wave);
        } else {
          // Sparse Memory-Wall Decode firing: cool cyan
          this.tmpColor.setRGB(0.25 * wave, 1.25 * wave, 1.85 * wave);
        }
      } else {
        // Idle dark silicon SM
        this.tmpColor.setRGB(0.05, 0.08, 0.13);
      }
      this.smInst.setColorAt(i, this.tmpColor);
    }
    this.smInst.instanceColor.needsUpdate = true;

    // 4. Update Copper Vapor Chamber, Fins, Die & HBM thermal glow
    if (this.thermalMode) {
      flirColor(Math.min(1, phys.heatNorm * 1.05), this.mats.die.emissive);
      this.mats.die.emissiveIntensity = 0.72;

      flirColor(Math.max(0.15, phys.heatNorm * 0.78), this.mats.copper.emissive);
      this.mats.copper.emissiveIntensity = 0.55;

      flirColor(Math.max(0.08, phys.heatNorm * 0.52), this.mats.fins.emissive);
      this.mats.fins.emissiveIntensity = 0.42;

      flirColor(Math.max(0.18, phys.heatNorm * 0.68), this.mats.hbmWafer.emissive);
      this.mats.hbmWafer.emissiveIntensity = 0.45;
    } else {
      // Subtle realistic incandescent warmth only when running very hot (>75°C)
      const hotGlow = Math.max(0, (phys.tempC - 75) / 22);
      this.mats.die.emissive.setRGB(1.0, 0.28, 0.06);
      this.mats.die.emissiveIntensity = hotGlow * 0.45;

      this.mats.copper.emissive.setRGB(0.95, 0.26, 0.04);
      this.mats.copper.emissiveIntensity = hotGlow * 0.30;

      this.mats.fins.emissive.setRGB(0.8, 0.2, 0.04);
      this.mats.fins.emissiveIntensity = hotGlow * 0.14;

      this.mats.hbmWafer.emissive.setRGB(0, 0, 0);
      this.mats.hbmWafer.emissiveIntensity = 0;
    }

    // HBM3e TSV seam glow intensity tracks HBM bandwidth
    const bwGlow = 0.45 + phys.bwUtil * 1.15;
    if (this.thermalMode) {
      flirColor(Math.max(0.25, phys.heatNorm * 0.75), this.tsvMat.color);
    } else {
      this.tsvMat.color.setRGB(0.45 * bwGlow, 0.35 * bwGlow, 1.35 * bwGlow);
    }

    // Status LED bar on front of shroud
    if (phys.throttlePct > 4) {
      this.ledMat.color.setRGB(2.2, 0.25, 0.25);
    } else if (phys.mfu > 0.5) {
      this.ledMat.color.setRGB(0.2, 1.9, 1.25);
    } else {
      this.ledMat.color.setRGB(0.25, 1.4, 2.2);
    }

    // 5. Animate 3D Optical Data Pulses across CoWoS-S Interposer (HBM -> Compute Die)
    const activePulses = Math.max(18, Math.round(phys.bwUtil * this.dataPackets.length));
    const speedScale = 0.5 + phys.bwUtil * 1.35;
    for (let i = 0; i < this.dataPackets.length; i++) {
      const p = this.dataPackets[i];
      p.t = (p.t + dt * p.speed * speedScale) % 1;

      if (i < activePulses) {
        const x = THREE.MathUtils.lerp(p.startX, p.endX, p.t);
        const y = 0.06 + Math.sin(p.t * Math.PI) * 0.045;
        this.dummy.position.set(x, y, p.z);
        this.dummy.rotation.set(0, 0, 0);
        this.dummy.scale.set(1, 1, 1);
        this.dummy.updateMatrix();
        this.pulseInst.setMatrixAt(i, this.dummy.matrix);

        if (p.isKv) {
          this.tmpColor.setRGB(0.95, 0.45, 2.0); // Violet KV-Cache stream
        } else {
          this.tmpColor.setRGB(0.22, 1.45, 2.2); // Cyan Weights stream
        }
        this.pulseInst.setColorAt(i, this.tmpColor);
      } else {
        this.dummy.scale.set(0, 0, 0);
        this.dummy.updateMatrix();
        this.pulseInst.setMatrixAt(i, this.dummy.matrix);
      }
    }
    this.pulseInst.instanceMatrix.needsUpdate = true;
    this.pulseInst.instanceColor.needsUpdate = true;

    // 6. Animate Subtle Rising Thermal Heat-Flux Plume
    const dieY = this.layers[3].position.y + 0.15;
    const topY = this.layers[5].position.y + 0.95;
    const heatVisible = Math.round((this.thermalMode ? 1.0 : Math.max(0.08, phys.heatNorm)) * this.heatParticles.length);

    for (let i = 0; i < this.heatParticles.length; i++) {
      const hp = this.heatParticles[i];
      hp.t = (hp.t + dt * hp.speed * (0.6 + phys.heatNorm * 1.4)) % 1;
      if (i < heatVisible && (phys.heatNorm > 0.22 || this.thermalMode)) {
        const y = THREE.MathUtils.lerp(dieY, topY, hp.t);
        const drift = Math.sin(elapsed * 1.5 + hp.swirl) * 0.12 * hp.t;
        const s = Math.sin(hp.t * Math.PI) * (0.65 + phys.heatNorm * 0.7);
        this.dummy.position.set(hp.x + drift, y, hp.z + drift * 0.6);
        this.dummy.scale.set(s, s, s);
        this.dummy.updateMatrix();
        this.heatInst.setMatrixAt(i, this.dummy.matrix);

        flirColor(Math.max(0.45, phys.heatNorm * (1.1 - hp.t * 0.45)), this.tmpColor);
        this.heatInst.setColorAt(i, this.tmpColor);
      } else {
        this.dummy.scale.set(0, 0, 0);
        this.dummy.updateMatrix();
        this.heatInst.setMatrixAt(i, this.dummy.matrix);
      }
    }
    this.heatInst.instanceMatrix.needsUpdate = true;
    this.heatInst.instanceColor.needsUpdate = true;

    this.controls.update();
    this.composer.render();
  }

  _bindEvents() {
    window.addEventListener("resize", () => {
      const w = this.container.clientWidth || window.innerWidth;
      const h = this.container.clientHeight || window.innerHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
      this.composer.setSize(w, h);
    });
  }
}