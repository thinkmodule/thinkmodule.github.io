/**
 * ThinkModule — Synchronized Multi-View Rendering System
 * ============================================================================
 * Drives 4 synchronized views from a single headless WorldState:
 *   View 01: Three.js 3D Real-Time Highway Environment
 *   View 02: Three.js 3D Occluded LiDAR Point Cloud Reconstruction
 *   View 03: HiDPI 2D Object Detection, Occlusion Cones & Tracking (Clickable)
 *   View 04: HiDPI 2D Rule-Based Autonomous Driving Intelligence & Signal Plots
 *
 * Strict Synchronization Rule:
 *   Zero internal animation clocks. Every wheel rotation, streetlamp, road dash,
 *   turn signal, and sensor sweep is computed purely from state.clock.simTime
 *   and state.ego.s. When paused, the entire physical world is mathematically
 *   frozen while cameras and inspectors remain fully responsive.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  LANE_CENTERS,
  LANE_WIDTH,
  ROAD_HALF_WIDTH,
  LAMP_SPACING,
  LIDAR_MOUNT_HEIGHT,
  LIDAR_MAX_RANGE,
  QUALITY_PRESETS,
} from "./sim-engine.js";

const STATUS_COLORS = {
  CONFIRMED: "#34d399",
  PARTIAL: "#f59e0b",
  OCCLUDED: "#94a3b8",
  COASTING: "#f59e0b",
  SELECTED: "#38bdf8",
  CRITICAL: "#ef4444",
};

/**
 * Builds a clean, architecturally proportioned procedural 3D vehicle mesh
 * with 4 articulated wheel pivots (rotating around X by exact -wheelAngle).
 */
function createVehicleMesh(type, colorHex, isEgo = false) {
  const group = new THREE.Group();

  const dims =
    type === "truck"
      ? { l: 9.8, w: 2.52, h: 3.45, r: 0.5, cabL: 2.25 }
      : type === "van"
        ? { l: 5.8, w: 2.08, h: 2.38, r: 0.39 }
        : type === "suv"
          ? { l: 4.95, w: 2.02, h: 1.76, r: 0.38 }
          : { l: isEgo ? 4.82 : 4.65, w: isEgo ? 1.94 : 1.88, h: isEgo ? 1.48 : 1.44, r: isEgo ? 0.36 : 0.34 };

  const bodyMat = new THREE.MeshStandardMaterial({
    color: isEgo ? 0xd8e0ea : colorHex,
    roughness: isEgo ? 0.28 : 0.42,
    metalness: isEgo ? 0.45 : 0.35,
  });

  const trimMat = new THREE.MeshStandardMaterial({
    color: 0x161a22,
    roughness: 0.78,
    metalness: 0.15,
  });

  const glassMat = new THREE.MeshStandardMaterial({
    color: 0x0e1624,
    roughness: 0.14,
    metalness: 0.75,
  });

  const headMat = new THREE.MeshBasicMaterial({ color: 0xeaf4ff });
  const tailMat = new THREE.MeshBasicMaterial({ color: 0x7c1d24 });
  const turnLeftMat = new THREE.MeshBasicMaterial({ color: 0x3b2a14 });
  const turnRightMat = new THREE.MeshBasicMaterial({ color: 0x3b2a14 });

  if (type === "truck") {
    // Cargo Box
    const boxL = dims.l - dims.cabL - 0.25;
    const cargoGeo = new THREE.BoxGeometry(dims.w, dims.h - 0.88, boxL);
    const cargo = new THREE.Mesh(cargoGeo, bodyMat);
    cargo.position.set(0, 0.88 + (dims.h - 0.88) * 0.5, dims.cabL * 0.5 + 0.12);
    group.add(cargo);

    // Tractor Cab
    const cabMat = new THREE.MeshStandardMaterial({ color: 0x3a4350, roughness: 0.4, metalness: 0.35 });
    const cabGeo = new THREE.BoxGeometry(dims.w * 0.94, 2.25, dims.cabL);
    const cab = new THREE.Mesh(cabGeo, cabMat);
    cab.position.set(0, 0.65 + 1.125, -dims.l * 0.5 + dims.cabL * 0.5);
    group.add(cab);

    // Windshield
    const ws = new THREE.Mesh(new THREE.BoxGeometry(dims.w * 0.84, 0.85, 0.08), glassMat);
    ws.position.set(0, 2.08, -dims.l * 0.5 + 0.02);
    group.add(ws);

    // Chassis rail
    const chassis = new THREE.Mesh(new THREE.BoxGeometry(dims.w * 0.78, 0.35, dims.l * 0.94), trimMat);
    chassis.position.set(0, 0.58, 0);
    group.add(chassis);
  } else if (type === "van") {
    const body = new THREE.Mesh(new THREE.BoxGeometry(dims.w, dims.h - 0.48, dims.l * 0.94), bodyMat);
    body.position.set(0, 0.42 + (dims.h - 0.48) * 0.5, 0.08);
    group.add(body);

    const ws = new THREE.Mesh(new THREE.BoxGeometry(dims.w * 0.88, 0.82, 0.08), glassMat);
    ws.position.set(0, 1.56, -dims.l * 0.465);
    ws.rotation.x = 0.24;
    group.add(ws);
  } else {
    // Sedan / SUV lower chassis & sculpted cabin
    const lowerH = type === "suv" ? 0.68 : 0.56;
    const lower = new THREE.Mesh(new THREE.BoxGeometry(dims.w, lowerH, dims.l), bodyMat);
    lower.position.set(0, dims.r * 0.65 + lowerH * 0.5, 0);
    group.add(lower);

    const cabinH = dims.h - (dims.r * 0.65 + lowerH);
    const cabinL = type === "suv" ? dims.l * 0.62 : dims.l * 0.48;
    const cabin = new THREE.Mesh(new THREE.BoxGeometry(dims.w * 0.84, cabinH, cabinL), glassMat);
    cabin.position.set(0, dims.r * 0.65 + lowerH + cabinH * 0.48, type === "suv" ? 0.28 : 0.18);
    group.add(cabin);

    const roof = new THREE.Mesh(new THREE.BoxGeometry(dims.w * 0.82, 0.07, cabinL * 0.92), bodyMat);
    roof.position.set(0, dims.h - 0.03, type === "suv" ? 0.28 : 0.18);
    group.add(roof);
  }

  // Headlights & Taillights
  const lightY = type === "truck" ? 0.95 : 0.64;
  const headGeo = new THREE.BoxGeometry(0.38, 0.12, 0.06);
  const hlL = new THREE.Mesh(headGeo, headMat);
  hlL.position.set(-dims.w * 0.34, lightY, -dims.l * 0.5 - 0.01);
  const hlR = hlL.clone();
  hlR.position.x = dims.w * 0.34;
  group.add(hlL, hlR);

  const tailGeo = new THREE.BoxGeometry(0.42, 0.13, 0.06);
  const tlL = new THREE.Mesh(tailGeo, tailMat);
  tlL.position.set(-dims.w * 0.34, lightY + 0.05, dims.l * 0.5 + 0.01);
  const tlR = tlL.clone();
  tlR.position.x = dims.w * 0.34;
  group.add(tlL, tlR);

  // Turn signals
  const turnGeo = new THREE.BoxGeometry(0.16, 0.1, 0.07);
  const tsL = new THREE.Mesh(turnGeo, turnLeftMat);
  tsL.position.set(-dims.w * 0.44, lightY + 0.05, dims.l * 0.5 + 0.015);
  const tsR = new THREE.Mesh(turnGeo, turnRightMat);
  tsR.position.set(dims.w * 0.44, lightY + 0.05, dims.l * 0.5 + 0.015);
  group.add(tsL, tsR);

  // Articulated 4 Wheels with metallic spokes so rotation & Pause-freeze are visually unmistakable
  const wheels = [];
  const wheelOffsetsZ =
    type === "truck"
      ? [-dims.l * 0.37, dims.l * 0.24, dims.l * 0.38]
      : [-dims.l * 0.31, dims.l * 0.31];
  const wheelOffsetX = dims.w * 0.5 - 0.09;
  const tireMat = new THREE.MeshStandardMaterial({ color: 0x111318, roughness: 0.9 });
  const rimMat = new THREE.MeshStandardMaterial({ color: 0x9aa4b4, roughness: 0.3, metalness: 0.7 });
  const tireGeo = new THREE.CylinderGeometry(dims.r, dims.r, 0.24, 18);
  tireGeo.rotateZ(Math.PI / 2);
  const spokeGeo = new THREE.BoxGeometry(0.26, dims.r * 1.42, 0.055);
  const spoke2Geo = new THREE.BoxGeometry(0.26, 0.055, dims.r * 1.42);

  for (const wz of wheelOffsetsZ) {
    for (const side of [-1, 1]) {
      const pivot = new THREE.Group();
      pivot.position.set(side * wheelOffsetX, dims.r, wz);
      pivot.add(new THREE.Mesh(tireGeo, tireMat));
      pivot.add(new THREE.Mesh(spokeGeo, rimMat));
      pivot.add(new THREE.Mesh(spoke2Geo, rimMat));
      group.add(pivot);
      wheels.push(pivot);
    }
  }

  // Roof-mounted Autonomous Sensor Mast & Rotating LiDAR Puck (Ego only)
  let lidarSpinner = null;
  if (isEgo) {
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.16, 0.24, 16), trimMat);
    mast.position.set(0, dims.h + 0.12, 0.0);
    group.add(mast);

    lidarSpinner = new THREE.Group();
    lidarSpinner.position.set(0, LIDAR_MOUNT_HEIGHT - 0.04, 0.0);
    const puck = new THREE.Mesh(
      new THREE.CylinderGeometry(0.14, 0.14, 0.16, 20),
      new THREE.MeshStandardMaterial({ color: 0x1b2330, roughness: 0.3, metalness: 0.8 })
    );
    const opticWindow = new THREE.Mesh(
      new THREE.CylinderGeometry(0.144, 0.144, 0.06, 20),
      new THREE.MeshBasicMaterial({ color: 0x38bdf8 })
    );
    const emitterDot = new THREE.Mesh(
      new THREE.BoxGeometry(0.05, 0.07, 0.16),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    emitterDot.position.z = -0.07;
    lidarSpinner.add(puck, opticWindow, emitterDot);
    group.add(lidarSpinner);
  }

  return {
    group,
    wheels,
    lidarSpinner,
    bodyMat,
    tailMat,
    turnLeftMat,
    turnRightMat,
    type,
  };
}

/* ============================================================================
   VIEW 01 — Real-Time 3D Highway Environment Renderer
   ============================================================================ */
class Env3DRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setClearColor(0x070a0f, 1);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x070a0f, 0.0044);

    this.camera = new THREE.PerspectiveCamera(46, 16 / 9, 0.2, 420);
    this.camera.position.set(0, 6.2, 13.5);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.485;
    this.controls.minDistance = 4;
    this.controls.maxDistance = 130;
    this.controls.enabled = false;

    this.camMode = "follow";
    this._camTargetPos = new THREE.Vector3();
    this._camLookAt = new THREE.Vector3();
    this._maxDpr = 1.5;

    this._buildLights();
    this._buildHighway();
    this._buildStreetlamps();
    this._buildEgo();
    this._buildVehiclePool();
    this._buildOverlays();
  }

  _buildLights() {
    this.scene.add(new THREE.HemisphereLight(0xaec4e0, 0x161b24, 0.95));
    const dir = new THREE.DirectionalLight(0xeef4ff, 1.35);
    dir.position.set(22, 42, 18);
    this.scene.add(dir);

    const rim = new THREE.DirectionalLight(0x6488b8, 0.45);
    rim.position.set(-28, 18, -32);
    this.scene.add(rim);
  }

  _buildHighway() {
    // Surrounding ground plane
    const terrain = new THREE.Mesh(
      new THREE.PlaneGeometry(460, 560),
      new THREE.MeshStandardMaterial({ color: 0x0a0d12, roughness: 0.96 })
    );
    terrain.rotation.x = -Math.PI / 2;
    terrain.position.y = -0.04;
    this.scene.add(terrain);

    // Asphalt Deck
    const roadW = ROAD_HALF_WIDTH * 2;
    const road = new THREE.Mesh(
      new THREE.PlaneGeometry(roadW, 560),
      new THREE.MeshStandardMaterial({ color: 0x171b22, roughness: 0.88, metalness: 0.05 })
    );
    road.rotation.x = -Math.PI / 2;
    road.position.y = 0;
    this.scene.add(road);

    // Solid Edge Lines
    const edgeMat = new THREE.MeshBasicMaterial({ color: 0x7b8594 });
    for (const side of [-1, 1]) {
      const edge = new THREE.Mesh(new THREE.PlaneGeometry(0.16, 560), edgeMat);
      edge.rotation.x = -Math.PI / 2;
      edge.position.set(side * (ROAD_HALF_WIDTH - 0.55), 0.012, 0);
      this.scene.add(edge);
    }

    // World-Anchored Dashed Lane Dividers (moved strictly via ego.s — 100% frozen on Pause!)
    this.dashCycle = 12.0; // 4m dash + 8m gap
    this.dashGroup = new THREE.Group();
    const dashGeo = new THREE.PlaneGeometry(0.14, 4.0);
    dashGeo.rotateX(-Math.PI / 2);
    const dashMat = new THREE.MeshBasicMaterial({ color: 0x9aa4b4 });
    const dividerXs = [-LANE_WIDTH * 0.5, LANE_WIDTH * 0.5];
    for (let z = -240; z <= 180; z += this.dashCycle) {
      for (const dx of dividerXs) {
        const dash = new THREE.Mesh(dashGeo, dashMat);
        dash.position.set(dx, 0.014, z);
        this.dashGroup.add(dash);
      }
    }
    this.scene.add(this.dashGroup);

    // Continuous Steel Guardrails
    const railMat = new THREE.MeshStandardMaterial({ color: 0x3b434f, roughness: 0.55, metalness: 0.5 });
    for (const side of [-1, 1]) {
      const rail = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.34, 560), railMat);
      rail.position.set(side * ROAD_HALF_WIDTH, 0.62, 0);
      this.scene.add(rail);
    }
  }

  _buildStreetlamps() {
    // 10 pairs of world-anchored highway lamps spaced by exact LAMP_SPACING (40m)
    this.lampPairs = [];
    const poleMat = new THREE.MeshStandardMaterial({ color: 0x363e4a, roughness: 0.5, metalness: 0.6 });
    const headMat = new THREE.MeshBasicMaterial({ color: 0xdfe9f7 });
    const poolMat = new THREE.MeshBasicMaterial({
      color: 0xbcd0ea,
      transparent: true,
      opacity: 0.055,
      depthWrite: false,
    });
    const poleGeo = new THREE.CylinderGeometry(0.08, 0.11, 7.6, 8);
    const armGeo = new THREE.BoxGeometry(2.1, 0.09, 0.09);
    const lumGeo = new THREE.BoxGeometry(0.58, 0.08, 0.24);
    const poolGeo = new THREE.CircleGeometry(5.6, 20);
    poolGeo.rotateX(-Math.PI / 2);

    for (let i = 0; i < 11; i++) {
      const pair = new THREE.Group();
      for (const side of [-1, 1]) {
        const pole = new THREE.Mesh(poleGeo, poleMat);
        pole.position.set(side * (ROAD_HALF_WIDTH + 0.65), 3.8, 0);
        pair.add(pole);

        const arm = new THREE.Mesh(armGeo, poleMat);
        arm.position.set(side * (ROAD_HALF_WIDTH - 0.35), 7.55, 0);
        pair.add(arm);

        const lum = new THREE.Mesh(lumGeo, headMat);
        lum.position.set(side * (ROAD_HALF_WIDTH - 1.25), 7.49, 0);
        pair.add(lum);

        const pool = new THREE.Mesh(poolGeo, poolMat);
        pool.position.set(side * (ROAD_HALF_WIDTH - 2.4), 0.016, 0);
        pair.add(pool);
      }
      this.scene.add(pair);
      this.lampPairs.push(pair);
    }
  }

  _buildEgo() {
    this.egoVisual = createVehicleMesh("sedan", 0xd8e0ea, true);
    this.scene.add(this.egoVisual.group);
  }

  _buildVehiclePool() {
    // Pre-build reusable mesh cache per pool slot & type to avoid runtime allocation
    this.poolSlots = [];
    for (let i = 0; i < 18; i++) {
      const container = new THREE.Group();
      container.visible = false;
      const variants = {
        sedan: createVehicleMesh("sedan", 0x6e7b8b),
        suv: createVehicleMesh("suv", 0x566270),
        van: createVehicleMesh("van", 0x9aa3ad),
        truck: createVehicleMesh("truck", 0xa9b2bc),
      };
      for (const k of Object.keys(variants)) {
        variants[k].group.visible = false;
        container.add(variants[k].group);
      }
      this.scene.add(container);
      this.poolSlots.push({ container, variants, activeType: null });
    }
  }

  _buildOverlays() {
    // Sparse LiDAR sweep beam visualization
    const maxRayVerts = 256 * 2;
    this.rayPosArr = new Float32Array(maxRayVerts * 3);
    this.rayColArr = new Float32Array(maxRayVerts * 3);
    const rayGeo = new THREE.BufferGeometry();
    rayGeo.setAttribute("position", new THREE.BufferAttribute(this.rayPosArr, 3));
    rayGeo.setAttribute("color", new THREE.BufferAttribute(this.rayColArr, 3));
    this.rayLines = new THREE.LineSegments(
      rayGeo,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.42 })
    );
    this.scene.add(this.rayLines);

    // Pooled 3D Tracking Wireframe Bounding Boxes
    this.boxPool = [];
    const unitBoxEdges = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    for (let i = 0; i < 18; i++) {
      const mat = new THREE.LineBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.9 });
      const box = new THREE.LineSegments(unitBoxEdges, mat);
      box.visible = false;
      this.scene.add(box);
      this.boxPool.push(box);
    }

    // Planned Longitudinal Path Ribbon
    const pathPts = new Float32Array(34 * 3);
    const pathGeo = new THREE.BufferGeometry();
    pathGeo.setAttribute("position", new THREE.BufferAttribute(pathPts, 3));
    this.pathLine = new THREE.Line(
      pathGeo,
      new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.78 })
    );
    this.scene.add(this.pathLine);

    // Corridor edges
    const leftCor = new Float32Array(34 * 3);
    const rightCor = new Float32Array(34 * 3);
    const lcGeo = new THREE.BufferGeometry();
    lcGeo.setAttribute("position", new THREE.BufferAttribute(leftCor, 3));
    const rcGeo = new THREE.BufferGeometry();
    rcGeo.setAttribute("position", new THREE.BufferAttribute(rightCor, 3));
    const corMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.28 });
    this.corridorLeft = new THREE.Line(lcGeo, corMat);
    this.corridorRight = new THREE.Line(rcGeo, corMat);
    this.scene.add(this.corridorLeft, this.corridorRight);

    // Debug Coordinate Axes & Range Rings
    this.debugGroup = new THREE.Group();
    this.debugGroup.visible = false;
    this.debugGroup.add(new THREE.AxesHelper(4.5));
    for (const r of [30, 60, 90, 120]) {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(r - 0.14, r + 0.14, 64),
        new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.22, side: THREE.DoubleSide })
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.04;
      this.debugGroup.add(ring);
    }
    this.scene.add(this.debugGroup);
  }

  setCameraMode(mode) {
    this.camMode = mode;
    this.controls.enabled = mode === "orbit";
    if (mode === "orbit") {
      this.controls.target.set(0, 1.2, -14);
      this.controls.update();
    }
  }

  resetCamera() {
    if (this.camMode === "orbit") {
      this.camera.position.set(14, 11, 18);
      this.controls.target.set(0, 1.0, -16);
      this.controls.update();
    } else {
      this.setCameraMode("follow");
    }
  }

  setQuality(presetKey) {
    const q = QUALITY_PRESETS[presetKey] || QUALITY_PRESETS.balanced;
    this._maxDpr = q.maxDpr;
    this.resize();
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const w = Math.max(10, Math.floor(rect.width));
    const h = Math.max(10, Math.floor(rect.height));
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, this._maxDpr));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render(state, settings) {
    const ego = state.ego;
    const simTime = state.clock.simTime;

    // 1. World-Anchored Road Dashes & Streetlamps (Strictly driven by ego.s!)
    this.dashGroup.position.z = ego.s % this.dashCycle;

    const firstLampIdx = Math.floor((ego.s - 120) / LAMP_SPACING);
    for (let i = 0; i < this.lampPairs.length; i++) {
      const worldLampS = (firstLampIdx + i) * LAMP_SPACING;
      this.lampPairs[i].position.z = -(worldLampS - ego.s);
    }

    // 2. Ego Vehicle Pose, Exact Wheel Rotation & LiDAR Puck Spin
    this.egoVisual.group.position.set(ego.x, 0, 0);
    for (const w of this.egoVisual.wheels) {
      w.rotation.x = -ego.wheelAngle;
    }
    if (this.egoVisual.lidarSpinner) {
      this.egoVisual.lidarSpinner.rotation.y = state.clock.scanAngleRad;
    }
    this.egoVisual.tailMat.color.setHex(ego.brakeLight ? 0xff2b38 : 0x7c1d24);

    // 3. Synchronized Traffic Vehicles (Zero uncoordinated timers)
    const blinkOn = Math.floor(simTime * 2.4) % 2 === 0;
    for (let i = 0; i < state.vehicles.length; i++) {
      const v = state.vehicles[i];
      const slot = this.poolSlots[i];
      if (!v.active) {
        slot.container.visible = false;
        continue;
      }

      slot.container.visible = true;
      slot.container.position.set(v.x, 0, -v.relS);
      slot.container.rotation.y = -v.steeringAngle * 0.9;

      if (slot.activeType !== v.type) {
        for (const k of Object.keys(slot.variants)) {
          slot.variants[k].group.visible = k === v.type;
        }
        slot.activeType = v.type;
      }

      const vis = slot.variants[v.type];
      vis.bodyMat.color.setHex(v.colorHex);
      vis.tailMat.color.setHex(v.brakeLight ? 0xff2b38 : 0x7c1d24);
      vis.turnLeftMat.color.setHex(v.turnSignal === -1 && blinkOn ? 0xffa600 : 0x3b2a14);
      vis.turnRightMat.color.setHex(v.turnSignal === 1 && blinkOn ? 0xffa600 : 0x3b2a14);

      for (const w of vis.wheels) {
        w.rotation.x = -v.wheelAngle;
      }
    }

    // 4. LiDAR Sweep Overlay
    const showRays = settings.showLidar && settings.v1Rays && state.lidar.calibrationProgress > 0.05;
    this.rayLines.visible = showRays;
    if (showRays) {
      const n = state.lidar.raySampleCount;
      const ends = state.lidar.raySampleEnds;
      const hits = state.lidar.raySampleHits;
      for (let i = 0; i < n; i++) {
        const i6 = i * 6;
        const e3 = i * 3;
        this.rayPosArr[i6] = ego.x;
        this.rayPosArr[i6 + 1] = LIDAR_MOUNT_HEIGHT;
        this.rayPosArr[i6 + 2] = 0;
        this.rayPosArr[i6 + 3] = ends[e3];
        this.rayPosArr[i6 + 4] = ends[e3 + 1];
        this.rayPosArr[i6 + 5] = ends[e3 + 2];

        const isVeh = hits[i] === 2;
        const r = isVeh ? 0.22 : 0.14;
        const g = isVeh ? 0.85 : 0.45;
        const b = isVeh ? 0.98 : 0.65;
        this.rayColArr[i6] = r;
        this.rayColArr[i6 + 1] = g;
        this.rayColArr[i6 + 2] = b;
        this.rayColArr[i6 + 3] = r;
        this.rayColArr[i6 + 4] = g;
        this.rayColArr[i6 + 5] = b;
      }
      this.rayLines.geometry.setDrawRange(0, n * 2);
      this.rayLines.geometry.attributes.position.needsUpdate = true;
      this.rayLines.geometry.attributes.color.needsUpdate = true;
    }

    // 5. 3D Bounding Boxes Synchronized to Tracks
    const showBoxes = settings.showDetection && settings.v1Boxes;
    for (let i = 0; i < this.boxPool.length; i++) {
      const box = this.boxPool[i];
      if (!showBoxes || i >= state.tracks.length) {
        box.visible = false;
        continue;
      }
      const trk = state.tracks[i];
      if (trk.status === "OCCLUDED" && !settings.debugMode) {
        box.visible = false;
        continue;
      }
      box.visible = true;
      box.position.set(trk.bbox.cx, trk.bbox.cy, trk.bbox.cz);
      box.scale.set(trk.bbox.w + 0.16, trk.bbox.h + 0.14, trk.bbox.l + 0.18);
      const isSelected = settings.selectedTrackId === trk.id;
      const colorHex = isSelected
        ? 0x38bdf8
        : trk.ttc < 3.0
          ? 0xef4444
          : trk.status === "PARTIAL"
            ? 0xf59e0b
            : 0x34d399;
      box.material.color.setHex(colorHex);
    }

    // 6. Planned Path Ribbon
    const showPath = settings.v1Path && ego.plannedPath.length > 1;
    this.pathLine.visible = showPath;
    this.corridorLeft.visible = showPath;
    this.corridorRight.visible = showPath;
    if (showPath) {
      const pAttr = this.pathLine.geometry.attributes.position;
      const lAttr = this.corridorLeft.geometry.attributes.position;
      const rAttr = this.corridorRight.geometry.attributes.position;
      const halfW = ego.width * 0.54;
      for (let i = 0; i < ego.plannedPath.length; i++) {
        const pt = ego.plannedPath[i];
        pAttr.setXYZ(i, pt.x, 0.08, pt.z);
        lAttr.setXYZ(i, pt.x - halfW, 0.06, pt.z);
        rAttr.setXYZ(i, pt.x + halfW, 0.06, pt.z);
      }
      pAttr.needsUpdate = true;
      lAttr.needsUpdate = true;
      rAttr.needsUpdate = true;
      this.pathLine.geometry.setDrawRange(0, ego.plannedPath.length);
      this.corridorLeft.geometry.setDrawRange(0, ego.plannedPath.length);
      this.corridorRight.geometry.setDrawRange(0, ego.plannedPath.length);

      const pathCol =
        ego.controlState === "AEB_BRAKING"
          ? 0xef4444
          : ego.controlState === "CUT_IN_YIELD" || ego.controlState === "ACC_FOLLOW"
            ? 0xf59e0b
            : 0x38bdf8;
      this.pathLine.material.color.setHex(pathCol);
      this.corridorLeft.material.color.setHex(pathCol);
    }

    // 7. Debug Overlay
    this.debugGroup.visible = settings.debugMode;
    this.debugGroup.position.set(ego.x, 0, 0);

    // 8. Camera Controller (Remains interactive even when simulation is paused)
    this._updateCamera(state, settings);
    this.renderer.render(this.scene, this.camera);
  }

  _updateCamera(state, settings) {
    const ego = state.ego;
    if (this.camMode === "orbit") {
      this.controls.update();
      return;
    }

    if (this.camMode === "follow") {
      this._camTargetPos.set(ego.x + 2.4, 5.4, 13.8);
      this._camLookAt.set(ego.x, 1.1, -24);
    } else if (this.camMode === "bev") {
      this._camTargetPos.set(0, 62, -16);
      this._camLookAt.set(0, 0, -24);
    } else if (this.camMode === "sensor") {
      this._camTargetPos.set(ego.x, LIDAR_MOUNT_HEIGHT + 0.35, -0.35);
      this._camLookAt.set(ego.x, 1.1, -48);
    } else if (this.camMode === "track") {
      const target =
        state.tracks.find((t) => t.id === settings.selectedTrackId) ||
        state.tracks.find((t) => t.status !== "OCCLUDED") ||
        state.tracks[0];
      if (target) {
        this._camTargetPos.set(target.bbox.cx + 8.5, 6.2, target.bbox.cz + 11.5);
        this._camLookAt.set(target.bbox.cx, target.bbox.cy, target.bbox.cz);
      } else {
        this._camTargetPos.set(ego.x + 3.0, 5.8, 14.0);
        this._camLookAt.set(ego.x, 1.1, -20);
      }
    }

    const lerpRate = this.camMode === "sensor" ? 0.45 : 0.1;
    this.camera.position.lerp(this._camTargetPos, lerpRate);
    this.camera.lookAt(this._camLookAt);
  }
}

/* ============================================================================
   VIEW 02 — 3D Occluded LiDAR Point Cloud Reconstruction Renderer
   ============================================================================ */
class LiDAR3DRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setClearColor(0x05070b, 1);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(48, 16 / 9, 0.2, 360);
    this.camera.position.set(16, 22, 26);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0.8, -18);
    this.controls.minDistance = 5;
    this.controls.maxDistance = 150;
    this._maxDpr = 1.5;

    this._buildStage();
    this._buildPointCloud();
    this._buildFrustumAndBoxes();
  }

  _buildStage() {
    // Concentric Range Rings (20m - 120m)
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x1e293b,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.7,
    });
    for (const r of [20, 40, 60, 80, 100, 120]) {
      const ring = new THREE.Mesh(new THREE.RingGeometry(r - 0.1, r + 0.1, 72), ringMat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.01;
      this.scene.add(ring);
    }

    // Subtle polar axes
    const axisMat = new THREE.LineBasicMaterial({ color: 0x1a2333 });
    const axisPts = [
      new THREE.Vector3(-120, 0.02, 0), new THREE.Vector3(120, 0.02, 0),
      new THREE.Vector3(0, 0.02, -120), new THREE.Vector3(0, 0.02, 120),
    ];
    this.scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(axisPts), axisMat));

    // Ego Wireframe Silhouette at Origin
    const egoWire = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1.94, 1.48, 4.82)),
      new THREE.LineBasicMaterial({ color: 0xe2e8f0, transparent: true, opacity: 0.7 })
    );
    egoWire.position.set(0, 0.74, 0);
    this.scene.add(egoWire);

    const sensorOrigin = new THREE.Mesh(
      new THREE.SphereGeometry(0.22, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0x38bdf8 })
    );
    sensorOrigin.position.set(0, LIDAR_MOUNT_HEIGHT, 0);
    this.scene.add(sensorOrigin);

    this.debugAxes = new THREE.AxesHelper(6);
    this.debugAxes.visible = false;
    this.scene.add(this.debugAxes);
  }

  _buildPointCloud() {
    const maxPts = 24 * 280;
    this.posArr = new Float32Array(maxPts * 3);
    this.colArr = new Float32Array(maxPts * 3);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.posArr, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(this.colArr, 3));

    this.pointsMat = new THREE.PointsMaterial({
      size: 2.2,
      sizeAttenuation: false,
      vertexColors: true,
      transparent: true,
      opacity: 0.92,
    });

    this.pointCloud = new THREE.Points(geo, this.pointsMat);
    this.scene.add(this.pointCloud);
  }

  _buildFrustumAndBoxes() {
    // Sensor FOV Frustum (forward 90 deg reference + horizon ring)
    const frustumPts = [];
    const r = 65;
    const angles = [-Math.PI * 0.3, Math.PI * 0.3, Math.PI * 0.7, -Math.PI * 0.7];
    for (const a of angles) {
      frustumPts.push(
        new THREE.Vector3(0, LIDAR_MOUNT_HEIGHT, 0),
        new THREE.Vector3(Math.sin(a) * r, 0.2, -Math.cos(a) * r)
      );
    }
    this.frustum = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(frustumPts),
      new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.2 })
    );
    this.scene.add(this.frustum);

    // Synchronized Track Wireframes inside Point Cloud
    this.boxPool = [];
    const unitEdges = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    for (let i = 0; i < 18; i++) {
      const box = new THREE.LineSegments(
        unitEdges,
        new THREE.LineBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.78 })
      );
      box.visible = false;
      this.scene.add(box);
      this.boxPool.push(box);
    }
  }

  resetCamera() {
    this.camera.position.set(16, 22, 26);
    this.controls.target.set(0, 0.8, -18);
    this.controls.update();
  }

  setQuality(presetKey) {
    const q = QUALITY_PRESETS[presetKey] || QUALITY_PRESETS.balanced;
    this._maxDpr = q.maxDpr;
    this.resize();
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const w = Math.max(10, Math.floor(rect.width));
    const h = Math.max(10, Math.floor(rect.height));
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, this._maxDpr));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render(state, settings) {
    const ls = state.lidar;
    this.pointsMat.size = settings.lidarPointSize;
    this.frustum.visible = settings.lidarFrustum && settings.showLidar;
    this.debugAxes.visible = settings.debugMode;

    const selectedVehId = settings.selectedTrackId
      ? parseInt(settings.selectedTrackId.replace("TRK-", ""), 10)
      : -1;

    if (!settings.showLidar || ls.activePoints === 0) {
      this.pointCloud.geometry.setDrawRange(0, 0);
    } else {
      const n = ls.activePoints;
      const srcPos = ls.pointsPos;
      const srcDist = ls.pointsDist;
      const srcInt = ls.pointsIntensity;
      const srcObj = ls.pointsObjId;
      const mode = settings.lidarColorMode;

      for (let i = 0; i < n; i++) {
        const i3 = i * 3;
        this.posArr[i3] = srcPos[i3];
        this.posArr[i3 + 1] = srcPos[i3 + 1];
        this.posArr[i3 + 2] = srcPos[i3 + 2];

        const objId = srcObj[i];
        let r = 0.25, g = 0.55, b = 0.78;

        if (objId === selectedVehId && selectedVehId > 0) {
          r = 0.98; g = 0.82; b = 0.25;
        } else if (mode === "distance") {
          const t = Math.min(1, srcDist[i] / LIDAR_MAX_RANGE);
          if (objId >= 0) {
            // Vehicle returns pop cleanly with near-cyan to cool-blue ramp
            r = 0.2 + (1 - t) * 0.25;
            g = 0.92 - t * 0.35;
            b = 0.98 - t * 0.15;
          } else if (objId === -2) {
            // Guardrails
            r = 0.36; g = 0.44; b = 0.54;
          } else {
            // Road surface
            r = 0.13 + (1 - t) * 0.1;
            g = 0.24 + (1 - t) * 0.16;
            b = 0.34 + (1 - t) * 0.18;
          }
        } else if (mode === "intensity") {
          const v = Math.min(1, srcInt[i]);
          r = 0.18 + v * 0.78;
          g = 0.24 + v * 0.64;
          b = 0.32 + (1 - v) * 0.35;
        } else {
          // Semantic object classification mode
          if (objId >= 0) {
            r = 0.22; g = 0.82; b = 0.98;
          } else if (objId === -2) {
            r = 0.52; g = 0.56; b = 0.64;
          } else {
            r = 0.16; g = 0.20; b = 0.26;
          }
        }

        this.colArr[i3] = r;
        this.colArr[i3 + 1] = g;
        this.colArr[i3 + 2] = b;
      }

      this.pointCloud.geometry.setDrawRange(0, n);
      this.pointCloud.geometry.attributes.position.needsUpdate = true;
      this.pointCloud.geometry.attributes.color.needsUpdate = true;
    }

    // Track bounding boxes inside Point Cloud
    const showBoxes = settings.showDetection;
    for (let i = 0; i < this.boxPool.length; i++) {
      const box = this.boxPool[i];
      if (!showBoxes || i >= state.tracks.length) {
        box.visible = false;
        continue;
      }
      const trk = state.tracks[i];
      if (trk.status === "OCCLUDED") {
        box.visible = false;
        continue;
      }
      box.visible = true;
      box.position.set(trk.bbox.cx, trk.bbox.cy, trk.bbox.cz);
      box.scale.set(trk.bbox.w + 0.2, trk.bbox.h + 0.15, trk.bbox.l + 0.2);
      box.material.color.setHex(settings.selectedTrackId === trk.id ? 0xfbbf24 : 0x34d399);
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

/* ============================================================================
   VIEW 03 — Analytical Object Detection, Occlusion & Tracking (2D HiDPI)
   ============================================================================ */
class Tracking2DRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.w = 300;
    this.h = 200;
    this.dpr = 1;
    this._hitRects = [];
  }

  resize(maxDpr = 2) {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.w = Math.max(10, Math.floor(rect.width));
    this.h = Math.max(10, Math.floor(rect.height));
    this.dpr = Math.min(window.devicePixelRatio || 1, maxDpr);
    this.canvas.width = Math.floor(this.w * this.dpr);
    this.canvas.height = Math.floor(this.h * this.dpr);
  }

  pickTrackAt(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = clientX - rect.left;
    const my = clientY - rect.top;
    for (let i = this._hitRects.length - 1; i >= 0; i--) {
      const h = this._hitRects[i];
      if (mx >= h.x && mx <= h.x + h.w && my >= h.y && my <= h.y + h.h) {
        return h.id;
      }
    }
    return null;
  }

  render(state, settings) {
    const ctx = this.ctx;
    const w = this.w;
    const h = this.h;
    ctx.save();
    ctx.scale(this.dpr, this.dpr);

    // Background
    ctx.fillStyle = "#06080d";
    ctx.fillRect(0, 0, w, h);

    // Coordinate mapping: Longitudinal [-36m, +104m], centered horizontally on highway
    const totalRangeM = 140;
    const pxPerM = Math.min(h / totalRangeM, (w - 40) / 34);
    const originX = w * 0.5;
    const egoY = h * 0.74;
    const toX = (worldX) => originX + worldX * pxPerM;
    const toY = (relS) => egoY - relS * pxPerM;

    // 1. Highway Surface & Lanes
    const roadLeft = toX(-ROAD_HALF_WIDTH);
    const roadWidthPx = ROAD_HALF_WIDTH * 2 * pxPerM;
    ctx.fillStyle = "#0d1118";
    ctx.fillRect(roadLeft, 0, roadWidthPx, h);

    // Guardrail edges
    ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(roadLeft, 0);
    ctx.lineTo(roadLeft, h);
    ctx.moveTo(roadLeft + roadWidthPx, 0);
    ctx.lineTo(roadLeft + roadWidthPx, h);
    ctx.stroke();

    // World-anchored dashed lane dividers (frozen on Pause!)
    ctx.save();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.22)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4 * pxPerM, 8 * pxPerM]);
    ctx.lineDashOffset = -(state.ego.s % 12) * pxPerM;
    for (const dx of [-LANE_WIDTH * 0.5, LANE_WIDTH * 0.5]) {
      const lx = toX(dx);
      ctx.beginPath();
      ctx.moveTo(lx, 0);
      ctx.lineTo(lx, h);
      ctx.stroke();
    }
    ctx.restore();

    // 2. Concentric Range Rings (20m steps)
    ctx.strokeStyle = "rgba(56, 189, 248, 0.11)";
    ctx.lineWidth = 1;
    ctx.font = '500 9px "JetBrains Mono", monospace';
    ctx.fillStyle = "rgba(148, 163, 184, 0.45)";
    for (const r of [20, 40, 60, 80, 100]) {
      const rPx = r * pxPerM;
      ctx.beginPath();
      ctx.arc(toX(state.ego.x), egoY, rPx, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillText(`${r}m`, toX(ROAD_HALF_WIDTH) + 6, toY(r) + 3);
    }

    // Debug metric grid
    if (settings.debugMode) {
      ctx.strokeStyle = "rgba(56, 189, 248, 0.16)";
      ctx.beginPath();
      ctx.moveTo(toX(state.ego.x), 0);
      ctx.lineTo(toX(state.ego.x), h);
      ctx.moveTo(0, egoY);
      ctx.lineTo(w, egoY);
      ctx.stroke();
    }

    // 3. Geometric LiDAR Occlusion Shadow Cones behind visible vehicles
    if (settings.v3Occlusion) {
      const ox = toX(state.ego.x);
      const oy = egoY;
      for (const trk of state.tracks) {
        if (trk.status === "OCCLUDED") continue;
        const hw = trk.bbox.w * 0.5;
        const hl = trk.bbox.l * 0.5;
        const corners = [
          { x: toX(trk.x - hw), y: toY(trk.relS - hl) },
          { x: toX(trk.x + hw), y: toY(trk.relS - hl) },
          { x: toX(trk.x - hw), y: toY(trk.relS + hl) },
          { x: toX(trk.x + hw), y: toY(trk.relS + hl) },
        ];
        // Find angularly outermost corners relative to sensor origin
        let minA = Infinity, maxA = -Infinity, cMin = corners[0], cMax = corners[0];
        const baseA = Math.atan2(toY(trk.relS) - oy, toX(trk.x) - ox);
        for (const c of corners) {
          let a = Math.atan2(c.y - oy, c.x - ox) - baseA;
          while (a > Math.PI) a -= Math.PI * 2;
          while (a < -Math.PI) a += Math.PI * 2;
          if (a < minA) { minA = a; cMin = c; }
          if (a > maxA) { maxA = a; cMax = c; }
        }
        const far = 130 * pxPerM;
        const a1 = baseA + minA;
        const a2 = baseA + maxA;
        ctx.fillStyle = "rgba(245, 158, 11, 0.075)";
        ctx.beginPath();
        ctx.moveTo(cMin.x, cMin.y);
        ctx.lineTo(ox + Math.cos(a1) * far, oy + Math.sin(a1) * far);
        ctx.lineTo(ox + Math.cos(a2) * far, oy + Math.sin(a2) * far);
        ctx.lineTo(cMax.x, cMax.y);
        ctx.closePath();
        ctx.fill();
      }
    }

    // 4. Ego Vehicle Footprint
    const egoW = state.ego.width * pxPerM;
    const egoL = state.ego.length * pxPerM;
    ctx.fillStyle = "rgba(226, 232, 240, 0.16)";
    ctx.strokeStyle = "#f1f4f8";
    ctx.lineWidth = 1.6;
    ctx.fillRect(toX(state.ego.x) - egoW * 0.5, egoY - egoL * 0.5, egoW, egoL);
    ctx.strokeRect(toX(state.ego.x) - egoW * 0.5, egoY - egoL * 0.5, egoW, egoL);

    ctx.fillStyle = "#38bdf8";
    ctx.beginPath();
    ctx.arc(toX(state.ego.x), egoY, 2.8, 0, Math.PI * 2);
    ctx.fill();

    // 5. Tracked Vehicles, Trajectories, Velocity Vectors & Click Hit-Testing
    this._hitRects = [];
    for (const trk of state.tracks) {
      const cx = toX(trk.x);
      const cy = toY(trk.relS);
      const bw = Math.max(8, trk.bbox.w * pxPerM);
      const bl = Math.max(14, trk.bbox.l * pxPerM);
      const isSelected = settings.selectedTrackId === trk.id;
      const isCritical = trk.ttc < 3.0;
      const col = isSelected
        ? STATUS_COLORS.SELECTED
        : isCritical
          ? STATUS_COLORS.CRITICAL
          : STATUS_COLORS[trk.status] || STATUS_COLORS.CONFIRMED;

      // Trajectory history
      if (settings.showTracking && settings.v3Trajectories && trk.history.length > 1) {
        ctx.strokeStyle = col;
        ctx.globalAlpha = 0.45;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        for (let i = 0; i < trk.history.length; i++) {
          const pt = trk.history[i];
          const hx = toX(pt.x);
          const hy = toY(pt.s - state.ego.s);
          if (i === 0) ctx.moveTo(hx, hy);
          else ctx.lineTo(hx, hy);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      // Bounding Box
      if (settings.showDetection && settings.v3Boxes) {
        ctx.save();
        if (trk.status === "OCCLUDED") {
          ctx.setLineDash([3, 3]);
        }
        ctx.fillStyle = isSelected ? "rgba(56, 189, 248, 0.22)" : "rgba(52, 211, 153, 0.1)";
        if (trk.status === "OCCLUDED") ctx.fillStyle = "rgba(148, 163, 184, 0.08)";
        if (isCritical) ctx.fillStyle = "rgba(239, 68, 68, 0.18)";

        ctx.strokeStyle = col;
        ctx.lineWidth = isSelected ? 2.2 : 1.35;
        ctx.fillRect(cx - bw * 0.5, cy - bl * 0.5, bw, bl);
        ctx.strokeRect(cx - bw * 0.5, cy - bl * 0.5, bw, bl);
        ctx.restore();

        // Relative velocity arrow
        if (Math.abs(trk.relSpeed) > 0.6 && trk.status !== "OCCLUDED") {
          const vy = -trk.relSpeed * 1.4 * pxPerM * 0.35;
          ctx.strokeStyle = col;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(cx, cy + vy);
          ctx.stroke();
        }
      }

      // Store clickable hit box (with generous padding for easy inspection)
      this._hitRects.push({
        id: trk.id,
        x: cx - bw * 0.5 - 8,
        y: cy - bl * 0.5 - 8,
        w: bw + 16,
        h: bl + 16,
      });

      // Compact Telemetry Label
      if (settings.showTracking && settings.v3Labels && cy > 14 && cy < h - 14) {
        const alignRight = trk.x < 0;
        const lx = alignRight ? cx - bw * 0.5 - 6 : cx + bw * 0.5 + 6;
        ctx.font = '500 9.5px "JetBrains Mono", monospace';
        ctx.textAlign = alignRight ? "right" : "left";
        ctx.fillStyle = col;
        const statusSuffix = trk.status === "OCCLUDED" ? " [OCCL]" : "";
        ctx.fillText(`${trk.id}${statusSuffix}`, lx, cy - 2);

        ctx.font = '400 8.5px "JetBrains Mono", monospace';
        ctx.fillStyle = "#9aa3b2";
        const sign = trk.relSpeed >= 0 ? "+" : "";
        const line2 =
          trk.status === "OCCLUDED"
            ? `${trk.distance.toFixed(0)}m · shadow`
            : `${trk.distance.toFixed(1)}m · ${sign}${trk.relSpeed.toFixed(1)}m/s`;
        ctx.fillText(line2, lx, cy + 9);
      }
    }

    // Legend Footer (top-left subtle)
    ctx.textAlign = "left";
    ctx.font = '500 9px "JetBrains Mono", monospace';
    ctx.fillStyle = "rgba(154, 163, 178, 0.75)";
    ctx.fillText("BEV TRACKER · CLICK VEHICLE TO INSPECT", 10, 16);

    ctx.restore();
  }
}

/* ============================================================================
   VIEW 04 — Autonomous Driving Intelligence & Kinematic Plots (2D HiDPI)
   ============================================================================ */
class Intelligence2DRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.w = 300;
    this.h = 200;
    this.dpr = 1;
  }

  resize(maxDpr = 2) {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.w = Math.max(10, Math.floor(rect.width));
    this.h = Math.max(10, Math.floor(rect.height));
    this.dpr = Math.min(window.devicePixelRatio || 1, maxDpr);
    this.canvas.width = Math.floor(this.w * this.dpr);
    this.canvas.height = Math.floor(this.h * this.dpr);
  }

  render(state, settings) {
    const ctx = this.ctx;
    const w = this.w;
    const h = this.h;
    const ego = state.ego;
    const pad = 12;
    const bottomReserve = settings.expandedView === 4 ? 36 : 8;
    const usableH = h - pad * 2 - bottomReserve;

    ctx.save();
    ctx.scale(this.dpr, this.dpr);

    ctx.fillStyle = "#06080d";
    ctx.fillRect(0, 0, w, h);

    // ------------------------------------------------------------------------
    // Zone 1: Rule-Based Control State & Lane Occupancy Matrix (Top 30%)
    // ------------------------------------------------------------------------
    const z1H = Math.max(54, usableH * 0.3);
    const stateCol =
      ego.controlState === "AEB_BRAKING"
        ? "#ef4444"
        : ego.controlState === "CUT_IN_YIELD" || ego.controlState === "ACC_FOLLOW"
          ? "#f59e0b"
          : ego.controlState === "INIT" || ego.controlState === "CALIBRATING"
            ? "#38bdf8"
            : "#34d399";

    ctx.fillStyle = "#0d1118";
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.lineWidth = 1;
    this._roundRect(ctx, pad, pad, w - pad * 2, z1H, 7);
    ctx.fill();
    ctx.stroke();

    // State indicator bar
    ctx.fillStyle = stateCol;
    ctx.fillRect(pad + 1, pad + 8, 3, z1H - 16);

    ctx.font = '600 10px "JetBrains Mono", monospace';
    ctx.fillStyle = stateCol;
    ctx.fillText(`CTRL STATE: ${ego.controlState.replace(/_/g, " ")}`, pad + 12, pad + 18);

    ctx.font = '400 10px "Inter", sans-serif';
    ctx.fillStyle = "#9aa3b2";
    ctx.fillText(ego.controlReason, pad + 12, pad + 33, w - pad * 2 - 140);

    // Lane Occupancy Pills (Right side of Zone 1)
    const laneLabels = ["L", "C", "R"];
    const laneBoxW = 28;
    const startLX = w - pad - 10 - laneBoxW * 3 - 8;
    ctx.font = '500 8.5px "JetBrains Mono", monospace';
    ctx.fillStyle = "#646d7d";
    ctx.fillText("LANE OCC", startLX, pad + 14);
    for (let i = 0; i < 3; i++) {
      const bx = startLX + i * (laneBoxW + 4);
      const by = pad + 20;
      const occ = ego.laneOccupancy[i];
      ctx.fillStyle = occ ? "rgba(245, 158, 11, 0.18)" : "rgba(52, 211, 153, 0.1)";
      ctx.strokeStyle = occ ? "rgba(245, 158, 11, 0.55)" : "rgba(52, 211, 153, 0.35)";
      this._roundRect(ctx, bx, by, laneBoxW, 20, 4);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = occ ? "#f59e0b" : "#34d399";
      ctx.textAlign = "center";
      ctx.fillText(laneLabels[i], bx + laneBoxW * 0.5, by + 13);
    }
    ctx.textAlign = "left";

    // ------------------------------------------------------------------------
    // Zone 2: Live Kinematic Signal Plot (Speed vs Target & Headway) (Middle 46%)
    // ------------------------------------------------------------------------
    const z2Y = pad + z1H + 8;
    const z2H = Math.max(70, usableH * 0.45);
    ctx.fillStyle = "#0a0d13";
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    this._roundRect(ctx, pad, z2Y, w - pad * 2, z2H, 7);
    ctx.fill();
    ctx.stroke();

    const plotX = pad + 34;
    const plotY = z2Y + 20;
    const plotW = w - pad * 2 - 44;
    const plotH = z2H - 30;

    // Header
    ctx.font = '500 9px "JetBrains Mono", monospace';
    ctx.fillStyle = "#9aa3b2";
    ctx.fillText("LONGITUDINAL KINEMATICS (km/h)", pad + 10, z2Y + 13);

    ctx.textAlign = "right";
    ctx.fillStyle = "#38bdf8";
    ctx.fillText(`EGO ${(ego.speed * 3.6).toFixed(1)}`, w - pad - 80, z2Y + 13);
    ctx.fillStyle = "#646d7d";
    ctx.fillText(`TGT ${(ego.targetSpeed * 3.6).toFixed(0)}`, w - pad - 10, z2Y + 13);
    ctx.textAlign = "left";

    // Grid lines (0, 60, 120 km/h)
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.font = '400 8px "JetBrains Mono", monospace';
    ctx.fillStyle = "#5d6676";
    for (const val of [0, 60, 120]) {
      const gy = plotY + plotH - (val / 135) * plotH;
      ctx.beginPath();
      ctx.moveTo(plotX, gy);
      ctx.lineTo(plotX + plotW, gy);
      ctx.stroke();
      ctx.fillText(String(val), pad + 8, gy + 3);
    }

    const hist = ego.speedHistory;
    if (hist.length > 1) {
      // Target Speed Dashed Trace
      ctx.save();
      ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < hist.length; i++) {
        const px = plotX + (i / Math.max(1, hist.length - 1)) * plotW;
        const py = plotY + plotH - (Math.min(135, hist[i].targetKmh) / 135) * plotH;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.restore();

      // Actual Ego Speed Trace
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 1.75;
      ctx.beginPath();
      for (let i = 0; i < hist.length; i++) {
        const px = plotX + (i / Math.max(1, hist.length - 1)) * plotW;
        const py = plotY + plotH - (Math.min(135, hist[i].speedKmh) / 135) * plotH;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // ------------------------------------------------------------------------
    // Zone 3: Quantitative Headway / TTC Safety Envelope & Pipeline (Bottom)
    // ------------------------------------------------------------------------
    const z3Y = z2Y + z2H + 8;
    const z3H = Math.max(38, h - z3Y - bottomReserve - pad);
    if (z3H >= 34) {
      ctx.fillStyle = "#0d1118";
      ctx.strokeStyle = "rgba(255,255,255,0.07)";
      this._roundRect(ctx, pad, z3Y, w - pad * 2, z3H, 7);
      ctx.fill();
      ctx.stroke();

      const gapStr = Number.isFinite(ego.leadGap) ? `${ego.leadGap.toFixed(1)} m` : "Clear (>110 m)";
      const safeStr = `${ego.safeFollowDist.toFixed(1)} m`;
      const ttcStr = Number.isFinite(ego.minTtc) ? `${ego.minTtc.toFixed(2)} s` : "∞";

      ctx.font = '500 9px "JetBrains Mono", monospace';
      ctx.fillStyle = "#9aa3b2";
      ctx.fillText(`FOLLOW GAP: ${gapStr}`, pad + 10, z3Y + 15);
      ctx.fillText(`SAFE ENV: ${safeStr}`, pad + (w - pad * 2) * 0.42, z3Y + 15);
      ctx.fillStyle = ego.minTtc < 3.0 ? "#ef4444" : "#34d399";
      ctx.fillText(`TTC: ${ttcStr}`, pad + (w - pad * 2) * 0.76, z3Y + 15);

      // Headway Safety Progress Bar
      const barX = pad + 10;
      const barY = z3Y + 23;
      const barW = w - pad * 2 - 20;
      const barH = 6;
      ctx.fillStyle = "rgba(255,255,255,0.08)";
      this._roundRect(ctx, barX, barY, barW, barH, 3);
      ctx.fill();

      const ratio = Number.isFinite(ego.leadGap) ? Math.min(1, ego.leadGap / Math.max(15, ego.safeFollowDist * 1.4)) : 1;
      ctx.fillStyle = ratio < 0.38 ? "#ef4444" : ratio < 0.72 ? "#f59e0b" : "#34d399";
      this._roundRect(ctx, barX, barY, Math.max(4, barW * ratio), barH, 3);
      ctx.fill();
    }

    ctx.restore();
  }

  _roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
}

/* ============================================================================
   Unified MultiViewRenderer Coordinator
   ============================================================================ */
export class MultiViewRenderer {
  constructor(canvases) {
    this.view1 = new Env3DRenderer(canvases.v1);
    this.view2 = new LiDAR3DRenderer(canvases.v2);
    this.view3 = new Tracking2DRenderer(canvases.v3);
    this.view4 = new Intelligence2DRenderer(canvases.v4);
    this._maxDpr = 1.5;
  }

  setQuality(presetKey) {
    const q = QUALITY_PRESETS[presetKey] || QUALITY_PRESETS.balanced;
    this._maxDpr = q.maxDpr;
    this.view1.setQuality(presetKey);
    this.view2.setQuality(presetKey);
    this.resizeAll();
  }

  resizeAll() {
    this.view1.resize();
    this.view2.resize();
    this.view3.resize(this._maxDpr);
    this.view4.resize(this._maxDpr);
  }

  renderAll(state, settings) {
    this.view1.render(state, settings);
    this.view2.render(state, settings);
    this.view3.render(state, settings);
    this.view4.render(state, settings);
  }
}