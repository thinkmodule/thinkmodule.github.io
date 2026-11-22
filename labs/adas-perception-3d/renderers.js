/**
 * ThinkModule — High-Resolution 3D Autonomous Driving & LiDAR Cinema Renderer
 * ============================================================================
 * Features:
 *  - Sculpted aerodynamic 3D vehicle geometries with 5-spoke alloy rims, LED
 *    lightbars, brake lights, turn signals & soft ground contact shadows
 *  - Procedural PBR highway asphalt, world-anchored streetlamps & sign gantries
 *    (100% frozen on Pause)
 *  - True 3D occluded LiDAR point cloud + dual-zone occlusion shader (solid
 *    translucent amber on asphalt, crisp dotted fade beyond highway barriers)
 *  - Synchronized left & right vehicle shadow cones in both 3D & BEV Mini-Map
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  LANE_WIDTH,
  ROAD_HALF_WIDTH,
  LAMP_SPACING,
  GANTRY_SPACING,
  LIDAR_MOUNT_HEIGHT,
  LIDAR_MAX_RANGE,
  QUALITY_PRESETS,
} from "./sim-engine.js";

function createAsphaltTexture() {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 512;
  const ctx = c.getContext("2d");

  ctx.fillStyle = "#191e27";
  ctx.fillRect(0, 0, 512, 512);

  const laneU = [0.22, 0.5, 0.78];
  for (const u of laneU) {
    for (const track of [-0.045, 0.045]) {
      const gx = (u + track) * 512;
      const grad = ctx.createLinearGradient(gx - 24, 0, gx + 24, 0);
      grad.addColorStop(0, "rgba(8, 10, 15, 0)");
      grad.addColorStop(0.5, "rgba(8, 10, 15, 0.34)");
      grad.addColorStop(1, "rgba(8, 10, 15, 0)");
      ctx.fillStyle = grad;
      ctx.fillRect(gx - 24, 0, 48, 512);
    }
  }

  const img = ctx.getImageData(0, 0, 512, 512);
  const d = img.data;
  let seed = 918273;
  for (let i = 0; i < d.length; i += 4) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const n = ((seed >>> 24) / 255 - 0.5) * 20;
    d[i] = Math.max(0, Math.min(255, d[i] + n));
    d[i + 1] = Math.max(0, Math.min(255, d[i + 1] + n));
    d[i + 2] = Math.max(0, Math.min(255, d[i + 2] + n));
  }
  ctx.putImageData(img, 0, 0);

  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(1, 40);
  tex.anisotropy = 8;
  return tex;
}

function createSoftGlowSprite(innerColor = "rgba(255,255,255,1)", outerColor = "rgba(255,255,255,0)") {
  const c = document.createElement("canvas");
  c.width = 64;
  c.height = 64;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(32, 32, 2, 32, 32, 31);
  g.addColorStop(0, innerColor);
  g.addColorStop(0.42, "rgba(255,255,255,0.65)");
  g.addColorStop(1, outerColor);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}

function createContactShadowTexture() {
  const c = document.createElement("canvas");
  c.width = 128;
  c.height = 256;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(64, 128, 12, 64, 128, 62);
  g.addColorStop(0, "rgba(0,0,0,0.88)");
  g.addColorStop(0.65, "rgba(0,0,0,0.48)");
  g.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = g;
  ctx.scale(1, 2);
  ctx.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

function createHighwaySignTexture() {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 128;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#0e3b68";
  ctx.fillRect(0, 0, 512, 128);
  ctx.strokeStyle = "#dbe7f5";
  ctx.lineWidth = 6;
  ctx.strokeRect(8, 8, 496, 112);

  ctx.fillStyle = "#f4f8ff";
  ctx.font = '600 32px "Inter", sans-serif';
  ctx.fillText("A81  STUTTGART  ·  MÜNCHEN", 34, 60);
  ctx.font = '500 22px "JetBrains Mono", monospace';
  ctx.fillStyle = "#93c5fd";
  ctx.fillText("AUTONOMOUS CORRIDOR  ·  L3/L4", 34, 96);

  const tex = new THREE.CanvasTexture(c);
  tex.anisotropy = 4;
  return tex;
}

const sharedShadowTex = createContactShadowTexture();
const sharedLightGlowTex = createSoftGlowSprite();

function createSculptedProfileGeo(points, width, bevel = 0.09) {
  const shape = new THREE.Shape();
  shape.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    shape.lineTo(points[i][0], points[i][1]);
  }
  shape.closePath();

  const innerW = Math.max(0.2, width - bevel * 2);
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: innerW,
    bevelEnabled: true,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 3,
    curveSegments: 8,
  });

  geo.translate(0, 0, -innerW * 0.5);
  geo.rotateY(-Math.PI / 2);
  geo.computeVertexNormals();
  return geo;
}

function createWheelAssembly(radius) {
  const pivot = new THREE.Group();

  const tireMat = new THREE.MeshStandardMaterial({ color: 0x101319, roughness: 0.88, metalness: 0.08 });
  const rimMat = new THREE.MeshStandardMaterial({ color: 0xc2ccd8, roughness: 0.22, metalness: 0.85 });
  const hubMat = new THREE.MeshStandardMaterial({ color: 0x1c232e, roughness: 0.4, metalness: 0.6 });

  const tireW = radius > 0.44 ? 0.3 : 0.25;
  const tireGeo = new THREE.CylinderGeometry(radius, radius, tireW, 24);
  tireGeo.rotateZ(Math.PI / 2);
  pivot.add(new THREE.Mesh(tireGeo, tireMat));

  const barrelGeo = new THREE.CylinderGeometry(radius * 0.66, radius * 0.66, tireW + 0.014, 20);
  barrelGeo.rotateZ(Math.PI / 2);
  pivot.add(new THREE.Mesh(barrelGeo, hubMat));

  const spokeGeo = new THREE.BoxGeometry(tireW + 0.026, radius * 1.18, 0.048);
  for (let i = 0; i < 5; i++) {
    const spoke = new THREE.Mesh(spokeGeo, rimMat);
    spoke.rotation.x = (i / 5) * Math.PI;
    pivot.add(spoke);
  }

  const lipGeo = new THREE.TorusGeometry(radius * 0.64, 0.024, 8, 24);
  lipGeo.rotateY(Math.PI / 2);
  const lipL = new THREE.Mesh(lipGeo, rimMat);
  lipL.position.x = -tireW * 0.48;
  const lipR = lipL.clone();
  lipR.position.x = tireW * 0.48;
  pivot.add(lipL, lipR);

  return pivot;
}

function createSculptedVehicle(type, colorHex, isEgo = false) {
  const group = new THREE.Group();

  const bodyMat = new THREE.MeshStandardMaterial({
    color: isEgo ? 0xe6edf6 : colorHex,
    roughness: isEgo ? 0.18 : 0.26,
    metalness: isEgo ? 0.58 : 0.48,
  });
  const glassMat = new THREE.MeshStandardMaterial({ color: 0x0b1220, roughness: 0.08, metalness: 0.9 });
  const darkTrimMat = new THREE.MeshStandardMaterial({ color: 0x12161e, roughness: 0.72, metalness: 0.2 });

  const headMat = new THREE.MeshBasicMaterial({ color: 0xf0f7ff });
  const tailMat = new THREE.MeshBasicMaterial({ color: 0x8f1822 });
  const turnLeftMat = new THREE.MeshBasicMaterial({ color: 0x3a2810 });
  const turnRightMat = new THREE.MeshBasicMaterial({ color: 0x3a2810 });

  let length = 4.85;
  let width = 1.92;
  let wheelR = 0.35;
  let wheelZs = [-1.45, 1.45];
  let lightY = 0.68;

  if (type === "truck") {
    length = 11.4;
    width = 2.54;
    wheelR = 0.5;
    wheelZs = [-4.25, -1.65, 3.3, 4.55];
    lightY = 0.95;

    const trailerL = 8.4;
    const trailerMat = new THREE.MeshStandardMaterial({ color: colorHex, roughness: 0.42, metalness: 0.28 });
    const trailer = new THREE.Mesh(new THREE.BoxGeometry(width, 2.55, trailerL), trailerMat);
    trailer.position.set(0, 1.02 + 1.275, 1.48);
    group.add(trailer);

    const ribGeo = new THREE.BoxGeometry(width + 0.04, 2.48, 0.06);
    const ribMat = new THREE.MeshStandardMaterial({ color: 0x8e99a8, roughness: 0.4, metalness: 0.5 });
    for (let z = -2.2; z <= 5.2; z += 1.45) {
      const rib = new THREE.Mesh(ribGeo, ribMat);
      rib.position.set(0, 2.28, z);
      group.add(rib);
    }

    const skirt = new THREE.Mesh(new THREE.BoxGeometry(width * 0.92, 0.54, trailerL * 0.92), darkTrimMat);
    skirt.position.set(0, 0.72, 1.48);
    group.add(skirt);

    const cabProfile = [
      [-5.65, 0.52], [-5.65, 1.65], [-5.35, 2.68],
      [-4.6, 3.45], [-2.95, 3.45], [-2.95, 0.52],
    ];
    const cabMat = new THREE.MeshStandardMaterial({ color: 0x2c3645, roughness: 0.3, metalness: 0.55 });
    group.add(new THREE.Mesh(createSculptedProfileGeo(cabProfile, width * 0.94, 0.1), cabMat));

    const cabGlass = new THREE.Mesh(new THREE.BoxGeometry(width * 0.84, 0.86, 0.12), glassMat);
    cabGlass.position.set(0, 2.15, -5.45);
    cabGlass.rotation.x = 0.24;
    group.add(cabGlass);

    const grille = new THREE.Mesh(
      new THREE.BoxGeometry(1.5, 0.78, 0.06),
      new THREE.MeshStandardMaterial({ color: 0x9aa6b8, roughness: 0.25, metalness: 0.85 })
    );
    grille.position.set(0, 1.15, -5.66);
    group.add(grille);
  } else {
    if (type === "coupe") {
      length = 4.65;
      width = 1.9;
      wheelR = 0.34;
      wheelZs = [-1.42, 1.4];
      lightY = 0.64;

      const lowerPts = [
        [-2.3, 0.28], [-2.22, 0.64], [-1.35, 0.82], [1.45, 0.86],
        [2.26, 0.82], [2.32, 0.52], [2.18, 0.28],
      ];
      const cabinPts = [[-1.12, 0.78], [-0.32, 1.35], [0.85, 1.33], [1.78, 0.82]];
      group.add(new THREE.Mesh(createSculptedProfileGeo(lowerPts, width, 0.1), bodyMat));
      group.add(new THREE.Mesh(createSculptedProfileGeo(cabinPts, width * 0.8, 0.08), glassMat));
      const roof = new THREE.Mesh(new THREE.BoxGeometry(width * 0.74, 0.06, 1.15), bodyMat);
      roof.position.set(0, 1.36, 0.24);
      group.add(roof);
    } else if (type === "suv") {
      length = 5.02;
      width = 2.04;
      wheelR = 0.38;
      wheelZs = [-1.52, 1.52];
      lightY = 0.82;

      const lowerPts = [
        [-2.48, 0.34], [-2.42, 0.86], [-1.38, 1.04], [2.38, 1.06],
        [2.48, 0.68], [2.36, 0.34],
      ];
      const cabinPts = [[-1.24, 1.0], [-0.56, 1.72], [1.62, 1.71], [2.18, 1.04]];
      group.add(new THREE.Mesh(createSculptedProfileGeo(lowerPts, width, 0.11), bodyMat));
      group.add(new THREE.Mesh(createSculptedProfileGeo(cabinPts, width * 0.84, 0.08), glassMat));
      const roof = new THREE.Mesh(new THREE.BoxGeometry(width * 0.78, 0.07, 2.1), bodyMat);
      roof.position.set(0, 1.73, 0.52);
      group.add(roof);
    } else {
      length = isEgo ? 4.86 : 4.85;
      width = isEgo ? 1.94 : 1.92;
      wheelR = isEgo ? 0.36 : 0.35;
      wheelZs = [-1.48, 1.48];
      lightY = 0.68;

      const lowerPts = [
        [-2.42, 0.28], [-2.34, 0.68], [-1.35, 0.87], [1.62, 0.9],
        [2.38, 0.84], [2.42, 0.48], [2.26, 0.28],
      ];
      const cabinPts = [[-1.16, 0.82], [-0.36, 1.42], [0.96, 1.4], [1.92, 0.86]];
      group.add(new THREE.Mesh(createSculptedProfileGeo(lowerPts, width, 0.1), bodyMat));
      group.add(new THREE.Mesh(createSculptedProfileGeo(cabinPts, width * 0.82, 0.08), glassMat));
      const roof = new THREE.Mesh(new THREE.BoxGeometry(width * 0.76, 0.06, 1.32), bodyMat);
      roof.position.set(0, 1.43, 0.3);
      group.add(roof);
    }

    const sill = new THREE.Mesh(new THREE.BoxGeometry(width * 0.96, 0.13, length * 0.92), darkTrimMat);
    sill.position.set(0, 0.28, 0);
    group.add(sill);

    const mirGeo = new THREE.BoxGeometry(0.18, 0.09, 0.14);
    const mirL = new THREE.Mesh(mirGeo, bodyMat);
    mirL.position.set(-width * 0.52, lightY + 0.34, -0.58);
    const mirR = mirL.clone();
    mirR.position.x = width * 0.52;
    group.add(mirL, mirR);
  }

  const frontZ = -length * 0.5 - 0.01;
  const rearZ = length * 0.5 + 0.01;

  if (type === "truck") {
    const hlGeo = new THREE.BoxGeometry(0.42, 0.14, 0.06);
    const hlL = new THREE.Mesh(hlGeo, headMat);
    hlL.position.set(-width * 0.35, lightY, frontZ);
    const hlR = hlL.clone();
    hlR.position.x = width * 0.35;
    group.add(hlL, hlR);

    const tlGeo = new THREE.BoxGeometry(0.46, 0.14, 0.06);
    const tlL = new THREE.Mesh(tlGeo, tailMat);
    tlL.position.set(-width * 0.36, 0.9, rearZ);
    const tlR = tlL.clone();
    tlR.position.x = width * 0.36;
    group.add(tlL, tlR);
  } else {
    const frontBar = new THREE.Mesh(new THREE.BoxGeometry(width * 0.84, 0.075, 0.05), headMat);
    frontBar.position.set(0, lightY, frontZ);
    group.add(frontBar);

    const rearBar = new THREE.Mesh(new THREE.BoxGeometry(width * 0.86, 0.085, 0.06), tailMat);
    rearBar.position.set(0, lightY + 0.06, rearZ);
    group.add(rearBar);
  }

  const tsGeo = new THREE.BoxGeometry(0.18, 0.09, 0.07);
  const tsL = new THREE.Mesh(tsGeo, turnLeftMat);
  tsL.position.set(-width * 0.43, lightY + 0.06, rearZ + 0.01);
  const tsR = new THREE.Mesh(tsGeo, turnRightMat);
  tsR.position.set(width * 0.43, lightY + 0.06, rearZ + 0.01);
  group.add(tsL, tsR);

  const shadowPlane = new THREE.Mesh(
    new THREE.PlaneGeometry(width * 1.42, length * 1.18),
    new THREE.MeshBasicMaterial({ map: sharedShadowTex, transparent: true, opacity: 0.78, depthWrite: false })
  );
  shadowPlane.rotation.x = -Math.PI / 2;
  shadowPlane.position.y = 0.018;
  group.add(shadowPlane);

  const wheels = [];
  const archMat = new THREE.MeshBasicMaterial({ color: 0x06080c });
  const archGeo = new THREE.CylinderGeometry(wheelR * 1.12, wheelR * 1.12, 0.3, 18);
  archGeo.rotateZ(Math.PI / 2);
  const wheelOffsetX = width * 0.5 - 0.12;

  for (const wz of wheelZs) {
    for (const side of [-1, 1]) {
      const arch = new THREE.Mesh(archGeo, archMat);
      arch.position.set(side * (wheelOffsetX - 0.06), wheelR + 0.03, wz);
      group.add(arch);

      const wPivot = createWheelAssembly(wheelR);
      wPivot.position.set(side * wheelOffsetX, wheelR, wz);
      group.add(wPivot);
      wheels.push(wPivot);
    }
  }

  let lidarSpinner = null;
  if (isEgo) {
    const pod = new THREE.Mesh(
      new THREE.BoxGeometry(0.92, 0.14, 0.62),
      new THREE.MeshStandardMaterial({ color: 0x151b26, roughness: 0.25, metalness: 0.75 })
    );
    pod.position.set(0, 1.5, 0.1);
    group.add(pod);

    for (const side of [-1, 1]) {
      const camPod = new THREE.Mesh(
        new THREE.SphereGeometry(0.065, 12, 12),
        new THREE.MeshStandardMaterial({ color: 0x0c111b, roughness: 0.1, metalness: 0.9 })
      );
      camPod.position.set(side * 0.44, 1.52, -0.14);
      group.add(camPod);
    }

    lidarSpinner = new THREE.Group();
    lidarSpinner.position.set(0, LIDAR_MOUNT_HEIGHT - 0.05, 0.08);
    const housing = new THREE.Mesh(
      new THREE.CylinderGeometry(0.145, 0.165, 0.18, 28),
      new THREE.MeshStandardMaterial({ color: 0x18202c, roughness: 0.22, metalness: 0.85 })
    );
    const opticRing = new THREE.Mesh(
      new THREE.CylinderGeometry(0.15, 0.15, 0.065, 28),
      new THREE.MeshBasicMaterial({ color: 0x38bdf8 })
    );
    const laserEmitter = new THREE.Mesh(
      new THREE.BoxGeometry(0.05, 0.075, 0.16),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    laserEmitter.position.z = -0.075;
    lidarSpinner.add(housing, opticRing, laserEmitter);
    group.add(lidarSpinner);

    const beamGeo = new THREE.PlaneGeometry(3.8, 26);
    beamGeo.rotateX(-Math.PI / 2);
    const beamMat = new THREE.MeshBasicMaterial({
      map: sharedLightGlowTex,
      color: 0xdbeeff,
      transparent: true,
      opacity: 0.13,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const beam = new THREE.Mesh(beamGeo, beamMat);
    beam.position.set(0, 0.03, -15.5);
    group.add(beam);
  }

  return { group, wheels, lidarSpinner, bodyMat, tailMat, turnLeftMat, turnRightMat, type };
}

/* ============================================================================
   MAIN CINEMA 3D RENDERER
   ============================================================================ */
export class Cinema3DRenderer {
  constructor(canvas, labelsLayer, onUserOrbitStart) {
    this.canvas = canvas;
    this.labelsLayer = labelsLayer;
    this.onUserOrbitStart = onUserOrbitStart;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setClearColor(0x060a12, 1);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.24;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x080d18, 0.0036);

    this.camera = new THREE.PerspectiveCamera(45, 16 / 9, 0.2, 460);
    this.camera.position.set(6.5, 5.2, 14.5);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.maxPolarAngle = Math.PI * 0.482;
    this.controls.minPolarAngle = 0.08;
    this.controls.minDistance = 4.2;
    this.controls.maxDistance = 125;
    this.controls.target.set(0, 1.0, -14);

    this.controls.addEventListener("start", () => {
      if (this.camMode !== "orbit") {
        this.camMode = "orbit";
        if (this.onUserOrbitStart) this.onUserOrbitStart("orbit");
      }
    });

    this.camMode = "cinematic";
    this.visualMode = "cinema_ar";
    this._camTargetPos = new THREE.Vector3();
    this._camTargetLook = new THREE.Vector3();
    this._maxDpr = 1.75;
    this._projVec = new THREE.Vector3();

    this.envGroup = new THREE.Group();
    this.lidarGroup = new THREE.Group();
    this.arGroup = new THREE.Group();
    this.fusionGroup = new THREE.Group();
    this.occlusionGroup = new THREE.Group();
    this.scene.add(this.envGroup, this.lidarGroup, this.arGroup, this.fusionGroup, this.occlusionGroup);

    this._buildSkyAndLighting();
    this._buildHighwayEnvironment();
    this._buildStreetlampsAndGantries();
    this._buildVehicles();
    this._buildLiDARPointCloud();
    this._buildARAndFusionOverlays();
    this._buildLabelPool();
  }

  _buildSkyAndLighting() {
    const skyCanvas = document.createElement("canvas");
    skyCanvas.width = 4;
    skyCanvas.height = 256;
    const sctx = skyCanvas.getContext("2d");
    const grad = sctx.createLinearGradient(0, 0, 0, 256);
    grad.addColorStop(0.0, "#050914");
    grad.addColorStop(0.45, "#0b1426");
    grad.addColorStop(0.55, "#152540");
    grad.addColorStop(0.62, "#1c314f");
    grad.addColorStop(0.7, "#090e18");
    grad.addColorStop(1.0, "#05070c");
    sctx.fillStyle = grad;
    sctx.fillRect(0, 0, 4, 256);
    const skyTex = new THREE.CanvasTexture(skyCanvas);

    const skyDome = new THREE.Mesh(
      new THREE.SphereGeometry(380, 24, 16),
      new THREE.MeshBasicMaterial({ map: skyTex, side: THREE.BackSide, fog: false })
    );
    this.envGroup.add(skyDome);

    this.scene.add(new THREE.HemisphereLight(0xbcd4f2, 0x161c28, 1.15));

    const keyLight = new THREE.DirectionalLight(0xf3f7ff, 1.65);
    keyLight.position.set(24, 46, 22);
    this.scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x74a4de, 0.72);
    rimLight.position.set(-32, 18, -36);
    this.scene.add(rimLight);

    const fillLight = new THREE.DirectionalLight(0x9fb8d8, 0.42);
    fillLight.position.set(0, 12, 34);
    this.scene.add(fillLight);
  }

  _buildHighwayEnvironment() {
    const terrain = new THREE.Mesh(
      new THREE.PlaneGeometry(520, 560),
      new THREE.MeshStandardMaterial({ color: 0x090d14, roughness: 0.96 })
    );
    terrain.rotation.x = -Math.PI / 2;
    terrain.position.y = -0.05;
    this.envGroup.add(terrain);

    this.asphaltTex = createAsphaltTexture();
    const roadW = ROAD_HALF_WIDTH * 2;
    const road = new THREE.Mesh(
      new THREE.PlaneGeometry(roadW, 560),
      new THREE.MeshStandardMaterial({ map: this.asphaltTex, roughness: 0.82, metalness: 0.08 })
    );
    road.rotation.x = -Math.PI / 2;
    road.position.y = 0;
    this.envGroup.add(road);

    const edgeMat = new THREE.MeshBasicMaterial({ color: 0xa6b2c2 });
    for (const side of [-1, 1]) {
      const edge = new THREE.Mesh(new THREE.PlaneGeometry(0.18, 560), edgeMat);
      edge.rotation.x = -Math.PI / 2;
      edge.position.set(side * (ROAD_HALF_WIDTH - 0.62), 0.012, 0);
      this.envGroup.add(edge);
    }

    this.dashCycle = 12.0;
    this.dashGroup = new THREE.Group();
    const dashGeo = new THREE.PlaneGeometry(0.16, 4.2);
    dashGeo.rotateX(-Math.PI / 2);
    const dashMat = new THREE.MeshBasicMaterial({ color: 0xc2ccd8 });
    for (let z = -252; z <= 180; z += this.dashCycle) {
      for (const dx of [-LANE_WIDTH * 0.5, LANE_WIDTH * 0.5]) {
        const dash = new THREE.Mesh(dashGeo, dashMat);
        dash.position.set(dx, 0.014, z);
        this.dashGroup.add(dash);
      }
    }
    this.envGroup.add(this.dashGroup);

    const concreteMat = new THREE.MeshStandardMaterial({ color: 0x2a323e, roughness: 0.85, metalness: 0.1 });
    const steelMat = new THREE.MeshStandardMaterial({ color: 0x586578, roughness: 0.4, metalness: 0.72 });
    for (const side of [-1, 1]) {
      const barrier = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.78, 560), concreteMat);
      barrier.position.set(side * (ROAD_HALF_WIDTH + 0.12), 0.39, 0);
      this.envGroup.add(barrier);

      const rail = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.18, 560), steelMat);
      rail.position.set(side * (ROAD_HALF_WIDTH - 0.14), 0.72, 0);
      this.envGroup.add(rail);
    }
  }

  _buildStreetlampsAndGantries() {
    this.lampPairs = [];
    const poleMat = new THREE.MeshStandardMaterial({ color: 0x3b4556, roughness: 0.45, metalness: 0.65 });
    const lumMat = new THREE.MeshBasicMaterial({ color: 0xeef5ff });
    const glowMat = new THREE.SpriteMaterial({
      map: sharedLightGlowTex,
      color: 0xcbe4ff,
      transparent: true,
      opacity: 0.42,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const poolMat = new THREE.MeshBasicMaterial({
      map: sharedLightGlowTex,
      color: 0xbcd8f8,
      transparent: true,
      opacity: 0.09,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    const poleGeo = new THREE.CylinderGeometry(0.08, 0.12, 8.2, 10);
    const armGeo = new THREE.BoxGeometry(2.3, 0.1, 0.1);
    const headGeo = new THREE.BoxGeometry(0.65, 0.09, 0.28);
    const poolGeo = new THREE.PlaneGeometry(11.5, 11.5);
    poolGeo.rotateX(-Math.PI / 2);

    for (let i = 0; i < 12; i++) {
      const pair = new THREE.Group();
      for (const side of [-1, 1]) {
        const pole = new THREE.Mesh(poleGeo, poleMat);
        pole.position.set(side * (ROAD_HALF_WIDTH + 0.75), 4.1, 0);
        pair.add(pole);

        const arm = new THREE.Mesh(armGeo, poleMat);
        arm.position.set(side * (ROAD_HALF_WIDTH - 0.3), 8.15, 0);
        pair.add(arm);

        const head = new THREE.Mesh(headGeo, lumMat);
        head.position.set(side * (ROAD_HALF_WIDTH - 1.25), 8.08, 0);
        pair.add(head);

        const sprite = new THREE.Sprite(glowMat);
        sprite.position.set(side * (ROAD_HALF_WIDTH - 1.25), 8.02, 0);
        sprite.scale.set(2.4, 2.4, 1);
        pair.add(sprite);

        const pool = new THREE.Mesh(poolGeo, poolMat);
        pool.position.set(side * (ROAD_HALF_WIDTH - 2.4), 0.02, 0);
        pair.add(pool);
      }
      this.envGroup.add(pair);
      this.lampPairs.push(pair);
    }

    this.gantries = [];
    const signTex = createHighwaySignTexture();
    const trussMat = new THREE.MeshStandardMaterial({ color: 0x384252, roughness: 0.45, metalness: 0.7 });
    const signMat = new THREE.MeshStandardMaterial({
      map: signTex,
      roughness: 0.35,
      metalness: 0.2,
      emissive: 0x0a2540,
      emissiveIntensity: 0.55,
    });
    const pillarGeo = new THREE.BoxGeometry(0.42, 7.6, 0.42);
    const beamGeo = new THREE.BoxGeometry(ROAD_HALF_WIDTH * 2 + 2.8, 0.45, 0.45);
    const boardGeo = new THREE.BoxGeometry(8.6, 2.1, 0.16);

    for (let i = 0; i < 3; i++) {
      const g = new THREE.Group();
      for (const side of [-1, 1]) {
        const p = new THREE.Mesh(pillarGeo, trussMat);
        p.position.set(side * (ROAD_HALF_WIDTH + 1.2), 3.8, 0);
        g.add(p);
      }
      const beam = new THREE.Mesh(beamGeo, trussMat);
      beam.position.set(0, 7.4, 0);
      g.add(beam);

      const board = new THREE.Mesh(boardGeo, signMat);
      board.position.set(0, 6.8, 0.28);
      g.add(board);

      this.envGroup.add(g);
      this.gantries.push(g);
    }
  }

  _buildVehicles() {
    this.egoVisual = createSculptedVehicle("sedan", 0xe6edf6, true);
    this.envGroup.add(this.egoVisual.group);

    this.egoWireframe = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1.94, 1.44, 4.86)),
      new THREE.LineBasicMaterial({ color: 0xe2e8f0, transparent: true, opacity: 0.78 })
    );
    this.egoWireframe.position.set(0, 0.72, 0);
    this.lidarGroup.add(this.egoWireframe);

    this.poolSlots = [];
    for (let i = 0; i < 14; i++) {
      const container = new THREE.Group();
      container.visible = false;
      const variants = {
        coupe: createSculptedVehicle("coupe", 0x8899ad),
        sedan: createSculptedVehicle("sedan", 0x6e7b8b),
        suv: createSculptedVehicle("suv", 0x4c5866),
        truck: createSculptedVehicle("truck", 0xc4cdd8),
      };
      for (const k of Object.keys(variants)) {
        variants[k].group.visible = false;
        container.add(variants[k].group);
      }
      this.envGroup.add(container);
      this.poolSlots.push({ container, variants, activeType: null });
    }
  }

  _buildLiDARPointCloud() {
    const maxPts = 24 * 280;
    this.pcPosArr = new Float32Array(maxPts * 3);
    this.pcColArr = new Float32Array(maxPts * 3);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.pcPosArr, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(this.pcColArr, 3));

    this.pcMat = new THREE.PointsMaterial({
      size: 0.34,
      map: sharedLightGlowTex,
      vertexColors: true,
      transparent: true,
      opacity: 0.94,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.pointCloud = new THREE.Points(geo, this.pcMat);
    this.lidarGroup.add(this.pointCloud);

    this.rangeRings = new THREE.Group();
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.16,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    for (const r of [20, 40, 60, 80, 100]) {
      const ring = new THREE.Mesh(new THREE.RingGeometry(r - 0.1, r + 0.1, 72), ringMat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.03;
      this.rangeRings.add(ring);
    }
    this.lidarGroup.add(this.rangeRings);

    this.pulseRing = new THREE.Mesh(
      new THREE.RingGeometry(0.96, 1.0, 72),
      new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.32,
        side: THREE.DoubleSide,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      })
    );
    this.pulseRing.rotation.x = -Math.PI / 2;
    this.pulseRing.position.y = 0.04;
    this.arGroup.add(this.pulseRing);
  }

  _buildARAndFusionOverlays() {
    // 1. 3D Tracking Bounding Boxes
    this.boxPool = [];
    const unitEdges = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    for (let i = 0; i < 14; i++) {
      const box = new THREE.LineSegments(
        unitEdges,
        new THREE.LineBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.85 })
      );
      box.visible = false;
      this.arGroup.add(box);
      this.boxPool.push(box);
    }

    // 2. Planned Path Corridor Ribbon
    const segs = 16;
    const pathPos = new Float32Array((segs + 1) * 2 * 3);
    const indices = [];
    for (let i = 0; i < segs; i++) {
      const a = i * 2;
      indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
    const pathGeo = new THREE.BufferGeometry();
    pathGeo.setAttribute("position", new THREE.BufferAttribute(pathPos, 3));
    pathGeo.setIndex(indices);
    this.pathMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.16,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.pathRibbon = new THREE.Mesh(pathGeo, this.pathMat);
    this.arGroup.add(this.pathRibbon);

    const edgePosL = new Float32Array((segs + 1) * 3);
    const edgePosR = new Float32Array((segs + 1) * 3);
    const lGeo = new THREE.BufferGeometry();
    lGeo.setAttribute("position", new THREE.BufferAttribute(edgePosL, 3));
    const rGeo = new THREE.BufferGeometry();
    rGeo.setAttribute("position", new THREE.BufferAttribute(edgePosR, 3));
    this.pathEdgeMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.72 });
    this.pathEdgeL = new THREE.Line(lGeo, this.pathEdgeMat);
    this.pathEdgeR = new THREE.Line(rGeo, this.pathEdgeMat);
    this.arGroup.add(this.pathEdgeL, this.pathEdgeR);

    // 3. Forward 77 GHz Radar Cone (Sensor Fusion Mode)
    const radarShape = new THREE.Shape();
    radarShape.moveTo(0, 0);
    const rDist = 75;
    const rAngle = (16 * Math.PI) / 180;
    radarShape.lineTo(-Math.sin(rAngle) * rDist, Math.cos(rAngle) * rDist);
    radarShape.absarc(0, 0, rDist, Math.PI * 0.5 + rAngle, Math.PI * 0.5 - rAngle, true);
    radarShape.closePath();
    const radarGeo = new THREE.ShapeGeometry(radarShape, 24);
    radarGeo.rotateX(-Math.PI / 2);
    this.radarCone = new THREE.Mesh(
      radarGeo,
      new THREE.MeshBasicMaterial({
        color: 0xa855f7,
        transparent: true,
        opacity: 0.14,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    this.radarCone.position.y = 0.05;
    this.fusionGroup.add(this.radarCone);

    // 4. Dual-Zone 3D LiDAR Occlusion Shader:
    //    Solid translucent amber on asphalt (|x| <= barrier), crisp dotted fade outside barriers
    this.occlusionMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      uniforms: {
        uColor: { value: new THREE.Color(0xf59e0b) },
        uRoadHalfWidth: { value: ROAD_HALF_WIDTH },
        uOpacity: { value: 0.2 },
      },
      vertexShader: `
        varying vec3 vPos;
        void main() {
          vPos = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        uniform float uRoadHalfWidth;
        uniform float uOpacity;
        varying vec3 vPos;
        void main() {
          float dist = length(vPos.xz);
          float distFade = smoothstep(94.0, 8.0, dist);
          float absX = abs(vPos.x);

          if (absX <= uRoadHalfWidth) {
            gl_FragColor = vec4(uColor, uOpacity * distFade);
          } else {
            vec2 cell = fract(vPos.xz * 0.82) - 0.5;
            float dotMask = 1.0 - smoothstep(0.15, 0.23, length(cell));
            float outerFade = smoothstep(uRoadHalfWidth + 34.0, uRoadHalfWidth + 0.5, absX);
            float alpha = uOpacity * 1.35 * dotMask * distFade * outerFade;
            if (alpha < 0.01) discard;
            gl_FragColor = vec4(uColor, alpha);
          }
        }
      `,
    });

    this.occlusionWedges = [];
    for (let i = 0; i < 12; i++) {
      const wGeo = new THREE.BufferGeometry();
      wGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(4 * 3), 3));
      wGeo.setIndex([0, 1, 2, 0, 2, 3]);
      const wedge = new THREE.Mesh(wGeo, this.occlusionMat);
      wedge.visible = false;
      this.occlusionGroup.add(wedge);
      this.occlusionWedges.push(wedge);
    }
  }

  _buildLabelPool() {
    this.labelEls = [];
    for (let i = 0; i < 14; i++) {
      const el = document.createElement("div");
      el.className = "veh-badge hidden";
      el.innerHTML = `<span class="vb-id"></span><span class="vb-meta"></span>`;
      this.labelsLayer.appendChild(el);
      this.labelEls.push({
        el,
        idSpan: el.querySelector(".vb-id"),
        metaSpan: el.querySelector(".vb-meta"),
      });
    }
  }

  setVisualMode(mode) {
    this.visualMode = mode;
  }

  setCameraMode(mode) {
    this.camMode = mode;
    if (mode === "orbit") {
      this.controls.target.set(0, 1.1, -12);
      this.controls.update();
    }
  }

  setQuality(presetKey) {
    const q = QUALITY_PRESETS[presetKey] || QUALITY_PRESETS.high;
    this._maxDpr = q.maxDpr;
    this.resize();
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.width = Math.max(10, Math.floor(rect.width));
    this.height = Math.max(10, Math.floor(rect.height));
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, this._maxDpr));
    this.renderer.setSize(this.width, this.height, false);
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
  }

  _updateWorldGeometry(state) {
    const ego = state.ego;
    const simTime = state.clock.simTime;

    // 1. World-Anchored Road Texture, Dashes, Lamps & Gantries (100% Frozen on Pause)
    this.asphaltTex.offset.y = (ego.s / 14.0) % 1;
    this.dashGroup.position.z = ego.s % this.dashCycle;

    const firstLampIdx = Math.floor((ego.s - 110) / LAMP_SPACING);
    for (let i = 0; i < this.lampPairs.length; i++) {
      const lampS = (firstLampIdx + i) * LAMP_SPACING;
      this.lampPairs[i].position.z = -(lampS - ego.s);
    }

    const firstGantryIdx = Math.floor((ego.s - 80) / GANTRY_SPACING);
    for (let i = 0; i < this.gantries.length; i++) {
      const gantryS = (firstGantryIdx + i) * GANTRY_SPACING + 95;
      this.gantries[i].position.z = -(gantryS - ego.s);
    }

    // 2. Ego Pose, Exact Wheel Rotation & Spinning LiDAR Dome
    this.egoVisual.group.position.set(ego.x, 0, 0);
    for (const w of this.egoVisual.wheels) {
      w.rotation.x = -ego.wheelAngle;
    }
    if (this.egoVisual.lidarSpinner) {
      this.egoVisual.lidarSpinner.rotation.y = state.clock.scanAngleRad;
    }
    this.egoVisual.tailMat.color.setHex(ego.brakeLight ? 0xff2432 : 0x8f1822);

    // 3. Synchronized Traffic Vehicles
    const blinkOn = Math.floor(simTime * 2.5) % 2 === 0;
    for (let i = 0; i < state.vehicles.length; i++) {
      const v = state.vehicles[i];
      const slot = this.poolSlots[i];
      if (!v.active) {
        slot.container.visible = false;
        continue;
      }

      slot.container.visible = true;
      slot.container.position.set(v.x, 0, -v.relS);
      slot.container.rotation.y = -v.steeringAngle * 0.85;

      if (slot.activeType !== v.type) {
        for (const k of Object.keys(slot.variants)) {
          slot.variants[k].group.visible = k === v.type;
        }
        slot.activeType = v.type;
      }

      const vis = slot.variants[v.type];
      vis.bodyMat.color.setHex(v.colorHex);
      vis.tailMat.color.setHex(v.brakeLight ? 0xff2432 : 0x8f1822);
      vis.turnLeftMat.color.setHex(v.turnSignal === -1 && blinkOn ? 0xffaa00 : 0x3a2810);
      vis.turnRightMat.color.setHex(v.turnSignal === 1 && blinkOn ? 0xffaa00 : 0x3a2810);

      for (const w of vis.wheels) {
        w.rotation.x = -v.wheelAngle;
      }
    }

    // 4. Expanding LiDAR Scan Pulse Ring
    const pRad = Math.max(1.5, state.lidar.pulseRadius);
    this.pulseRing.scale.set(pRad, pRad, 1);
    this.pulseRing.material.opacity = Math.max(0.02, 0.32 * (1 - pRad / LIDAR_MAX_RANGE));

    // 5. Update 3D LiDAR Point Cloud Buffer
    const ls = state.lidar;
    const n = ls.activePoints;
    const srcPos = ls.pointsPos;
    const srcDist = ls.pointsDist;
    const srcObj = ls.pointsObjId;

    for (let i = 0; i < n; i++) {
      const i3 = i * 3;
      this.pcPosArr[i3] = srcPos[i3];
      this.pcPosArr[i3 + 1] = srcPos[i3 + 1];
      this.pcPosArr[i3 + 2] = srcPos[i3 + 2];

      const objId = srcObj[i];
      const t = Math.min(1, srcDist[i] / LIDAR_MAX_RANGE);
      if (objId >= 0) {
        this.pcColArr[i3] = 0.22 + (1 - t) * 0.25;
        this.pcColArr[i3 + 1] = 0.94 - t * 0.26;
        this.pcColArr[i3 + 2] = 1.0 - t * 0.14;
      } else if (objId === -2) {
        this.pcColArr[i3] = 0.34;
        this.pcColArr[i3 + 1] = 0.45;
        this.pcColArr[i3 + 2] = 0.6;
      } else {
        this.pcColArr[i3] = 0.1 + (1 - t) * 0.12;
        this.pcColArr[i3 + 1] = 0.28 + (1 - t) * 0.2;
        this.pcColArr[i3 + 2] = 0.44 + (1 - t) * 0.24;
      }
    }
    this.pointCloud.geometry.setDrawRange(0, n);
    this.pointCloud.geometry.attributes.position.needsUpdate = true;
    this.pointCloud.geometry.attributes.color.needsUpdate = true;

    // 6. 3D Tracking Bounding Boxes
    for (let i = 0; i < this.boxPool.length; i++) {
      const box = this.boxPool[i];
      if (i >= state.tracks.length) {
        box.visible = false;
        continue;
      }
      const trk = state.tracks[i];
      if (trk.status === "OCCLUDED" && this.visualMode !== "sensor_fusion") {
        box.visible = false;
        continue;
      }
      box.visible = true;
      box.position.set(trk.bbox.cx, trk.bbox.cy, trk.bbox.cz);
      box.scale.set(trk.bbox.w + 0.18, trk.bbox.h + 0.14, trk.bbox.l + 0.22);
      const hex =
        trk.ttc < 2.8
          ? 0xef4444
          : trk.status === "OCCLUDED"
            ? 0x94a3b8
            : trk.status === "PARTIAL"
              ? 0xf59e0b
              : 0x34d399;
      box.material.color.setHex(hex);
    }

    // 7. Planned Path Corridor Ribbon
    if (ego.plannedPath.length > 1) {
      const halfW = ego.width * 0.56;
      const pAttr = this.pathRibbon.geometry.attributes.position;
      const lAttr = this.pathEdgeL.geometry.attributes.position;
      const rAttr = this.pathEdgeR.geometry.attributes.position;
      for (let i = 0; i < ego.plannedPath.length; i++) {
        const pt = ego.plannedPath[i];
        pAttr.setXYZ(i * 2, pt.x - halfW, 0.045, pt.z);
        pAttr.setXYZ(i * 2 + 1, pt.x + halfW, 0.045, pt.z);
        lAttr.setXYZ(i, pt.x - halfW, 0.055, pt.z);
        rAttr.setXYZ(i, pt.x + halfW, 0.055, pt.z);
      }
      pAttr.needsUpdate = true;
      lAttr.needsUpdate = true;
      rAttr.needsUpdate = true;

      const corridorCol =
        ego.controlState === "AEB_BRAKING"
          ? 0xef4444
          : ego.controlState === "CUT_IN_YIELD" || ego.controlState === "ACC_FOLLOW"
            ? 0xf59e0b
            : 0x38bdf8;
      this.pathMat.color.setHex(corridorCol);
      this.pathEdgeMat.color.setHex(corridorCol);
    }

    // 8. Dual-Zone Geometric Occlusion Shadows for ALL Tracked Vehicles (Left & Right)
    let wedgeIdx = 0;
    for (const trk of state.tracks) {
      if (wedgeIdx >= this.occlusionWedges.length) break;
      if (trk.status === "OCCLUDED" || trk.distance < 3.0 || trk.distance > 78 || trk.relS < -12) {
        continue;
      }

      const hw = trk.bbox.w * 0.5;
      const hl = trk.bbox.l * 0.5;
      const corners = [
        { x: trk.x - hw, z: trk.bbox.cz - hl },
        { x: trk.x + hw, z: trk.bbox.cz - hl },
        { x: trk.x - hw, z: trk.bbox.cz + hl },
        { x: trk.x + hw, z: trk.bbox.cz + hl },
      ];
      const baseA = Math.atan2(trk.bbox.cz, trk.x - ego.x);
      let minA = Infinity, maxA = -Infinity, cMin = corners[0], cMax = corners[0];
      for (const c of corners) {
        let a = Math.atan2(c.z, c.x - ego.x) - baseA;
        while (a > Math.PI) a -= Math.PI * 2;
        while (a < -Math.PI) a += Math.PI * 2;
        if (a < minA) { minA = a; cMin = c; }
        if (a > maxA) { maxA = a; cMax = c; }
      }
      if (maxA - minA > Math.PI * 0.85) continue;

      const farDist = 92;
      const a1 = baseA + minA;
      const a2 = baseA + maxA;
      const wMesh = this.occlusionWedges[wedgeIdx++];
      wMesh.visible = true;
      const pos = wMesh.geometry.attributes.position;
      pos.setXYZ(0, cMin.x, 0.04, cMin.z);
      pos.setXYZ(1, ego.x + Math.cos(a1) * farDist, 0.04, Math.sin(a1) * farDist);
      pos.setXYZ(2, ego.x + Math.cos(a2) * farDist, 0.04, Math.sin(a2) * farDist);
      pos.setXYZ(3, cMax.x, 0.04, cMax.z);
      pos.needsUpdate = true;
    }
    for (let i = wedgeIdx; i < this.occlusionWedges.length; i++) {
      this.occlusionWedges[i].visible = false;
    }
  }

  _updateCamera(state) {
    if (this.camMode === "orbit") {
      this.controls.update();
      return;
    }

    const ego = state.ego;
    const intro = state.clock.introProgress;
    const easeIntro = 1 - Math.pow(1 - intro, 3);

    if (this.camMode === "cinematic") {
      const startX = 9.5;
      const endX = 3.6;
      const startY = 7.4;
      const endY = 4.6;
      const startZ = 15.5;
      const endZ = 12.2;
      this._camTargetPos.set(
        ego.x + startX + (endX - startX) * easeIntro,
        startY + (endY - startY) * easeIntro,
        startZ + (endZ - startZ) * easeIntro
      );
      this._camTargetLook.set(ego.x, 1.05, -20);
    } else if (this.camMode === "bev") {
      this._camTargetPos.set(0, 56, -10);
      this._camTargetLook.set(0, 0, -22);
    } else if (this.camMode === "hood") {
      this._camTargetPos.set(ego.x, LIDAR_MOUNT_HEIGHT + 0.28, -0.3);
      this._camTargetLook.set(ego.x, 1.1, -55);
    }

    const rate = this.camMode === "hood" ? 0.5 : 0.1;
    this.camera.position.lerp(this._camTargetPos, rate);
    this.controls.target.lerp(this._camTargetLook, rate);
    this.camera.lookAt(this.controls.target);
  }

  _projectLabels(state, viewportWidth, viewportOffsetX = 0) {
    const h = this.height;
    for (let i = 0; i < this.labelEls.length; i++) {
      const item = this.labelEls[i];
      if (i >= state.tracks.length) {
        item.el.classList.add("hidden");
        continue;
      }

      const trk = state.tracks[i];
      if (
        (trk.status === "OCCLUDED" && this.visualMode !== "sensor_fusion" && state.scenario.id !== "truck_occlusion") ||
        trk.distance > 90
      ) {
        item.el.classList.add("hidden");
        continue;
      }

      this._projVec.set(trk.bbox.cx, trk.bbox.h + 0.55, trk.bbox.cz);
      this._projVec.project(this.camera);

      if (this._projVec.z > 1 || this._projVec.x < -0.96 || this._projVec.x > 0.96 || this._projVec.y < -0.9 || this._projVec.y > 0.88) {
        item.el.classList.add("hidden");
        continue;
      }

      const sx = viewportOffsetX + (this._projVec.x * 0.5 + 0.5) * viewportWidth;
      const sy = (-this._projVec.y * 0.5 + 0.5) * h;

      item.el.className = "veh-badge";
      if (trk.ttc < 2.8) item.el.classList.add("status-critical");
      else if (trk.status === "OCCLUDED") item.el.classList.add("status-occluded");
      else if (trk.status === "PARTIAL") item.el.classList.add("status-partial");

      item.idSpan.textContent = trk.id;
      const sign = trk.relSpeed >= 0 ? "+" : "";
      item.metaSpan.textContent =
        trk.status === "OCCLUDED"
          ? `OCCLUDED · ${trk.distance.toFixed(0)}m`
          : `${trk.distance.toFixed(0)}m · ${sign}${(trk.relSpeed * 3.6).toFixed(0)} km/h`;

      item.el.style.transform = `translate3d(${sx.toFixed(1)}px, ${sy.toFixed(1)}px, 0) translate(-50%, -100%)`;
    }
  }

  render(state) {
    this._updateWorldGeometry(state);
    this._updateCamera(state);

    const w = this.width;
    const h = this.height;

    if (this.visualMode === "split_compare") {
      const halfW = Math.floor(w * 0.5);
      this.renderer.setScissorTest(true);

      // Left Half: 3D Reality + AR + Dual-Zone Occlusion Shadows
      this.camera.aspect = halfW / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setViewport(0, 0, halfW, h);
      this.renderer.setScissor(0, 0, halfW, h);
      this.renderer.setClearColor(0x060a12, 1);
      this.scene.fog.density = 0.0036;
      this.envGroup.visible = true;
      this.lidarGroup.visible = false;
      this.arGroup.visible = true;
      this.fusionGroup.visible = false;
      this.occlusionGroup.visible = true;
      this.occlusionMat.uniforms.uOpacity.value = 0.2;
      this.renderer.render(this.scene, this.camera);
      this._projectLabels(state, halfW, 0);

      // Right Half: Pure 3D Occluded LiDAR Point Cloud
      this.camera.aspect = (w - halfW) / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setViewport(halfW, 0, w - halfW, h);
      this.renderer.setScissor(halfW, 0, w - halfW, h);
      this.renderer.setClearColor(0x03050a, 1);
      this.scene.fog.density = 0.0014;
      this.envGroup.visible = false;
      this.lidarGroup.visible = true;
      this.egoWireframe.visible = true;
      this.rangeRings.visible = true;
      this.arGroup.visible = true;
      this.fusionGroup.visible = false;
      this.occlusionGroup.visible = false;
      this.pcMat.size = 0.42;
      this.renderer.render(this.scene, this.camera);

      this.renderer.setScissorTest(false);
      return;
    }

    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setViewport(0, 0, w, h);

    if (this.visualMode === "lidar_cloud") {
      this.renderer.setClearColor(0x03050a, 1);
      this.scene.fog.density = 0.0014;
      this.envGroup.visible = false;
      this.lidarGroup.visible = true;
      this.egoWireframe.visible = true;
      this.rangeRings.visible = true;
      this.arGroup.visible = true;
      this.fusionGroup.visible = false;
      this.occlusionGroup.visible = false;
      this.pcMat.size = 0.44;
    } else if (this.visualMode === "sensor_fusion") {
      this.renderer.setClearColor(0x050810, 1);
      this.scene.fog.density = 0.003;
      this.envGroup.visible = true;
      this.lidarGroup.visible = true;
      this.egoWireframe.visible = false;
      this.rangeRings.visible = true;
      this.arGroup.visible = true;
      this.fusionGroup.visible = true;
      this.radarCone.visible = true;
      this.occlusionGroup.visible = true;
      this.occlusionMat.uniforms.uOpacity.value = 0.25;
      this.pcMat.size = 0.34;
    } else {
      // 3D Studio + AR
      this.renderer.setClearColor(0x060a12, 1);
      this.scene.fog.density = 0.0036;
      this.envGroup.visible = true;
      this.lidarGroup.visible = true;
      this.egoWireframe.visible = false;
      this.rangeRings.visible = false;
      this.arGroup.visible = true;
      this.fusionGroup.visible = false;
      this.occlusionGroup.visible = true;
      this.occlusionMat.uniforms.uOpacity.value =
        state.scenario.id === "truck_occlusion" ? 0.24 : 0.16;
      this.pcMat.size = 0.26;
    }

    this.renderer.render(this.scene, this.camera);
    this._projectLabels(state, w, 0);
  }
}

/* ============================================================================
   SYNCHRONIZED LIVE BEV RADAR MINI-MAP RENDERER
   ============================================================================ */
export class BEVRadarRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.w = 176;
    this.h = 186;
    this.dpr = 1;
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.w = Math.max(10, Math.floor(rect.width));
    this.h = Math.max(10, Math.floor(rect.height));
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.floor(this.w * this.dpr);
    this.canvas.height = Math.floor(this.h * this.dpr);
  }

  render(state) {
    const ctx = this.ctx;
    const w = this.w;
    const h = this.h;
    ctx.save();
    ctx.scale(this.dpr, this.dpr);

    ctx.fillStyle = "#05080f";
    ctx.fillRect(0, 0, w, h);

    const pxPerM = h / 128;
    const originX = w * 0.5;
    const egoY = h * 0.74;
    const toX = (x) => originX + x * pxPerM * 1.45;
    const toY = (relS) => egoY - relS * pxPerM;

    const roadL = toX(-ROAD_HALF_WIDTH);
    const roadR = toX(ROAD_HALF_WIDTH);
    ctx.fillStyle = "#0d121d";
    ctx.fillRect(roadL, 0, roadR - roadL, h);

    ctx.save();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4 * pxPerM, 8 * pxPerM]);
    ctx.lineDashOffset = -(state.ego.s % 12) * pxPerM;
    for (const dx of [-LANE_WIDTH * 0.5, LANE_WIDTH * 0.5]) {
      ctx.beginPath();
      ctx.moveTo(toX(dx), 0);
      ctx.lineTo(toX(dx), h);
      ctx.stroke();
    }
    ctx.restore();

    ctx.strokeStyle = "rgba(56, 189, 248, 0.14)";
    for (const r of [25, 50, 75]) {
      ctx.beginPath();
      ctx.arc(toX(state.ego.x), egoY, r * pxPerM, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Dual-Zone Occlusion Shadow Cones for ALL tracked vehicles (Left & Right)
    const ox = toX(state.ego.x);
    const oy = egoY;
    for (const trk of state.tracks) {
      if (trk.status === "OCCLUDED" || trk.distance < 3.0 || trk.distance > 78 || trk.relS < -12) {
        continue;
      }
      const hw = trk.bbox.w * 0.5;
      const hl = trk.bbox.l * 0.5;
      const corners = [
        { x: toX(trk.x - hw), y: toY(trk.relS - hl) },
        { x: toX(trk.x + hw), y: toY(trk.relS - hl) },
        { x: toX(trk.x - hw), y: toY(trk.relS + hl) },
        { x: toX(trk.x + hw), y: toY(trk.relS + hl) },
      ];
      const baseA = Math.atan2(toY(trk.relS) - oy, toX(trk.x) - ox);
      let minA = Infinity, maxA = -Infinity, cMin = corners[0], cMax = corners[0];
      for (const c of corners) {
        let a = Math.atan2(c.y - oy, c.x - ox) - baseA;
        while (a > Math.PI) a -= Math.PI * 2;
        while (a < -Math.PI) a += Math.PI * 2;
        if (a < minA) { minA = a; cMin = c; }
        if (a > maxA) { maxA = a; cMax = c; }
      }
      if (maxA - minA > Math.PI * 0.85) continue;

      const far = 135;
      const p1x = ox + Math.cos(baseA + minA) * far;
      const p1y = oy + Math.sin(baseA + minA) * far;
      const p2x = ox + Math.cos(baseA + maxA) * far;
      const p2y = oy + Math.sin(baseA + maxA) * far;

      // Subtle dashed outline extending beyond highway
      ctx.save();
      ctx.fillStyle = "rgba(245, 158, 11, 0.06)";
      ctx.strokeStyle = "rgba(245, 158, 11, 0.24)";
      ctx.lineWidth = 0.8;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(cMin.x, cMin.y);
      ctx.lineTo(p1x, p1y);
      ctx.lineTo(p2x, p2y);
      ctx.lineTo(cMax.x, cMax.y);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      // Solid amber shadow clipped cleanly onto the highway deck
      ctx.save();
      ctx.beginPath();
      ctx.rect(roadL, 0, roadR - roadL, h);
      ctx.clip();
      ctx.fillStyle = "rgba(245, 158, 11, 0.19)";
      ctx.beginPath();
      ctx.moveTo(cMin.x, cMin.y);
      ctx.lineTo(p1x, p1y);
      ctx.lineTo(p2x, p2y);
      ctx.lineTo(cMax.x, cMax.y);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    // Tracked Vehicles
    for (const trk of state.tracks) {
      const cx = toX(trk.x);
      const cy = toY(trk.relS);
      const bw = Math.max(5, trk.bbox.w * pxPerM * 1.45);
      const bl = Math.max(8, trk.bbox.l * pxPerM);
      ctx.save();
      if (trk.status === "OCCLUDED") ctx.setLineDash([2, 2]);
      const col =
        trk.ttc < 2.8
          ? "#ef4444"
          : trk.status === "OCCLUDED"
            ? "#94a3b8"
            : trk.status === "PARTIAL"
              ? "#f59e0b"
              : "#34d399";
      ctx.fillStyle = trk.status === "OCCLUDED" ? "rgba(148,163,184,0.16)" : "rgba(52,211,153,0.22)";
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.25;
      ctx.fillRect(cx - bw * 0.5, cy - bl * 0.5, bw, bl);
      ctx.strokeRect(cx - bw * 0.5, cy - bl * 0.5, bw, bl);
      ctx.restore();
    }

    // Ego Vehicle
    const egoW = state.ego.width * pxPerM * 1.45;
    const egoL = state.ego.length * pxPerM;
    ctx.fillStyle = "#f1f5f9";
    ctx.fillRect(toX(state.ego.x) - egoW * 0.5, egoY - egoL * 0.5, egoW, egoL);
    ctx.fillStyle = "#38bdf8";
    ctx.beginPath();
    ctx.arc(toX(state.ego.x), egoY, 2.3, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }
}