/**
 * ThinkModule Lab — Photorealistic WebGL PBR Autonomous Driving & LiDAR Scene
 * Features:
 *   - Studio-grade PBR clearcoat vehicles, articulated walking pedestrian, infinite highway
 *   - Real-time 4,600-point 3D LiDAR point cloud with true ray-vs-AABB occlusion shadows
 *   - Translucent RGB Camera FOV frustum, animated 77 GHz Radar wave arcs & 3D bounding boxes
 *   - Dynamic predictive braking corridor ribbon & 3 view modes (Reality+AR, LiDAR Cloud, Camera Raw)
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { createRoadTextures, createLidarDotTexture } from "./textures.js";

const LIDAR_MAX_PTS = 4800;

function rayBlocked2D(x1, z1, actors, ignoreId) {
  for (let i = 0; i < actors.length; i++) {
    const a = actors[i];
    if (a.id === ignoreId) continue;
    const minX = a.x - a.w * 0.5;
    const maxX = a.x + a.w * 0.5;
    const minZ = -a.dist - a.l * 0.5;
    const maxZ = -a.dist + a.l * 0.5;

    let tMin = 0, tMax = 1;
    if (Math.abs(x1) < 1e-6) {
      if (0 < minX || 0 > maxX) continue;
    } else {
      let t1 = minX / x1, t2 = maxX / x1;
      if (t1 > t2) [t1, t2] = [t2, t1];
      tMin = Math.max(tMin, t1);
      tMax = Math.min(tMax, t2);
      if (tMin > tMax) continue;
    }

    if (Math.abs(z1) < 1e-6) {
      if (0 < minZ || 0 > maxZ) continue;
    } else {
      let t1 = minZ / z1, t2 = maxZ / z1;
      if (t1 > t2) [t1, t2] = [t2, t1];
      tMin = Math.max(tMin, t1);
      tMax = Math.min(tMax, t2);
      if (tMin > tMax) continue;
    }

    if (tMin > 0.04 && tMin < 0.96) return true;
  }
  return false;
}

export class AdasScene3D {
  constructor(container) {
    this.container = container;
    this.viewMode = "fusion"; // "fusion" | "lidar" | "camera"
    this.camPreset = "chase";
    this.camLerp = false;
    this.camGoalPos = new THREE.Vector3(7.4, 5.2, 10.6);
    this.camGoalTarget = new THREE.Vector3(0, 1.0, -11.0);

    this.actorMeshes = new Map();
    this.bboxMeshes = new Map();
    this.streetLights = [];
    this.egoWheels = [];
    this.visiblePointsCount = 0;

    this._initRenderer();
    this._initLightsAndEnvironment();
    this._buildHighway();
    this._buildEgoVehicle();
    this._buildSensorVisuals();
    this._buildLidarPointCloud();
    this._buildTrajectoryCorridor();
    this._bindResize();
  }

  _initRenderer() {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x050912);
    this.scene.fog = new THREE.FogExp2(0x09111f, 0.011);

    this.camera = new THREE.PerspectiveCamera(42, w / h, 0.15, 220);
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
    this.renderer.toneMappingExposure = 1.06;
    this.container.appendChild(this.renderer.domElement);

    // Studio IBL environment so clearcoat vehicle paint gleams
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.05).texture;
    this.scene.environmentIntensity = 0.72;
    pmrem.dispose();

    // Subtle selective bloom for LED lights, LiDAR points, and bounding boxes
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(w, h), 0.34, 0.38, 0.86);
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(new OutputPass());

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.minDistance = 4.0;
    this.controls.maxDistance = 58.0;
    this.controls.maxPolarAngle = Math.PI * 0.485;
    this.controls.target.copy(this.camGoalTarget);

    this.controls.addEventListener("start", () => {
      this.camLerp = false;
    });
  }

  _initLightsAndEnvironment() {
    this.hemiLight = new THREE.HemisphereLight(0x88b4e6, 0x0b1220, 0.65);
    this.scene.add(this.hemiLight);

    const moonKey = new THREE.DirectionalLight(0xdcebff, 1.65);
    moonKey.position.set(16, 26, 14);
    moonKey.castShadow = true;
    moonKey.shadow.mapSize.set(2048, 2048);
    moonKey.shadow.camera.near = 2;
    moonKey.shadow.camera.far = 90;
    moonKey.shadow.camera.left = -24;
    moonKey.shadow.camera.right = 24;
    moonKey.shadow.camera.top = 48;
    moonKey.shadow.camera.bottom = -20;
    moonKey.shadow.bias = -0.0005;
    this.scene.add(moonKey);

    // Ego LED Headlight SpotLight casting forward illumination on asphalt
    const headSpot = new THREE.SpotLight(0xdff3ff, 12.0, 42, Math.PI * 0.24, 0.65, 1.3);
    headSpot.position.set(0, 0.85, -2.0);
    headSpot.target.position.set(0, 0, -24);
    this.scene.add(headSpot);
    this.scene.add(headSpot.target);
  }

  _buildHighway() {
    this.worldGroup = new THREE.Group();
    this.scene.add(this.worldGroup);

    const { map, roughnessMap } = createRoadTextures();
    this.roadTex = map;
    this.roadRoughTex = roughnessMap;

    // 3-lane highway surface (13.6m wide x 150m long)
    this.roadMat = new THREE.MeshStandardMaterial({
      map,
      roughnessMap,
      roughness: 0.72,
      metalness: 0.14
    });
    const road = new THREE.Mesh(new THREE.PlaneGeometry(13.6, 150), this.roadMat);
    road.rotation.x = -Math.PI / 2;
    road.position.set(0, 0, -45);
    road.receiveShadow = true;
    this.worldGroup.add(road);

    // Outer dark terrain ground plane
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(180, 180),
      new THREE.MeshStandardMaterial({ color: 0x060a12, roughness: 0.92, metalness: 0.05 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(0, -0.03, -45);
    this.worldGroup.add(ground);

    // Subtle LiDAR-mode architectural metric grid (visible mainly in LiDAR Cloud mode)
    this.metricGrid = new THREE.GridHelper(120, 60, 0x132b4d, 0x0a1628);
    this.metricGrid.position.set(0, 0.015, -30);
    this.metricGrid.visible = false;
    this.scene.add(this.metricGrid);

    // Sleek concrete highway guardrails on both shoulders
    const railMat = new THREE.MeshStandardMaterial({ color: 0x1d2635, roughness: 0.78, metalness: 0.22 });
    const railGeo = new THREE.BoxGeometry(0.45, 0.78, 150);
    [-6.95, 6.95].forEach(x => {
      const r = new THREE.Mesh(railGeo, railMat);
      r.position.set(x, 0.39, -45);
      r.receiveShadow = true;
      this.worldGroup.add(r);
    });

    // Minimalist scrolling highway streetlamps
    const poleMat = new THREE.MeshStandardMaterial({ color: 0x263142, metalness: 0.75, roughness: 0.35 });
    const lampMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.4, 1.75, 2.2), toneMapped: false });

    for (let i = 0; i < 10; i++) {
      const zPos = 18 - i * 14;
      [-7.4, 7.4].forEach(side => {
        const g = new THREE.Group();
        const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.1, 6.2, 8), poleMat);
        pole.position.y = 3.1;
        g.add(pole);

        const arm = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.08, 0.09), poleMat);
        arm.position.set(side < 0 ? 0.75 : -0.75, 6.1, 0);
        g.add(arm);

        const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.07, 0.18), lampMat);
        lamp.position.set(side < 0 ? 1.35 : -1.35, 6.02, 0);
        g.add(lamp);

        g.position.set(side, 0, zPos);
        this.worldGroup.add(g);
        this.streetLights.push(g);
      });
    }
  }

  _createWheel(radius = 0.35, width = 0.24) {
    const g = new THREE.Group();
    const tire = new THREE.Mesh(
      new THREE.CylinderGeometry(radius, radius, width, 20),
      new THREE.MeshStandardMaterial({ color: 0x101318, roughness: 0.85, metalness: 0.1 })
    );
    tire.rotation.z = Math.PI / 2;
    tire.castShadow = true;
    g.add(tire);

    const rim = new THREE.Mesh(
      new THREE.CylinderGeometry(radius * 0.64, radius * 0.64, width + 0.015, 14),
      new THREE.MeshStandardMaterial({ color: 0x9aa8bd, metalness: 0.9, roughness: 0.25 })
    );
    rim.rotation.z = Math.PI / 2;
    g.add(rim);

    // Spoke marker so rotation is visibly crisp
    const spoke = new THREE.Mesh(
      new THREE.BoxGeometry(width + 0.025, radius * 1.12, 0.055),
      new THREE.MeshStandardMaterial({ color: 0x2a3342, metalness: 0.85, roughness: 0.3 })
    );
    g.add(spoke);
    return g;
  }

  _buildEgoVehicle() {
    this.egoGroup = new THREE.Group();
    this.scene.add(this.egoGroup);

    const bodyPaint = new THREE.MeshPhysicalMaterial({
      color: 0x1c2534,
      metalness: 0.68,
      roughness: 0.22,
      clearcoat: 1.0,
      clearcoatRoughness: 0.08
    });
    const glassMat = new THREE.MeshPhysicalMaterial({
      color: 0x070d16,
      metalness: 0.15,
      roughness: 0.08,
      clearcoat: 1.0
    });
    const darkTrim = new THREE.MeshStandardMaterial({ color: 0x0d121a, roughness: 0.6, metalness: 0.3 });

    // Lower sculpted aerodynamic chassis
    const lower = new THREE.Mesh(new THREE.BoxGeometry(1.92, 0.56, 4.58), bodyPaint);
    lower.position.set(0, 0.58, 0);
    lower.castShadow = true;
    lower.receiveShadow = true;
    this.egoGroup.add(lower);

    // Front sloping hood wedge
    const hood = new THREE.Mesh(new THREE.BoxGeometry(1.86, 0.22, 1.35), bodyPaint);
    hood.position.set(0, 0.84, -1.52);
    hood.rotation.x = 0.14;
    hood.castShadow = true;
    this.egoGroup.add(hood);

    // Sleek fastback cabin greenhouse
    const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.58, 0.54, 2.15), glassMat);
    cabin.position.set(0, 1.12, 0.12);
    cabin.castShadow = true;
    this.egoGroup.add(cabin);

    // Roof cap
    const roof = new THREE.Mesh(new THREE.BoxGeometry(1.52, 0.08, 1.55), bodyPaint);
    roof.position.set(0, 1.40, 0.15);
    this.egoGroup.add(roof);

    // Front Bumper 77GHz Radar Grille Accent
    const radarMod = new THREE.Mesh(
      new THREE.BoxGeometry(0.42, 0.12, 0.06),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(0.75, 0.5, 1.8), toneMapped: false })
    );
    radarMod.position.set(0, 0.44, -2.30);
    this.egoGroup.add(radarMod);

    // Full-width HDR Front LED DRL Lightbar
    const drl = new THREE.Mesh(
      new THREE.BoxGeometry(1.76, 0.065, 0.06),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(1.6, 2.0, 2.5), toneMapped: false })
    );
    drl.position.set(0, 0.72, -2.29);
    this.egoGroup.add(drl);

    // Dynamic Rear LED Brake Lightbar (brightens intensely on AEB braking!)
    this.egoBrakeMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.1, 0.12, 0.12), toneMapped: false });
    const tailbar = new THREE.Mesh(new THREE.BoxGeometry(1.78, 0.08, 0.06), this.egoBrakeMat);
    tailbar.position.set(0, 0.78, 2.29);
    this.egoGroup.add(tailbar);

    // Roof Autonomous Sensor Pod + Spinning 64-Ch LiDAR Dome
    const podBase = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.36, 0.12, 20), darkTrim);
    podBase.position.set(0, 1.50, -0.05);
    this.egoGroup.add(podBase);

    this.lidarTurret = new THREE.Group();
    this.lidarTurret.position.set(0, 1.66, -0.05);
    const dome = new THREE.Mesh(new THREE.CylinderGeometry(0.20, 0.21, 0.19, 24), darkTrim);
    this.lidarTurret.add(dome);

    const laserRing = new THREE.Mesh(
      new THREE.CylinderGeometry(0.208, 0.208, 0.05, 24),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(0.25, 1.65, 2.4), toneMapped: false })
    );
    this.lidarTurret.add(laserRing);
    this.egoGroup.add(this.lidarTurret);

    // 4 Rotating Alloy Wheels
    [[-0.96, -1.42], [0.96, -1.42], [-0.96, 1.42], [0.96, 1.42]].forEach(([wx, wz]) => {
      const wheel = this._createWheel(0.36, 0.24);
      wheel.position.set(wx, 0.36, wz);
      this.egoGroup.add(wheel);
      this.egoWheels.push(wheel);
    });
  }

  _buildSensorVisuals() {
    // 1. RGB Camera FOV Frustum Wireframe (Amber)
    const camPts = [
      new THREE.Vector3(0, 1.24, -0.75), new THREE.Vector3(-11.5, 4.2, -32),
      new THREE.Vector3(0, 1.24, -0.75), new THREE.Vector3( 11.5, 4.2, -32),
      new THREE.Vector3(0, 1.24, -0.75), new THREE.Vector3(-11.5, 0.1, -32),
      new THREE.Vector3(0, 1.24, -0.75), new THREE.Vector3( 11.5, 0.1, -32),
      new THREE.Vector3(-11.5, 4.2, -32), new THREE.Vector3(11.5, 4.2, -32),
      new THREE.Vector3(-11.5, 0.1, -32), new THREE.Vector3(11.5, 0.1, -32),
      new THREE.Vector3(-11.5, 0.1, -32), new THREE.Vector3(-11.5, 4.2, -32),
      new THREE.Vector3( 11.5, 0.1, -32), new THREE.Vector3( 11.5, 4.2, -32)
    ];
    this.camFrustumMat = new THREE.LineBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.24 });
    this.camFrustum = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(camPts), this.camFrustumMat);
    this.scene.add(this.camFrustum);

    // 2. 77 GHz mmWave Radar Expanding Arcs (Violet)
    this.radarGroup = new THREE.Group();
    this.radarArcs = [];
    for (let i = 0; i < 4; i++) {
      const curve = new THREE.EllipseCurve(0, 0, 1, 1, -Math.PI * 0.72, -Math.PI * 0.28, false, 0);
      const pts2d = curve.getPoints(28);
      const pts3d = pts2d.map(p => new THREE.Vector3(p.x, 0.48, p.y));
      const mat = new THREE.LineBasicMaterial({ color: 0xa78bfa, transparent: true, opacity: 0.55 });
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts3d), mat);
      line.position.z = -2.3;
      this.radarGroup.add(line);
      this.radarArcs.push({ line, mat, phase: i / 4 });
    }
    this.scene.add(this.radarGroup);

    // 3. Rotating LiDAR Optical Sweep Beam Cone
    const sweepGeo = new THREE.ConeGeometry(16, 0.12, 32, 1, true, 0, Math.PI * 0.18);
    sweepGeo.rotateX(Math.PI / 2);
    this.lidarSweepMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.14,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    this.lidarSweep = new THREE.Mesh(sweepGeo, this.lidarSweepMat);
    this.lidarSweep.position.set(0, 1.66, -0.05);
    this.scene.add(this.lidarSweep);
  }

  _buildLidarPointCloud() {
    this.lidarPositions = new Float32Array(LIDAR_MAX_PTS * 3);
    this.lidarColors = new Float32Array(LIDAR_MAX_PTS * 3);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.lidarPositions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(this.lidarColors, 3));

    this.lidarPointsMat = new THREE.PointsMaterial({
      size: 0.24,
      map: createLidarDotTexture(),
      vertexColors: true,
      transparent: true,
      opacity: 0.92,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    this.lidarPoints = new THREE.Points(geo, this.lidarPointsMat);
    this.lidarPoints.frustumCulled = false;
    this.scene.add(this.lidarPoints);
  }

  _buildTrajectoryCorridor() {
    // Glowing planned path ribbon directly in front of the Ego vehicle
    const geo = new THREE.PlaneGeometry(1.96, 1.0);
    geo.rotateX(-Math.PI / 2);
    this.corridorMat = new THREE.MeshBasicMaterial({
      color: 0x34d399,
      transparent: true,
      opacity: 0.18,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    this.corridorMesh = new THREE.Mesh(geo, this.corridorMat);
    this.corridorMesh.position.set(0, 0.04, -12);
    this.scene.add(this.corridorMesh);

    const edgeGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(1.98, 0.02, 1.0));
    this.corridorEdgeMat = new THREE.LineBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.65 });
    this.corridorEdge = new THREE.LineSegments(edgeGeo, this.corridorEdgeMat);
    this.corridorEdge.position.set(0, 0.05, -12);
    this.scene.add(this.corridorEdge);
  }

  _buildActorMesh(actor) {
    const g = new THREE.Group();
    g.userData = { wheels: [], brakeMat: null, blinkerMat: null, limbs: null };

    if (actor.type === "truck") {
      const cabMat = new THREE.MeshPhysicalMaterial({ color: 0x2a3648, metalness: 0.55, roughness: 0.3, clearcoat: 0.8 });
      const trailerMat = new THREE.MeshStandardMaterial({ color: 0xb9c5d6, metalness: 0.42, roughness: 0.45 });
      const glassMat = new THREE.MeshPhysicalMaterial({ color: 0x070d16, roughness: 0.1, clearcoat: 1.0 });

      // Box trailer
      const trailer = new THREE.Mesh(new THREE.BoxGeometry(2.42, 2.75, 6.85), trailerMat);
      trailer.position.set(0, 2.02, 1.0);
      trailer.castShadow = true;
      trailer.receiveShadow = true;
      g.add(trailer);

      // Tractor cab (at front -Z)
      const cab = new THREE.Mesh(new THREE.BoxGeometry(2.34, 2.25, 2.15), cabMat);
      cab.position.set(0, 1.65, -3.52);
      cab.castShadow = true;
      g.add(cab);

      const wind = new THREE.Mesh(new THREE.BoxGeometry(2.12, 0.85, 0.12), glassMat);
      wind.position.set(0, 2.15, -4.58);
      g.add(wind);

      // Red rear marker / brake lights
      const brakeMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.4, 0.15, 0.15), toneMapped: false });
      const tail = new THREE.Mesh(new THREE.BoxGeometry(2.1, 0.12, 0.08), brakeMat);
      tail.position.set(0, 0.92, 4.44);
      g.add(tail);
      g.userData.brakeMat = brakeMat;

      // 6 Heavy Truck Wheels
      [-3.5, 0.8, 3.2].forEach(wz => {
        [-1.18, 1.18].forEach(wx => {
          const w = this._createWheel(0.48, 0.28);
          w.position.set(wx, 0.48, wz);
          g.add(w);
          g.userData.wheels.push(w);
        });
      });
    } else if (actor.type === "sedan" || actor.type === "suv") {
      const isSuv = actor.type === "suv";
      const paintHex = isSuv ? 0x3b4d6b : 0x5c6f8a;
      const paint = new THREE.MeshPhysicalMaterial({
        color: paintHex,
        metalness: 0.62,
        roughness: 0.24,
        clearcoat: 1.0
      });
      const glass = new THREE.MeshPhysicalMaterial({ color: 0x070c14, roughness: 0.1, clearcoat: 1.0 });

      const body = new THREE.Mesh(new THREE.BoxGeometry(actor.w, isSuv ? 0.72 : 0.56, actor.l), paint);
      body.position.set(0, isSuv ? 0.68 : 0.58, 0);
      body.castShadow = true;
      body.receiveShadow = true;
      g.add(body);

      const cab = new THREE.Mesh(
        new THREE.BoxGeometry(actor.w * 0.84, isSuv ? 0.62 : 0.52, isSuv ? 2.55 : 2.05),
        glass
      );
      cab.position.set(0, isSuv ? 1.28 : 1.08, 0.15);
      cab.castShadow = true;
      g.add(cab);

      const brakeMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.2, 0.12, 0.12), toneMapped: false });
      const tail = new THREE.Mesh(new THREE.BoxGeometry(actor.w * 0.9, 0.08, 0.06), brakeMat);
      tail.position.set(0, isSuv ? 0.88 : 0.76, actor.l * 0.5 + 0.01);
      g.add(tail);
      g.userData.brakeMat = brakeMat;

      // Turn-signal amber indicator (right rear/front for Cut-In SUV)
      const blinkMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.2, 0.12, 0.0), toneMapped: false });
      const blinkR = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.09, 0.08), blinkMat);
      blinkR.position.set(actor.w * 0.42, isSuv ? 0.86 : 0.75, actor.l * 0.5 + 0.02);
      g.add(blinkR);
      g.userData.blinkerMat = blinkMat;

      [[-actor.w * 0.49, -1.42], [actor.w * 0.49, -1.42], [-actor.w * 0.49, 1.42], [actor.w * 0.49, 1.42]].forEach(([wx, wz]) => {
        const w = this._createWheel(isSuv ? 0.39 : 0.35, 0.23);
        w.position.set(wx, isSuv ? 0.39 : 0.35, wz);
        g.add(w);
        g.userData.wheels.push(w);
      });
    } else if (actor.type === "pedestrian") {
      // Clean articulated architectural humanoid figure
      const coatMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.65, metalness: 0.05 });
      const pantsMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.78 });
      const headMat = new THREE.MeshStandardMaterial({ color: 0xd6b59c, roughness: 0.6 });

      const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.19, 0.44, 8, 12), coatMat);
      torso.position.y = 1.15;
      torso.castShadow = true;
      g.add(torso);

      const head = new THREE.Mesh(new THREE.SphereGeometry(0.135, 14, 14), headMat);
      head.position.y = 1.62;
      head.castShadow = true;
      g.add(head);

      const makeLimb = (rad, len, mat, ox, oy) => {
        const pivot = new THREE.Group();
        pivot.position.set(ox, oy, 0);
        const mesh = new THREE.Mesh(new THREE.CapsuleGeometry(rad, len, 6, 8), mat);
        mesh.position.y = -len * 0.5;
        mesh.castShadow = true;
        pivot.add(mesh);
        g.add(pivot);
        return pivot;
      };

      const legL = makeLimb(0.075, 0.68, pantsMat, -0.1, 0.82);
      const legR = makeLimb(0.075, 0.68, pantsMat,  0.1, 0.82);
      const armL = makeLimb(0.055, 0.48, coatMat, -0.25, 1.38);
      const armR = makeLimb(0.055, 0.48, coatMat,  0.25, 1.38);
      g.userData.limbs = { legL, legR, armL, armR };
      // Face left (-X) across the road
      g.rotation.y = Math.PI / 2;
    }

    return g;
  }

  _syncActors(actors, dt, elapsed) {
    const activeIds = new Set(actors.map(a => a.id));

    // Remove actors from previous scenarios
    for (const [id, mesh] of this.actorMeshes.entries()) {
      if (!activeIds.has(id)) {
        this.worldGroup.remove(mesh);
        this.actorMeshes.delete(id);
        const bb = this.bboxMeshes.get(id);
        if (bb) {
          this.scene.remove(bb);
          this.bboxMeshes.delete(id);
        }
      }
    }

    // Add or update active actors & their 3D Perception Bounding Boxes
    actors.forEach(a => {
      let mesh = this.actorMeshes.get(a.id);
      if (!mesh) {
        mesh = this._buildActorMesh(a);
        this.worldGroup.add(mesh);
        this.actorMeshes.set(a.id, mesh);

        const boxGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(a.w + 0.16, a.h + 0.14, a.l + 0.16));
        const boxMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.9 });
        const box = new THREE.LineSegments(boxGeo, boxMat);
        this.scene.add(box);
        this.bboxMeshes.set(a.id, box);
      }

      mesh.position.set(a.x, 0, -a.dist);

      // Spin actor wheels
      if (mesh.userData.wheels.length) {
        const spin = (a.speed * dt) / 0.36;
        mesh.userData.wheels.forEach(w => { w.rotation.x -= spin; });
      }
      // Animate pedestrian walking stride
      if (mesh.userData.limbs) {
        const s = Math.sin(a.walkPhase) * 0.55;
        mesh.userData.limbs.legL.rotation.x = s;
        mesh.userData.limbs.legR.rotation.x = -s;
        mesh.userData.limbs.armL.rotation.x = -s * 0.7;
        mesh.userData.limbs.armR.rotation.x = s * 0.7;
      }
      // Dynamic brake light & blinker glow
      if (mesh.userData.brakeMat) {
        mesh.userData.brakeMat.color.setRGB(a.braking ? 2.4 : 0.95, 0.12, 0.12);
      }
      if (mesh.userData.blinkerMat) {
        const blinkOn = a.blinker && Math.sin(elapsed * 11) > 0;
        mesh.userData.blinkerMat.color.setRGB(blinkOn ? 2.2 : 0.15, blinkOn ? 1.2 : 0.08, 0.02);
      }

      // 3D Perception Bounding Box visibility & risk color
      const box = this.bboxMeshes.get(a.id);
      if (box) {
        box.position.set(a.x, a.h * 0.5 + 0.05, -a.dist);
        const showBox = this.viewMode !== "camera" && a.fusedConf >= 0.25;
        box.visible = showBox;
        if (showBox) {
          box.material.opacity = Math.min(0.95, 0.25 + a.fusedConf * 0.75);
          if (a.isThreat && a.ttc < 2.8) {
            box.material.color.setHex(0xf87171); // Critical Red
          } else if (a.isThreat) {
            box.material.color.setHex(0xfbbf24); // Caution Amber
          } else if (a.type === "pedestrian") {
            box.material.color.setHex(0x34d399); // VRU Emerald
          } else {
            box.material.color.setHex(0x38bdf8); // Tracked Vehicle Cyan
          }
        }
      }
    });
  }

  _updateLidarCloud(actors, sim, elapsed) {
    const showLidar = sim.sensors.lidar && this.viewMode !== "camera";
    this.lidarPoints.visible = showLidar;
    if (!showLidar) {
      this.visiblePointsCount = 0;
      return;
    }

    const pos = this.lidarPositions;
    const col = this.lidarColors;
    const fog = sim.fogNorm;
    let idx = 0;

    const pushPt = (x, y, z, r, g, b) => {
      if (idx >= LIDAR_MAX_PTS) return;
      const o = idx * 3;
      pos[o] = x; pos[o + 1] = y; pos[o + 2] = z;
      col[o] = r; col[o + 1] = g; col[o + 2] = b;
      idx++;
    };

    // 1. Concentric 64-Channel LiDAR Ground Scan Rings with TRUE Vehicle Occlusion Shadows!
    const rings = 26;
    const ptsPerRing = 84;
    const spinOffset = elapsed * 1.6;
    for (let ri = 0; ri < rings; ri++) {
      const dist = 3.2 + Math.pow(ri / (rings - 1), 1.48) * 48.0;
      // Slight range attenuation in dense fog
      if (fog > 0.45 && dist > 56 - fog * 22) continue;

      for (let pi = 0; pi < ptsPerRing; pi++) {
        const ang = (pi / ptsPerRing) * Math.PI * 2 + spinOffset * 0.08;
        const x = Math.cos(ang) * dist;
        const z = Math.sin(ang) * dist;
        if (Math.abs(x) > 7.2 || z > 18 || z < -54) continue;

        // Cast shadow behind vehicles so dark occlusion wedges appear realistically!
        if (rayBlocked2D(x, z, actors, null)) continue;

        const nearRoadEdge = Math.abs(x) > 6.65;
        const y = nearRoadEdge ? 0.35 + ((pi * 7) % 5) * 0.09 : 0.05;
        const fade = Math.max(0.18, 1 - dist / 56);
        if (nearRoadEdge) {
          pushPt(x, y, z, 0.25 * fade, 0.65 * fade, 0.95 * fade);
        } else {
          pushPt(x, y, z, 0.12 * fade, 0.45 * fade, 0.72 * fade);
        }
      }
    }

    // 2. High-Density 3D Surface Returns on Visible Actors (Truck, Cars, Pedestrian)
    actors.forEach(a => {
      if (a.dist < -12 || a.dist > 56) return;
      const samples = a.type === "truck" ? 720 : a.type === "pedestrian" ? 280 : 420;
      const az = -a.dist;
      const isCrit = a.isThreat && a.ttc < 2.8;

      for (let s = 0; s < samples; s++) {
        // Deterministic Fibonacci-like point distribution on visible actor faces
        const u = ((s * 0.6180339887) % 1) - 0.5;
        const v = (s / samples);
        const face = s % 3;

        let px, py, pz;
        if (face === 0) {
          // Rear face facing Ego (+Z side of actor)
          px = a.x + u * a.w;
          py = 0.12 + v * (a.h - 0.12);
          pz = az + a.l * 0.5;
        } else if (face === 1) {
          // Side face facing Ego centerline
          const sideSign = a.x >= 0 ? -1 : 1;
          px = a.x + sideSign * a.w * 0.5;
          py = 0.12 + v * (a.h - 0.12);
          pz = az + u * a.l;
        } else {
          // Roof / top surface
          px = a.x + u * a.w;
          py = a.h;
          pz = az + (((s * 0.381966) % 1) - 0.5) * a.l;
        }

        // True occlusion check per laser point!
        if (rayBlocked2D(px, pz, actors, a.id)) continue;

        if (isCrit) {
          pushPt(px, py, pz, 1.0, 0.28, 0.28);
        } else if (a.type === "pedestrian") {
          pushPt(px, py, pz, 0.28, 1.0, 0.68);
        } else {
          // Height-coded LiDAR intensity (cyan base -> emerald/amber top)
          const hNorm = Math.min(1, py / 3.2);
          pushPt(px, py, pz, 0.25 + hNorm * 0.72, 0.85, 1.0 - hNorm * 0.45);
        }
      }
    });

    // 3. Realistic LiDAR Fog Backscatter Noise when Fog > 35%
    if (fog > 0.35) {
      const noisePts = Math.round((fog - 0.35) * 420);
      for (let n = 0; n < noisePts; n++) {
        const ang = ((n * 2.39996 + elapsed * 3.1) % (Math.PI * 2));
        const r = 2.0 + ((n * 0.7548) % 1) * 14.0;
        const y = 0.4 + ((n * 0.5698) % 1) * 2.2;
        pushPt(Math.cos(ang) * r, y, Math.sin(ang) * r, 0.45, 0.68, 0.85);
      }
    }

    this.visiblePointsCount = idx;
    this.lidarPoints.geometry.setDrawRange(0, idx);
    this.lidarPoints.geometry.attributes.position.needsUpdate = true;
    this.lidarPoints.geometry.attributes.color.needsUpdate = true;
  }

  setViewMode(mode) {
    this.viewMode = mode;
    const isLidarOnly = mode === "lidar";
    const isCamRaw = mode === "camera";

    // In LiDAR Cloud mode, fade physical world into minimalist dark stealth mode so glowing points dominate!
    this.worldGroup.visible = !isLidarOnly;
    this.metricGrid.visible = isLidarOnly;
    this.lidarPointsMat.size = isLidarOnly ? 0.30 : 0.20;
    this.lidarPointsMat.opacity = isLidarOnly ? 0.96 : 0.68;

    this.corridorMesh.visible = !isCamRaw;
    this.corridorEdge.visible = !isCamRaw;
  }

  setCameraPreset(preset) {
    this.camPreset = preset;
    if (preset === "chase") {
      this.camGoalPos.set(7.4, 5.2, 10.6);
      this.camGoalTarget.set(0, 1.0, -11.0);
    } else if (preset === "bev") {
      this.camGoalPos.set(0.0, 36.0, -7.5);
      this.camGoalTarget.set(0, 0.0, -13.5);
    } else if (preset === "hood") {
      this.camGoalPos.set(0.0, 1.72, -0.55);
      this.camGoalTarget.set(0, 1.05, -24.0);
    }
    this.camLerp = true;
  }

  update(dt, elapsed, sim, tel) {
    // 1. Camera smooth preset fly-to
    if (this.camLerp) {
      this.camera.position.lerp(this.camGoalPos, Math.min(1, dt * 5.2));
      this.controls.target.lerp(this.camGoalTarget, Math.min(1, dt * 5.2));
      if (this.camera.position.distanceTo(this.camGoalPos) < 0.05) {
        this.camLerp = false;
      }
    }

    // 2. Volumetric Fog Density (Extra intense in Camera Raw mode; clean in LiDAR Cloud mode)
    const fog = sim.fogNorm;
    if (this.viewMode === "lidar") {
      this.scene.fog.density = 0.008;
      this.scene.fog.color.setHex(0x04070e);
      this.scene.background.setHex(0x04070e);
    } else {
      const mult = this.viewMode === "camera" ? 0.072 : 0.044;
      this.scene.fog.density = 0.008 + fog * fog * mult;
      const fogGray = 0.04 + fog * 0.28;
      this.scene.fog.color.setRGB(fogGray * 0.85, fogGray * 0.95, fogGray * 1.12);
      this.scene.background.copy(this.scene.fog.color);
    }

    // 3. Scroll highway asphalt & streetlights strictly matched to Ego speed
    const tileLen = 150 / 9;
    this.roadTex.offset.y = (tel.roadDistance / tileLen) % 1;
    this.roadRoughTex.offset.y = this.roadTex.offset.y;

    for (let i = 0; i < this.streetLights.length; i++) {
      const sl = this.streetLights[i];
      sl.position.z += tel.egoSpeedMs * dt;
      if (sl.position.z > 24) sl.position.z -= 140;
    }

    // 4. Spin Ego wheels, roof LiDAR turret, and update AEB brake lights
    const wheelSpin = (tel.egoSpeedMs * dt) / 0.36;
    this.egoWheels.forEach(w => { w.rotation.x -= wheelSpin; });
    this.lidarTurret.rotation.y += dt * (sim.sensors.lidar ? 12.0 : 0.4);
    this.egoBrakeMat.color.setRGB(tel.aebActive || tel.aebDecel > 1.5 ? 2.8 : 0.9, 0.1, 0.1);

    // 5. Update Sensor Overlays (Camera FOV, Radar Arcs, LiDAR Sweep)
    const overlaysAllowed = this.viewMode !== "camera";
    this.camFrustum.visible = overlaysAllowed && sim.sensors.camera;
    this.camFrustumMat.opacity = Math.max(0.05, 0.26 * (1 - fog * 0.75));

    this.lidarSweep.visible = overlaysAllowed && sim.sensors.lidar;
    if (this.lidarSweep.visible) {
      this.lidarSweep.rotation.y = elapsed * 4.5;
    }

    this.radarGroup.visible = overlaysAllowed && sim.sensors.radar;
    if (this.radarGroup.visible) {
      this.radarArcs.forEach(arc => {
        const p = (elapsed * 0.65 + arc.phase) % 1;
        const radius = 2.0 + p * 34.0;
        arc.line.scale.set(radius, 1, radius);
        arc.mat.opacity = (1 - p) * 0.62;
      });
    }

    // 6. Update Planned Path / Braking Distance Corridor
    const stopLen = Math.max(3.5, Math.min(34, 4.0 + (tel.egoSpeedMs * tel.egoSpeedMs) / 14.5));
    this.corridorMesh.scale.set(1, 1, stopLen);
    this.corridorMesh.position.z = -2.35 - stopLen * 0.5;
    this.corridorEdge.scale.set(1, 1, stopLen);
    this.corridorEdge.position.z = -2.35 - stopLen * 0.5;

    const pathCol = tel.aebActive || tel.collisionFlag ? 0xf87171 : tel.fcwActive ? 0xfbbf24 : 0x34d399;
    this.corridorMat.color.setHex(pathCol);
    this.corridorEdgeMat.color.setHex(pathCol);

    // 7. Sync Dynamic Actors, Bounding Boxes & True Occluded LiDAR Point Cloud
    this._syncActors(tel.actors, dt, elapsed);
    this._updateLidarCloud(tel.actors, sim, elapsed);

    this.controls.update();
    this.composer.render();
  }

  _bindResize() {
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