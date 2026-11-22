/**
 * ThinkModule Lab — Photorealistic WebGL PBR 3D LiDAR Hardware X-Ray Scene
 * Features:
 *   - 5 exploded mechanical & optical layers (CNC heatsink, FPGA/TDC PCB, 12-pole BLDC motor,
 *     Emitter + 45° dielectric mirror + Aspheric glass RX lens + Gold APD detector, IR smoked dome)
 *   - Live Time-of-Flight slow-motion photon pulse propagation (Emitter -> Mirror -> Target -> Lens -> APD)
 *   - Telescopic movable 3D Target Plate with realistic surface materials (85% Sign / 42% Car / 8% Tire)
 *   - 360° Live Scan mode with multi-channel laser fan & real-time 3D point cloud returns
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

function createCircularPcbTexture() {
  const size = 1024;
  const c = document.createElement("canvas");
  c.width = size; c.height = size;
  const ctx = c.getContext("2d");

  ctx.fillStyle = "#081517";
  ctx.fillRect(0, 0, size, size);

  const cx = 512, cy = 512;
  // Concentric high-speed differential clock rings
  for (let r = 130; r <= 470; r += 26) {
    ctx.strokeStyle = r % 52 === 0 ? "rgba(212, 168, 78, 0.42)" : "rgba(48, 122, 134, 0.48)";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Radial signal traces from central FPGA to perimeter connectors
  for (let i = 0; i < 48; i++) {
    const a = (i / 48) * Math.PI * 2;
    ctx.strokeStyle = i % 4 === 0 ? "rgba(212, 168, 78, 0.52)" : "rgba(56, 189, 248, 0.35)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * 125, cy + Math.sin(a) * 125);
    ctx.lineTo(cx + Math.cos(a) * 465, cy + Math.sin(a) * 465);
    ctx.stroke();

    // Gold plated micro-via at outer rim
    ctx.fillStyle = "#cfa049";
    ctx.beginPath();
    ctx.arc(cx + Math.cos(a) * 475, cy + Math.sin(a) * 475, 5, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "rgba(220, 234, 248, 0.55)";
  ctx.font = "700 22px ui-monospace, Menlo, monospace";
  ctx.textAlign = "center";
  ctx.fillText("TDC <50ps // FPGA DSP REV 3.1", cx, 118);

  const tex = new THREE.CanvasTexture(c);
  tex.anisotropy = 8;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function createTargetStripeTexture() {
  const c = document.createElement("canvas");
  c.width = 512; c.height = 512;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, 512, 512);

  // Clean calibration crosshair target for visual precision
  ctx.strokeStyle = "rgba(15, 23, 42, 0.25)";
  ctx.lineWidth = 4;
  ctx.strokeRect(24, 24, 464, 464);

  ctx.strokeStyle = "rgba(56, 189, 248, 0.45)";
  ctx.lineWidth = 3;
  [85, 165].forEach(r => {
    ctx.beginPath();
    ctx.arc(256, 256, r, 0, Math.PI * 2);
    ctx.stroke();
  });
  ctx.beginPath();
  ctx.moveTo(256, 48); ctx.lineTo(256, 464);
  ctx.moveTo(48, 256); ctx.lineTo(464, 256);
  ctx.stroke();

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export class LidarHardwareScene {
  constructor(container) {
    this.container = container;
    this.simMode = "tof"; // "tof" | "scan"
    this.explodeTarget = 0.45;
    this.explodeCurrent = 0.45;
    this.autoOrbit = true;

    this.camLerp = false;
    this.camGoalPos = new THREE.Vector3(5.6, 3.6, 6.9);
    this.camGoalTarget = new THREE.Vector3(1.5, 0.35, 0);

    this.tofPhase = 0; // 0..1 continuous photon round-trip cycle
    this.rotorAngle = 0;
    this.layers = [];
    this.scanPointsData = [];

    this._initRenderer();
    this._initLightsAndBench();
    this._buildLidarLayers();
    this._buildOpticalBenchTarget();
    this._buildPhotonVisuals();
    this._buildScanArena();
    this._bindResize();
  }

  _initRenderer() {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x04070d, 0.026);

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
    this.renderer.toneMappingExposure = 1.1;
    this.container.appendChild(this.renderer.domElement);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.9;
    pmrem.dispose();

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(w, h), 0.42, 0.42, 0.84);
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(new OutputPass());

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.minDistance = 2.0;
    this.controls.maxDistance = 18.0;
    this.controls.maxPolarAngle = Math.PI * 0.95;
    this.controls.target.copy(this.camGoalTarget);

    this.controls.addEventListener("start", () => {
      this.camLerp = false;
    });
  }

  _initLightsAndBench() {
    const key = new THREE.DirectionalLight(0xf2f7ff, 2.1);
    key.position.set(6, 11, 7);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -8;
    key.shadow.camera.right = 10;
    key.shadow.camera.top = 8;
    key.shadow.camera.bottom = -8;
    key.shadow.bias = -0.0004;
    this.scene.add(key);

    const rim = new THREE.DirectionalLight(0x38bdf8, 1.1);
    rim.position.set(-7, 4, -6);
    this.scene.add(rim);

    const fill = new THREE.PointLight(0xffb478, 1.1, 16);
    fill.position.set(4, -1.2, -4);
    this.scene.add(fill);

    // Studio floor + precision optical bench grid
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(46, 46),
      new THREE.MeshStandardMaterial({ color: 0x050912, roughness: 0.72, metalness: 0.2 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -2.35;
    floor.receiveShadow = true;
    this.scene.add(floor);

    const grid = new THREE.GridHelper(28, 42, 0x162640, 0x0a1322);
    grid.position.y = -2.34;
    this.scene.add(grid);

    this.lidarRoot = new THREE.Group();
    this.scene.add(this.lidarRoot);
  }

  _buildLidarLayers() {
    this.mats = {
      cncDark: new THREE.MeshStandardMaterial({ color: 0x1b222e, metalness: 0.86, roughness: 0.30 }),
      titanium: new THREE.MeshPhysicalMaterial({ color: 0x263040, metalness: 0.82, roughness: 0.24, clearcoat: 0.4 }),
      copper: new THREE.MeshStandardMaterial({ color: 0xc87a48, metalness: 0.94, roughness: 0.24 }),
      gold: new THREE.MeshStandardMaterial({ color: 0xdca94a, metalness: 0.96, roughness: 0.18 }),
      steel: new THREE.MeshStandardMaterial({ color: 0xc8d4e4, metalness: 0.95, roughness: 0.18 }),
      mirror: new THREE.MeshPhysicalMaterial({
        color: 0xe6f6ff,
        metalness: 0.98,
        roughness: 0.04,
        clearcoat: 1.0,
        iridescence: 0.75
      }),
      lensGlass: new THREE.MeshPhysicalMaterial({
        color: 0x9be2ff,
        metalness: 0.05,
        roughness: 0.04,
        transmission: 0.86,
        ior: 1.52,
        thickness: 0.45,
        transparent: true,
        opacity: 0.88
      }),
      irDome: new THREE.MeshPhysicalMaterial({
        color: 0x0b1624,
        metalness: 0.12,
        roughness: 0.08,
        clearcoat: 1.0,
        transparent: true,
        opacity: 0.28,
        depthWrite: false,
        side: THREE.DoubleSide
      })
    };

    const addLayer = (baseY, explodeY) => {
      const g = new THREE.Group();
      g.userData = { baseY, explodeY };
      this.lidarRoot.add(g);
      this.layers.push(g);
      return g;
    };

    // =========================================================================
    // LAYER 0: CNC ALUMINUM HEATSINK BASE & M12 AUTOMOTIVE CONNECTOR
    // =========================================================================
    const L0 = addLayer(-0.88, -1.95);
    const baseCore = new THREE.Mesh(new THREE.CylinderGeometry(1.48, 1.56, 0.36, 40), this.mats.cncDark);
    baseCore.castShadow = true;
    baseCore.receiveShadow = true;
    L0.add(baseCore);

    // 24 Radial Precision Cooling Fins
    const finGeo = new THREE.BoxGeometry(0.05, 0.30, 0.34);
    for (let i = 0; i < 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      const fin = new THREE.Mesh(finGeo, this.mats.titanium);
      fin.position.set(Math.cos(a) * 1.50, 0, Math.sin(a) * 1.50);
      fin.rotation.y = -a + Math.PI / 2;
      fin.castShadow = true;
      L0.add(fin);
    }

    // Copper base thermal spreader ring
    const cuRing = new THREE.Mesh(new THREE.CylinderGeometry(1.22, 1.22, 0.04, 36), this.mats.copper);
    cuRing.position.y = 0.19;
    L0.add(cuRing);

    // M12 Automotive Ethernet/Power Bulkhead Plug (-Z side)
    const conn = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.42, 16), this.mats.steel);
    conn.rotation.x = Math.PI / 2;
    conn.position.set(0, -0.02, -1.68);
    L0.add(conn);

    // =========================================================================
    // LAYER 1: FPGA & PICOSECOND TDC SIGNAL PROCESSING PCB
    // =========================================================================
    const L1 = addLayer(-0.56, -1.12);
    const pcbTex = createCircularPcbTexture();
    const pcbDisc = new THREE.Mesh(
      new THREE.CylinderGeometry(1.36, 1.36, 0.07, 48),
      new THREE.MeshStandardMaterial({ map: pcbTex, roughness: 0.45, metalness: 0.32 })
    );
    pcbDisc.castShadow = true;
    pcbDisc.receiveShadow = true;
    L1.add(pcbDisc);

    // Central Xilinx/Altera-style FPGA BGA Die Package
    const fpgaPkg = new THREE.Mesh(new THREE.BoxGeometry(0.76, 0.10, 0.76), this.mats.cncDark);
    fpgaPkg.position.y = 0.08;
    fpgaPkg.castShadow = true;
    L1.add(fpgaPkg);

    const fpgaLid = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.02, 0.56), this.mats.steel);
    fpgaLid.position.y = 0.14;
    L1.add(fpgaLid);

    // TDC (Time-to-Digital Converter) Precision Timing Chip + Gold Shield
    const tdcChip = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.09, 0.28), this.mats.gold);
    tdcChip.position.set(0.68, 0.075, 0.35);
    L1.add(tdcChip);

    // Pulsing FPGA Sync LEDs
    this.fpgaLedMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.2, 1.8, 1.4), toneMapped: false });
    [[-0.85, 0.3], [-0.85, -0.3], [0.85, -0.35]].forEach(([x, z]) => {
      const led = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.05, 0.08), this.fpgaLedMat);
      led.position.set(x, 0.06, z);
      L1.add(led);
    });

    // =========================================================================
    // LAYER 2: 12-POLE BRUSHLESS MOTOR & OPTICAL AZIMUTH ENCODER
    // =========================================================================
    const L2 = addLayer(-0.22, -0.32);

    // Motor housing ring
    const motorPlate = new THREE.Mesh(new THREE.CylinderGeometry(1.18, 1.24, 0.08, 40), this.mats.titanium);
    motorPlate.castShadow = true;
    L2.add(motorPlate);

    // 12 Copper-Wound Stator Coils
    const coilGeo = new THREE.CylinderGeometry(0.11, 0.11, 0.22, 14);
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      const coil = new THREE.Mesh(coilGeo, this.mats.copper);
      coil.position.set(Math.cos(a) * 0.62, 0.14, Math.sin(a) * 0.62);
      coil.castShadow = true;
      L2.add(coil);
    }

    // Slotted Optical Encoder Ring (spins with rotor to measure exact azimuth angle θ)
    this.encoderGroup = new THREE.Group();
    this.encoderGroup.position.y = 0.08;
    const encDisc = new THREE.Mesh(new THREE.RingGeometry(0.84, 1.05, 48), this.mats.steel);
    encDisc.rotation.x = -Math.PI / 2;
    this.encoderGroup.add(encDisc);

    // 24 Dark Optical Encoder Tick Slots
    const tickGeo = new THREE.BoxGeometry(0.03, 0.015, 0.16);
    for (let i = 0; i < 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      const tick = new THREE.Mesh(tickGeo, this.mats.cncDark);
      tick.position.set(Math.cos(a) * 0.94, 0.008, Math.sin(a) * 0.94);
      tick.rotation.y = -a;
      this.encoderGroup.add(tick);
    }
    L2.add(this.encoderGroup);

    // Central Stainless Rotor Shaft
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.44, 24), this.mats.steel);
    shaft.position.y = 0.18;
    L2.add(shaft);

    // =========================================================================
    // LAYER 3: OPTO-MECHANICAL CORE (LASER EMITTER, 45° MIRROR, RX LENS & APD)
    // =========================================================================
    const L3 = addLayer(0.32, 0.56);

    // Rotating Optical Turret Group (spins in 360° Live Scan mode; locks to 0° in ToF Slow-Mo!)
    this.opticalTurret = new THREE.Group();
    L3.add(this.opticalTurret);

    // Turret structural deck plate
    const deck = new THREE.Mesh(new THREE.CylinderGeometry(1.16, 1.16, 0.07, 40), this.mats.cncDark);
    deck.position.y = -0.34;
    deck.castShadow = true;
    this.opticalTurret.add(deck);

    // 1) LASER EMITTER DIODE & GOLD COLLIMATOR (Center vertical shaft pointing UP to 45° Mirror!)
    const emitterHousing = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.26, 0.24, 20), this.mats.gold);
    emitterHousing.position.set(0, -0.20, 0);
    emitterHousing.castShadow = true;
    this.opticalTurret.add(emitterHousing);

    this.emitterApertureMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.3, 1.6, 2.4), toneMapped: false });
    const aperture = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 0.04, 16), this.emitterApertureMat);
    aperture.position.set(0, -0.07, 0);
    this.opticalTurret.add(aperture);

    // 2) PRECISION 45° DEFLECTION MIRROR (Deflects vertical +Y laser 90° horizontally along +X!)
    const mirrorMount = new THREE.Group();
    mirrorMount.position.set(0, 0.22, 0);

    const mirrorBacking = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.07, 0.50), this.mats.titanium);
    mirrorBacking.rotation.z = Math.PI / 4; // Normal faces (+X, -Y) so +Y beam reflects cleanly to +X!
    mirrorBacking.castShadow = true;
    mirrorMount.add(mirrorBacking);

    const mirrorFace = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.02, 0.46), this.mats.mirror);
    mirrorFace.rotation.z = Math.PI / 4;
    mirrorFace.position.set(0.032, -0.032, 0);
    mirrorMount.add(mirrorFace);

    // Support struts holding mirror mount
    [-0.27, 0.27].forEach(z => {
      const strut = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.58, 0.05), this.mats.titanium);
      strut.position.set(-0.20, -0.05, z);
      mirrorMount.add(strut);
    });
    this.opticalTurret.add(mirrorMount);

    // 3) RECEIVING ASPHERIC GLASS OBJECTIVE LENS (Mounted facing +X at z = +0.52)
    const rxGroup = new THREE.Group();
    rxGroup.position.set(0.68, 0.02, 0.52);

    const lensBarrel = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, 0.18, 28, 1, true), this.mats.titanium);
    lensBarrel.rotation.z = Math.PI / 2;
    rxGroup.add(lensBarrel);

    const asphericLens = new THREE.Mesh(new THREE.SphereGeometry(0.31, 24, 16), this.mats.lensGlass);
    asphericLens.scale.set(0.38, 1.0, 1.0);
    rxGroup.add(asphericLens);

    // Lens retaining ring
    const lensRing = new THREE.Mesh(new THREE.TorusGeometry(0.32, 0.028, 12, 32), this.mats.gold);
    lensRing.rotation.y = Math.PI / 2;
    lensRing.position.x = 0.08;
    rxGroup.add(lensRing);
    this.opticalTurret.add(rxGroup);

    // 4) AVALANCHE PHOTODIODE (APD) DETECTOR CHIP (Directly behind RX lens focal point!)
    const apdCarrier = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.36, 0.36), this.mats.gold);
    apdCarrier.position.set(0.06, 0.02, 0.52);
    apdCarrier.castShadow = true;
    this.opticalTurret.add(apdCarrier);

    this.apdSensorMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.12, 0.35, 0.45), toneMapped: false });
    const apdActive = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.22, 0.22), this.apdSensorMat);
    apdActive.position.set(0.14, 0.02, 0.52);
    this.opticalTurret.add(apdActive);

    // =========================================================================
    // LAYER 4: IR-TRANSPARENT SMOKED OPTICAL WINDOW & CNC TOP CAP
    // =========================================================================
    const L4 = addLayer(0.42, 1.88);

    const glassCyl = new THREE.Mesh(new THREE.CylinderGeometry(1.42, 1.42, 1.12, 48, 1, true), this.mats.irDome);
    L4.add(glassCyl);

    const topCap = new THREE.Mesh(new THREE.CylinderGeometry(1.38, 1.46, 0.22, 48), this.mats.titanium);
    topCap.position.y = 0.64;
    topCap.castShadow = true;
    L4.add(topCap);

    const topAccentRing = new THREE.Mesh(new THREE.TorusGeometry(1.18, 0.022, 10, 48), this.mats.steel);
    topAccentRing.rotation.x = Math.PI / 2;
    topAccentRing.position.y = 0.76;
    L4.add(topAccentRing);

    // Subtle vertical CAD alignment guides when exploded
    this.guideMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.16 });
    const gPts = [];
    [[1.22, 0], [-1.22, 0], [0, 1.22], [0, -1.22]].forEach(([x, z]) => {
      gPts.push(new THREE.Vector3(x, -1.95, z), new THREE.Vector3(x, 2.35, z));
    });
    this.guides = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(gPts), this.guideMat);
    this.lidarRoot.add(this.guides);
  }

  _buildOpticalBenchTarget() {
    // Architectural Optical Bench Rail along +X showing real target distance
    this.benchGroup = new THREE.Group();
    this.scene.add(this.benchGroup);

    const rail = new THREE.Mesh(
      new THREE.BoxGeometry(6.8, 0.08, 0.42),
      new THREE.MeshStandardMaterial({ color: 0x131c2b, metalness: 0.75, roughness: 0.38 })
    );
    rail.position.set(5.4, -2.30, 0);
    rail.receiveShadow = true;
    this.benchGroup.add(rail);

    // Movable Telescopic Target Stand
    this.targetAssembly = new THREE.Group();
    this.targetAssembly.position.set(4.5, 0, 0);
    this.benchGroup.add(this.targetAssembly);

    const sled = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.16, 0.56), this.mats.titanium);
    sled.position.y = -2.22;
    this.targetAssembly.add(sled);

    this.targetMast = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.065, 2.8, 12), this.mats.steel);
    this.targetMast.position.y = -0.82;
    this.targetAssembly.add(this.targetMast);

    // Target Reflector Panel (material changes live for 85% Sign / 42% Car / 8% Tire!)
    this.targetMat = new THREE.MeshPhysicalMaterial({
      color: 0x3a6496,
      map: createTargetStripeTexture(),
      roughness: 0.22,
      metalness: 0.68,
      clearcoat: 0.85
    });
    this.targetPanel = new THREE.Mesh(new THREE.BoxGeometry(0.14, 1.45, 1.55), this.targetMat);
    this.targetPanel.position.set(0, 0.55, 0);
    this.targetPanel.castShadow = true;
    this.targetPanel.receiveShadow = true;
    this.targetAssembly.add(this.targetPanel);

    // Glowing Laser Hit-Spot Ring on Target Face
    this.hitSpotMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(0.3, 1.8, 2.4),
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide,
      toneMapped: false
    });
    this.hitSpot = new THREE.Mesh(new THREE.RingGeometry(0.04, 0.14, 24), this.hitSpotMat);
    this.hitSpot.rotation.y = -Math.PI / 2;
    this.hitSpot.position.set(-0.085, 0.55, 0);
    this.targetAssembly.add(this.hitSpot);
  }

  _buildPhotonVisuals() {
    // 1. Subtle continuous TX & RX optical guide rays (in ToF Slow-Motion mode)
    this.tofVisualGroup = new THREE.Group();
    this.scene.add(this.tofVisualGroup);

    this.txGuideMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.34 });
    this.txGuideLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 0.5, 0),
        new THREE.Vector3(4.5, 0.5, 0)
      ]),
      this.txGuideMat
    );
    this.tofVisualGroup.add(this.txGuideLine);

    this.rxGuideMat = new THREE.LineBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.24 });
    this.rxGuideLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(4.5, 0.5, 0),
        new THREE.Vector3(0.68, 0.35, 0.52),
        new THREE.Vector3(0.14, 0.35, 0.52)
      ]),
      this.rxGuideMat
    );
    this.tofVisualGroup.add(this.rxGuideLine);

    // 2. Glowing Flying Laser Photon Pulse (Outgoing TX packet)
    this.txPulseMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.3, 1.7, 2.5), toneMapped: false });
    this.txPulse = new THREE.Mesh(new THREE.CapsuleGeometry(0.05, 0.28, 8, 12), this.txPulseMat);
    this.tofVisualGroup.add(this.txPulse);

    // 3. Returning Echo Wavefront Ring + Converging RX Photon Packet
    this.rxPulseMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(0.25, 2.1, 1.4),
      transparent: true,
      opacity: 0.92,
      toneMapped: false
    });
    this.rxPulse = new THREE.Mesh(new THREE.SphereGeometry(0.085, 14, 14), this.rxPulseMat);
    this.tofVisualGroup.add(this.rxPulse);

    this.rxWaveMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(0.25, 1.9, 1.3),
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide,
      toneMapped: false
    });
    this.rxWaveRing = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.22, 24), this.rxWaveMat);
    this.rxWaveRing.rotation.y = Math.PI / 2;
    this.tofVisualGroup.add(this.rxWaveRing);
  }

  _buildScanArena() {
    // Visible only in "360° Live Scan" mode: multi-channel vertical fan + miniature surrounding objects
    this.scanGroup = new THREE.Group();
    this.scanGroup.visible = false;
    this.scene.add(this.scanGroup);

    // 9-beam vertical laser fan rotating with the optical turret
    this.fanGroup = new THREE.Group();
    this.fanMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.65 });
    const fanPts = [];
    for (let ch = -4; ch <= 4; ch++) {
      const pitch = ch * 0.065;
      fanPts.push(
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(Math.cos(pitch) * 5.6, Math.sin(pitch) * 5.6, 0)
      );
    }
    this.fanLines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(fanPts), this.fanMat);
    this.fanGroup.add(this.fanLines);
    this.scanGroup.add(this.fanGroup);

    // Subtle dark architectural calibration ring obstacles around sensor (radius ~5.0)
    const obsMat = new THREE.MeshStandardMaterial({
      color: 0x131d2d,
      roughness: 0.65,
      metalness: 0.35,
      transparent: true,
      opacity: 0.45
    });
    const shapes = [
      { pos: [4.8, -1.35, -1.6], geo: new THREE.BoxGeometry(1.1, 1.9, 2.2) },
      { pos: [-4.6, -1.45, 1.8], geo: new THREE.BoxGeometry(1.4, 1.7, 1.8) },
      { pos: [0.4, -1.55, -5.0], geo: new THREE.CylinderGeometry(0.6, 0.75, 1.5, 20) },
      { pos: [-1.4, -1.65, 4.9], geo: new THREE.BoxGeometry(2.4, 1.3, 0.9) }
    ];
    shapes.forEach(s => {
      const m = new THREE.Mesh(s.geo, obsMat);
      m.position.set(...s.pos);
      this.scanGroup.add(m);
    });

    // Live glowing 3D scanned point ring painted around the arena
    const pCount = 900;
    const pArr = new Float32Array(pCount * 3);
    const cArr = new Float32Array(pCount * 3);
    for (let i = 0; i < pCount; i++) {
      const a = (i / pCount) * Math.PI * 2;
      const ring = (i % 9) - 4;
      const r = 4.9 + Math.sin(a * 4) * 0.35;
      pArr[i * 3] = Math.cos(a) * r;
      pArr[i * 3 + 1] = 0.25 + ring * 0.26;
      pArr[i * 3 + 2] = Math.sin(a) * r;
      cArr[i * 3] = 0.2; cArr[i * 3 + 1] = 0.85; cArr[i * 3 + 2] = 1.0;
    }
    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute("position", new THREE.BufferAttribute(pArr, 3));
    pGeo.setAttribute("color", new THREE.BufferAttribute(cArr, 3));
    this.scanPtsMat = new THREE.PointsMaterial({
      size: 0.12,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.scanPoints = new THREE.Points(pGeo, this.scanPtsMat);
    this.scanGroup.add(this.scanPoints);
  }

  setSimMode(mode) {
    this.simMode = mode;
    const isTof = mode === "tof";
    this.benchGroup.visible = isTof;
    this.tofVisualGroup.visible = isTof;
    this.scanGroup.visible = !isTof;
  }

  setExplode(norm01) {
    this.explodeTarget = Math.max(0, Math.min(1, norm01));
  }

  setAutoOrbit(enabled) {
    this.autoOrbit = !!enabled;
  }

  focusPart(partKey) {
    if (partKey === "all") {
      this.camGoalPos.set(5.6, 3.6, 6.9);
      this.camGoalTarget.set(1.5, 0.35, 0);
      this.camLerp = true;
      return;
    }

    // Ensure enough explode separation to inspect interior layers clearly
    if (partKey !== "window" && this.explodeTarget < 0.52) {
      this.explodeTarget = 0.68;
    }

    const yOf = idx => THREE.MathUtils.lerp(this.layers[idx].userData.baseY, this.layers[idx].userData.explodeY, this.explodeTarget);

    if (partKey === "window") {
      const y = yOf(4) + 0.3;
      this.camGoalTarget.set(0, y, 0);
      this.camGoalPos.set(3.4, y + 1.4, 3.8);
    } else if (partKey === "emitter") {
      const y = yOf(3) - 0.15;
      this.camGoalTarget.set(0, y, 0);
      this.camGoalPos.set(2.4, y + 0.8, 2.7);
    } else if (partKey === "mirror") {
      const y = yOf(3) + 0.22;
      this.camGoalTarget.set(0, y, 0);
      this.camGoalPos.set(2.3, y + 0.9, 2.5);
    } else if (partKey === "receiver") {
      const y = yOf(3) + 0.02;
      this.camGoalTarget.set(0.42, y, 0.52);
      this.camGoalPos.set(2.4, y + 0.9, 2.8);
    } else if (partKey === "motor") {
      const y = yOf(2) + 0.1;
      this.camGoalTarget.set(0, y, 0);
      this.camGoalPos.set(2.8, y + 1.2, 3.1);
    } else if (partKey === "fpga") {
      const y = yOf(1) + 0.1;
      this.camGoalTarget.set(0, y, 0);
      this.camGoalPos.set(2.9, y + 1.4, 3.2);
    }
    this.camLerp = true;
  }

  update(dt, elapsed, phys) {
    // 1. Smooth Exploded View vertical interpolation
    this.explodeCurrent += (this.explodeTarget - this.explodeCurrent) * Math.min(1, dt * 6.5);
    for (let i = 0; i < this.layers.length; i++) {
      const L = this.layers[i];
      L.position.y = THREE.MathUtils.lerp(L.userData.baseY, L.userData.explodeY, this.explodeCurrent);
    }
    this.guideMat.opacity = Math.max(0, (this.explodeCurrent - 0.08) * 0.22);

    // 2. Smooth camera fly-to or studio orbit
    if (this.camLerp) {
      this.camera.position.lerp(this.camGoalPos, Math.min(1, dt * 4.8));
      this.controls.target.lerp(this.camGoalTarget, Math.min(1, dt * 4.8));
      if (this.camera.position.distanceTo(this.camGoalPos) < 0.03) this.camLerp = false;
    } else if (this.autoOrbit) {
      const a = dt * 0.16;
      const dx = this.camera.position.x - this.controls.target.x;
      const dz = this.camera.position.z - this.controls.target.z;
      const ca = Math.cos(a), sa = Math.sin(a);
      this.camera.position.x = this.controls.target.x + dx * ca - dz * sa;
      this.camera.position.z = this.controls.target.z + dx * sa + dz * ca;
    }

    // 3. Layer 3 optical coordinates (World Y)
    const l3Y = this.layers[3].position.y;
    const beamY = l3Y + 0.22;
    const rxY = l3Y + 0.02;

    // Position 3D Target Plate along optical rail based on real distance [5..120m -> X 2.6..8.2]
    const distFrac = (phys.distanceM - 5) / 115;
    const targetX = 2.65 + distFrac * 5.65;
    this.targetAssembly.position.x = targetX;
    this.targetPanel.position.y = beamY;
    this.hitSpot.position.y = beamY;

    // Keep telescopic mast cleanly connecting floor sled (-2.22) to target panel bottom
    const mastBottom = -2.22;
    const mastTop = beamY - 0.72;
    const mastLen = Math.max(0.2, mastTop - mastBottom);
    this.targetMast.scale.y = mastLen / 2.8;
    this.targetMast.position.y = mastBottom + mastLen * 0.5;

    // Update target PBR surface appearance
    this.targetMat.color.setHex(phys.target.colorHex);
    this.targetMat.roughness = phys.target.roughness;
    this.targetMat.metalness = phys.target.metalness;

    // Update wavelength beam colors
    const wl = phys.wavelength;
    this.emitterApertureMat.color.setRGB(...wl.beamRgb);
    this.txGuideMat.color.setHex(wl.beamHex);
    this.txPulseMat.color.setRGB(...wl.beamRgb);
    this.hitSpotMat.color.setRGB(...wl.beamRgb);
    this.fanMat.color.setHex(wl.beamHex);

    // 4. Mode-specific rotation & photon flight
    let apdFlash = 0;

    if (this.simMode === "tof") {
      // Lock turret cleanly at 0° facing +X target so the user sees the complete optical path!
      this.rotorAngle += (0 - this.rotorAngle) * Math.min(1, dt * 6);
      this.opticalTurret.rotation.y = this.rotorAngle;
      this.encoderGroup.rotation.y = this.rotorAngle;

      // Update optical guide lines to track live explode Y & target X
      const txAttr = this.txGuideLine.geometry.attributes.position;
      txAttr.setXYZ(0, 0, l3Y - 0.07, 0);
      txAttr.setXYZ(1, 0, beamY, 0);
      txAttr.setXYZ(2, targetX - 0.08, beamY, 0);
      txAttr.needsUpdate = true;

      const rxAttr = this.rxGuideLine.geometry.attributes.position;
      rxAttr.setXYZ(0, targetX - 0.08, beamY, 0);
      rxAttr.setXYZ(1, 0.68, rxY, 0.52);
      rxAttr.setXYZ(2, 0.14, rxY, 0.52);
      rxAttr.needsUpdate = true;

      // Advance slow-motion photon pulse (slower round-trip at larger distance!)
      const cycleSpeed = 0.52 - distFrac * 0.24;
      this.tofPhase = (this.tofPhase + dt * cycleSpeed) % 1;
      const p = this.tofPhase;

      // Phase breakdown:
      // 0.00..0.10: Emitter -> 45° Mirror (Vertical)
      // 0.10..0.48: 45° Mirror -> Target Plate (Outgoing TX)
      // 0.48..0.86: Target Plate -> Aspheric Lens (Returning RX Echo)
      // 0.86..0.94: Aspheric Lens -> APD Chip (Focusing)
      // 0.94..1.00: APD Electronic Flash & FPGA TDC Readout
      if (p < 0.10) {
        const u = p / 0.10;
        this.txPulse.visible = true;
        this.rxPulse.visible = false;
        this.rxWaveRing.visible = false;
        this.txPulse.rotation.set(0, 0, 0);
        this.txPulse.position.set(0, THREE.MathUtils.lerp(l3Y - 0.07, beamY, u), 0);
      } else if (p < 0.48) {
        const u = (p - 0.10) / 0.38;
        this.txPulse.visible = true;
        this.rxPulse.visible = false;
        this.rxWaveRing.visible = false;
        this.txPulse.rotation.set(0, 0, Math.PI / 2);
        this.txPulse.position.set(THREE.MathUtils.lerp(0, targetX - 0.08, u), beamY, 0);
      } else if (p < 0.86) {
        const u = (p - 0.48) / 0.38;
        this.txPulse.visible = false;
        this.rxPulse.visible = true;
        this.rxWaveRing.visible = true;

        const rxX = THREE.MathUtils.lerp(targetX - 0.08, 0.68, u);
        const rxCurY = THREE.MathUtils.lerp(beamY, rxY, u);
        const rxZ = THREE.MathUtils.lerp(0, 0.52, u);
        this.rxPulse.position.set(rxX, rxCurY, rxZ);
        this.rxWaveRing.position.set(rxX, rxCurY, rxZ);

        // Echo brightness reflects real LiDAR Range Equation SNR!
        const eScale = phys.detected ? (0.45 + phys.echoNorm * 0.95) : 0.25;
        this.rxPulse.scale.setScalar(eScale);
        this.rxWaveRing.scale.setScalar(0.5 + u * 1.45);
        this.rxPulseMat.color.setRGB(
          wl.echoRgb[0] * (0.25 + phys.echoNorm * 0.95),
          wl.echoRgb[1] * (0.25 + phys.echoNorm * 0.95),
          wl.echoRgb[2] * (0.25 + phys.echoNorm * 0.95)
        );
        this.rxWaveMat.opacity = phys.detected ? (1 - u * 0.45) * (0.3 + phys.echoNorm * 0.55) : 0.14;
      } else if (p < 0.95) {
        const u = (p - 0.86) / 0.09;
        this.txPulse.visible = false;
        this.rxPulse.visible = true;
        this.rxWaveRing.visible = false;
        this.rxPulse.position.set(THREE.MathUtils.lerp(0.68, 0.14, u), rxY, 0.52);
        this.rxPulse.scale.setScalar(Math.max(0.25, (1 - u * 0.65) * (0.4 + phys.echoNorm * 0.8)));
      } else {
        this.txPulse.visible = false;
        this.rxPulse.visible = false;
        this.rxWaveRing.visible = false;
        apdFlash = phys.detected ? 1.0 : 0.12;
      }

      // Target hit-spot pulse right at p = 0.48
      const hitDist = Math.abs(p - 0.48);
      const hitGlow = Math.max(0.22, 1 - hitDist * 14);
      this.hitSpotMat.opacity = hitGlow;
      this.hitSpot.scale.setScalar(0.85 + hitGlow * 0.55);
    } else {
      // 360° Live Scan Mode: smooth optical turret rotation + live rotating fan
      this.rotorAngle = (this.rotorAngle + dt * 2.2) % (Math.PI * 2);
      this.opticalTurret.rotation.y = this.rotorAngle;
      this.encoderGroup.rotation.y = this.rotorAngle;
      this.fanGroup.position.y = beamY;
      this.fanGroup.rotation.y = this.rotorAngle;
      apdFlash = 0.55 + 0.45 * Math.sin(elapsed * 18);
    }

    // 5. APD Detector & FPGA LED Electronic Spike Glow
    this.apdSensorMat.color.setRGB(
      0.12 + apdFlash * 0.45,
      0.32 + apdFlash * 2.1,
      0.42 + apdFlash * 1.4
    );
    this.fpgaLedMat.color.setRGB(
      0.15 + apdFlash * 0.3,
      0.65 + apdFlash * 1.6,
      0.85 + apdFlash * 1.2
    );

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