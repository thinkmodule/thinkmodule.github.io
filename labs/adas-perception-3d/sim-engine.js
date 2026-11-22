/**
 * ThinkModule — Autonomous Driving & LiDAR Simulation Engine
 * ============================================================================
 * Pure headless, deterministic, fixed-timestep simulation core.
 * Strictly decoupled from Three.js rendering and DOM manipulation.
 *
 * Coordinate Convention (SI units: meters, seconds, radians):
 *  - s : Longitudinal highway coordinate (m, positive = forward along highway)
 *  - x : Lateral coordinate (m, negative = left/fast lane, positive = right/slow lane)
 *  - y : Vertical coordinate (m, 0 = asphalt surface)
 *  - 3D Rendering mapping: Three.js X = x, Y = y, Z = -(s - ego.s)
 *
 * Transparency Note:
 *  - Planning & control use explicit rule-based IDM (Intelligent Driver Model)
 *    and analytical TTC-based AEB logic — NOT a trained neural policy.
 *  - Detection confidence is modeled analytically from LiDAR return density
 *    and geometric visibility.
 */

export const FIXED_DT = 1 / 60;
export const MAX_WALL_DT = 0.1; // Clamps frame drops & browser tab switches
export const LANE_WIDTH = 3.7;
export const LANE_CENTERS = [-3.7, 0.0, 3.7]; // 0: Fast/Left, 1: Center, 2: Slow/Right
export const ROAD_HALF_WIDTH = 6.35;
export const LAMP_SPACING = 40.0;
export const EGO_WHEEL_RADIUS = 0.36;
export const LIDAR_MOUNT_HEIGHT = 1.88;
export const LIDAR_MAX_RANGE = 120.0;
export const LIDAR_SCAN_HZ = 10.0;
export const LIDAR_SCAN_INTERVAL = 1 / LIDAR_SCAN_HZ; // 0.1 s (6 fixed ticks)

export const QUALITY_PRESETS = {
  high: {
    id: "high",
    label: "High",
    channels: 16,
    azSteps: 180,
    maxDpr: 2,
    shadows: true,
    raySampleStride: 24,
  },
  balanced: {
    id: "balanced",
    label: "Balanced",
    channels: 12,
    azSteps: 136,
    maxDpr: 1.5,
    shadows: false,
    raySampleStride: 28,
  },
  low: {
    id: "low",
    label: "Low (Laptop)",
    channels: 8,
    azSteps: 96,
    maxDpr: 1,
    shadows: false,
    raySampleStride: 32,
  },
};

export const VEHICLE_SPECS = {
  sedan: {
    type: "sedan",
    label: "Sedan",
    length: 4.65,
    width: 1.88,
    height: 1.44,
    wheelRadius: 0.34,
    reflectivity: 0.72,
    palette: [0x6e7b8b, 0x4b5868, 0x8b95a5, 0x3b4754, 0x5d6d7e],
  },
  suv: {
    type: "suv",
    label: "SUV",
    length: 4.95,
    width: 2.02,
    height: 1.76,
    wheelRadius: 0.38,
    reflectivity: 0.76,
    palette: [0x46525e, 0x626c78, 0x38424d, 0x76818e],
  },
  van: {
    type: "van",
    label: "Commercial Van",
    length: 5.8,
    width: 2.08,
    height: 2.38,
    wheelRadius: 0.39,
    reflectivity: 0.84,
    palette: [0x9aa3ad, 0x7f8a96, 0xb2bac2],
  },
  truck: {
    type: "truck",
    label: "Heavy Box Truck",
    length: 9.8,
    width: 2.52,
    height: 3.45,
    wheelRadius: 0.5,
    reflectivity: 0.92,
    palette: [0xa9b2bc, 0x8d97a3, 0xbfc6ce],
  },
};

export const SCENARIOS = [
  {
    id: "guided_sequence",
    code: "01",
    name: "01 · Guided R&D Sequence",
    shortName: "Guided R&D Sequence",
    duration: 56.0,
    seed: 104729,
    description:
      "Cohesive multi-phase autonomous highway sequence: system boot, LiDAR calibration, controlled launch, overtaking tracking, cut-in ACC adaptation, and dense multi-object monitoring.",
    phases: [
      { start: 0, end: 3.5, name: "System Initialization", detail: "Powering perception bus, verifying vehicle kinematics & reference frame.", spotlight: 1 },
      { start: 3.5, end: 7.5, name: "LiDAR Calibration", detail: "Spinning roof-mounted 360° LiDAR to 10 Hz and validating ground-plane alignment.", spotlight: 2 },
      { start: 7.5, end: 13.5, name: "Controlled Departure", detail: "Smooth longitudinal acceleration from standstill to 100 km/h highway target.", spotlight: 1 },
      { start: 13.5, end: 25.0, name: "Overtaking & Tracking", detail: "Faster left-lane traffic enters the 120 m FOV; persistent tracks estimate relative velocity.", spotlight: 3 },
      { start: 25.0, end: 39.0, name: "Cut-In & ACC Control", detail: "Adjacent vehicle merges ahead; rule-based IDM controller modulates speed to preserve headway.", spotlight: 4 },
      { start: 39.0, end: 56.0, name: "Multi-Vehicle Monitoring", detail: "Simultaneous multi-lane tracking, occlusion evaluation, and continuous path verification.", spotlight: 3 },
    ],
  },
  {
    id: "overtaking",
    code: "02",
    name: "02 · Highway Overtaking",
    shortName: "Highway Overtaking",
    duration: 42.0,
    seed: 208311,
    description:
      "Ego cruises steadily at 95 km/h while higher-speed vehicles overtake on the left lane and commercial traffic travels on the right.",
    phases: [
      { start: 0, end: 4.0, name: "Steady Cruise Baseline", detail: "Ego maintains 95 km/h in center lane with 10 Hz LiDAR sweep active.", spotlight: 1 },
      { start: 4.0, end: 18.0, name: "Rear Approach & Closing", detail: "Fast left-lane sedan enters rear 120 m envelope with positive closing rate.", spotlight: 2 },
      { start: 18.0, end: 30.0, name: "Lateral Pass & Tracking", detail: "Continuous 3D bounding box lock and relative velocity inversion during pass.", spotlight: 3 },
      { start: 30.0, end: 42.0, name: "Receding Traffic Flow", detail: "Sustained highway traffic continues across all three lanes.", spotlight: 4 },
    ],
  },
  {
    id: "acc_following",
    code: "03",
    name: "03 · Adaptive Cruise Control",
    shortName: "Adaptive Cruise Control",
    duration: 42.0,
    seed: 309127,
    description:
      "Ego approaches a slower commercial van in the same lane and smoothly regulates deceleration via rule-based IDM headway control.",
    phases: [
      { start: 0, end: 8.0, name: "Free-Flow Approach", detail: "Ego cruises at 112 km/h approaching a slower 76 km/h lead vehicle.", spotlight: 1 },
      { start: 8.0, end: 22.0, name: "ACC Speed Regulation", detail: "Closing gap triggers smooth deceleration to match lead velocity and 1.8 s time headway.", spotlight: 4 },
      { start: 22.0, end: 42.0, name: "Stabilized Headway Hold", detail: "Maintains equilibrium safety gap while monitoring adjacent passing traffic.", spotlight: 3 },
    ],
  },
  {
    id: "lane_change",
    code: "04",
    name: "04 · Lane Change Cut-In",
    shortName: "Lane Change Cut-In",
    duration: 42.0,
    seed: 412843,
    description:
      "A vehicle from the right lane initiates a lateral cut-in maneuver in front of the Ego vehicle; perception detects lateral motion and ACC yields safely.",
    phases: [
      { start: 0, end: 7.0, name: "Parallel Traffic Flow", detail: "Ego at 105 km/h monitors right-lane vehicle approaching a slower truck.", spotlight: 1 },
      { start: 7.0, end: 16.0, name: "Lateral Cut-In Detection", detail: "Right-lane sedan merges into center lane; tracker registers lateral velocity & lane intrusion.", spotlight: 3 },
      { start: 16.0, end: 29.0, name: "Yield & Gap Recovery", detail: "Rule-based controller reduces speed smoothly to re-establish safe following envelope.", spotlight: 4 },
      { start: 29.0, end: 42.0, name: "Post-Maneuver Cruise", detail: "Traffic stabilizes as cut-in vehicle accelerates and resumes normal highway flow.", spotlight: 2 },
    ],
  },
  {
    id: "dense_traffic",
    code: "05",
    name: "05 · Dense Traffic Monitoring",
    shortName: "Dense Traffic Monitoring",
    duration: 48.0,
    seed: 519461,
    description:
      "High-density multi-lane highway traffic demonstrating simultaneous multi-object LiDAR reconstruction, track association, and clutter-free monitoring.",
    phases: [
      { start: 0, end: 12.0, name: "Multi-Lane Acquisition", detail: "Acquiring 8+ concurrent vehicle tracks across left, center, and right lanes.", spotlight: 2 },
      { start: 12.0, end: 30.0, name: "Multi-Object Tracking", detail: "Maintaining distinct track IDs, relative velocities, and safety envelopes under high density.", spotlight: 3 },
      { start: 30.0, end: 48.0, name: "Continuous Congestion Flow", detail: "Adaptive speed modulation in response to variable downstream flow.", spotlight: 4 },
    ],
  },
  {
    id: "sensor_occlusion",
    code: "06",
    name: "06 · LiDAR Sensor Occlusion",
    shortName: "LiDAR Sensor Occlusion",
    duration: 42.0,
    seed: 623917,
    description:
      "A heavy box truck geometrically blocks LiDAR rays from reaching a vehicle traveling behind it, illustrating real line-of-sight shadow cones and re-acquisition.",
    phases: [
      { start: 0, end: 10.0, name: "Heavy Occluder Ahead", detail: "Box truck blocks optical line-of-sight along the right-center corridor.", spotlight: 2 },
      { start: 10.0, end: 23.0, name: "Shadow Cone Occlusion", detail: "Secondary sedan lies inside the truck's geometric ray shadow — zero direct returns.", spotlight: 2 },
      { start: 23.0, end: 42.0, name: "Emergence & Re-Acquisition", detail: "Target emerges laterally from behind the truck; point density climbs and track locks.", spotlight: 3 },
    ],
  },
  {
    id: "aeb_braking",
    code: "07",
    name: "07 · Emergency Braking (AEB)",
    shortName: "Emergency Braking (AEB)",
    duration: 34.0,
    seed: 731587,
    description:
      "A lead vehicle executes sudden hard braking (-7.2 m/s²). Ego calculates Time-to-Collision (TTC) and triggers rule-based autonomous emergency braking.",
    phases: [
      { start: 0, end: 7.0, name: "Nominal Following", detail: "Ego follows lead sedan at 100 km/h with nominal clearance.", spotlight: 1 },
      { start: 7.0, end: 12.5, name: "Sudden Lead Deceleration", detail: "Lead vehicle brakes aggressively; TTC drops rapidly below 2.6 s intervention threshold.", spotlight: 4 },
      { start: 12.5, end: 22.0, name: "AEB Full Intervention", detail: "Rule-based AEB commands -8.5 m/s² deceleration to prevent rear-end collision.", spotlight: 4 },
      { start: 22.0, end: 34.0, name: "Safe Standstill & Recovery", detail: "Vehicles stabilize with positive clearance; system verifies safe release and resumes.", spotlight: 3 },
    ],
  },
];

/**
 * Deterministic Seeded PRNG (Mulberry32)
 * Guarantees identical traffic patterns across restarts and timeline scrubbing.
 */
class SeededRNG {
  constructor(seed = 123456) {
    this.state = seed >>> 0;
  }
  next() {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  range(min, max) {
    return min + this.next() * (max - min);
  }
  pick(arr) {
    return arr[Math.floor(this.next() * arr.length)];
  }
  cloneState() {
    return this.state;
  }
  restoreState(s) {
    this.state = s >>> 0;
  }
}

/**
 * Analytical Ray-AABB Slab Intersection (3D)
 * Returns hit distance t > 0 or -1 if no intersection within [0.5, maxT].
 */
function intersectRayAABB(ox, oy, oz, dx, dy, dz, minX, maxX, minY, maxY, minZ, maxZ, maxT) {
  let tmin = 0.4;
  let tmax = maxT;

  if (Math.abs(dx) < 1e-8) {
    if (ox < minX || ox > maxX) return -1;
  } else {
    const invD = 1 / dx;
    let t1 = (minX - ox) * invD;
    let t2 = (maxX - ox) * invD;
    if (t1 > t2) {
      const tmp = t1;
      t1 = t2;
      t2 = tmp;
    }
    if (t1 > tmin) tmin = t1;
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return -1;
  }

  if (Math.abs(dy) < 1e-8) {
    if (oy < minY || oy > maxY) return -1;
  } else {
    const invD = 1 / dy;
    let t1 = (minY - oy) * invD;
    let t2 = (maxY - oy) * invD;
    if (t1 > t2) {
      const tmp = t1;
      t1 = t2;
      t2 = tmp;
    }
    if (t1 > tmin) tmin = t1;
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return -1;
  }

  if (Math.abs(dz) < 1e-8) {
    if (oz < minZ || oz > maxZ) return -1;
  } else {
    const invD = 1 / dz;
    let t1 = (minZ - oz) * invD;
    let t2 = (maxZ - oz) * invD;
    if (t1 > t2) {
      const tmp = t1;
      t1 = t2;
      t2 = tmp;
    }
    if (t1 > tmin) tmin = t1;
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return -1;
  }

  return tmin <= tmax && tmin > 0.4 ? tmin : -1;
}

export class SimulationEngine {
  constructor(options = {}) {
    this.quality = QUALITY_PRESETS[options.quality || "balanced"] || QUALITY_PRESETS.balanced;
    this.trafficDensity = options.trafficDensity || "moderate"; // 'low' | 'moderate' | 'dense'
    this.lidarNoiseEnabled = false;
    this.lidarDensityScale = 1.0;

    // Pre-allocated point cloud typed arrays (sized for maximum quality * 1.5x density)
    const maxPts = 24 * 280;
    this.lidarBuffers = {
      pos: new Float32Array(maxPts * 3),
      dist: new Float32Array(maxPts),
      intensity: new Float32Array(maxPts),
      objId: new Int16Array(maxPts), // -1 = ground, -2 = guardrail/lamp, >=0 = vehicle id
      raySampleEnds: new Float32Array(256 * 3),
      raySampleHits: new Uint8Array(256), // 0=miss, 1=ground, 2=vehicle
    };

    // Pooled vehicle slots (reused cleanly across scenarios — no runtime GC churn)
    this.poolSize = 18;
    this.vehiclePool = Array.from({ length: this.poolSize }, (_, i) => this._createEmptyVehicle(i));

    // Clock separation: wall vs simulation vs sensor sampling
    this._lastWallMs = null;
    this._accumulator = 0;

    // Deterministic snapshot cache for glitch-free scrubbing
    this._snapshots = new Map(); // secondInt -> snapshot
    this._lastSnapshotSec = -1;

    this.setScenario(options.scenarioId || "guided_sequence");
  }

  _createEmptyVehicle(poolIndex) {
    return {
      poolIndex,
      id: 0,
      active: false,
      scripted: false,
      scriptRole: null,
      type: "sedan",
      label: "Sedan",
      colorHex: 0x6e7b8b,
      s: 0,
      x: 0,
      relS: 0,
      speed: 25,
      targetSpeed: 25,
      accel: 0,
      laneIndex: 1,
      targetLane: 1,
      laneChangeProgress: 1,
      laneChangeStartX: 0,
      laneChangeDuration: 3.0,
      laneChangeTimer: 0,
      wheelAngle: 0,
      wheelRadius: 0.34,
      length: 4.65,
      width: 1.88,
      height: 1.44,
      reflectivity: 0.72,
      brakeLight: false,
      turnSignal: 0, // -1 left, 0 off, 1 right
      steeringAngle: 0,
    };
  }

  setScenario(scenarioId) {
    const found = SCENARIOS.find((s) => s.id === scenarioId) || SCENARIOS[0];
    this.scenarioDef = found;
    this.restart();
  }

  setTrafficDensity(density) {
    if (["low", "moderate", "dense"].includes(density)) {
      this.trafficDensity = density;
    }
  }

  setQuality(presetKey) {
    if (QUALITY_PRESETS[presetKey]) {
      this.quality = QUALITY_PRESETS[presetKey];
      this._runLiDARScan();
      this._updatePerceptionAndTracking(0);
    }
  }

  setLidarNoise(enabled) {
    this.lidarNoiseEnabled = Boolean(enabled);
    this._runLiDARScan();
  }

  setLidarDensityScale(scale) {
    this.lidarDensityScale = Math.max(0.5, Math.min(1.5, Number(scale) || 1.0));
    this._runLiDARScan();
    this._updatePerceptionAndTracking(0);
  }

  setSpeed(mult) {
    const allowed = [0.25, 0.5, 1, 2];
    if (allowed.includes(mult)) {
      this.clock.speed = mult;
    }
  }

  setPaused(paused) {
    this.clock.paused = Boolean(paused);
    this._accumulator = 0;
  }

  togglePause() {
    this.setPaused(!this.clock.paused);
    return this.clock.paused;
  }

  /**
   * Resets wall-clock baseline after browser tab visibility change
   * so returning to the tab never causes a simulation jump.
   */
  notifyVisibilityResume(nowMs) {
    this._lastWallMs = nowMs;
    this._accumulator = 0;
  }

  restart() {
    this.rng = new SeededRNG(this.scenarioDef.seed);
    this._nextVehicleId = 1;
    this._accumulator = 0;
    this._lastWallMs = null;
    this._snapshots.clear();
    this._lastSnapshotSec = -1;

    this.clock = {
      simTime: 0,
      duration: this.scenarioDef.duration,
      paused: false,
      speed: this.clock ? this.clock.speed : 1.0,
      fixedDt: FIXED_DT,
      tickCount: 0,
      sensorScanHz: LIDAR_SCAN_HZ,
      lastScanTime: -1,
      scanAngleRad: 0,
    };

    // Reset vehicle pool completely (zero stale state on restart)
    for (const v of this.vehiclePool) {
      v.active = false;
      v.scripted = false;
      v.scriptRole = null;
      v.id = 0;
    }

    // Initialize Ego Vehicle state according to scenario
    const startsFromStandstill = this.scenarioDef.id === "guided_sequence";
    const initialSpeedMs = startsFromStandstill
      ? 0.0
      : this.scenarioDef.id === "overtaking"
        ? 95 / 3.6
        : this.scenarioDef.id === "acc_following"
          ? 112 / 3.6
          : this.scenarioDef.id === "lane_change"
            ? 105 / 3.6
            : this.scenarioDef.id === "dense_traffic"
              ? 88 / 3.6
              : this.scenarioDef.id === "sensor_occlusion"
                ? 92 / 3.6
                : 100 / 3.6; // aeb_braking

    const targetSpeedMs =
      this.scenarioDef.id === "guided_sequence"
        ? 100 / 3.6
        : initialSpeedMs;

    this.ego = {
      s: 0,
      x: LANE_CENTERS[1],
      speed: initialSpeedMs,
      targetSpeed: targetSpeedMs,
      baseTargetSpeed: targetSpeedMs,
      accel: 0,
      wheelAngle: 0,
      wheelRadius: EGO_WHEEL_RADIUS,
      length: 4.82,
      width: 1.94,
      height: 1.48,
      laneIndex: 1,
      controlState: startsFromStandstill ? "INIT" : "CRUISING",
      controlReason: startsFromStandstill
        ? "Perception bus initializing"
        : "Nominal highway cruise",
      safeFollowDist: 38.0,
      leadVehicleId: null,
      leadGap: Infinity,
      minTtc: Infinity,
      brakeLight: startsFromStandstill,
      plannedPath: [],
      laneOccupancy: [false, false, false],
      speedHistory: [],
    };

    this.lidarState = {
      sensorHeight: LIDAR_MOUNT_HEIGHT,
      maxRange: LIDAR_MAX_RANGE,
      hfovDeg: 360,
      vfovMinDeg: -17,
      vfovMaxDeg: 6,
      channels: this.quality.channels,
      azSteps: this.quality.azSteps,
      scanHz: LIDAR_SCAN_HZ,
      noiseEnabled: this.lidarNoiseEnabled,
      calibrationProgress: startsFromStandstill ? 0 : 1,
      activePoints: 0,
      hitVehiclePoints: 0,
      hitGroundPoints: 0,
      raySampleCount: 0,
      pointsPos: this.lidarBuffers.pos,
      pointsDist: this.lidarBuffers.dist,
      pointsIntensity: this.lidarBuffers.intensity,
      pointsObjId: this.lidarBuffers.objId,
      raySampleEnds: this.lidarBuffers.raySampleEnds,
      raySampleHits: this.lidarBuffers.raySampleHits,
    };

    this._trackMap = new Map();
    this.tracks = [];
    this.events = [];
    this._lastLoggedState = "";
    this._lastLoggedPhase = -1;
    this._spawnCooldown = 0.5;

    this._seedInitialScenarioTraffic();
    this._updatePhaseMetadata();
    this._runLiDARScan();
    this._updatePerceptionAndTracking(0);
    this._updateAutonomousPlanner(0);
    this._recordTelemetrySample();
    this._saveSnapshot(0);
    this._logEvent("SYSTEM", `Scenario initialized: ${this.scenarioDef.shortName}`);
  }

  _acquireVehicle(cfg) {
    const slot = this.vehiclePool.find((v) => !v.active);
    if (!slot) return null;

    const spec = VEHICLE_SPECS[cfg.type] || VEHICLE_SPECS.sedan;
    const lane = cfg.laneIndex ?? 1;

    slot.active = true;
    slot.id = this._nextVehicleId++;
    slot.scripted = Boolean(cfg.scripted);
    slot.scriptRole = cfg.scriptRole || null;
    slot.type = spec.type;
    slot.label = spec.label;
    slot.colorHex = cfg.colorHex ?? this.rng.pick(spec.palette);
    slot.s = this.ego.s + (cfg.relS ?? 60);
    slot.laneIndex = lane;
    slot.targetLane = lane;
    slot.x = cfg.x ?? LANE_CENTERS[lane];
    slot.relS = slot.s - this.ego.s;
    slot.speed = cfg.speed ?? 25;
    slot.targetSpeed = cfg.targetSpeed ?? slot.speed;
    slot.accel = 0;
    slot.laneChangeProgress = 1;
    slot.laneChangeStartX = slot.x;
    slot.laneChangeDuration = 3.2;
    slot.laneChangeTimer = 0;
    slot.wheelAngle = this.rng.range(0, Math.PI * 2);
    slot.wheelRadius = spec.wheelRadius;
    slot.length = spec.length;
    slot.width = spec.width;
    slot.height = spec.height;
    slot.reflectivity = spec.reflectivity;
    slot.brakeLight = false;
    slot.turnSignal = 0;
    slot.steeringAngle = 0;

    return slot;
  }

  _seedInitialScenarioTraffic() {
    const id = this.scenarioDef.id;

    if (id === "guided_sequence") {
      // Calm highway environment with realistic passing vehicles already flowing
      this._acquireVehicle({ type: "truck", laneIndex: 2, relS: 48, speed: 78 / 3.6 });
      this._acquireVehicle({ type: "sedan", laneIndex: 0, relS: -75, speed: 124 / 3.6, scripted: true, scriptRole: "guided_overtaker" });
      this._acquireVehicle({ type: "suv", laneIndex: 2, relS: 112, speed: 84 / 3.6 });
      this._acquireVehicle({ type: "sedan", laneIndex: 0, relS: 88, speed: 118 / 3.6 });
      // Cut-in actor positioned to align around t = 25s
      this._acquireVehicle({ type: "suv", laneIndex: 2, relS: -14, speed: 92 / 3.6, scripted: true, scriptRole: "guided_cutin", colorHex: 0x7b8ea3 });
    } else if (id === "overtaking") {
      this._acquireVehicle({ type: "sedan", laneIndex: 0, relS: -68, speed: 126 / 3.6, scripted: true, scriptRole: "fast_overtaker", colorHex: 0x8aa0b8 });
      this._acquireVehicle({ type: "suv", laneIndex: 0, relS: -130, speed: 122 / 3.6 });
      this._acquireVehicle({ type: "truck", laneIndex: 2, relS: 54, speed: 80 / 3.6 });
      this._acquireVehicle({ type: "van", laneIndex: 2, relS: -28, speed: 84 / 3.6 });
      this._acquireVehicle({ type: "sedan", laneIndex: 1, relS: 92, speed: 96 / 3.6 });
    } else if (id === "acc_following") {
      this._acquireVehicle({ type: "van", laneIndex: 1, relS: 72, speed: 76 / 3.6, scripted: true, scriptRole: "acc_lead", colorHex: 0xa6b0ba });
      this._acquireVehicle({ type: "sedan", laneIndex: 0, relS: -45, speed: 120 / 3.6 });
      this._acquireVehicle({ type: "truck", laneIndex: 2, relS: 38, speed: 74 / 3.6 });
      this._acquireVehicle({ type: "suv", laneIndex: 0, relS: 62, speed: 116 / 3.6 });
    } else if (id === "lane_change") {
      this._acquireVehicle({ type: "sedan", laneIndex: 2, relS: 26, speed: 96 / 3.6, scripted: true, scriptRole: "cutin_actor", colorHex: 0x8ea2b7 });
      this._acquireVehicle({ type: "truck", laneIndex: 2, relS: 68, speed: 76 / 3.6, scripted: true, scriptRole: "slow_truck_right" });
      this._acquireVehicle({ type: "suv", laneIndex: 0, relS: -36, speed: 122 / 3.6 });
      this._acquireVehicle({ type: "sedan", laneIndex: 0, relS: 74, speed: 118 / 3.6 });
    } else if (id === "dense_traffic") {
      this._acquireVehicle({ type: "sedan", laneIndex: 1, relS: 44, speed: 86 / 3.6 });
      this._acquireVehicle({ type: "suv", laneIndex: 1, relS: 92, speed: 88 / 3.6 });
      this._acquireVehicle({ type: "truck", laneIndex: 2, relS: 24, speed: 76 / 3.6 });
      this._acquireVehicle({ type: "van", laneIndex: 2, relS: 76, speed: 78 / 3.6 });
      this._acquireVehicle({ type: "sedan", laneIndex: 2, relS: -32, speed: 80 / 3.6 });
      this._acquireVehicle({ type: "sedan", laneIndex: 0, relS: -18, speed: 108 / 3.6 });
      this._acquireVehicle({ type: "suv", laneIndex: 0, relS: 34, speed: 106 / 3.6 });
      this._acquireVehicle({ type: "sedan", laneIndex: 0, relS: 88, speed: 110 / 3.6 });
    } else if (id === "sensor_occlusion") {
      // Heavy box truck right ahead-right casts a massive geometric shadow cone over the sedan behind it
      this._acquireVehicle({ type: "truck", laneIndex: 2, relS: 26, speed: 88 / 3.6, scripted: true, scriptRole: "occluder_truck", colorHex: 0xb6bec8 });
      this._acquireVehicle({ type: "sedan", laneIndex: 2, relS: 48, speed: 92 / 3.6, scripted: true, scriptRole: "occluded_target", colorHex: 0x7292b2 });
      this._acquireVehicle({ type: "suv", laneIndex: 0, relS: -40, speed: 118 / 3.6 });
      this._acquireVehicle({ type: "sedan", laneIndex: 0, relS: 68, speed: 114 / 3.6 });
    } else if (id === "aeb_braking") {
      this._acquireVehicle({ type: "sedan", laneIndex: 1, relS: 42, speed: 100 / 3.6, scripted: true, scriptRole: "aeb_lead", colorHex: 0x9aa8b8 });
      this._acquireVehicle({ type: "truck", laneIndex: 2, relS: 64, speed: 82 / 3.6 });
      this._acquireVehicle({ type: "suv", laneIndex: 0, relS: -52, speed: 116 / 3.6 });
    }
  }

  /**
   * Main Wall-Clock Entry Point called by requestAnimationFrame.
   * Strict separation: if paused, returns immediately without modifying simulation state.
   */
  updateWallClock(nowMs) {
    if (this._lastWallMs === null) {
      this._lastWallMs = nowMs;
      return { stepped: false, stepsCount: 0 };
    }

    let wallDt = (nowMs - this._lastWallMs) / 1000;
    this._lastWallMs = nowMs;

    if (wallDt < 0 || !Number.isFinite(wallDt)) wallDt = 0;
    if (wallDt > MAX_WALL_DT) wallDt = MAX_WALL_DT;

    if (this.clock.paused) {
      this._accumulator = 0;
      return { stepped: false, stepsCount: 0 };
    }

    this._accumulator += wallDt * this.clock.speed;
    let steps = 0;
    const maxSubsteps = 8;

    while (this._accumulator >= FIXED_DT && steps < maxSubsteps) {
      this._stepFixed(FIXED_DT);
      this._accumulator -= FIXED_DT;
      steps++;
    }

    if (steps === maxSubsteps) {
      this._accumulator = 0;
    }

    return { stepped: steps > 0, stepsCount: steps };
  }

  /**
   * Advances simulation by exactly one fixed step (1/60s), usable even while paused.
   */
  stepSingleFrame() {
    this._stepFixed(FIXED_DT);
    this._runLiDARScan();
    this._updatePerceptionAndTracking(0);
  }

  /**
   * Deterministic Timeline Scrubbing to any targetTime in [0, duration].
   */
  seekToTime(targetTime) {
    const clamped = Math.max(0, Math.min(this.clock.duration, targetTime));
    const wasPaused = this.clock.paused;

    // Find closest snapshot at or before clamped time
    let bestSec = 0;
    for (const sec of this._snapshots.keys()) {
      if (sec <= clamped && sec >= bestSec) {
        bestSec = sec;
      }
    }

    const snap = this._snapshots.get(bestSec);
    if (snap) {
      this._restoreSnapshot(snap);
    } else {
      this.restart();
    }

    // Fast-forward deterministically using exact fixed steps
    const ticksNeeded = Math.round((clamped - this.clock.simTime) / FIXED_DT);
    for (let i = 0; i < ticksNeeded; i++) {
      this._stepFixed(FIXED_DT, true);
    }

    // Always run a fresh LiDAR scan at the scrubbed instant for 100% spatial consistency
    this._runLiDARScan();
    this._updatePerceptionAndTracking(0);
    this._updateAutonomousPlanner(0);

    this.clock.paused = wasPaused;
    this._accumulator = 0;
  }

  _stepFixed(dt, isFastForward = false) {
    // Loop gracefully at end of scenario
    if (this.clock.simTime >= this.clock.duration) {
      const preservedPaused = this.clock.paused;
      const preservedSpeed = this.clock.speed;
      this.restart();
      this.clock.paused = preservedPaused;
      this.clock.speed = preservedSpeed;
      return;
    }

    this.clock.simTime = Math.min(this.clock.duration, this.clock.simTime + dt);
    this.clock.tickCount++;

    // 1. Update Scenario Phase & Calibration
    this._updatePhaseMetadata();

    // 2. Update Autonomous Planner & Ego Kinematics
    this._updateAutonomousPlanner(dt);
    this._integrateEgoDynamics(dt);

    // 3. Update Traffic Vehicles (scripted actors + dynamic IDM traffic + pooling)
    this._updateTraffic(dt);

    // 4. Sensor Sampling Clock (10 Hz LiDAR sweep, synchronized to simTime)
    const sensorDue =
      this.clock.lastScanTime < 0 ||
      this.clock.simTime - this.clock.lastScanTime >= LIDAR_SCAN_INTERVAL - 1e-6;

    if (this.lidarState.calibrationProgress > 0.05) {
      this.clock.scanAngleRad =
        (this.clock.scanAngleRad + Math.PI * 2 * LIDAR_SCAN_HZ * dt) % (Math.PI * 2);
    }

    if (sensorDue) {
      this.clock.lastScanTime = this.clock.simTime;
      if (!isFastForward) {
        this._runLiDARScan();
      } else {
        // Lightweight visibility check during fast scrub so tracks stay coherent
        this._runLiDARScanLightweight();
      }
      this._updatePerceptionAndTracking(LIDAR_SCAN_INTERVAL);
    } else {
      // Keep relative positions of existing tracks strictly synchronized between 10 Hz sweeps
      this._syncTrackVisualPositions();
    }

    // 5. Record Telemetry History every 0.25 s
    if (this.clock.tickCount % 15 === 0) {
      this._recordTelemetrySample();
    }

    // 6. Save Deterministic Snapshot every 1.0 s
    const currentSec = Math.floor(this.clock.simTime);
    if (currentSec > this._lastSnapshotSec) {
      this._saveSnapshot(currentSec);
    }
  }

  _updatePhaseMetadata() {
    const t = this.clock.simTime;
    const phases = this.scenarioDef.phases;
    let activeIdx = 0;
    for (let i = 0; i < phases.length; i++) {
      if (t >= phases[i].start) activeIdx = i;
    }
    const p = phases[activeIdx];
    this.currentPhase = {
      index: activeIdx,
      total: phases.length,
      name: p.name,
      detail: p.detail,
      spotlightView: p.spotlight,
    };

    if (this.scenarioDef.id === "guided_sequence") {
      if (t < 3.5) {
        this.lidarState.calibrationProgress = 0;
      } else if (t < 7.5) {
        this.lidarState.calibrationProgress = Math.min(1, (t - 3.5) / 3.2);
      } else {
        this.lidarState.calibrationProgress = 1;
      }
    } else {
      this.lidarState.calibrationProgress = 1;
    }

    if (activeIdx !== this._lastLoggedPhase) {
      this._lastLoggedPhase = activeIdx;
      this._logEvent("SYSTEM", `Phase 0${activeIdx + 1}: ${p.name}`);
    }
  }

  /**
   * Rule-Based Autonomous Longitudinal Planner (IDM + TTC-based AEB)
   * Explicitly rule-based mathematical control — never fakes ML predictions.
   */
  _updateAutonomousPlanner(dt) {
    const t = this.clock.simTime;
    const ego = this.ego;

    // Determine lane occupancy within [-18 m, +55 m]
    ego.laneOccupancy[0] = false;
    ego.laneOccupancy[1] = false;
    ego.laneOccupancy[2] = false;

    let leadVehicle = null;
    let leadGap = Infinity;
    let minTtc = Infinity;

    for (const v of this.vehiclePool) {
      if (!v.active) continue;
      const relS = v.s - ego.s;

      // Lane occupancy check
      for (let l = 0; l < 3; l++) {
        if (Math.abs(v.x - LANE_CENTERS[l]) < 1.85 && relS > -18 && relS < 60) {
          ego.laneOccupancy[l] = true;
        }
      }

      // Same-lane lead vehicle detection (also catches vehicles actively cutting into ego lane)
      const lateralOffset = Math.abs(v.x - ego.x);
      const isInEgoCorridor = lateralOffset < (ego.width * 0.5 + v.width * 0.5) * 0.88;

      if (isInEgoCorridor && relS > 0) {
        const bumperGap = relS - (ego.length * 0.5 + v.length * 0.5);
        if (bumperGap < leadGap) {
          leadGap = Math.max(0.1, bumperGap);
          leadVehicle = v;
        }
        const closingSpeed = ego.speed - v.speed;
        if (closingSpeed > 0.4) {
          const ttc = Math.max(0.1, bumperGap / closingSpeed);
          if (ttc < minTtc) minTtc = ttc;
        }
      }
    }

    ego.leadVehicleId = leadVehicle ? leadVehicle.id : null;
    ego.leadGap = leadGap;
    ego.minTtc = minTtc;

    // Guided Sequence startup phases (Standstill during System Init & Sensor Calibration)
    if (this.scenarioDef.id === "guided_sequence" && t < 7.5) {
      ego.targetSpeed = 0;
      ego.accel = 0;
      ego.speed = 0;
      ego.brakeLight = true;
      ego.controlState = t < 3.5 ? "INIT" : "CALIBRATING";
      ego.controlReason =
        t < 3.5 ? "System diagnostics & bus sync" : "360° LiDAR horizon calibration";
      this._buildPlannedPath();
      return;
    }

    // Desired free-flow speed
    const desiredV = ego.baseTargetSpeed;
    const minStandstillGap = 5.5;
    const desiredTimeHeadway = 1.65; // seconds
    const maxAccel = 2.4; // m/s^2
    const comfortDecel = 2.8; // m/s^2
    const aebDecel = 8.6; // m/s^2

    ego.safeFollowDist = minStandstillGap + ego.speed * desiredTimeHeadway;

    // Intelligent Driver Model (IDM) longitudinal acceleration
    const freeTerm = 1 - Math.pow(ego.speed / Math.max(1, desiredV), 4);
    let cmdAccel = maxAccel * freeTerm;
    let state = this.scenarioDef.id === "guided_sequence" && t < 13.5 ? "LAUNCH" : "CRUISING";
    let reason = state === "LAUNCH" ? "Controlled highway merge acceleration" : "Clear corridor — tracking target speed";

    if (leadVehicle && leadGap < 110) {
      const deltaV = ego.speed - leadVehicle.speed;
      const sStar =
        minStandstillGap +
        Math.max(
          0,
          ego.speed * desiredTimeHeadway +
            (ego.speed * deltaV) / (2 * Math.sqrt(maxAccel * comfortDecel))
        );
      const interactionTerm = Math.pow(sStar / Math.max(1.2, leadGap), 2);
      const idmAccel = maxAccel * (freeTerm - interactionTerm);

      // Rule-based AEB Trigger
      if (minTtc < 2.6 && (leadGap < 26 || deltaV > 6.5)) {
        cmdAccel = -aebDecel;
        state = "AEB_BRAKING";
        reason = `TTC ${minTtc.toFixed(2)}s — Emergency braking active`;
      } else if (leadGap < ego.safeFollowDist * 1.15 && deltaV > 0.8 && Math.abs(leadVehicle.turnSignal) > 0) {
        cmdAccel = Math.max(-4.8, idmAccel);
        state = "CUT_IN_YIELD";
        reason = `Yielding to lateral cut-in (TRK-${String(leadVehicle.id).padStart(2, "0")})`;
      } else if (idmAccel < cmdAccel - 0.15 || leadGap < ego.safeFollowDist * 1.25) {
        cmdAccel = Math.max(-4.5, idmAccel);
        state = "ACC_FOLLOW";
        reason = `Maintaining ${desiredTimeHeadway.toFixed(1)}s headway to TRK-${String(leadVehicle.id).padStart(2, "0")}`;
      }
    }

    // Clamp & smooth jerk
    cmdAccel = Math.max(-aebDecel, Math.min(maxAccel, cmdAccel));
    if (dt > 0) {
      const maxJerk = state === "AEB_BRAKING" ? 28.0 : 6.5;
      const diff = cmdAccel - ego.accel;
      const maxStep = maxJerk * dt;
      ego.accel += Math.max(-maxStep, Math.min(maxStep, diff));
    }

    // Prevent negative velocity
    if (ego.speed <= 0.05 && ego.accel < 0) {
      ego.accel = 0;
      ego.speed = 0;
    }

    ego.targetSpeed = desiredV;
    ego.controlState = state;
    ego.controlReason = reason;
    ego.brakeLight = ego.accel < -0.65 || ego.speed < 0.2;

    if (state !== this._lastLoggedState) {
      this._lastLoggedState = state;
      const cat = state === "AEB_BRAKING" ? "WARNING" : "CONTROL";
      this._logEvent(cat, `${state.replace(/_/g, " ")} — ${reason}`);
    }

    this._buildPlannedPath();
  }

  _integrateEgoDynamics(dt) {
    const ego = this.ego;
    ego.speed = Math.max(0, ego.speed + ego.accel * dt);
    const ds = ego.speed * dt;
    ego.s += ds;

    // Physically exact wheel angular displacement: dTheta = ds / R_wheel
    if (ds > 0) {
      ego.wheelAngle = (ego.wheelAngle + ds / ego.wheelRadius) % (Math.PI * 2);
    }
  }

  _buildPlannedPath() {
    const ego = this.ego;
    const pts = [];
    const horizon = Math.min(85, Math.max(18, ego.speed * 2.5));
    const clampedHorizon = Math.min(horizon, Math.max(6, ego.leadGap - 2.0));
    const steps = 16;
    for (let i = 0; i <= steps; i++) {
      const frac = i / steps;
      pts.push({
        x: ego.x,
        z: -(frac * clampedHorizon),
        dist: frac * clampedHorizon,
      });
    }
    ego.plannedPath = pts;
  }

  _updateTraffic(dt) {
    const t = this.clock.simTime;
    const ego = this.ego;

    // Scripted scenario maneuvers (deterministic & smooth)
    for (const v of this.vehiclePool) {
      if (!v.active) continue;

      if (v.scripted) {
        if (v.scriptRole === "guided_cutin") {
          // Keeps pace near ego until t=25s, then executes smooth cut-in into center lane
          if (t < 24.5) {
            const desiredRel = 16;
            const err = desiredRel - (v.s - ego.s);
            v.targetSpeed = Math.max(18, Math.min(34, ego.speed + err * 0.45));
          } else if (t >= 24.5 && v.targetLane === 2) {
            v.targetLane = 1;
            v.laneChangeStartX = v.x;
            v.laneChangeTimer = 0;
            v.laneChangeProgress = 0;
            v.turnSignal = -1;
            v.targetSpeed = 84 / 3.6;
            this._logEvent("PERCEPTION", `Lateral lane change initiated by TRK-${String(v.id).padStart(2, "0")}`);
          } else if (t > 36) {
            v.targetSpeed = 104 / 3.6;
          }
        } else if (v.scriptRole === "cutin_actor") {
          if (t < 7.2) {
            v.targetSpeed = 96 / 3.6;
          } else if (t >= 7.2 && v.targetLane === 2) {
            v.targetLane = 1;
            v.laneChangeStartX = v.x;
            v.laneChangeTimer = 0;
            v.laneChangeProgress = 0;
            v.turnSignal = -1;
            v.targetSpeed = 88 / 3.6;
            this._logEvent("PERCEPTION", `Cut-in detected: TRK-${String(v.id).padStart(2, "0")} merging into Ego lane`);
          } else if (t > 24.0) {
            v.targetSpeed = 108 / 3.6;
          }
        } else if (v.scriptRole === "occluded_target") {
          // Stays hidden in truck's LiDAR shadow until t=20s, then moves laterally into center lane
          if (t < 19.0) {
            v.targetSpeed = 88 / 3.6;
          } else if (t >= 19.0 && v.targetLane === 2) {
            v.targetLane = 1;
            v.laneChangeStartX = v.x;
            v.laneChangeTimer = 0;
            v.laneChangeProgress = 0;
            v.turnSignal = -1;
            v.targetSpeed = 96 / 3.6;
            this._logEvent("PERCEPTION", `Target TRK-${String(v.id).padStart(2, "0")} emerging from truck occlusion shadow`);
          }
        } else if (v.scriptRole === "aeb_lead") {
          if (t < 7.2) {
            v.targetSpeed = 100 / 3.6;
          } else if (t >= 7.2 && t < 17.5) {
            v.targetSpeed = 6 / 3.6;
            v.accel = -7.2;
          } else {
            v.targetSpeed = 88 / 3.6;
          }
        }
      }

      // Compute safe longitudinal acceleration against nearest vehicle ahead in same lane
      let aheadGap = Infinity;
      let aheadSpeed = v.targetSpeed;
      for (const other of this.vehiclePool) {
        if (!other.active || other === v) continue;
        if (Math.abs(other.x - v.x) < 2.2) {
          const gap = other.s - v.s - (v.length * 0.5 + other.length * 0.5);
          if (gap > 0 && gap < aheadGap) {
            aheadGap = gap;
            aheadSpeed = other.speed;
          }
        }
      }
      // Also respect Ego vehicle when in the same lane behind Ego
      if (Math.abs(ego.x - v.x) < 2.2) {
        const gapToEgo = ego.s - v.s - (v.length * 0.5 + ego.length * 0.5);
        if (gapToEgo > 0 && gapToEgo < aheadGap) {
          aheadGap = gapToEgo;
          aheadSpeed = ego.speed;
        }
      }

      const isHardBrakingLead = v.scriptRole === "aeb_lead" && t >= 7.2 && t < 17.5;
      if (!isHardBrakingLead) {
        const desiredGap = 8.0 + v.speed * 1.4;
        let accel = Math.max(-2.5, Math.min(2.0, (v.targetSpeed - v.speed) * 0.7));
        if (aheadGap < desiredGap * 1.35) {
          const dv = v.speed - aheadSpeed;
          const brake = -2.2 * Math.pow(desiredGap / Math.max(2.0, aheadGap), 1.6) - dv * 0.35;
          accel = Math.min(accel, Math.max(-5.5, brake));
        }
        v.accel = accel;
      }

      v.speed = Math.max(0, v.speed + v.accel * dt);
      const ds = v.speed * dt;
      v.s += ds;
      v.relS = v.s - ego.s;
      v.brakeLight = v.accel < -1.1;

      // Exact wheel rotation linked to vehicle speed
      if (ds > 0) {
        v.wheelAngle = (v.wheelAngle + ds / v.wheelRadius) % (Math.PI * 2);
      }

      // Smooth cosine-eased lane change kinematics
      if (v.laneChangeProgress < 1) {
        v.laneChangeTimer += dt;
        const p = Math.min(1, v.laneChangeTimer / v.laneChangeDuration);
        v.laneChangeProgress = p;
        const targetX = LANE_CENTERS[v.targetLane];
        const ease = 0.5 - 0.5 * Math.cos(Math.PI * p);
        const prevX = v.x;
        v.x = v.laneChangeStartX + (targetX - v.laneChangeStartX) * ease;
        const latVel = (v.x - prevX) / Math.max(1e-4, dt);
        v.steeringAngle = Math.atan2(latVel, Math.max(4, v.speed));
        if (p >= 1) {
          v.laneIndex = v.targetLane;
          v.x = targetX;
          v.turnSignal = 0;
          v.steeringAngle = 0;
        }
      } else {
        v.steeringAngle = 0;
      }

      // Despawn non-scripted vehicles well outside visible & sensor range (no visible popping)
      if (!v.scripted && (v.relS > 220 || v.relS < -175)) {
        v.active = false;
      }
    }

    // Dynamic Traffic Spawner — guarantees sustained, realistic highway flow
    this._spawnCooldown -= dt;
    if (this._spawnCooldown <= 0) {
      const targetCount =
        this.scenarioDef.id === "dense_traffic"
          ? 12
          : this.trafficDensity === "low"
            ? 5
            : this.trafficDensity === "dense"
              ? 11
              : 8;

      const activeCount = this.vehiclePool.reduce((acc, v) => acc + (v.active ? 1 : 0), 0);
      if (activeCount < targetCount) {
        this._spawnAmbientVehicle();
      }
      this._spawnCooldown =
        this.trafficDensity === "dense" ? 1.15 : this.trafficDensity === "low" ? 2.6 : 1.75;
    }
  }

  _spawnAmbientVehicle() {
    // Choose a lane and spawn well beyond LiDAR & camera horizon so vehicles never pop in
    const laneChoices =
      this.scenarioDef.id === "acc_following" || this.scenarioDef.id === "aeb_braking"
        ? [0, 2] // Keep center lane clear for scripted longitudinal test
        : [0, 0, 1, 2, 2];

    const lane = this.rng.pick(laneChoices);
    const speedKmh =
      lane === 0
        ? this.rng.range(112, 134)
        : lane === 1
          ? this.rng.range(92, 108)
          : this.rng.range(74, 90);
    const speedMs = speedKmh / 3.6;

    // Fast vehicles enter from far behind; slower vehicles enter from far ahead
    const fromBehind = speedMs > this.ego.speed + 2.0;
    const relS = fromBehind ? this.rng.range(-155, -128) : this.rng.range(148, 185);

    // Ensure safe clearance from existing vehicles in that lane
    for (const other of this.vehiclePool) {
      if (!other.active) continue;
      if (Math.abs(other.x - LANE_CENTERS[lane]) < 2.2) {
        if (Math.abs(other.relS - relS) < 34) return;
      }
    }

    const type =
      lane === 2
        ? this.rng.pick(["truck", "van", "suv", "sedan"])
        : lane === 0
          ? this.rng.pick(["sedan", "sedan", "suv"])
          : this.rng.pick(["sedan", "suv", "van"]);

    this._acquireVehicle({
      type,
      laneIndex: lane,
      relS,
      speed: speedMs,
    });
  }

  /**
   * True 3D Occluded LiDAR Raycaster (Analytical Ray-AABB + Ground + Guardrails)
   * Every point corresponds strictly to real geometry in the world.
   */
  _runLiDARScan() {
    const ls = this.lidarState;
    const calib = ls.calibrationProgress;
    const posBuf = this.lidarBuffers.pos;
    const distBuf = this.lidarBuffers.dist;
    const intBuf = this.lidarBuffers.intensity;
    const objBuf = this.lidarBuffers.objId;
    const rayEndBuf = this.lidarBuffers.raySampleEnds;
    const rayHitBuf = this.lidarBuffers.raySampleHits;

    this._vehDirectHits = new Map();
    this._vehUnoccludedHits = new Map();

    if (calib <= 0.02) {
      ls.activePoints = 0;
      ls.hitVehiclePoints = 0;
      ls.hitGroundPoints = 0;
      ls.raySampleCount = 0;
      return;
    }

    const channels = Math.max(6, Math.round(this.quality.channels * this.lidarDensityScale));
    const azSteps = Math.max(72, Math.round(this.quality.azSteps * this.lidarDensityScale));
    ls.channels = channels;
    ls.azSteps = azSteps;

    const maxRange = LIDAR_MAX_RANGE * Math.min(1, 0.25 + calib * 0.75);
    const ox = this.ego.x;
    const oy = LIDAR_MOUNT_HEIGHT;
    const oz = 0;

    // Precompute AABBs and angular bounding sectors for active vehicles within range
    const boxes = [];
    for (const v of this.vehiclePool) {
      if (!v.active) continue;
      const cz = -(v.s - this.ego.s);
      if (Math.abs(cz) > maxRange + v.length) continue;

      const hw = v.width * 0.5;
      const hl = v.length * 0.5;
      const minX = v.x - hw;
      const maxX = v.x + hw;
      const minY = 0.08;
      const maxY = v.height;
      const minZ = cz - hl;
      const maxZ = cz + hl;

      const cx = v.x - ox;
      const centerDist = Math.hypot(cx, cz);
      const boundRadius = Math.hypot(hw, hl) + 0.5;
      const centerAz = Math.atan2(cx, -cz);
      const halfSpan = centerDist > boundRadius ? Math.asin(Math.min(1, boundRadius / centerDist)) : Math.PI;

      boxes.push({
        id: v.id,
        minX,
        maxX,
        minY,
        maxY,
        minZ,
        maxZ,
        reflectivity: v.reflectivity,
        centerAz,
        halfSpan,
      });
    }

    const vMinRad = (ls.vfovMinDeg * Math.PI) / 180;
    const vMaxRad = (ls.vfovMaxDeg * Math.PI) / 180;
    const maxBufferPts = distBuf.length;

    let ptCount = 0;
    let vehPts = 0;
    let gndPts = 0;
    let sampleCount = 0;
    const sampleStride = this.quality.raySampleStride;
    const noiseAmp = this.lidarNoiseEnabled ? 0.08 : 0.0;

    // Deterministic noise hash per scan
    let noiseSeed = (this.clock.tickCount * 2654435761) >>> 0;
    const fastNoise = () => {
      noiseSeed = (Math.imul(noiseSeed ^ (noiseSeed >>> 15), 2246822519) + 3266489917) >>> 0;
      return (noiseSeed / 4294967296 - 0.5) * 2;
    };

    for (let a = 0; a < azSteps; a++) {
      const az = -Math.PI + (a / azSteps) * Math.PI * 2;
      const sinAz = Math.sin(az);
      const cosAz = Math.cos(az);

      // Filter candidate vehicles whose angular span covers this azimuth
      const candidates = [];
      for (let b = 0; b < boxes.length; b++) {
        let dAz = az - boxes[b].centerAz;
        while (dAz > Math.PI) dAz -= Math.PI * 2;
        while (dAz < -Math.PI) dAz += Math.PI * 2;
        if (Math.abs(dAz) <= boxes[b].halfSpan) {
          candidates.push(boxes[b]);
        }
      }

      for (let c = 0; c < channels; c++) {
        if (ptCount >= maxBufferPts) break;
        const frac = channels > 1 ? c / (channels - 1) : 0.5;
        const elev = vMinRad + frac * (vMaxRad - vMinRad);
        const cosEl = Math.cos(elev);
        const sinEl = Math.sin(elev);

        const dx = sinAz * cosEl;
        const dy = sinEl;
        const dz = -cosAz * cosEl;

        let bestT = maxRange;
        let hitObjId = -99; // miss
        let hitIntensity = 0.25;

        // 1. Ground plane intersection (y = 0)
        if (dy < -0.008) {
          const tG = -oy / dy;
          if (tG > 2.2 && tG < bestT) {
            const gx = ox + dx * tG;
            if (Math.abs(gx) <= ROAD_HALF_WIDTH + 1.2) {
              bestT = tG;
              hitObjId = -1;
              hitIntensity = 0.28 + 0.14 * Math.exp(-tG / 45);
            }
          }
        }

        // 2. Highway Guardrails (x = +/- ROAD_HALF_WIDTH, y in [0.15, 0.92])
        if (Math.abs(dx) > 0.02) {
          const wallX = dx > 0 ? ROAD_HALF_WIDTH : -ROAD_HALF_WIDTH;
          const tW = (wallX - ox) / dx;
          if (tW > 2.0 && tW < bestT) {
            const wy = oy + dy * tW;
            if (wy >= 0.12 && wy <= 0.95) {
              bestT = tW;
              hitObjId = -2;
              hitIntensity = 0.55;
            }
          }
        }

        // 3. Vehicle AABB Intersections (True Occlusion — smallest t wins!)
        let hitVeh = null;
        for (let b = 0; b < candidates.length; b++) {
          const box = candidates[b];
          const tV = intersectRayAABB(
            ox, oy, oz, dx, dy, dz,
            box.minX, box.maxX, box.minY, box.maxY, box.minZ, box.maxZ,
            maxRange
          );
          if (tV > 0) {
            // Track unoccluded potential hits so we can measure exact occlusion ratio
            this._vehUnoccludedHits.set(box.id, (this._vehUnoccludedHits.get(box.id) || 0) + 1);
            if (tV < bestT) {
              bestT = tV;
              hitObjId = box.id;
              hitIntensity = box.reflectivity * (0.65 + 0.35 * Math.exp(-tV / 75));
              hitVeh = box;
            }
          }
        }

        if (hitVeh) {
          this._vehDirectHits.set(hitVeh.id, (this._vehDirectHits.get(hitVeh.id) || 0) + 1);
          vehPts++;
        } else if (hitObjId !== -99) {
          gndPts++;
        }

        if (hitObjId !== -99) {
          const noisyT = noiseAmp > 0 ? Math.max(0.5, bestT + fastNoise() * noiseAmp) : bestT;
          const px = ox + dx * noisyT;
          const py = Math.max(0.02, oy + dy * noisyT);
          const pz = oz + dz * noisyT;

          const i3 = ptCount * 3;
          posBuf[i3] = px;
          posBuf[i3 + 1] = py;
          posBuf[i3 + 2] = pz;
          distBuf[ptCount] = noisyT;
          intBuf[ptCount] = hitIntensity;
          objBuf[ptCount] = hitObjId;
          ptCount++;
        }

        // Store sparse ray fan endpoints for optional View 01 / View 02 beam visualization
        if (c === Math.floor(channels * 0.45) && a % Math.max(2, Math.floor(azSteps / 48)) === 0 && sampleCount < 256) {
          const s3 = sampleCount * 3;
          const rT = hitObjId !== -99 ? bestT : Math.min(maxRange, 42);
          rayEndBuf[s3] = ox + dx * rT;
          rayEndBuf[s3 + 1] = Math.max(0.05, oy + dy * rT);
          rayEndBuf[s3 + 2] = oz + dz * rT;
          rayHitBuf[sampleCount] = hitVeh ? 2 : hitObjId !== -99 ? 1 : 0;
          sampleCount++;
        }
      }
    }

    ls.activePoints = ptCount;
    ls.hitVehiclePoints = vehPts;
    ls.hitGroundPoints = gndPts;
    ls.raySampleCount = sampleCount;
    ls.noiseEnabled = this.lidarNoiseEnabled;
  }

  /**
   * Fast visibility approximation used only during timeline scrubbing intermediate steps.
   */
  _runLiDARScanLightweight() {
    this._vehDirectHits = new Map();
    this._vehUnoccludedHits = new Map();
    for (const v of this.vehiclePool) {
      if (!v.active) continue;
      const dist = Math.hypot(v.x - this.ego.x, v.s - this.ego.s);
      if (dist <= LIDAR_MAX_RANGE) {
        const estHits = Math.max(4, Math.round(1400 / Math.max(10, dist)));
        this._vehDirectHits.set(v.id, estHits);
        this._vehUnoccludedHits.set(v.id, estHits);
      }
    }
  }

  /**
   * Object Detection & Multi-Object Tracking derived strictly from LiDAR returns.
   */
  _updatePerceptionAndTracking(dtScan) {
    const activeIds = new Set();

    for (const v of this.vehiclePool) {
      if (!v.active) continue;
      const relS = v.s - this.ego.s;
      const relX = v.x - this.ego.x;
      const dist = Math.hypot(relX, relS);
      if (dist > LIDAR_MAX_RANGE + 5) continue;

      const hits = this._vehDirectHits.get(v.id) || 0;
      const potential = this._vehUnoccludedHits.get(v.id) || 0;
      const occludedRatio =
        potential > 0 ? Math.max(0, Math.min(1, 1 - hits / potential)) : dist < LIDAR_MAX_RANGE ? 1 : 0;

      let existing = this._trackMap.get(v.id);

      // Acquire track if sufficient LiDAR points hit the object
      if (!existing && hits >= 4) {
        existing = {
          id: `TRK-${String(v.id).padStart(2, "0")}`,
          vehicleId: v.id,
          type: v.type,
          label: v.label,
          age: 0,
          missedScans: 0,
          history: [],
        };
        this._trackMap.set(v.id, existing);
      }

      // Also expose fully occluded objects inside sensor range ONLY if previously seen or in occlusion scenario
      if (!existing && occludedRatio > 0.85 && potential >= 4 && this.scenarioDef.id === "sensor_occlusion") {
        existing = {
          id: `TRK-${String(v.id).padStart(2, "0")}`,
          vehicleId: v.id,
          type: v.type,
          label: v.label,
          age: 0,
          missedScans: 3,
          history: [],
        };
        this._trackMap.set(v.id, existing);
      }

      if (!existing) continue;

      activeIds.add(v.id);
      existing.age += dtScan;
      if (hits >= 4) {
        existing.missedScans = 0;
      } else {
        existing.missedScans++;
      }

      const relSpeed = v.speed - this.ego.speed; // Negative = closing toward ego
      const closingSpeed = -relSpeed;
      const inSameCorridor = Math.abs(relX) < 2.6 && relS > 0;
      const bumperGap = Math.max(0.1, relS - (this.ego.length * 0.5 + v.length * 0.5));
      const ttc = inSameCorridor && closingSpeed > 0.5 ? bumperGap / closingSpeed : Infinity;

      // Analytical confidence modeled from return density & range (NOT a neural network score)
      const hitFactor = Math.min(1, hits / 34);
      const rangeFactor = Math.max(0.25, 1 - dist / (LIDAR_MAX_RANGE * 1.18));
      const confidence = hits >= 4 ? Math.min(0.99, 0.32 + hitFactor * 0.56 + rangeFactor * 0.11) : 0.0;

      let status = "CONFIRMED";
      if (hits < 4 && occludedRatio > 0.7) {
        status = "OCCLUDED";
      } else if (hits < 4) {
        status = "COASTING";
      } else if (occludedRatio > 0.35) {
        status = "PARTIAL";
      }

      existing.relS = relS;
      existing.x = v.x;
      existing.distance = dist;
      existing.relSpeed = relSpeed;
      existing.absSpeed = v.speed;
      existing.ttc = ttc;
      existing.hitCount = hits;
      existing.potentialHits = potential;
      existing.occludedRatio = occludedRatio;
      existing.confidence = confidence;
      existing.status = status;
      existing.turnSignal = v.turnSignal;
      existing.brakeLight = v.brakeLight;
      existing.bbox = {
        cx: v.x,
        cy: v.height * 0.5,
        cz: -relS,
        l: v.length,
        w: v.width,
        h: v.height,
      };

      if (dtScan > 0 || existing.history.length === 0) {
        existing.history.push({ x: v.x, z: -relS, s: v.s });
        if (existing.history.length > 26) existing.history.shift();
      }
    }

    // Prune stale tracks cleanly
    for (const [vid, trk] of this._trackMap.entries()) {
      if (!activeIds.has(vid) || (trk.missedScans > 18 && this.scenarioDef.id !== "sensor_occlusion")) {
        this._trackMap.delete(vid);
      }
    }

    this.tracks = Array.from(this._trackMap.values()).sort((a, b) => a.distance - b.distance);
  }

  _syncTrackVisualPositions() {
    for (const trk of this.tracks) {
      const v = this.vehiclePool.find((veh) => veh.active && veh.id === trk.vehicleId);
      if (!v) continue;
      const relS = v.s - this.ego.s;
      trk.relS = relS;
      trk.x = v.x;
      trk.distance = Math.hypot(v.x - this.ego.x, relS);
      trk.relSpeed = v.speed - this.ego.speed;
      trk.absSpeed = v.speed;
      trk.bbox.cx = v.x;
      trk.bbox.cz = -relS;
    }
  }

  _recordTelemetrySample() {
    this.ego.speedHistory.push({
      t: this.clock.simTime,
      speedKmh: this.ego.speed * 3.6,
      targetKmh: this.ego.targetSpeed * 3.6,
      accel: this.ego.accel,
      leadGap: Number.isFinite(this.ego.leadGap) ? Math.min(120, this.ego.leadGap) : 120,
    });
    if (this.ego.speedHistory.length > 90) {
      this.ego.speedHistory.shift();
    }
  }

  _logEvent(category, message) {
    this.events.unshift({
      time: this.clock.simTime,
      category,
      message,
    });
    if (this.events.length > 24) {
      this.events.pop();
    }
  }

  _saveSnapshot(secInt) {
    this._lastSnapshotSec = secInt;
    this._snapshots.set(secInt, {
      rngState: this.rng.cloneState(),
      nextVehicleId: this._nextVehicleId,
      spawnCooldown: this._spawnCooldown,
      lastLoggedState: this._lastLoggedState,
      lastLoggedPhase: this._lastLoggedPhase,
      clock: { ...this.clock },
      ego: {
        ...this.ego,
        laneOccupancy: [...this.ego.laneOccupancy],
        plannedPath: this.ego.plannedPath.map((p) => ({ ...p })),
        speedHistory: this.ego.speedHistory.map((h) => ({ ...h })),
      },
      vehicles: this.vehiclePool.map((v) => ({ ...v })),
      events: this.events.map((e) => ({ ...e })),
    });
  }

  _restoreSnapshot(snap) {
    this.rng.restoreState(snap.rngState);
    this._nextVehicleId = snap.nextVehicleId;
    this._spawnCooldown = snap.spawnCooldown;
    this._lastLoggedState = snap.lastLoggedState;
    this._lastLoggedPhase = snap.lastLoggedPhase;
    this.clock = { ...snap.clock };
    this.ego = {
      ...snap.ego,
      laneOccupancy: [...snap.ego.laneOccupancy],
      plannedPath: snap.ego.plannedPath.map((p) => ({ ...p })),
      speedHistory: snap.ego.speedHistory.map((h) => ({ ...h })),
    };
    for (let i = 0; i < this.vehiclePool.length; i++) {
      Object.assign(this.vehiclePool[i], snap.vehicles[i]);
    }
    this.events = snap.events.map((e) => ({ ...e }));
    this._trackMap.clear();
    this.tracks = [];
    this._lastSnapshotSec = Math.floor(this.clock.simTime);
  }

  getState() {
    const nearest = this.tracks.find((t) => t.status !== "OCCLUDED") || this.tracks[0] || null;
    const confirmedTracks = this.tracks.filter((t) => t.status === "CONFIRMED" || t.status === "PARTIAL").length;

    return {
      clock: this.clock,
      scenario: {
        id: this.scenarioDef.id,
        name: this.scenarioDef.name,
        shortName: this.scenarioDef.shortName,
        description: this.scenarioDef.description,
        phases: this.scenarioDef.phases,
        phase: this.currentPhase,
      },
      ego: this.ego,
      vehicles: this.vehiclePool,
      lidar: this.lidarState,
      tracks: this.tracks,
      events: this.events,
      telemetry: {
        egoSpeedKmh: this.ego.speed * 3.6,
        egoSpeedMs: this.ego.speed,
        accelMs2: this.ego.accel,
        scanRateHz: this.lidarState.calibrationProgress > 0.05 ? this.clock.sensorScanHz : 0,
        activeTracks: confirmedTracks,
        totalTracked: this.tracks.length,
        detectionRangeM: LIDAR_MAX_RANGE * this.lidarState.calibrationProgress,
        minTtcS: this.ego.minTtc,
        nearestDistM: nearest ? nearest.distance : Infinity,
        nearestRelVelMs: nearest ? nearest.relSpeed : 0,
        controlState: this.ego.controlState,
        simTimeS: this.clock.simTime,
      },
    };
  }
}