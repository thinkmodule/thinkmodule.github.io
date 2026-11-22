/**
 * ThinkModule — Autonomous Driving & 3D LiDAR Deterministic Simulation Engine
 * ============================================================================
 * Headless fixed-timestep (60 Hz) simulation engine with:
 *  - Strict separation of Wall-Clock, Simulation-Time, and 10 Hz LiDAR Sampling
 *  - Calm 1.6 s cinematic sensor spin-up on load / scenario switch
 *  - Guaranteed physical anti-overlap collision constraint (no vehicle clipping)
 *  - Analytical 3D Ray-AABB occluded LiDAR raycaster (up to 6,000+ real points)
 *  - Explicit rule-based IDM headway & TTC-based AEB longitudinal control
 *  - Deterministic seeded PRNG + 1 Hz snapshot cache for exact timeline scrubbing
 */

export const FIXED_DT = 1 / 60;
export const MAX_WALL_DT = 0.08;
export const LANE_WIDTH = 3.75;
export const LANE_CENTERS = [-3.75, 0.0, 3.75]; // 0: Left/Fast, 1: Center, 2: Right/Slow
export const ROAD_HALF_WIDTH = 6.6;
export const LAMP_SPACING = 38.0;
export const GANTRY_SPACING = 150.0;
export const EGO_WHEEL_RADIUS = 0.36;
export const LIDAR_MOUNT_HEIGHT = 1.76;
export const LIDAR_MAX_RANGE = 115.0;
export const LIDAR_SCAN_HZ = 10.0;
export const LIDAR_SCAN_INTERVAL = 1 / LIDAR_SCAN_HZ;
export const INTRO_HOLD_SEC = 1.6; // Calm establishing beat before full motion

export const QUALITY_PRESETS = {
  ultra: {
    id: "ultra",
    label: "Ultra HD",
    channels: 22,
    azSteps: 260,
    maxDpr: 2,
  },
  high: {
    id: "high",
    label: "High HD",
    channels: 16,
    azSteps: 200,
    maxDpr: 1.75,
  },
  laptop: {
    id: "laptop",
    label: "Balanced",
    channels: 12,
    azSteps: 148,
    maxDpr: 1.25,
  },
};

export const VEHICLE_SPECS = {
  coupe: {
    type: "coupe",
    label: "Sport Coupe",
    length: 4.65,
    width: 1.9,
    height: 1.38,
    wheelRadius: 0.34,
    reflectivity: 0.78,
    palette: [0x8899ad, 0x3b4d63, 0xa8b4c2, 0x2c3847, 0x67798e],
  },
  sedan: {
    type: "sedan",
    label: "Executive Sedan",
    length: 4.85,
    width: 1.92,
    height: 1.46,
    wheelRadius: 0.35,
    reflectivity: 0.75,
    palette: [0x6e7b8b, 0x495464, 0x9aa6b5, 0x38414e, 0x5c6878],
  },
  suv: {
    type: "suv",
    label: "Premium SUV",
    length: 5.02,
    width: 2.04,
    height: 1.76,
    wheelRadius: 0.38,
    reflectivity: 0.8,
    palette: [0x4c5866, 0x73808f, 0x343e4a, 0x8994a2],
  },
  truck: {
    type: "truck",
    label: "Articulated Truck",
    length: 11.4,
    width: 2.54,
    height: 3.55,
    wheelRadius: 0.5,
    reflectivity: 0.92,
    palette: [0xc4cdd8, 0x9aa6b4, 0xb0bac6],
  },
};

export const SCENARIOS = [
  {
    id: "highway_flow",
    num: "01",
    shortTitle: "Highway Overtake",
    title: "Multi-Lane Highway Flow & 360° Tracking",
    duration: 36.0,
    seed: 104729,
    summary: "Real-world 3-lane autonomous cruising with continuous multi-vehicle perception.",
    phases: [
      {
        start: 0,
        end: 1.8,
        tag: "SENSOR LOCK",
        title: "Establishing 360° Perception Field",
        text: "Roof-mounted LiDAR dome spins up to 10 Hz and locks onto highway lanes and surrounding traffic geometry before nominal cruise.",
        sensorNote: "360° HFOV · 115 m Range · Ground Plane Calibrated",
      },
      {
        start: 1.8,
        end: 12.0,
        tag: "FAST OVERTAKE",
        title: "High-Speed Left-Lane Approach",
        text: "A faster sport coupe approaches from behind in the left lane at 128 km/h. Multi-object tracking measures positive closing rate and maintains lane separation.",
        sensorNote: "LiDAR Point Density Rising · Persistent Track Lock",
      },
      {
        start: 12.0,
        end: 24.0,
        tag: "MULTI-OBJECT FLOW",
        title: "Simultaneous 3-Lane Traffic Monitoring",
        text: "Ego holds 100 km/h in the center lane while tracking fast sedans on the left and heavy commercial trucks on the right without occlusion loss.",
        sensorNote: "Real-Time AABB Classification · Relative Velocity Vectors",
      },
      {
        start: 24.0,
        end: 36.0,
        tag: "SUSTAINED CRUISE",
        title: "Continuous Autonomous Corridor Verification",
        text: "Rule-based planner continuously evaluates lane occupancy and maintains a clear forward safety envelope as highway traffic flows smoothly past.",
        sensorNote: "IDM Corridor Clear · Zero-Latency State Sync",
      },
    ],
  },
  {
    id: "cut_in_acc",
    num: "02",
    shortTitle: "Cut-In & ACC Yield",
    title: "Sudden Lane Cut-In & Adaptive Cruise Control",
    duration: 32.0,
    seed: 412843,
    summary: "Adjacent vehicle merges ahead; perception tracks lateral motion and ACC yields cleanly.",
    phases: [
      {
        start: 0,
        end: 1.8,
        tag: "SENSOR LOCK",
        title: "Nominal Highway Baseline",
        text: "Ego stabilizes at 106 km/h in the center lane while monitoring a coupe in the right lane closing on a slower articulated truck.",
        sensorNote: "Center Corridor Clear · Right-Lane Closing Observed",
      },
      {
        start: 1.8,
        end: 7.2,
        tag: "PRE-MANEUVER",
        title: "Right-Lane Compression Detected",
        text: "The adjacent coupe reaches the slower truck ahead and prepares to overtake into Ego's center lane. Forward radar and LiDAR keep both vehicles locked.",
        sensorNote: "Tracking Parallel Vehicles · Lateral Margin Monitored",
      },
      {
        start: 7.2,
        end: 15.5,
        tag: "LATERAL CUT-IN",
        title: "Cut-In Intrusion & Smooth Yield",
        text: "Amber left signal activates as the coupe crosses the lane divider 24 m ahead. Ego detects corridor intrusion, activates brake lights, and smoothly yields via IDM.",
        sensorNote: "ACC Yield Active · Target Headway 1.7 s Enforced",
      },
      {
        start: 15.5,
        end: 32.0,
        tag: "GAP RECOVERY",
        title: "Equilibrium Safety Envelope Restored",
        text: "Ego matches the merging vehicle's velocity at a comfortable following distance, then gradually accelerates back toward target speed once the lane opens.",
        sensorNote: "Smooth Jerk-Limited Recovery · Zero Tailgating",
      },
    ],
  },
  {
    id: "truck_occlusion",
    num: "03",
    shortTitle: "LiDAR Occlusion",
    title: "Heavy Truck Blind-Spot & Shadow Emergence",
    duration: 34.0,
    seed: 623917,
    summary: "Real 3D raycast occlusion behind a heavy box truck and live target re-acquisition.",
    phases: [
      {
        start: 0,
        end: 1.8,
        tag: "SENSOR LOCK",
        title: "Line-of-Sight Geometry Active",
        text: "Ego cruises at 94 km/h behind a tall articulated box truck in the right lane. Switch to Sensor Fusion or LiDAR Cloud to inspect the physical shadow cone.",
        sensorNote: "True 3D Ray-AABB Occlusion Enabled",
      },
      {
        start: 1.8,
        end: 13.5,
        tag: "FULL OCCLUSION",
        title: "Target Hidden Inside Geometric Shadow",
        text: "The truck's 3.55 m trailer blocks every LiDAR laser ray behind it. A silver sedan driving ahead in that shadow yields 0 direct laser returns and cannot be seen optically.",
        sensorNote: "Shadow Cone Active · 100% Optical Occlusion Behind Truck",
      },
      {
        start: 13.5,
        end: 22.0,
        tag: "EMERGENCE",
        title: "Lateral Emergence & Instant Point Lock",
        text: "The hidden sedan shifts left into the center lane. The instant its left fender clears the truck's silhouette, LiDAR rays strike it and track confidence climbs rapidly.",
        sensorNote: "Direct Laser Returns Acquired · Track Promoted to Confirmed",
      },
      {
        start: 22.0,
        end: 34.0,
        tag: "TRACK CONFIRMED",
        title: "Full Visibility & Adaptive Headway",
        text: "With unobstructed line-of-sight restored, Ego locks onto the emerged sedan with full 3D bounding box geometry and adapts speed cleanly.",
        sensorNote: "Full Contour Reconstruction · Continuous Tracking",
      },
    ],
  },
  {
    id: "aeb_emergency",
    num: "04",
    shortTitle: "Emergency AEB",
    title: "Hard Lead Braking & Autonomous Emergency Stop",
    duration: 28.0,
    seed: 731587,
    summary: "Lead SUV brakes abruptly; TTC drops below threshold and rule-based AEB prevents impact.",
    phases: [
      {
        start: 0,
        end: 1.8,
        tag: "SENSOR LOCK",
        title: "Nominal Highway Following",
        text: "Ego follows a lead SUV in the center lane at 98 km/h with nominal clearance and active longitudinal monitoring.",
        sensorNote: "Lead Track Locked · TTC Nominal (∞)",
      },
      {
        start: 1.8,
        end: 6.8,
        tag: "APPROACH",
        title: "Steady Lead Tracking",
        text: "Both vehicles cruise at matching velocity while surrounding highway traffic flows in the adjacent left and right lanes.",
        sensorNote: "Relative Velocity ~0.0 m/s · Path Corridor Cyan",
      },
      {
        start: 6.8,
        end: 15.2,
        tag: "AEB INTERVENTION",
        title: "Sudden Lead Deceleration — AEB Engaged",
        text: "The lead SUV brakes hard (-6.8 m/s²). Closing speed spikes and Time-to-Collision plunges below 2.5 s; rule-based AEB commands -8.5 m/s² emergency braking.",
        sensorNote: "CRITICAL TTC · Full Hydraulic Deceleration · Corridor Crimson",
      },
      {
        start: 15.2,
        end: 28.0,
        tag: "SAFE STOP & RESUME",
        title: "Collision Avoided With Positive Margin",
        text: "Ego stabilizes smoothly with a positive safety margin behind the stopped SUV. Once the lead vehicle clears, Ego releases brakes and accelerates smoothly back to cruise.",
        sensorNote: "Zero Contact Verified · Controlled Cruise Resume",
      },
    ],
  },
];

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

function intersectRayAABB(ox, oy, oz, dx, dy, dz, minX, maxX, minY, maxY, minZ, maxZ, maxT) {
  let tmin = 0.45;
  let tmax = maxT;

  if (Math.abs(dx) < 1e-8) {
    if (ox < minX || ox > maxX) return -1;
  } else {
    const invD = 1 / dx;
    let t1 = (minX - ox) * invD;
    let t2 = (maxX - ox) * invD;
    if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
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
    if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
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
    if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
    if (t1 > tmin) tmin = t1;
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return -1;
  }

  return tmin <= tmax && tmin > 0.45 ? tmin : -1;
}

export class SimulationEngine {
  constructor(options = {}) {
    this.quality = QUALITY_PRESETS[options.quality || "high"] || QUALITY_PRESETS.high;

    // Pre-allocated point cloud buffers sized for Ultra HD (22 * 260 = 5720 pts)
    const maxPts = 24 * 280;
    this.lidarBuffers = {
      pos: new Float32Array(maxPts * 3),
      dist: new Float32Array(maxPts),
      intensity: new Float32Array(maxPts),
      objId: new Int16Array(maxPts), // -1 ground, -2 guardrail, >=0 vehicle id
    };

    this.poolSize = 14;
    this.vehiclePool = Array.from({ length: this.poolSize }, (_, i) => this._createEmptyVehicle(i));

    this._lastWallMs = null;
    this._accumulator = 0;
    this._snapshots = new Map();
    this._lastSnapshotSec = -1;

    this.setScenario(options.scenarioId || "highway_flow");
  }

  _createEmptyVehicle(poolIndex) {
    return {
      poolIndex,
      id: 0,
      active: false,
      scripted: false,
      scriptRole: null,
      type: "sedan",
      label: "Executive Sedan",
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
      laneChangeDuration: 3.1,
      laneChangeTimer: 0,
      wheelAngle: 0,
      wheelRadius: 0.35,
      length: 4.85,
      width: 1.92,
      height: 1.46,
      reflectivity: 0.75,
      brakeLight: false,
      turnSignal: 0, // -1 left, 0 off, 1 right
      steeringAngle: 0,
    };
  }

  setScenario(scenarioId) {
    this.scenarioDef = SCENARIOS.find((s) => s.id === scenarioId) || SCENARIOS[0];
    this.restart();
  }

  setQuality(presetKey) {
    if (QUALITY_PRESETS[presetKey]) {
      this.quality = QUALITY_PRESETS[presetKey];
      this._runLiDARScan();
      this._updatePerceptionAndTracking(0);
    }
  }

  setSpeed(mult) {
    if ([0.25, 0.5, 1, 1.5].includes(mult)) {
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
      introProgress: 0, // 0..1 smooth cinematic reveal during first 1.6s
    };

    for (const v of this.vehiclePool) {
      v.active = false;
      v.scripted = false;
      v.scriptRole = null;
      v.id = 0;
    }

    const baseKmh =
      this.scenarioDef.id === "highway_flow"
        ? 100
        : this.scenarioDef.id === "cut_in_acc"
          ? 106
          : this.scenarioDef.id === "truck_occlusion"
            ? 94
            : 98; // aeb_emergency

    const baseMs = baseKmh / 3.6;

    this.ego = {
      s: 0,
      x: LANE_CENTERS[1],
      speed: baseMs,
      targetSpeed: baseMs,
      baseTargetSpeed: baseMs,
      accel: 0,
      wheelAngle: 0,
      wheelRadius: EGO_WHEEL_RADIUS,
      length: 4.86,
      width: 1.94,
      height: 1.44,
      laneIndex: 1,
      controlState: "SENSOR_LOCK",
      safeFollowDist: 42,
      leadVehicleId: null,
      leadGap: Infinity,
      minTtc: Infinity,
      brakeLight: false,
      plannedPath: [],
      laneOccupancy: [false, false, false],
    };

    this.lidarState = {
      sensorHeight: LIDAR_MOUNT_HEIGHT,
      maxRange: LIDAR_MAX_RANGE,
      channels: this.quality.channels,
      azSteps: this.quality.azSteps,
      scanHz: LIDAR_SCAN_HZ,
      pulseRadius: 0,
      activePoints: 0,
      hitVehiclePoints: 0,
      pointsPos: this.lidarBuffers.pos,
      pointsDist: this.lidarBuffers.dist,
      pointsIntensity: this.lidarBuffers.intensity,
      pointsObjId: this.lidarBuffers.objId,
    };

    this._trackMap = new Map();
    this.tracks = [];
    this._spawnCooldown = 1.4;

    this._seedScenarioTraffic();
    this._enforceNoOverlap();
    this._updatePhaseMetadata();
    this._runLiDARScan();
    this._updatePerceptionAndTracking(0);
    this._updateAutonomousPlanner(0);
    this._saveSnapshot(0);
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
    slot.laneChangeDuration = cfg.laneChangeDuration ?? 3.0;
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

  _seedScenarioTraffic() {
    const id = this.scenarioDef.id;

    if (id === "highway_flow") {
      // Rich, balanced 3-lane highway traffic with generous clear lanes
      this._acquireVehicle({ type: "coupe", laneIndex: 0, relS: -46, speed: 128 / 3.6, colorHex: 0x9ab0c9 });
      this._acquireVehicle({ type: "suv", laneIndex: 0, relS: 44, speed: 122 / 3.6, colorHex: 0x5c6d80 });
      this._acquireVehicle({ type: "truck", laneIndex: 2, relS: 36, speed: 80 / 3.6, colorHex: 0xc2cbd6 });
      this._acquireVehicle({ type: "sedan", laneIndex: 2, relS: -26, speed: 84 / 3.6, colorHex: 0x738294 });
      this._acquireVehicle({ type: "suv", laneIndex: 1, relS: 74, speed: 102 / 3.6, colorHex: 0x8997a8 });
      this._acquireVehicle({ type: "coupe", laneIndex: 0, relS: -112, speed: 130 / 3.6, colorHex: 0x6f849c });
    } else if (id === "cut_in_acc") {
      // Right-lane coupe smoothly overtakes slower truck by merging 24m ahead of Ego
      this._acquireVehicle({
        type: "coupe",
        laneIndex: 2,
        relS: 22,
        speed: 104 / 3.6,
        scripted: true,
        scriptRole: "cutin_coupe",
        colorHex: 0x99b2cc,
      });
      this._acquireVehicle({
        type: "truck",
        laneIndex: 2,
        relS: 72,
        speed: 78 / 3.6,
        scripted: true,
        scriptRole: "cutin_truck",
        colorHex: 0xc4cdd8,
      });
      this._acquireVehicle({ type: "sedan", laneIndex: 0, relS: -48, speed: 126 / 3.6 });
      this._acquireVehicle({ type: "suv", laneIndex: 0, relS: 54, speed: 122 / 3.6 });
    } else if (id === "truck_occlusion") {
      // Massive box truck directly creates geometric shadow cone over the sedan ahead of it
      this._acquireVehicle({
        type: "truck",
        laneIndex: 2,
        relS: 25,
        speed: 90 / 3.6,
        scripted: true,
        scriptRole: "occluder_truck",
        colorHex: 0xc7d0db,
      });
      this._acquireVehicle({
        type: "sedan",
        laneIndex: 2,
        relS: 48,
        speed: 92 / 3.6,
        scripted: true,
        scriptRole: "occluded_sedan",
        colorHex: 0x9eb8d4,
      });
      this._acquireVehicle({ type: "coupe", laneIndex: 0, relS: -42, speed: 122 / 3.6 });
      this._acquireVehicle({ type: "suv", laneIndex: 0, relS: 62, speed: 118 / 3.6 });
    } else if (id === "aeb_emergency") {
      // Lead SUV brakes abruptly at t = 6.8s while adjacent lanes flow smoothly
      this._acquireVehicle({
        type: "suv",
        laneIndex: 1,
        relS: 40,
        speed: 98 / 3.6,
        scripted: true,
        scriptRole: "aeb_lead_suv",
        colorHex: 0xa4b2c2,
      });
      this._acquireVehicle({ type: "truck", laneIndex: 2, relS: 58, speed: 82 / 3.6 });
      this._acquireVehicle({ type: "coupe", laneIndex: 0, relS: -54, speed: 122 / 3.6 });
      this._acquireVehicle({ type: "sedan", laneIndex: 0, relS: 48, speed: 118 / 3.6 });
    }
  }

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

  stepSingleFrame() {
    this._stepFixed(FIXED_DT);
    this._runLiDARScan();
    this._updatePerceptionAndTracking(0);
  }

  seekToTime(targetTime) {
    const clamped = Math.max(0, Math.min(this.clock.duration, targetTime));
    const wasPaused = this.clock.paused;

    let bestSec = 0;
    for (const sec of this._snapshots.keys()) {
      if (sec <= clamped && sec >= bestSec) bestSec = sec;
    }

    const snap = this._snapshots.get(bestSec);
    if (snap) this._restoreSnapshot(snap);
    else this.restart();

    const ticksNeeded = Math.round((clamped - this.clock.simTime) / FIXED_DT);
    for (let i = 0; i < ticksNeeded; i++) {
      this._stepFixed(FIXED_DT, true);
    }

    this._runLiDARScan();
    this._updatePerceptionAndTracking(0);
    this._updateAutonomousPlanner(0);

    this.clock.paused = wasPaused;
    this._accumulator = 0;
  }

  _stepFixed(dt, isFastForward = false) {
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
    this.clock.introProgress = Math.min(1, this.clock.simTime / INTRO_HOLD_SEC);

    // Calm delayed roll-in: during the first 1.6s, physical world motion ramps gently from 0 -> 1
    // while LiDAR spins up cleanly, preventing an abrupt chaotic start on page load.
    const motionScale =
      this.clock.simTime < 0.55
        ? 0.0
        : this.clock.simTime < INTRO_HOLD_SEC
          ? 0.5 - 0.5 * Math.cos(Math.PI * ((this.clock.simTime - 0.55) / (INTRO_HOLD_SEC - 0.55)))
          : 1.0;

    const physDt = dt * motionScale;

    this._updatePhaseMetadata();
    this._updateAutonomousPlanner(physDt);
    this._integrateEgoDynamics(physDt);
    this._updateTraffic(physDt);

    // Hard physical non-overlap enforcement — guarantees zero vehicle clipping ever
    this._enforceNoOverlap();

    // Synchronized 10 Hz LiDAR dome & pulse ring (frozen on Pause!)
    this.clock.scanAngleRad =
      (this.clock.scanAngleRad + Math.PI * 2 * LIDAR_SCAN_HZ * dt) % (Math.PI * 2);
    this.lidarState.pulseRadius =
      ((this.clock.simTime * 38.0) % LIDAR_MAX_RANGE);

    const sensorDue =
      this.clock.lastScanTime < 0 ||
      this.clock.simTime - this.clock.lastScanTime >= LIDAR_SCAN_INTERVAL - 1e-6;

    if (sensorDue) {
      this.clock.lastScanTime = this.clock.simTime;
      if (!isFastForward) this._runLiDARScan();
      else this._runLiDARScanLightweight();
      this._updatePerceptionAndTracking(LIDAR_SCAN_INTERVAL);
    } else {
      this._syncTrackVisualPositions();
    }

    const currentSec = Math.floor(this.clock.simTime);
    if (currentSec > this._lastSnapshotSec) {
      this._saveSnapshot(currentSec);
    }
  }

  _updatePhaseMetadata() {
    const t = this.clock.simTime;
    const phases = this.scenarioDef.phases;
    let idx = 0;
    for (let i = 0; i < phases.length; i++) {
      if (t >= phases[i].start) idx = i;
    }
    const p = phases[idx];
    this.currentPhase = {
      index: idx,
      total: phases.length,
      tag: p.tag,
      title: p.title,
      text: p.text,
      sensorNote: p.sensorNote,
    };
  }

  _updateAutonomousPlanner(dt) {
    const ego = this.ego;

    ego.laneOccupancy[0] = false;
    ego.laneOccupancy[1] = false;
    ego.laneOccupancy[2] = false;

    let leadVehicle = null;
    let leadGap = Infinity;
    let minTtc = Infinity;

    for (const v of this.vehiclePool) {
      if (!v.active) continue;
      const relS = v.s - ego.s;

      for (let l = 0; l < 3; l++) {
        if (Math.abs(v.x - LANE_CENTERS[l]) < 1.9 && relS > -18 && relS < 58) {
          ego.laneOccupancy[l] = true;
        }
      }

      const latOverlap = Math.abs(v.x - ego.x) < (ego.width + v.width) * 0.48;
      if (latOverlap && relS > 0) {
        const bumperGap = Math.max(0.1, relS - (ego.length * 0.5 + v.length * 0.5));
        if (bumperGap < leadGap) {
          leadGap = bumperGap;
          leadVehicle = v;
        }
        const closing = ego.speed - v.speed;
        if (closing > 0.35) {
          const ttc = Math.max(0.1, bumperGap / closing);
          if (ttc < minTtc) minTtc = ttc;
        }
      }
    }

    ego.leadVehicleId = leadVehicle ? leadVehicle.id : null;
    ego.leadGap = leadGap;
    ego.minTtc = minTtc;

    if (this.clock.simTime < 0.55) {
      ego.controlState = "SENSOR_LOCK";
      ego.accel = 0;
      ego.brakeLight = false;
      this._buildPlannedPath();
      return;
    }

    const desiredV = ego.baseTargetSpeed;
    const minStandstillGap = 5.5;
    const desiredTimeHeadway = 1.65;
    const maxAccel = 2.4;
    const comfortDecel = 2.8;
    const aebDecel = 8.6;

    ego.safeFollowDist = minStandstillGap + ego.speed * desiredTimeHeadway;
    const freeTerm = 1 - Math.pow(ego.speed / Math.max(1, desiredV), 4);
    let cmdAccel = maxAccel * freeTerm;
    let state = "CRUISING";

    if (leadVehicle && leadGap < 110) {
      const deltaV = ego.speed - leadVehicle.speed;
      const sStar =
        minStandstillGap +
        Math.max(
          0,
          ego.speed * desiredTimeHeadway +
            (ego.speed * deltaV) / (2 * Math.sqrt(maxAccel * comfortDecel))
        );
      const idmAccel = maxAccel * (freeTerm - Math.pow(sStar / Math.max(1.2, leadGap), 2));

      if ((minTtc < 2.6 && deltaV > 5.0) || (leadGap < 14 && deltaV > 2.5)) {
        cmdAccel = -aebDecel;
        state = "AEB_BRAKING";
      } else if (leadGap < ego.safeFollowDist * 1.15 && Math.abs(leadVehicle.turnSignal) > 0) {
        cmdAccel = Math.max(-4.6, idmAccel);
        state = "CUT_IN_YIELD";
      } else if (idmAccel < cmdAccel - 0.15 || leadGap < ego.safeFollowDist * 1.2) {
        cmdAccel = Math.max(-4.5, idmAccel);
        state = "ACC_FOLLOW";
      }
    }

    cmdAccel = Math.max(-aebDecel, Math.min(maxAccel, cmdAccel));
    if (dt > 0) {
      const maxJerk = state === "AEB_BRAKING" ? 30.0 : 7.0;
      const diff = cmdAccel - ego.accel;
      const step = maxJerk * dt;
      ego.accel += Math.max(-step, Math.min(step, diff));
    }

    if (ego.speed <= 0.08 && ego.accel < 0) {
      ego.accel = 0;
      ego.speed = 0;
    }

    ego.targetSpeed = desiredV;
    ego.controlState = state;
    ego.brakeLight = ego.accel < -0.65 || (ego.speed < 0.5 && leadGap < 20);
    this._buildPlannedPath();
  }

  _integrateEgoDynamics(dt) {
    if (dt <= 0) return;
    const ego = this.ego;
    ego.speed = Math.max(0, ego.speed + ego.accel * dt);
    const ds = ego.speed * dt;
    ego.s += ds;
    if (ds > 0) {
      ego.wheelAngle = (ego.wheelAngle + ds / ego.wheelRadius) % (Math.PI * 2);
    }
  }

  _buildPlannedPath() {
    const ego = this.ego;
    const pts = [];
    const horizon = Math.min(78, Math.max(16, ego.speed * 2.3));
    const clampedHorizon = Math.min(horizon, Math.max(5.5, ego.leadGap - 1.8));
    const steps = 16;
    for (let i = 0; i <= steps; i++) {
      const frac = i / steps;
      pts.push({ x: ego.x, z: -(frac * clampedHorizon) });
    }
    ego.plannedPath = pts;
  }

  _updateTraffic(dt) {
    if (dt <= 0) return;
    const t = this.clock.simTime;
    const ego = this.ego;

    for (const v of this.vehiclePool) {
      if (!v.active) continue;

      // Deterministic Scripted Maneuvers
      if (v.scripted) {
        if (v.scriptRole === "cutin_coupe") {
          if (t < 7.2) {
            // Keep clean position 22-25m ahead in right lane before merging
            const desiredRel = 24;
            const err = desiredRel - (v.s - ego.s);
            v.targetSpeed = Math.max(20, Math.min(34, ego.speed + err * 0.45));
          } else if (t >= 7.2 && v.targetLane === 2) {
            v.targetLane = 1;
            v.laneChangeStartX = v.x;
            v.laneChangeTimer = 0;
            v.laneChangeProgress = 0;
            v.turnSignal = -1;
            v.targetSpeed = 84 / 3.6;
          } else if (t > 21.0) {
            v.targetSpeed = 108 / 3.6;
          }
        } else if (v.scriptRole === "occluded_sedan") {
          if (t < 13.5) {
            // Stay aligned inside truck's geometric shadow
            const desiredRel = 49;
            const err = desiredRel - (v.s - ego.s);
            v.targetSpeed = Math.max(20, Math.min(32, ego.speed + err * 0.35));
          } else if (t >= 13.5 && v.targetLane === 2) {
            v.targetLane = 1;
            v.laneChangeStartX = v.x;
            v.laneChangeTimer = 0;
            v.laneChangeProgress = 0;
            v.turnSignal = -1;
            v.targetSpeed = 94 / 3.6;
          }
        } else if (v.scriptRole === "aeb_lead_suv") {
          if (t < 6.8) {
            v.targetSpeed = 98 / 3.6;
          } else if (t >= 6.8 && t < 17.5) {
            v.targetSpeed = 0;
            v.accel = v.speed > 0.2 ? -6.8 : 0;
          } else {
            v.targetSpeed = 90 / 3.6;
          }
        }
      }

      // Compute nearest vehicle ahead in same lateral corridor (including Ego!)
      let aheadGap = Infinity;
      let aheadSpeed = v.targetSpeed;
      for (const other of this.vehiclePool) {
        if (!other.active || other === v) continue;
        if (Math.abs(other.x - v.x) < (v.width + other.width) * 0.52) {
          const gap = other.s - v.s - (v.length * 0.5 + other.length * 0.5);
          if (gap > -1 && gap < aheadGap) {
            aheadGap = gap;
            aheadSpeed = other.speed;
          }
        }
      }

      if (Math.abs(ego.x - v.x) < (v.width + ego.width) * 0.52) {
        const gapToEgo = ego.s - v.s - (v.length * 0.5 + ego.length * 0.5);
        if (gapToEgo > -1 && gapToEgo < aheadGap) {
          aheadGap = gapToEgo;
          aheadSpeed = ego.speed;
        }
      }

      const isHardBrakingLead = v.scriptRole === "aeb_lead_suv" && t >= 6.8 && t < 17.5;
      if (!isHardBrakingLead) {
        const desiredGap = 9.0 + v.speed * 1.45;
        let accel = Math.max(-2.6, Math.min(2.2, (v.targetSpeed - v.speed) * 0.75));
        if (aheadGap < desiredGap * 1.4) {
          const dv = v.speed - aheadSpeed;
          const brake = -2.8 * Math.pow(desiredGap / Math.max(1.5, aheadGap), 1.7) - dv * 0.55;
          accel = Math.min(accel, Math.max(-8.5, brake));
        }
        v.accel = accel;
      }

      v.speed = Math.max(0, v.speed + v.accel * dt);
      const ds = v.speed * dt;
      v.s += ds;
      v.relS = v.s - ego.s;
      v.brakeLight = v.accel < -1.0 || (isHardBrakingLead && v.speed < 2.0);

      if (ds > 0) {
        v.wheelAngle = (v.wheelAngle + ds / v.wheelRadius) % (Math.PI * 2);
      }

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

      // Despawn outside horizon
      if (!v.scripted && (v.relS > 210 || v.relS < -170)) {
        v.active = false;
      }
    }

    // Continuous Ambient Highway Traffic Spawner (strictly safe lanes)
    this._spawnCooldown -= dt;
    if (this._spawnCooldown <= 0) {
      const activeCount = this.vehiclePool.reduce((acc, v) => acc + (v.active ? 1 : 0), 0);
      if (activeCount < 8) {
        this._spawnAmbientVehicle();
      }
      this._spawnCooldown = 1.9;
    }
  }

  _spawnAmbientVehicle() {
    // Only spawn ambient vehicles in Left (0) or Right (2) lanes to preserve Ego's scripted corridor
    const lane = this.rng.pick([0, 0, 2, 2]);
    const speedKmh = lane === 0 ? this.rng.range(116, 134) : this.rng.range(76, 88);
    const speedMs = speedKmh / 3.6;

    const fromBehind = speedMs > this.ego.speed + 2.5;
    const relS = fromBehind ? this.rng.range(-150, -122) : this.rng.range(140, 178);

    // Guarantee generous spacing so no car ever pops in or crowds another
    for (const other of this.vehiclePool) {
      if (!other.active) continue;
      if (Math.abs(other.x - LANE_CENTERS[lane]) < 2.2 && Math.abs(other.relS - relS) < 42) {
        return;
      }
    }

    const type =
      lane === 2
        ? this.rng.pick(["truck", "suv", "sedan"])
        : this.rng.pick(["coupe", "sedan", "suv"]);

    this._acquireVehicle({ type, laneIndex: lane, relS, speed: speedMs });
  }

  /**
   * Hard Physical Non-Overlap Solver
   * Makes it mathematically impossible for any vehicle to clip through Ego or another car.
   */
  _enforceNoOverlap() {
    const ego = this.ego;
    const MIN_GAP = 3.2;

    // 1. Enforce separation between every active vehicle and Ego
    for (const v of this.vehiclePool) {
      if (!v.active) continue;
      const latOverlap = Math.abs(v.x - ego.x) < (v.width + ego.width) * 0.5;
      if (!latOverlap) continue;

      const minCenterDist = v.length * 0.5 + ego.length * 0.5 + MIN_GAP;
      const relS = v.s - ego.s;

      if (relS >= 0 && relS < minCenterDist) {
        // Vehicle is directly ahead of Ego -> hold Ego safely behind
        ego.s = v.s - minCenterDist;
        ego.speed = Math.min(ego.speed, v.speed);
        ego.brakeLight = true;
      } else if (relS < 0 && -relS < minCenterDist) {
        // Vehicle is behind Ego -> hold vehicle safely behind Ego
        v.s = ego.s - minCenterDist;
        v.speed = Math.min(v.speed, ego.speed);
        v.brakeLight = true;
      }
      v.relS = v.s - ego.s;
    }

    // 2. Enforce separation between traffic vehicles in overlapping lateral bands
    const active = this.vehiclePool.filter((v) => v.active).sort((a, b) => b.s - a.s);
    for (let i = 0; i < active.length; i++) {
      const front = active[i];
      for (let j = i + 1; j < active.length; j++) {
        const rear = active[j];
        if (Math.abs(front.x - rear.x) >= (front.width + rear.width) * 0.5) continue;
        const minDist = front.length * 0.5 + rear.length * 0.5 + MIN_GAP;
        if (front.s - rear.s < minDist) {
          rear.s = front.s - minDist;
          rear.speed = Math.min(rear.speed, front.speed);
          rear.brakeLight = true;
          rear.relS = rear.s - ego.s;
        }
      }
    }
  }

  /**
   * High-Density 3D Occluded LiDAR Raycaster (Analytical Ray-AABB + Road + Guardrails)
   */
  _runLiDARScan() {
    const ls = this.lidarState;
    const posBuf = this.lidarBuffers.pos;
    const distBuf = this.lidarBuffers.dist;
    const intBuf = this.lidarBuffers.intensity;
    const objBuf = this.lidarBuffers.objId;

    this._vehDirectHits = new Map();
    this._vehUnoccludedHits = new Map();

    const channels = this.quality.channels;
    const azSteps = this.quality.azSteps;
    ls.channels = channels;
    ls.azSteps = azSteps;

    const maxRange = LIDAR_MAX_RANGE;
    const ox = this.ego.x;
    const oy = LIDAR_MOUNT_HEIGHT;
    const oz = 0;

    const boxes = [];
    for (const v of this.vehiclePool) {
      if (!v.active) continue;
      const cz = -(v.s - this.ego.s);
      if (Math.abs(cz) > maxRange + v.length) continue;

      const hw = v.width * 0.5;
      const hl = v.length * 0.5;
      const cx = v.x - ox;
      const centerDist = Math.hypot(cx, cz);
      const boundRadius = Math.hypot(hw, hl) + 0.45;
      const centerAz = Math.atan2(cx, -cz);
      const halfSpan =
        centerDist > boundRadius ? Math.asin(Math.min(1, boundRadius / centerDist)) : Math.PI;

      boxes.push({
        id: v.id,
        minX: v.x - hw,
        maxX: v.x + hw,
        minY: 0.1,
        maxY: v.height,
        minZ: cz - hl,
        maxZ: cz + hl,
        reflectivity: v.reflectivity,
        centerAz,
        halfSpan,
      });
    }

    const vMinRad = (-16.5 * Math.PI) / 180;
    const vMaxRad = (5.5 * Math.PI) / 180;
    const maxBufferPts = distBuf.length;

    let ptCount = 0;
    let vehPts = 0;

    for (let a = 0; a < azSteps; a++) {
      const az = -Math.PI + (a / azSteps) * Math.PI * 2;
      const sinAz = Math.sin(az);
      const cosAz = Math.cos(az);

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
        let hitObjId = -99;
        let hitIntensity = 0.25;

        // 1. Road Surface
        if (dy < -0.009) {
          const tG = -oy / dy;
          if (tG > 2.4 && tG < bestT) {
            const gx = ox + dx * tG;
            if (Math.abs(gx) <= ROAD_HALF_WIDTH + 0.8) {
              bestT = tG;
              hitObjId = -1;
              hitIntensity = 0.24 + 0.16 * Math.exp(-tG / 42);
            }
          }
        }

        // 2. Concrete Jersey Barriers / Guardrails
        if (Math.abs(dx) > 0.02) {
          const wallX = dx > 0 ? ROAD_HALF_WIDTH : -ROAD_HALF_WIDTH;
          const tW = (wallX - ox) / dx;
          if (tW > 2.0 && tW < bestT) {
            const wy = oy + dy * tW;
            if (wy >= 0.08 && wy <= 1.05) {
              bestT = tW;
              hitObjId = -2;
              hitIntensity = 0.58;
            }
          }
        }

        // 3. Vehicles (Smallest t wins -> true geometric shadow cones!)
        let hitVeh = null;
        for (let b = 0; b < candidates.length; b++) {
          const box = candidates[b];
          const tV = intersectRayAABB(
            ox, oy, oz, dx, dy, dz,
            box.minX, box.maxX, box.minY, box.maxY, box.minZ, box.maxZ,
            maxRange
          );
          if (tV > 0) {
            this._vehUnoccludedHits.set(box.id, (this._vehUnoccludedHits.get(box.id) || 0) + 1);
            if (tV < bestT) {
              bestT = tV;
              hitObjId = box.id;
              hitIntensity = box.reflectivity * (0.7 + 0.3 * Math.exp(-tV / 75));
              hitVeh = box;
            }
          }
        }

        if (hitVeh) {
          this._vehDirectHits.set(hitVeh.id, (this._vehDirectHits.get(hitVeh.id) || 0) + 1);
          vehPts++;
        }

        if (hitObjId !== -99) {
          const i3 = ptCount * 3;
          posBuf[i3] = ox + dx * bestT;
          posBuf[i3 + 1] = Math.max(0.02, oy + dy * bestT);
          posBuf[i3 + 2] = oz + dz * bestT;
          distBuf[ptCount] = bestT;
          intBuf[ptCount] = hitIntensity;
          objBuf[ptCount] = hitObjId;
          ptCount++;
        }
      }
    }

    ls.activePoints = ptCount;
    ls.hitVehiclePoints = vehPts;
  }

  _runLiDARScanLightweight() {
    this._vehDirectHits = new Map();
    this._vehUnoccludedHits = new Map();
    for (const v of this.vehiclePool) {
      if (!v.active) continue;
      const dist = Math.hypot(v.x - this.ego.x, v.s - this.ego.s);
      if (dist <= LIDAR_MAX_RANGE) {
        const est = Math.max(6, Math.round(1600 / Math.max(10, dist)));
        this._vehDirectHits.set(v.id, est);
        this._vehUnoccludedHits.set(v.id, est);
      }
    }
  }

  _updatePerceptionAndTracking(dtScan) {
    const activeIds = new Set();

    for (const v of this.vehiclePool) {
      if (!v.active) continue;
      const relS = v.s - this.ego.s;
      const relX = v.x - this.ego.x;
      const dist = Math.hypot(relX, relS);
      if (dist > LIDAR_MAX_RANGE + 4) continue;

      const hits = this._vehDirectHits.get(v.id) || 0;
      const potential = this._vehUnoccludedHits.get(v.id) || 0;
      const occludedRatio =
        potential > 0 ? Math.max(0, Math.min(1, 1 - hits / potential)) : dist < LIDAR_MAX_RANGE ? 1 : 0;

      let trk = this._trackMap.get(v.id);
      if (!trk && (hits >= 5 || (this.scenarioDef.id === "truck_occlusion" && occludedRatio > 0.8 && potential >= 4))) {
        trk = {
          id: `TRK-${String(v.id).padStart(2, "0")}`,
          vehicleId: v.id,
          type: v.type,
          label: v.label,
          missedScans: 0,
          history: [],
        };
        this._trackMap.set(v.id, trk);
      }
      if (!trk) continue;

      activeIds.add(v.id);
      trk.missedScans = hits >= 5 ? 0 : trk.missedScans + 1;

      const relSpeed = v.speed - this.ego.speed;
      const closing = -relSpeed;
      const sameCorridor = Math.abs(relX) < 2.5 && relS > 0;
      const bumperGap = Math.max(0.1, relS - (this.ego.length * 0.5 + v.length * 0.5));
      const ttc = sameCorridor && closing > 0.45 ? bumperGap / closing : Infinity;

      const hitFactor = Math.min(1, hits / 40);
      const rangeFactor = Math.max(0.25, 1 - dist / (LIDAR_MAX_RANGE * 1.15));
      const confidence = hits >= 5 ? Math.min(0.99, 0.35 + hitFactor * 0.53 + rangeFactor * 0.11) : 0;

      let status = "CONFIRMED";
      if (hits < 5 && occludedRatio > 0.7) status = "OCCLUDED";
      else if (hits < 5) status = "COASTING";
      else if (occludedRatio > 0.35) status = "PARTIAL";

      trk.relS = relS;
      trk.x = v.x;
      trk.distance = dist;
      trk.relSpeed = relSpeed;
      trk.absSpeed = v.speed;
      trk.ttc = ttc;
      trk.hitCount = hits;
      trk.occludedRatio = occludedRatio;
      trk.confidence = confidence;
      trk.status = status;
      trk.bbox = {
        cx: v.x,
        cy: v.height * 0.5,
        cz: -relS,
        l: v.length,
        w: v.width,
        h: v.height,
      };

      if (dtScan > 0 || trk.history.length === 0) {
        trk.history.push({ x: v.x, s: v.s });
        if (trk.history.length > 22) trk.history.shift();
      }
    }

    for (const [vid, trk] of this._trackMap.entries()) {
      if (!activeIds.has(vid) || (trk.missedScans > 16 && this.scenarioDef.id !== "truck_occlusion")) {
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

  _saveSnapshot(secInt) {
    this._lastSnapshotSec = secInt;
    this._snapshots.set(secInt, {
      rngState: this.rng.cloneState(),
      nextVehicleId: this._nextVehicleId,
      spawnCooldown: this._spawnCooldown,
      clock: { ...this.clock },
      ego: {
        ...this.ego,
        laneOccupancy: [...this.ego.laneOccupancy],
        plannedPath: this.ego.plannedPath.map((p) => ({ ...p })),
      },
      vehicles: this.vehiclePool.map((v) => ({ ...v })),
    });
  }

  _restoreSnapshot(snap) {
    this.rng.restoreState(snap.rngState);
    this._nextVehicleId = snap.nextVehicleId;
    this._spawnCooldown = snap.spawnCooldown;
    this.clock = { ...snap.clock };
    this.ego = {
      ...snap.ego,
      laneOccupancy: [...snap.ego.laneOccupancy],
      plannedPath: snap.ego.plannedPath.map((p) => ({ ...p })),
    };
    for (let i = 0; i < this.vehiclePool.length; i++) {
      Object.assign(this.vehiclePool[i], snap.vehicles[i]);
    }
    this._trackMap.clear();
    this.tracks = [];
    this._lastSnapshotSec = Math.floor(this.clock.simTime);
  }

  getState() {
    const confirmed = this.tracks.filter((t) => t.status === "CONFIRMED" || t.status === "PARTIAL").length;
    return {
      clock: this.clock,
      scenario: {
        id: this.scenarioDef.id,
        num: this.scenarioDef.num,
        title: this.scenarioDef.title,
        phases: this.scenarioDef.phases,
        phase: this.currentPhase,
      },
      ego: this.ego,
      vehicles: this.vehiclePool,
      lidar: this.lidarState,
      tracks: this.tracks,
      telemetry: {
        egoSpeedKmh: this.ego.speed * 3.6,
        accelMs2: this.ego.accel,
        activeTracks: confirmed,
        totalTracks: this.tracks.length,
        minTtcS: this.ego.minTtc,
        controlState: this.ego.controlState,
        simTimeS: this.clock.simTime,
        points: this.lidarState.activePoints,
      },
    };
  }
}