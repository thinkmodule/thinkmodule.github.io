/**
 * ThinkModule Lab — ADAS Vehicle Dynamics, Occlusion & EKF Sensor Fusion Engine
 * Simulates Euro NCAP VRU occluded pedestrian, highway cut-in, and multi-lane cruise.
 * Models physical optical camera degradation in fog, LiDAR line-of-sight occlusion,
 * 77 GHz mmWave Radar Doppler velocity penetration, TTC, and closed-loop AEB braking.
 */

export const SCENARIOS = {
  occluded_ped: {
    id: "occluded_ped",
    tag: "EURO NCAP VRU SCENARIO",
    title: "Occluded Pedestrian Test",
    desc: "A pedestrian steps out from behind the heavy truck in the adjacent lane. Raise Fog to 80% and disable LiDAR/Radar to see why Camera-Only struggles vs. full EKF Sensor Fusion.",
    targetSpeedMs: 13.9 // 50 km/h
  },
  cut_in: {
    id: "cut_in",
    tag: "HIGHWAY ADAS SCENARIO",
    title: "Aggressive Cut-In (90 km/h)",
    desc: "A slower SUV merges abruptly into the ego lane. 77 GHz Radar captures exact Doppler closing velocity immediately, triggering smooth predictive ACC/AEB deceleration.",
    targetSpeedMs: 25.0 // 90 km/h
  },
  highway: {
    id: "highway",
    tag: "MULTI-AGENT PERCEPTION",
    title: "Continuous 3-Lane Highway",
    desc: "Endless 3-lane traffic flow ideal for orbiting freely in 360°, comparing Reality vs. LiDAR Point Cloud vs. Camera Raw, and inspecting live occlusion shadows.",
    targetSpeedMs: 23.6 // 85 km/h
  }
};

// 2D segment-vs-AABB intersection for real line-of-sight occlusion (Origin -> Target)
function segmentHitsBox2D(x0, z0, x1, z1, minX, maxX, minZ, maxZ) {
  let tMin = 0.0;
  let tMax = 1.0;
  const dx = x1 - x0;
  const dz = z1 - z0;

  if (Math.abs(dx) < 1e-6) {
    if (x0 < minX || x0 > maxX) return false;
  } else {
    let t1 = (minX - x0) / dx;
    let t2 = (maxX - x0) / dx;
    if (t1 > t2) [t1, t2] = [t2, t1];
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    if (tMin > tMax) return false;
  }

  if (Math.abs(dz) < 1e-6) {
    if (z0 < minZ || z0 > maxZ) return false;
  } else {
    let t1 = (minZ - z0) / dz;
    let t2 = (maxZ - z0) / dz;
    if (t1 > t2) [t1, t2] = [t2, t1];
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    if (tMin > tMax) return false;
  }

  return tMin > 0.02 && tMin < 0.98;
}

export class AdasSimulation {
  constructor() {
    this.scenarioId = "occluded_ped";
    this.fogNorm = 0.15;
    this.sensors = { camera: true, lidar: true, radar: true };
    this.paused = false;

    this.resetScenario("occluded_ped");
  }

  resetScenario(id = this.scenarioId) {
    this.scenarioId = SCENARIOS[id] ? id : "occluded_ped";
    const sc = SCENARIOS[this.scenarioId];

    this.time = 0;
    this.egoSpeed = sc.targetSpeedMs;
    this.targetSpeed = sc.targetSpeedMs;
    this.roadDistance = 0;
    this.aebDecel = 0;
    this.aebActive = false;
    this.fcwActive = false;
    this.collisionFlag = false;
    this.safeStopFlag = false;
    this.minTtc = Infinity;
    this.criticalActor = null;
    this.lastSummary = {
      camConf: 0.92,
      lidarConf: 0.96,
      radarConf: 0.88,
      fusedConf: 0.99,
      relVel: 0
    };

    if (this.scenarioId === "occluded_ped") {
      this.actors = [
        {
          id: "truck",
          type: "truck",
          label: "HEAVY TRUCK",
          x: 3.55,
          dist: 24.5, // meters ahead (+dist maps to -Z in 3D)
          speed: 3.2,
          vx: 0,
          w: 2.45,
          h: 3.45,
          l: 9.2,
          blinker: 0,
          braking: true,
          walkPhase: 0
        },
        {
          id: "ped",
          type: "pedestrian",
          label: "PEDESTRIAN",
          x: 5.05,
          dist: 35.8,
          speed: 0.0,
          vx: -1.85,
          w: 0.68,
          h: 1.76,
          l: 0.65,
          blinker: 0,
          braking: false,
          walkPhase: 0
        },
        {
          id: "lead",
          type: "sedan",
          label: "SEDAN",
          x: -3.55,
          dist: 34.0,
          speed: 15.8,
          vx: 0,
          w: 1.88,
          h: 1.44,
          l: 4.55,
          blinker: 0,
          braking: false,
          walkPhase: 0
        }
      ];
    } else if (this.scenarioId === "cut_in") {
      this.actors = [
        {
          id: "cut_suv",
          type: "suv",
          label: "CUT-IN SUV",
          x: -3.55,
          dist: 23.0,
          speed: 14.2,
          vx: 0,
          w: 1.98,
          h: 1.66,
          l: 4.75,
          blinker: 1,
          braking: false,
          walkPhase: 0
        },
        {
          id: "truck",
          type: "truck",
          label: "FREIGHT TRUCK",
          x: 3.55,
          dist: 40.0,
          speed: 20.5,
          vx: 0,
          w: 2.45,
          h: 3.45,
          l: 9.2,
          blinker: 0,
          braking: false,
          walkPhase: 0
        },
        {
          id: "lead",
          type: "sedan",
          label: "LEAD SEDAN",
          x: 0.0,
          dist: 52.0,
          speed: 24.0,
          vx: 0,
          w: 1.88,
          h: 1.44,
          l: 4.55,
          blinker: 0,
          braking: false,
          walkPhase: 0
        }
      ];
    } else {
      // Continuous 3-lane highway
      this.actors = [
        {
          id: "lead",
          type: "sedan",
          label: "LEAD SEDAN",
          x: 0.0,
          dist: 36.0,
          speed: 23.2,
          vx: 0,
          w: 1.88,
          h: 1.44,
          l: 4.55,
          blinker: 0,
          braking: false,
          walkPhase: 0
        },
        {
          id: "truck",
          type: "truck",
          label: "HEAVY TRUCK",
          x: 3.55,
          dist: 26.0,
          speed: 19.2,
          vx: 0,
          w: 2.45,
          h: 3.45,
          l: 9.2,
          blinker: 0,
          braking: false,
          walkPhase: 0
        },
        {
          id: "cut_suv",
          type: "suv",
          label: "FAST SUV",
          x: -3.55,
          dist: 16.0,
          speed: 27.5,
          vx: 0,
          w: 1.98,
          h: 1.66,
          l: 4.75,
          blinker: 0,
          braking: false,
          walkPhase: 0
        }
      ];
    }

    this._evaluatePerception();
  }

  setFog(norm01) {
    this.fogNorm = Math.max(0, Math.min(1, norm01));
  }

  toggleSensor(key) {
    if (key in this.sensors) {
      this.sensors[key] = !this.sensors[key];
    }
  }

  step(dt) {
    if (this.paused) {
      this._evaluatePerception();
      return this.getTelemetry();
    }

    this.time += dt;

    // 1. Update actor maneuvers
    for (let i = 0; i < this.actors.length; i++) {
      const a = this.actors[i];

      if (this.scenarioId === "occluded_ped") {
        if (a.id === "ped") {
          // Pedestrian crosses steadily from right side behind truck into center ego lane
          if (a.x > -3.8 && !this.collisionFlag) {
            a.x += a.vx * dt;
            a.walkPhase += dt * 7.2;
          }
        }
        if (a.id === "truck") {
          // Truck slowly creeps/stops on right lane
          a.speed = Math.max(0, a.speed - dt * 1.1);
        }
      } else if (this.scenarioId === "cut_in") {
        if (a.id === "cut_suv") {
          // Smooth S-curve cut-in from left lane (-3.55) into ego lane (0.0) between t=0.8s and t=2.8s
          if (this.time > 0.65 && a.x < -0.02) {
            a.vx = Math.min(1.95, (0.0 - a.x) * 1.45);
            a.x = Math.min(0.0, a.x + a.vx * dt);
            a.blinker = 1;
            a.braking = this.time > 1.8 && this.time < 3.5;
            if (a.braking) a.speed = Math.max(11.5, a.speed - dt * 2.2);
          } else {
            a.vx = 0;
            a.blinker = 0;
            a.braking = false;
            if (this.time > 4.2) a.speed = Math.min(22.0, a.speed + dt * 2.5);
          }
        }
      } else if (this.scenarioId === "highway") {
        // Endless highway wrap-around
        if (a.dist < -18) a.dist = 66;
        if (a.dist > 72) a.dist = -12;
      }

      a.dist += (a.speed - this.egoSpeed) * dt;
    }

    // 2. Run per-sensor physical perception + EKF Sensor Fusion
    this._evaluatePerception();

    // 3. Longitudinal Control (ACC + AEB Braking based strictly on FUSED perception!)
    let desiredAccel = 0;
    this.aebActive = false;
    this.fcwActive = false;

    const crit = this.criticalActor;
    if (crit && crit.fusedConf >= 0.42) {
      const gap = Math.max(0.2, crit.dist - crit.l * 0.5 - 2.35);
      const closingSpeed = Math.max(0, this.egoSpeed - crit.speed);
      const ttc = closingSpeed > 0.15 ? gap / closingSpeed : Infinity;

      // Required stopping distance with human/AEB margin
      const reqStopDist = (closingSpeed * closingSpeed) / (2 * 7.8) + 2.8;

      if (ttc < 2.15 || gap < reqStopDist) {
        this.aebActive = true;
        this.fcwActive = true;
        // High-authority AEB hydraulic braking (up to -9.4 m/s²)
        const urgency = Math.max(1.0, reqStopDist / Math.max(gap, 1.0));
        desiredAccel = -Math.min(9.6, 7.6 * urgency);
      } else if (ttc < 3.6 || gap < reqStopDist * 1.55) {
        this.fcwActive = true;
        desiredAccel = -3.8;
      } else if (this.scenarioId === "cut_in" && Math.abs(crit.x) < 1.8 && gap < 22) {
        desiredAccel = -2.8;
      }
    }

    if (!this.aebActive && !this.fcwActive && !this.collisionFlag && !this.safeStopFlag) {
      // Smooth cruise speed recovery
      const speedErr = this.targetSpeed - this.egoSpeed;
      desiredAccel = Math.max(-2.0, Math.min(2.2, speedErr * 0.65));
    }

    // Physical collision check (if sensors were degraded/blinded and Ego didn't brake!)
    for (let i = 0; i < this.actors.length; i++) {
      const a = this.actors[i];
      const latClear = Math.abs(a.x) < (1.02 + a.w * 0.46);
      const longOverlap = a.dist - a.l * 0.5 < 2.25 && a.dist + a.l * 0.5 > -2.1;
      if (latClear && longOverlap && this.egoSpeed > 1.5) {
        this.collisionFlag = true;
        this.egoSpeed = 0;
        desiredAccel = 0;
      }
    }

    if (this.collisionFlag) {
      this.egoSpeed = 0;
      this.aebDecel = 0;
    } else {
      this.egoSpeed = Math.max(0, this.egoSpeed + desiredAccel * dt);
      this.aebDecel = desiredAccel < -0.2 ? -desiredAccel : 0;
    }

    // Check clean safe-stop victory state in Occluded Pedestrian test
    if (this.scenarioId === "occluded_ped" && !this.collisionFlag && this.egoSpeed < 0.08 && this.time > 1.2) {
      this.safeStopFlag = true;
    }

    this.roadDistance += this.egoSpeed * dt;
    return this.getTelemetry();
  }

  _evaluatePerception() {
    const fog = this.fogNorm;
    let bestHazard = null;
    let minTtc = Infinity;

    for (let i = 0; i < this.actors.length; i++) {
      const a = this.actors[i];
      const range = Math.hypot(a.x, Math.max(0.1, a.dist));
      const inFront = a.dist > -1.0 && a.dist < 68;

      // Check hard visual/LiDAR occlusion against other larger vehicles
      let occluded = false;
      for (let j = 0; j < this.actors.length; j++) {
        if (i === j) continue;
        const b = this.actors[j];
        if (b.h < 1.2 || b.dist > a.dist) continue;
        // Test ray from Ego front/roof (0, 0) to actor center (a.x, a.dist)
        if (
          segmentHitsBox2D(
            0, 0, a.x, a.dist,
            b.x - b.w * 0.5, b.x + b.w * 0.5,
            b.dist - b.l * 0.5, b.dist + b.l * 0.5
          )
        ) {
          occluded = true;
          break;
        }
      }
      a.occluded = occluded;

      // 1. RGB Camera Model: strong semantic clarity in clear weather, severely crippled by dense fog & occlusion
      let camConf = 0;
      if (this.sensors.camera && inFront && !occluded) {
        const fovOk = Math.abs(Math.atan2(a.x, Math.max(1, a.dist))) < 0.62;
        if (fovOk) {
          const distAtten = Math.exp(-range * (0.008 + fog * fog * 0.145));
          const fogContrastLoss = 1 - fog * 0.52;
          camConf = Math.max(0, Math.min(0.98, 0.96 * distAtten * fogContrastLoss));
        }
      }

      // 2. 64-Channel 1550nm LiDAR Model: crisp 3D geometry, immune to darkness, mild fog scatter, blocked by hard occlusion
      let lidarConf = 0;
      if (this.sensors.lidar && inFront && !occluded) {
        const rangeFactor = Math.max(0, 1 - range / 82);
        const fogAtten = 1 - fog * 0.24;
        const sizeBoost = a.type === "pedestrian" ? 0.92 : 1.0;
        lidarConf = Math.max(0, Math.min(0.99, 0.98 * rangeFactor * fogAtten * sizeBoost));
      }

      // 3. 77 GHz mmWave Radar Model: penetrates fog completely, partial under-chassis bounce near occlusion edge, lower semantic certainty
      let radarConf = 0;
      if (this.sensors.radar && inFront) {
        const fovOk = Math.abs(Math.atan2(a.x, Math.max(1, a.dist))) < 0.58;
        if (fovOk) {
          const rcs = a.type === "pedestrian" ? 0.68 : 0.95; // Metal vehicles have higher radar cross-section than pedestrians
          const occlFactor = occluded ? 0.28 : 1.0; // Slight under-truck ground bounce before visual clearance
          const rangeFactor = Math.max(0.15, 1 - range / 110);
          radarConf = Math.max(0, Math.min(0.96, rcs * occlFactor * rangeFactor * (1 - fog * 0.03)));
        }
      }

      // 4. EKF Bayesian Sensor Fusion (Independent evidence combination)
      const fusedConf = 1 - (1 - camConf) * (1 - lidarConf) * (1 - radarConf);

      a.camConf = camConf;
      a.lidarConf = lidarConf;
      a.radarConf = radarConf;
      a.fusedConf = fusedConf;
      a.relVel = a.speed - this.egoSpeed;

      // Predict path threat (in-lane or laterally moving into ego lane!)
      const predictedX = a.x + a.vx * 1.4;
      const inCorridor =
        Math.abs(a.x) < 1.65 + a.w * 0.45 ||
        Math.abs(predictedX) < 1.55 + a.w * 0.45 ||
        (a.x * predictedX < 0); // Crossing centerline

      const gap = Math.max(0.1, a.dist - a.l * 0.5 - 2.35);
      const closing = Math.max(0, this.egoSpeed - a.speed);
      a.ttc = inCorridor && a.dist > 0 && closing > 0.2 ? gap / closing : Infinity;
      a.isThreat = inCorridor && a.dist > 0 && a.dist < 42;

      if (a.isThreat && a.dist < (bestHazard ? bestHazard.dist : Infinity)) {
        bestHazard = a;
      }
      if (a.ttc < minTtc && a.fusedConf >= 0.35) {
        minTtc = a.ttc;
      }
    }

    this.criticalActor = bestHazard;
    this.minTtc = minTtc;

    // Pick primary tracked actor for the right-hand sensor confidence bars
    const focus = bestHazard || this.actors[0];
    if (focus) {
      this.lastSummary = {
        camConf: focus.camConf,
        lidarConf: focus.lidarConf,
        radarConf: focus.radarConf,
        fusedConf: focus.fusedConf,
        relVel: focus.relVel,
        focusLabel: focus.label,
        focusOccluded: focus.occluded
      };
    }
  }

  getTelemetry() {
    const activeSensorCount =
      (this.sensors.camera ? 1 : 0) +
      (this.sensors.lidar ? 1 : 0) +
      (this.sensors.radar ? 1 : 0);

    let statusText = "NOMINAL • EKF FUSION LOCK";
    let statusSeverity = "optimal";
    let insightText =
      "All active modalities fused cleanly. Observe the dark LiDAR shadow behind the truck and dynamic path corridor.";
    let insightSeverity = "info";

    if (activeSensorCount === 0) {
      statusText = "SENSORS DISABLED • BLIND";
      statusSeverity = "danger";
      insightText =
        "All perception sensors are turned off. The vehicle cannot detect obstacles or trigger AEB braking.";
      insightSeverity = "danger";
    } else if (this.collisionFlag) {
      statusText = "COLLISION • LATE PERCEPTION";
      statusSeverity = "danger";
      insightText =
        "Detection confidence stayed too low to trigger AEB in time! Enable LiDAR + 77 GHz Radar Fusion and click 'Replay Test' to see the difference.";
      insightSeverity = "danger";
    } else if (this.safeStopFlag) {
      statusText = "SAFE STOP • AEB SUCCESS";
      statusSeverity = "optimal";
      insightText =
        "AEB brought the vehicle to a controlled stop before the hazard! Click 'Replay Test' or raise Fog + disable LiDAR/Radar to test Camera-Only.";
      insightSeverity = "info";
    } else if (this.aebActive) {
      statusText = "AEB ENGAGED • EMERGENCY BRAKE";
      statusSeverity = "danger";
      insightText = `Autonomous Emergency Braking active (−${this.aebDecel.toFixed(1)} m/s²) on fused ${this.lastSummary.focusLabel} track (TTC ${this.minTtc < 20 ? this.minTtc.toFixed(1) + "s" : "LOW"}).`;
      insightSeverity = "danger";
    } else if (this.fcwActive) {
      statusText = "FCW • COLLISION WARNING";
      statusSeverity = "warn";
      insightText = `Forward Collision Warning triggered — closing velocity detected by sensor fusion. Preparing hydraulic brake pre-fill.`;
      insightSeverity = "warn";
    } else if (this.fogNorm > 0.65 && this.sensors.camera && !this.sensors.lidar && !this.sensors.radar) {
      statusText = "CAMERA-ONLY • FOG DEGRADED";
      statusSeverity = "warn";
      insightText =
        "Dense fog heavily attenuates visible light. Without LiDAR geometry or 77 GHz Radar Doppler, perception range collapses dangerously.";
      insightSeverity = "warn";
    } else if (this.lastSummary.focusOccluded) {
      statusText = "OCCLUSION SHADOW ACTIVE";
      statusSeverity = "warn";
      insightText =
        "The box truck blocks optical and LiDAR lines of sight to the pedestrian. Radar edge multipath provides early weak returns before visual emergence.";
      insightSeverity = "warn";
    }

    return {
      egoSpeedKmh: this.egoSpeed * 3.6,
      egoSpeedMs: this.egoSpeed,
      minTtc: this.minTtc,
      aebDecel: this.aebDecel,
      aebActive: this.aebActive,
      fcwActive: this.fcwActive,
      collisionFlag: this.collisionFlag,
      safeStopFlag: this.safeStopFlag,
      roadDistance: this.roadDistance,
      summary: this.lastSummary,
      statusText,
      statusSeverity,
      insightText,
      insightSeverity,
      actors: this.actors
    };
  }
}