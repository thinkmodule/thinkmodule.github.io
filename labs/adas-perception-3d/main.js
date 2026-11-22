/**
 * ThinkModule Lab — Module 06 Orchestrator (3D ADAS Perception + Sensor Fusion HUD)
 */
import { AdasSimulation, SCENARIOS } from "./physics.js";
import { AdasScene3D } from "./scene.js";

const $ = id => document.getElementById(id);

const dom = {
  viewport: $("viewport"),
  loadingOverlay: $("loadingOverlay"),
  camChaseBtn: $("camChaseBtn"),
  camBevBtn: $("camBevBtn"),
  camHoodBtn: $("camHoodBtn"),
  playPauseBtn: $("playPauseBtn"),
  replayBtn: $("replayBtn"),

  adasStatus: $("adasStatus"),
  telSpeed: $("telSpeed"),
  telTtc: $("telTtc"),
  telConf: $("telConf"),
  telBrake: $("telBrake"),
  telPoints: $("telPoints"),
  telDoppler: $("telDoppler"),

  scenTag: $("scenTag"),
  scenTitle: $("scenTitle"),
  scenDesc: $("scenDesc"),
  barCamVal: $("barCamVal"),
  barCamFill: $("barCamFill"),
  barLidarVal: $("barLidarVal"),
  barLidarFill: $("barLidarFill"),
  barRadarVal: $("barRadarVal"),
  barRadarFill: $("barRadarFill"),
  barFusedVal: $("barFusedVal"),
  barFusedFill: $("barFusedFill"),
  insightBox: $("insightBox"),

  scenarioGroup: $("scenarioGroup"),
  viewGroup: $("viewGroup"),
  fogSlider: $("fogSlider"),
  fogVal: $("fogVal"),
  sensorGroup: $("sensorGroup")
};

const sim = new AdasSimulation();
const scene3d = new AdasScene3D(dom.viewport);

let lastTs = 0;
let elapsed = 0;

function syncScenarioMeta() {
  const sc = SCENARIOS[sim.scenarioId];
  dom.scenTag.textContent = sc.tag;
  dom.scenTitle.textContent = sc.title;
  dom.scenDesc.textContent = sc.desc;

  dom.scenarioGroup.querySelectorAll(".seg").forEach(b => {
    b.classList.toggle("active", b.dataset.scen === sim.scenarioId);
  });
}

function updateHud(tel) {
  dom.adasStatus.textContent = tel.statusText;
  dom.adasStatus.className = `status-chip ${tel.statusSeverity}`;

  dom.telSpeed.textContent = `${Math.round(tel.egoSpeedKmh)} km/h`;
  dom.telSpeed.className = "m-val " + (tel.collisionFlag ? "danger" : "");

  const ttcStr = Number.isFinite(tel.minTtc) && tel.minTtc < 15 ? `${tel.minTtc.toFixed(1)} s` : "∞";
  dom.telTtc.textContent = ttcStr;
  dom.telTtc.className = "m-val " + (tel.minTtc < 2.2 ? "danger" : tel.minTtc < 3.6 ? "warn" : "cyan");

  const fPct = Math.round(tel.summary.fusedConf * 100);
  dom.telConf.textContent = `${fPct}%`;
  dom.telConf.className = "m-val " + (fPct < 45 ? "danger" : fPct < 75 ? "warn" : "accent");

  dom.telBrake.textContent = `${tel.aebDecel.toFixed(1)} m/s²`;
  dom.telBrake.className = "m-val " + (tel.aebActive ? "danger" : tel.aebDecel > 1.5 ? "warn" : "");

  dom.telPoints.textContent = `${scene3d.visiblePointsCount.toLocaleString()} pts`;

  const doppler = sim.sensors.radar ? `${tel.summary.relVel.toFixed(1)} m/s` : "OFF";
  dom.telDoppler.textContent = doppler;
  dom.telDoppler.className = "m-val " + (!sim.sensors.radar ? "warn" : "");

  // Right-hand Sensor Fusion Confidence Bars
  const setBar = (valEl, fillEl, norm, active) => {
    const pct = active ? Math.round(norm * 100) : 0;
    valEl.textContent = active ? `${pct}%` : "OFF";
    fillEl.style.width = `${pct}%`;
  };
  setBar(dom.barCamVal, dom.barCamFill, tel.summary.camConf, sim.sensors.camera);
  setBar(dom.barLidarVal, dom.barLidarFill, tel.summary.lidarConf, sim.sensors.lidar);
  setBar(dom.barRadarVal, dom.barRadarFill, tel.summary.radarConf, sim.sensors.radar);

  dom.barFusedVal.textContent = `${fPct}%`;
  dom.barFusedFill.style.width = `${fPct}%`;

  dom.insightBox.textContent = tel.insightText;
  dom.insightBox.className = `insight-box ${tel.insightSeverity === "info" ? "" : tel.insightSeverity}`;
}

// Scenario switcher
dom.scenarioGroup.addEventListener("click", evt => {
  const btn = evt.target.closest(".seg");
  if (!btn) return;
  sim.paused = false;
  dom.playPauseBtn.textContent = "Pause";
  sim.resetScenario(btn.dataset.scen);
  syncScenarioMeta();
});

// Perception View Mode switcher
dom.viewGroup.addEventListener("click", evt => {
  const btn = evt.target.closest(".seg");
  if (!btn) return;
  const mode = btn.dataset.view;
  scene3d.setViewMode(mode);
  dom.viewGroup.querySelectorAll(".seg").forEach(b => b.classList.toggle("active", b === btn));
  // Auto-switch to Hood Cam for Camera Raw so user literally sees through the windshield camera!
  if (mode === "camera") {
    selectCamera("hood");
  } else if (scene3d.camPreset === "hood") {
    selectCamera("chase");
  }
});

// Fog slider
dom.fogSlider.addEventListener("input", () => {
  const pct = parseInt(dom.fogSlider.value, 10);
  dom.fogVal.textContent = `${pct}%`;
  sim.setFog(pct / 100);
});

// Individual Sensor Toggles (Camera / LiDAR / 77GHz Radar)
dom.sensorGroup.addEventListener("click", evt => {
  const btn = evt.target.closest(".s-chip");
  if (!btn) return;
  const key = btn.dataset.sensor;
  sim.toggleSensor(key);
  btn.classList.toggle("active", sim.sensors[key]);
});

// Camera Presets
function selectCamera(preset) {
  scene3d.setCameraPreset(preset);
  [dom.camChaseBtn, dom.camBevBtn, dom.camHoodBtn].forEach(b => {
    b.classList.toggle("active", b.dataset.cam === preset);
  });
}

[dom.camChaseBtn, dom.camBevBtn, dom.camHoodBtn].forEach(btn => {
  btn.addEventListener("click", () => selectCamera(btn.dataset.cam));
});

// Pause & Replay
dom.playPauseBtn.addEventListener("click", () => {
  sim.paused = !sim.paused;
  dom.playPauseBtn.textContent = sim.paused ? "Play" : "Pause";
  dom.playPauseBtn.classList.toggle("active", sim.paused);
});

dom.replayBtn.addEventListener("click", () => {
  sim.paused = false;
  dom.playPauseBtn.textContent = "Pause";
  dom.playPauseBtn.classList.remove("active");
  sim.resetScenario(sim.scenarioId);
});

// Keyboard shortcuts (Space = Pause, R = Replay, 1/2/3 = View Modes)
window.addEventListener("keydown", evt => {
  if (evt.target.tagName === "INPUT") return;
  if (evt.code === "Space") {
    evt.preventDefault();
    dom.playPauseBtn.click();
  } else if (evt.key.toLowerCase() === "r") {
    dom.replayBtn.click();
  } else if (evt.key === "1") {
    dom.viewGroup.querySelector('[data-view="fusion"]').click();
  } else if (evt.key === "2") {
    dom.viewGroup.querySelector('[data-view="lidar"]').click();
  } else if (evt.key === "3") {
    dom.viewGroup.querySelector('[data-view="camera"]').click();
  }
});

// 60 FPS Render & Physics Loop
function animate(ts) {
  requestAnimationFrame(animate);
  if (!lastTs) lastTs = ts;
  const dt = Math.min((ts - lastTs) / 1000, 0.05);
  lastTs = ts;
  if (!sim.paused) elapsed += dt;

  const tel = sim.step(dt);
  scene3d.update(dt, elapsed, sim, tel);
  updateHud(tel);
}

syncScenarioMeta();
requestAnimationFrame(animate);

setTimeout(() => {
  dom.loadingOverlay.classList.add("hidden");
}, 260);