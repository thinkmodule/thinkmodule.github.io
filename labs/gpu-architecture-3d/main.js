/**
 * ThinkModule Lab — Module 05 Orchestrator (WebGL PBR Scene + Physics + Minimalist HUD)
 */
import { GpuPhysics, LAYER_INFO } from "./physics.js";
import { GpuScene3D } from "./scene.js";

const $ = id => document.getElementById(id);

const dom = {
  viewport: $("viewport"),
  loadingOverlay: $("loadingOverlay"),
  thermalBtn: $("thermalBtn"),
  orbitBtn: $("orbitBtn"),
  resetCamBtn: $("resetCamBtn"),

  telStatus: $("telStatus"),
  telTemp: $("telTemp"),
  telPower: $("telPower"),
  telClock: $("telClock"),
  telMfu: $("telMfu"),
  telBw: $("telBw"),
  telThrottle: $("telThrottle"),

  inspectTag: $("inspectTag"),
  inspectTitle: $("inspectTitle"),
  inspectDesc: $("inspectDesc"),
  inspectSpecs: $("inspectSpecs"),

  explodeSlider: $("explodeSlider"),
  explodeVal: $("explodeVal"),
  coolingSlider: $("coolingSlider"),
  coolingVal: $("coolingVal"),
  workloadGroup: $("workloadGroup"),
  layerGroup: $("layerGroup")
};

const physics = new GpuPhysics();
const scene3d = new GpuScene3D(dom.viewport);

let thermalOn = false;
let autoOrbitOn = true;
let lastTs = 0;
let elapsed = 0;

function updateInspector(layerKey) {
  const info = LAYER_INFO[layerKey] || LAYER_INFO.all;
  dom.inspectTag.textContent = info.tag;
  dom.inspectTitle.textContent = info.title;
  dom.inspectDesc.textContent = info.desc;
  dom.inspectSpecs.innerHTML = info.specs
    .map(s => `<span class="spec-pill">${s}</span>`)
    .join("");

  dom.layerGroup.querySelectorAll(".l-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.layer === String(layerKey));
  });
}

function updateTelemetryHUD(p) {
  dom.telStatus.textContent = p.statusText;
  dom.telStatus.className = `status-chip ${p.statusSeverity}`;

  dom.telTemp.textContent = `${p.tempC.toFixed(1)} °C`;
  dom.telTemp.className = "m-val " + (p.tempC >= 84 ? "danger" : p.tempC >= 74 ? "warn" : "");

  dom.telPower.textContent = `${Math.round(p.powerW)} W`;
  dom.telClock.textContent = `${Math.round(p.clockMHz)} MHz`;
  dom.telClock.className = "m-val " + (p.throttlePct > 4 ? "danger" : "");

  dom.telMfu.textContent = `${(p.mfu * 100).toFixed(1)}%`;
  dom.telMfu.className = "m-val " + (p.mfu > 0.5 ? "accent" : "warn");

  dom.telBw.textContent = `${p.bwTBs.toFixed(2)} TB/s`;

  dom.telThrottle.textContent = p.throttlePct > 0.5 ? `-${p.throttlePct.toFixed(1)}%` : "0.0%";
  dom.telThrottle.className = "m-val " + (p.throttlePct > 4 ? "danger" : "accent");
}

// Controls binding
dom.explodeSlider.addEventListener("input", () => {
  const val = parseInt(dom.explodeSlider.value, 10);
  dom.explodeVal.textContent = `${val}%`;
  scene3d.setExplode(val / 100);
});

dom.coolingSlider.addEventListener("input", () => {
  const val = parseInt(dom.coolingSlider.value, 10);
  dom.coolingVal.textContent = `${val}%`;
  physics.setCooling(val);
});

dom.workloadGroup.addEventListener("click", evt => {
  const btn = evt.target.closest(".seg");
  if (!btn) return;
  physics.setWorkload(btn.dataset.mode);
  dom.workloadGroup.querySelectorAll(".seg").forEach(b => b.classList.toggle("active", b === btn));
});

dom.layerGroup.addEventListener("click", evt => {
  const btn = evt.target.closest(".l-btn");
  if (!btn) return;
  const key = btn.dataset.layer;
  updateInspector(key);
  scene3d.focusLayer(key);

  // Sync explode slider if focusing an internal layer auto-expanded the chip
  const syncedPct = Math.round(scene3d.explodeTarget * 100);
  dom.explodeSlider.value = String(syncedPct);
  dom.explodeVal.textContent = `${syncedPct}%`;
});

function toggleThermal() {
  thermalOn = !thermalOn;
  scene3d.setThermalMode(thermalOn);
  dom.thermalBtn.textContent = thermalOn ? "Thermal IR: ON" : "Thermal IR: OFF";
  dom.thermalBtn.classList.toggle("thermal-on", thermalOn);
}

dom.thermalBtn.addEventListener("click", toggleThermal);

dom.orbitBtn.addEventListener("click", () => {
  autoOrbitOn = !autoOrbitOn;
  scene3d.setAutoOrbit(autoOrbitOn);
  dom.orbitBtn.classList.toggle("active", autoOrbitOn);
});

dom.resetCamBtn.addEventListener("click", () => {
  updateInspector("all");
  scene3d.resetCamera();
});

// Keyboard shortcuts (T = Thermal IR, R = Reset View, E = Toggle Explode)
window.addEventListener("keydown", evt => {
  if (evt.target.tagName === "INPUT") return;
  const k = evt.key.toLowerCase();
  if (k === "t") toggleThermal();
  if (k === "r") {
    updateInspector("all");
    scene3d.resetCamera();
  }
  if (k === "e") {
    const next = scene3d.explodeTarget > 0.35 ? 0 : 85;
    dom.explodeSlider.value = String(next);
    dom.explodeVal.textContent = `${next}%`;
    scene3d.setExplode(next / 100);
  }
});

// Main 60 FPS Render Loop
function animate(ts) {
  requestAnimationFrame(animate);
  if (!lastTs) lastTs = ts;
  const dt = Math.min((ts - lastTs) / 1000, 0.05);
  lastTs = ts;
  elapsed += dt;

  const phys = physics.step(dt);
  updateTelemetryHUD(phys);
  scene3d.update(dt, elapsed, phys);
}

updateInspector("all");
requestAnimationFrame(animate);

// Smooth fade-out of loading screen once first WebGL frame compiles
setTimeout(() => {
  dom.loadingOverlay.classList.add("hidden");
}, 260);