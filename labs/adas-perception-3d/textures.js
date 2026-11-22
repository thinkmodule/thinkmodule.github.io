/**
 * ThinkModule Lab — Procedural Automotive Highway & LiDAR Sprite Textures
 * Generates crisp asphalt grain, reflective lane markings, and glowing LiDAR point sprites.
 */
import * as THREE from "three";

export function createRoadTextures() {
  const w = 1024, h = 1024;
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d");

  const rC = document.createElement("canvas");
  rC.width = w; rC.height = h;
  const rctx = rC.getContext("2d");

  // Deep architectural slate-asphalt base
  ctx.fillStyle = "#11161f";
  ctx.fillRect(0, 0, w, h);

  rctx.fillStyle = "#b8b8b8";
  rctx.fillRect(0, 0, w, h);

  // Subtle asphalt aggregate micro-noise + tire wear tracks
  for (let i = 0; i < 26000; i++) {
    const x = Math.random() * w;
    const y = Math.random() * h;
    const v = 14 + Math.floor(Math.random() * 20);
    ctx.fillStyle = `rgb(${v},${v + 4},${v + 10})`;
    ctx.fillRect(x, y, 2, 2);
  }

  // Subtle smoother tire tracks in all 3 lanes (creates sleek wet-sheen specular highlights!)
  const laneCenters = [w * 0.24, w * 0.50, w * 0.76];
  laneCenters.forEach(cx => {
    [-46, 46].forEach(off => {
      const grad = rctx.createLinearGradient(cx + off - 28, 0, cx + off + 28, 0);
      grad.addColorStop(0, "rgba(184,184,184,0)");
      grad.addColorStop(0.5, "rgba(75,75,75,0.85)");
      grad.addColorStop(1, "rgba(184,184,184,0)");
      rctx.fillStyle = grad;
      rctx.fillRect(cx + off - 28, 0, 56, h);
    });
  });

  // Solid outer shoulder lines
  ctx.fillStyle = "rgba(218, 228, 242, 0.78)";
  ctx.fillRect(w * 0.095, 0, 10, h);
  ctx.fillRect(w * 0.905 - 10, 0, 10, h);

  // Dashed lane divider lines (between Left / Ego / Right lanes)
  const dividers = [w * 0.365, w * 0.635];
  ctx.fillStyle = "rgba(225, 235, 248, 0.82)";
  dividers.forEach(dx => {
    for (let y = 40; y < h; y += 256) {
      ctx.fillRect(dx - 5, y, 10, 118);
    }
  });

  const map = new THREE.CanvasTexture(c);
  map.wrapS = THREE.RepeatWrapping;
  map.wrapT = THREE.RepeatWrapping;
  map.repeat.set(1, 9);
  map.anisotropy = 8;
  map.colorSpace = THREE.SRGBColorSpace;

  const roughnessMap = new THREE.CanvasTexture(rC);
  roughnessMap.wrapS = THREE.RepeatWrapping;
  roughnessMap.wrapT = THREE.RepeatWrapping;
  roughnessMap.repeat.set(1, 9);
  roughnessMap.anisotropy = 8;

  return { map, roughnessMap };
}

export function createLidarDotTexture() {
  const size = 64;
  const c = document.createElement("canvas");
  c.width = size; c.height = size;
  const ctx = c.getContext("2d");

  const g = ctx.createRadialGradient(32, 32, 2, 32, 32, 30);
  g.addColorStop(0.0, "rgba(255, 255, 255, 1.0)");
  g.addColorStop(0.35, "rgba(255, 255, 255, 0.92)");
  g.addColorStop(0.7, "rgba(255, 255, 255, 0.25)");
  g.addColorStop(1.0, "rgba(255, 255, 255, 0.0)");

  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);

  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}