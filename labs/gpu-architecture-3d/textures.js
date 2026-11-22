/**
 * ThinkModule Lab — Procedural High-Resolution PBR Micro-Texture Generator
 * Synthesizes 1024x1024 PCB trace maps, CoWoS-S silicon bus lithography,
 * 132-SM GPU die floorplan, and directional brushed-metal maps in memory (zero external assets).
 */
import * as THREE from "three";

function makeCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return { canvas: c, ctx: c.getContext("2d") };
}

function finalizeTexture(canvas, isColor = true, repeatX = 1, repeatY = 1) {
  const tex = new THREE.CanvasTexture(canvas);
  tex.anisotropy = 8;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(repeatX, repeatY);
  if (isColor) tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

export function createPcbTextures() {
  const { canvas, ctx } = makeCanvas(1024, 1024);
  const { canvas: bumpC, ctx: bctx } = makeCanvas(1024, 1024);

  // Deep matte obsidian-emerald soldermask
  ctx.fillStyle = "#081214";
  ctx.fillRect(0, 0, 1024, 1024);
  bctx.fillStyle = "#505050";
  bctx.fillRect(0, 0, 1024, 1024);

  // Subtle woven fiberglass micro-grid
  ctx.strokeStyle = "rgba(26, 56, 62, 0.22)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 1024; i += 16) {
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 1024); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(1024, i); ctx.stroke();
  }

  // Differential-pair copper traces radiating outward from central BGA pad
  const drawTrace = (x1, y1, x2, y2, col, width) => {
    ctx.strokeStyle = col;
    ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    bctx.strokeStyle = "#b8b8b8";
    bctx.lineWidth = width;
    bctx.beginPath(); bctx.moveTo(x1, y1); bctx.lineTo(x2, y2); bctx.stroke();
  };

  for (let i = 0; i < 64; i++) {
    const off = 110 + i * 12.6;
    const isGold = i % 5 === 0;
    const col = isGold ? "rgba(196, 155, 76, 0.42)" : "rgba(42, 92, 98, 0.55)";
    drawTrace(off, 40, off, 210, col, 2.2);
    drawTrace(off, 814, off, 984, col, 2.2);
    drawTrace(40, off, 200, off, col, 2.2);
    drawTrace(824, off, 984, off, col, 2.2);
  }

  // Gold-plated via rings around perimeter
  for (let y = 70; y < 960; y += 38) {
    for (let x = 70; x < 960; x += 38) {
      const inCenter = x > 235 && x < 790 && y > 235 && y < 790;
      if (inCenter) continue;
      ctx.fillStyle = "#b88c3e";
      ctx.beginPath(); ctx.arc(x, y, 4.2, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#04080a";
      ctx.beginPath(); ctx.arc(x, y, 1.9, 0, Math.PI * 2); ctx.fill();

      bctx.fillStyle = "#ffffff";
      bctx.beginPath(); bctx.arc(x, y, 4.2, 0, Math.PI * 2); bctx.fill();
    }
  }

  // Crisp industrial silkscreen border & fiducial labels
  ctx.strokeStyle = "rgba(215, 228, 240, 0.35)";
  ctx.lineWidth = 2;
  ctx.strokeRect(24, 24, 976, 976);
  ctx.strokeRect(218, 218, 588, 588);

  ctx.fillStyle = "rgba(220, 232, 245, 0.55)";
  ctx.font = "600 18px ui-monospace, Menlo, monospace";
  ctx.fillText("THINKMODULE // SXM5-AI REV 4.2", 44, 54);
  ctx.fillText("24-PHASE DIGITAL VRM  |  900 GB/S NVLINK", 44, 990);

  return {
    map: finalizeTexture(canvas, true),
    bumpMap: finalizeTexture(bumpC, false)
  };
}

export function createInterposerTexture() {
  const { canvas, ctx } = makeCanvas(1024, 1024);

  // Dark iridescent silicon wafer background
  const grad = ctx.createLinearGradient(0, 0, 1024, 1024);
  grad.addColorStop(0, "#0b1420");
  grad.addColorStop(0.5, "#112233");
  grad.addColorStop(1, "#0a1524");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 1024, 1024);

  // High-density 6,144-bit horizontal silicon routing bridges between HBM stacks and central Die
  ctx.strokeStyle = "rgba(76, 178, 228, 0.38)";
  ctx.lineWidth = 1.5;
  const hbmBands = [180, 512, 844];
  hbmBands.forEach(cy => {
    for (let dy = -92; dy <= 92; dy += 6) {
      ctx.beginPath();
      ctx.moveTo(90, cy + dy);
      ctx.lineTo(934, cy + dy);
      ctx.stroke();
    }
  });

  // Vertical power/clock distribution spine
  ctx.strokeStyle = "rgba(180, 146, 82, 0.25)";
  for (let x = 290; x <= 734; x += 14) {
    ctx.beginPath();
    ctx.moveTo(x, 64);
    ctx.lineTo(x, 960);
    ctx.stroke();
  }

  // Pad outlines for 6 HBM3e stacks + central Compute Die
  ctx.strokeStyle = "rgba(160, 220, 255, 0.55)";
  ctx.lineWidth = 2.5;
  ctx.strokeRect(275, 115, 474, 794);

  ctx.strokeStyle = "rgba(167, 139, 250, 0.55)";
  [85, 785].forEach(bx => {
    hbmBands.forEach(cy => {
      ctx.strokeRect(bx, cy - 102, 154, 204);
    });
  });

  return finalizeTexture(canvas, true);
}

export function createDieLithographyTexture() {
  const { canvas, ctx } = makeCanvas(1024, 1024);

  // Mirror-dark monolithic silicon base
  ctx.fillStyle = "#090f1a";
  ctx.fillRect(0, 0, 1024, 1024);

  // Central 50MB L2 Cache & Crossbar Spine
  ctx.fillStyle = "#122033";
  ctx.fillRect(466, 26, 92, 972);
  ctx.strokeStyle = "rgba(56, 189, 248, 0.45)";
  ctx.lineWidth = 2;
  ctx.strokeRect(466, 26, 92, 972);

  // Fine L2 SRAM banks along spine
  ctx.fillStyle = "rgba(92, 162, 210, 0.28)";
  for (let y = 42; y < 970; y += 22) {
    ctx.fillRect(476, y, 30, 14);
    ctx.fillRect(518, y, 30, 14);
  }

  // 8x16 GPC / SM micro-lithography cells flanking the L2 spine
  const drawHalf = xStart => {
    for (let r = 0; r < 16; r++) {
      for (let c = 0; c < 4; c++) {
        const x = xStart + c * 102;
        const y = 32 + r * 60;
        ctx.fillStyle = "#0e192b";
        ctx.fillRect(x, y, 90, 50);
        ctx.strokeStyle = "rgba(110, 165, 220, 0.28)";
        ctx.lineWidth = 1.2;
        ctx.strokeRect(x, y, 90, 50);

        // Micro Tensor-Core sub-quadrants
        ctx.fillStyle = "rgba(56, 189, 248, 0.14)";
        ctx.fillRect(x + 6, y + 6, 36, 16);
        ctx.fillRect(x + 48, y + 6, 36, 16);
        ctx.fillRect(x + 6, y + 28, 36, 16);
        ctx.fillRect(x + 48, y + 28, 36, 16);
      }
    }
  };
  drawHalf(38);
  drawHalf(580);

  // Laser-etched border
  ctx.strokeStyle = "rgba(210, 232, 255, 0.4)";
  ctx.lineWidth = 4;
  ctx.strokeRect(10, 10, 1004, 1004);

  return finalizeTexture(canvas, true);
}

export function createBrushedMetalMap(baseHex = "#b87344", streakAlpha = 0.14) {
  const { canvas, ctx } = makeCanvas(512, 512);
  ctx.fillStyle = baseHex;
  ctx.fillRect(0, 0, 512, 512);

  for (let y = 0; y < 512; y++) {
    const bright = Math.random() > 0.5 ? 255 : 0;
    const a = Math.random() * streakAlpha;
    ctx.fillStyle = `rgba(${bright},${bright},${bright},${a.toFixed(3)})`;
    ctx.fillRect(0, y, 512, 1);
  }
  return finalizeTexture(canvas, true, 2, 2);
}