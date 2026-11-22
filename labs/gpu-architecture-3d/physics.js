/**
 * ThinkModule Lab — SXM-Class AI-GPU Thermal & Memory-Wall Physics Engine
 * Models closed-loop DVFS thermal throttling, junction temperature RC dynamics,
 * HBM3e bandwidth saturation (Memory Wall), and Tensor Core MFU.
 */

export const WORKLOADS = {
  decode: {
    id: "decode",
    label: "DECODE • MEMORY WALL",
    basePowerW: 415,
    baseMfu: 0.048,
    baseBwUtil: 0.92,
    description:
      "Autoregressive Batch-1 Decode saturates the 3.35 TB/s HBM3e silicon interposer while 95% of Tensor Cores stall idle waiting on weight transfers."
  },
  prefill: {
    id: "prefill",
    label: "PREFILL • COMPUTE SATURATED",
    basePowerW: 665,
    baseMfu: 0.87,
    baseBwUtil: 0.44,
    description:
      "Parallel prompt prefill feeds massive GEMM matrix tiles directly into all 132 Streaming Multiprocessors, driving high Tensor Core utilization and heat flux."
  },
  stress: {
    id: "stress",
    label: "FP8 BURN • 740W STRESS",
    basePowerW: 740,
    baseMfu: 0.96,
    baseBwUtil: 0.80,
    description:
      "Sustained FP8 matrix saturation pushes board power above nominal 700W TDP. Lower cooling capacity to observe closed-loop DVFS thermal throttling."
  }
};

export const LAYER_INFO = {
  all: {
    tag: "FULL 3D ASSEMBLY",
    title: "6-Layer CoWoS-S AI Package",
    desc: "Drag the Explode slider to inspect how data travels from 6× 8-Hi HBM3e stacks across the silicon interposer into 132 Tensor Core SMs — and how 700W of heat escapes vertically through copper and anodized fins.",
    specs: ["80 GB HBM3e", "3.35 TB/s", "700W TDP", "CoWoS-S 2.5D"]
  },
  5: {
    tag: "LAYER 06 • DISSIPATION",
    title: "Skived Aluminum Fin Array & CNC Shroud",
    desc: "36 razor-thin anodized aluminum cooling fins maximize surface area for high-static-pressure server airflow, pulling heat away from the underlying copper vapor chamber.",
    specs: ["36 Skived Fins", "6063-T6 Alloy", "CNC Titanium Shroud"]
  },
  4: {
    tag: "LAYER 05 • HEAT SPREADING",
    title: "Nickel-Plated Copper Vapor Chamber",
    desc: "Two-phase liquid-vapor micro-wick inside precision CNC copper spreads extreme 110 W/cm² localized hotspot flux from the central silicon die evenly across the entire fin stack.",
    specs: ["C11000 Copper", "Two-Phase Wick", "4× Torx Tension Springs"]
  },
  3: {
    tag: "LAYER 04 • COMPUTE & HBM3e",
    title: "132-SM Compute Die + 6× 8-Hi HBM3e Stacks",
    desc: "The 814 mm² monolithic 4nm compute die sits flanked by six 8-layer vertically stacked HBM3e DRAM cubes connected by thousands of micro-TSVs (Through-Silicon Vias).",
    specs: ["80B Transistors", "6× 8-Hi HBM3e", "1,979 TFLOPs FP8", "50 MB L2 Spine"]
  },
  2: {
    tag: "LAYER 03 • 2.5D BRIDGE",
    title: "CoWoS-S Passive Silicon Interposer",
    desc: "Ultra-fine sub-micron copper redistribution wires etched directly into silicon create a massive 6,144-bit wide parallel bus between HBM3e stacks and the compute die.",
    specs: ["6,144-bit Bus", "Sub-Micron Traces", "TSMC CoWoS-S"]
  },
  1: {
    tag: "LAYER 02 • PACKAGE & BGA",
    title: "Organic Substrate & BGA Solder Ball Grid",
    desc: "High-density multi-layer ABF substrate routes 700W of low-voltage current (<0.9V, >750 Amps) from hundreds of eutectic solder balls into the silicon interposer.",
    specs: ["16-Layer ABF", "Micro-BGA Grid", ">750A Power Delivery"]
  },
  0: {
    tag: "LAYER 01 • CARRIER BOARD",
    title: "SXM Carrier PCB & 24-Phase VRM Array",
    desc: "Server-grade high-Tg PCB housing 24-phase digital VRM chokes, ceramic decoupling capacitor banks, and high-speed 900 GB/s NVLink mezzanine interconnects.",
    specs: ["24-Phase VRM", "900 GB/s NVLink", "High-Tg 14-Layer PCB"]
  }
};

export class GpuPhysics {
  constructor() {
    this.workloadId = "decode";
    this.coolingPct = 100;
    this.tempC = 58.0;
    this.powerW = 415;
    this.clockMHz = 1980;
    this.mfu = 0.048;
    this.bwTBs = 3.08;
    this.bwUtil = 0.92;
    this.throttlePct = 0;
  }

  setWorkload(id) {
    if (WORKLOADS[id]) this.workloadId = id;
  }

  setCooling(pct) {
    this.coolingPct = Math.max(20, Math.min(100, pct));
  }

  step(dt) {
    const w = WORKLOADS[this.workloadId];
    const ambientC = 27.0;
    const peakClock = 1980;
    const minClock = 1080;
    const throttleStartC = 82.0;
    const criticalC = 95.0;

    // Closed-loop DVFS throttling factor based on current junction temperature
    const overTemp = Math.max(0, this.tempC - throttleStartC);
    const throttleFrac = Math.min(0.45, (overTemp / (criticalC - throttleStartC)) * 0.45);
    const dvfsScale = 1.0 - throttleFrac;

    // Power scales super-linearly with clock (P ~ V^2 * f)
    const targetPower = w.basePowerW * Math.pow(dvfsScale, 1.45);

    // Thermal resistance R_th increases sharply as cooling capacity drops
    const coolFactor = 0.28 + 0.72 * (this.coolingPct / 100);
    const rTheta = 0.074 / coolFactor;
    const eqTemp = ambientC + targetPower * rTheta;

    // Smooth first-order silicon thermal inertia
    const tau = 1.45;
    const alpha = 1 - Math.exp(-dt / tau);
    this.tempC += (eqTemp - this.tempC) * alpha;
    this.tempC = Math.max(30, Math.min(102, this.tempC));

    this.powerW += (targetPower - this.powerW) * Math.min(1, dt * 4);
    this.clockMHz = minClock + (peakClock - minClock) * (1 - throttleFrac / 0.45);
    this.throttlePct = throttleFrac * 100;

    this.mfu += (w.baseMfu * dvfsScale - this.mfu) * Math.min(1, dt * 4.5);
    this.bwUtil += (w.baseBwUtil * (0.65 + 0.35 * dvfsScale) - this.bwUtil) * Math.min(1, dt * 4.5);
    this.bwTBs = 3.35 * this.bwUtil;

    // Normalized heat [0..1] for 3D thermal glow & FLIR shader
    const heatNorm = Math.max(0, Math.min(1, (this.tempC - 42) / 53));

    let statusText = w.label;
    let statusSeverity = this.workloadId === "prefill" ? "optimal" : "warn";
    if (this.throttlePct > 4.0) {
      statusText = `THERMAL THROTTLING (−${this.throttlePct.toFixed(0)}% CLK)`;
      statusSeverity = "danger";
    } else if (this.workloadId === "stress") {
      statusSeverity = "warn";
    }

    return {
      tempC: this.tempC,
      powerW: this.powerW,
      clockMHz: this.clockMHz,
      mfu: this.mfu,
      bwUtil: this.bwUtil,
      bwTBs: this.bwTBs,
      throttlePct: this.throttlePct,
      heatNorm,
      statusText,
      statusSeverity
    };
  }
}