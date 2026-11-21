/**
 * ThinkModule Lab — GPU Inference, KV-Cache & Roofline Analytical Engine (v1.0)
 * Pure ES6 first-principles hardware modeling:
 *   - Real GPU specs (B200, H100 SXM5, A100 SXM4, RTX 4090, Apple M3 Max)
 *   - Real LLM architectures (Llama-3 8B/70B, Qwen-2.5 32B, Mixtral 8x7B MoE, GPT-3 175B)
 *   - Exact KV-Cache, MHA/GQA/MLA scaling, MoE batch-dependent expert activation
 *   - Roofline Arithmetic Intensity, TTFT, TPOT, MFU, HBM Utilization, Cost/1M tokens
 *   - Batch Size 1..128 Pareto frontier sweep
 */
(function (global) {
  "use strict";

  const GPUS = [
    {
      id: "h100_sxm",
      name: "NVIDIA H100 SXM5 (80 GB HBM3)",
      shortName: "H100 SXM5",
      vramGB: 80,
      bwGBs: 3350,
      fp16Tflops: 989,
      fp8Tflops: 1979,
      hourlyCost: 2.95,
      sramMB: 50,
      arch: "Hopper"
    },
    {
      id: "b200_sxm",
      name: "NVIDIA B200 SXM (192 GB HBM3e)",
      shortName: "B200 SXM",
      vramGB: 192,
      bwGBs: 8000,
      fp16Tflops: 2250,
      fp8Tflops: 4500,
      hourlyCost: 5.50,
      sramMB: 96,
      arch: "Blackwell"
    },
    {
      id: "a100_sxm",
      name: "NVIDIA A100 SXM4 (80 GB HBM2e)",
      shortName: "A100 80GB",
      vramGB: 80,
      bwGBs: 2039,
      fp16Tflops: 312,
      fp8Tflops: 312,
      hourlyCost: 1.65,
      sramMB: 40,
      arch: "Ampere"
    },
    {
      id: "rtx_4090",
      name: "NVIDIA RTX 4090 (24 GB GDDR6X)",
      shortName: "RTX 4090",
      vramGB: 24,
      bwGBs: 1008,
      fp16Tflops: 330,
      fp8Tflops: 660,
      hourlyCost: 0.45,
      sramMB: 72,
      arch: "Ada Lovelace"
    },
    {
      id: "m3_max",
      name: "Apple M3 Max (128 GB Unified)",
      shortName: "M3 Max",
      vramGB: 128,
      bwGBs: 400,
      fp16Tflops: 28,
      fp8Tflops: 28,
      hourlyCost: 0.14,
      sramMB: 48,
      arch: "Apple Silicon"
    }
  ];

  const MODELS = [
    {
      id: "llama3_8b",
      name: "Llama-3 8B (GQA 4:1)",
      shortName: "Llama-3 8B",
      paramsB: 8.03,
      activeParamsB: 8.03,
      layers: 32,
      dModel: 4096,
      nHeads: 32,
      nKvHeads: 8,
      dHead: 128,
      isMoE: false
    },
    {
      id: "qwen25_32b",
      name: "Qwen-2.5 32B (GQA 5:1)",
      shortName: "Qwen-2.5 32B",
      paramsB: 32.5,
      activeParamsB: 32.5,
      layers: 64,
      dModel: 5120,
      nHeads: 40,
      nKvHeads: 8,
      dHead: 128,
      isMoE: false
    },
    {
      id: "mixtral_8x7b",
      name: "Mixtral 8x7B MoE (12.9B Active)",
      shortName: "Mixtral 8x7B",
      paramsB: 46.7,
      activeParamsB: 12.9,
      layers: 32,
      dModel: 4096,
      nHeads: 32,
      nKvHeads: 8,
      dHead: 128,
      isMoE: true
    },
    {
      id: "llama3_70b",
      name: "Llama-3 70B (GQA 8:1)",
      shortName: "Llama-3 70B",
      paramsB: 70.6,
      activeParamsB: 70.6,
      layers: 80,
      dModel: 8192,
      nHeads: 64,
      nKvHeads: 8,
      dHead: 128,
      isMoE: false
    },
    {
      id: "gpt3_175b",
      name: "GPT-3 175B (Legacy MHA 1:1)",
      shortName: "GPT-3 175B",
      paramsB: 175.0,
      activeParamsB: 175.0,
      layers: 96,
      dModel: 12288,
      nHeads: 96,
      nKvHeads: 96,
      dHead: 128,
      isMoE: false
    }
  ];

  const WEIGHT_QUANTS = {
    fp16: { label: "FP16 / BF16", bytes: 2.0, useFp8: false, overhead: 1.0 },
    fp8:  { label: "FP8 (E4M3)", bytes: 1.0, useFp8: true,  overhead: 1.03 },
    int8: { label: "INT8 SmoothQuant", bytes: 1.0, useFp8: false, overhead: 1.05 },
    int4: { label: "INT4 (AWQ/GPTQ)", bytes: 0.55, useFp8: false, overhead: 1.08 }
  };

  const KV_QUANTS = {
    fp16: { label: "FP16 KV", bytes: 2.0 },
    fp8:  { label: "FP8 KV",  bytes: 1.0 }
  };

  const BATCH_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128];
  const PROMPT_OPTIONS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];

  const PRESETS = {
    memory_wall: {
      gpuId: "h100_sxm",
      gpuCount: 1,
      modelId: "llama3_8b",
      attnMode: "gqa",
      weightQuant: "fp16",
      kvQuant: "fp16",
      allocMode: "paged",
      batchSize: 1,
      promptLen: 2048,
      genLen: 512
    },
    vllm_prod: {
      gpuId: "h100_sxm",
      gpuCount: 1,
      modelId: "llama3_70b",
      attnMode: "gqa",
      weightQuant: "int4",
      kvQuant: "fp8",
      allocMode: "paged",
      batchSize: 32,
      promptLen: 2048,
      genLen: 512
    },
    long_ctx: {
      gpuId: "h100_sxm",
      gpuCount: 1,
      modelId: "llama3_8b",
      attnMode: "gqa",
      weightQuant: "fp16",
      kvQuant: "fp16",
      allocMode: "contiguous",
      batchSize: 8,
      promptLen: 131072,
      genLen: 1024
    },
    mha_shock: {
      gpuId: "h100_sxm",
      gpuCount: 4,
      modelId: "gpt3_175b",
      attnMode: "mha",
      weightQuant: "fp8",
      kvQuant: "fp16",
      allocMode: "contiguous",
      batchSize: 16,
      promptLen: 8192,
      genLen: 512
    }
  };

  function getGpu(id) {
    return GPUS.find(g => g.id === id) || GPUS[0];
  }

  function getModel(id) {
    return MODELS.find(m => m.id === id) || MODELS[0];
  }

  function tpScalingEfficiency(gpuCount) {
    if (gpuCount <= 1) return 1.0;
    if (gpuCount === 2) return 0.94;
    if (gpuCount === 4) return 0.89;
    return 0.84;
  }

  function computeMetrics(cfg) {
    const gpu = getGpu(cfg.gpuId);
    const model = getModel(cfg.modelId);
    const wq = WEIGHT_QUANTS[cfg.weightQuant] || WEIGHT_QUANTS.fp16;
    const kvq = KV_QUANTS[cfg.kvQuant] || KV_QUANTS.fp16;

    const gpuCount = Math.max(1, cfg.gpuCount || 1);
    const batchSize = Math.max(1, cfg.batchSize || 1);
    const promptLen = Math.max(64, cfg.promptLen || 2048);
    const genLen = Math.max(32, cfg.genLen || 512);
    const seqLen = promptLen + genLen;

    // Effective hardware ceilings with realistic NVLink AllReduce TP overhead
    const tpEff = tpScalingEfficiency(gpuCount);
    const totalVramGB = gpu.vramGB * gpuCount;
    const effBwGBs = gpu.bwGBs * gpuCount * tpEff;
    const perGpuPeakTflops = wq.useFp8 ? gpu.fp8Tflops : gpu.fp16Tflops;
    const effPeakTflops = perGpuPeakTflops * gpuCount * tpEff;
    const ridgePoint = (effPeakTflops * 1000) / Math.max(effBwGBs, 1e-6);

    // Effective KV heads (MHA vs GQA vs DeepSeek-style MLA)
    let effKvHeads = model.nKvHeads;
    if (cfg.attnMode === "mha") {
      effKvHeads = model.nHeads;
    } else if (cfg.attnMode === "mla") {
      effKvHeads = Math.max(1, model.nKvHeads / 4);
    }

    // Memory footprint (GB)
    const weightsGB = model.paramsB * wq.bytes * wq.overhead;
    const kvBytesPerToken = 2 * model.layers * effKvHeads * model.dHead * kvq.bytes;
    const logicalKvGB = (kvBytesPerToken * seqLen * batchSize) / 1e9;

    // Contiguous pre-allocation wastes ~45-58% due to unknown output length + fragmentation
    const contiguousSlotLen = Math.ceil((seqLen * 1.85) / 512) * 512;
    const contiguousKvGB = (kvBytesPerToken * contiguousSlotLen * batchSize) / 1e9;

    // PagedAttention allocates 16-token blocks on demand (~8 tokens avg waste per sequence)
    const pagedKvGB = (kvBytesPerToken * (seqLen + 8) * batchSize) / 1e9;

    const allocatedKvGB = cfg.allocMode === "paged" ? pagedKvGB : contiguousKvGB;
    const wasteKvGB = Math.max(0, allocatedKvGB - logicalKvGB);
    const wastePct = allocatedKvGB > 0 ? (wasteKvGB / allocatedKvGB) * 100 : 0;

    // CUDA context + FlashAttention-2 linear activation buffer
    const activationGB =
      0.75 * gpuCount +
      (batchSize * Math.min(promptLen, 8192) * model.dModel * 2 * 4) / 1e9 * 0.14;

    const totalUsedGB = weightsGB + allocatedKvGB + activationGB;
    const vramUtilPct = (totalUsedGB / totalVramGB) * 100;
    const isOOM = totalUsedGB > totalVramGB;

    // MoE: higher batch size activates a larger fraction of unique experts per decode step
    let activeWeightRatio = model.activeParamsB / model.paramsB;
    if (model.isMoE) {
      activeWeightRatio = Math.min(
        1.0,
        activeWeightRatio + (1 - activeWeightRatio) * (1 - Math.exp(-batchSize / 12))
      );
    }
    const decodeWeightsReadBytes = model.paramsB * activeWeightRatio * 1e9 * wq.bytes;
    const prefillWeightsReadBytes = model.paramsB * Math.min(1, activeWeightRatio * 1.35) * 1e9 * wq.bytes;

    // ---- PREFILL PHASE (Processes promptLen * batchSize tokens in parallel) ----
    const prefillLinearFlops = 2 * model.activeParamsB * 1e9 * batchSize * promptLen;
    const prefillAttnFlops = 2 * model.layers * batchSize * model.dModel * promptLen * promptLen;
    const prefillFlops = prefillLinearFlops + prefillAttnFlops;

    const prefillKvWriteBytes = kvBytesPerToken * promptLen * batchSize;
    const prefillActBytes = batchSize * promptLen * model.dModel * 4 * model.layers * 0.22;
    const prefillBytes = Math.max(1, prefillWeightsReadBytes + prefillKvWriteBytes + prefillActBytes);

    const prefillAI = prefillFlops / prefillBytes;
    const prefillRoofTflops = Math.min(effPeakTflops, (effBwGBs * prefillAI) / 1000);
    const prefillAchievedTflops = prefillRoofTflops * 0.88;
    const ttftMs = (prefillFlops / (Math.max(prefillAchievedTflops, 1e-6) * 1e12)) * 1000;
    const prefillMfuPct = (prefillAchievedTflops / effPeakTflops) * 100;
    const prefillBwUtilPct = Math.min(
      100,
      ((prefillBytes / Math.max(ttftMs / 1000, 1e-9)) / (effBwGBs * 1e9)) * 100
    );
    const prefillRegime = prefillAI >= ridgePoint ? "Compute-Bound" : "Memory-Bound";

    // ---- DECODE PHASE (Generates 1 token per user across batchSize) ----
    const decodeLinearFlops = 2 * model.activeParamsB * 1e9 * batchSize;
    const decodeAttnFlops = 4 * model.layers * batchSize * model.dModel * seqLen;
    const decodeFlops = decodeLinearFlops + decodeAttnFlops;

    const decodeKvReadBytes = kvBytesPerToken * seqLen * batchSize;
    const decodeBytes = Math.max(1, decodeWeightsReadBytes + decodeKvReadBytes);

    const decodeAI = decodeFlops / decodeBytes;
    const decodeRoofTflops = Math.min(effPeakTflops, (effBwGBs * decodeAI) / 1000);
    const decodeAchievedTflops = decodeRoofTflops * 0.85;
    const tpotMs = (decodeFlops / (Math.max(decodeAchievedTflops, 1e-6) * 1e12)) * 1000;

    const userTokPerSec = 1000 / Math.max(tpotMs, 1e-6);
    const totalTokPerSec = userTokPerSec * batchSize;
    const decodeMfuPct = (decodeAchievedTflops / effPeakTflops) * 100;
    const decodeBwUtilPct = Math.min(
      100,
      ((decodeBytes / Math.max(tpotMs / 1000, 1e-9)) / (effBwGBs * 1e9)) * 100
    );
    const decodeRegime = decodeAI >= ridgePoint ? "Compute-Bound" : "Memory-Bound";

    // KV vs Weights traffic share during decode
    const kvTrafficSharePct = (decodeKvReadBytes / decodeBytes) * 100;
    const weightTrafficSharePct = 100 - kvTrafficSharePct;

    // Cloud serving cost per 1M generated tokens
    const clusterHourlyCost = gpu.hourlyCost * gpuCount;
    const costPer1MTok = (clusterHourlyCost / 3600) * (1e6 / Math.max(totalTokPerSec, 1e-6));

    // Senior engineering diagnosis
    const diagnosis = buildDiagnosis({
      cfg, gpu, model, gpuCount, batchSize, seqLen, isOOM,
      totalUsedGB, totalVramGB, weightsGB, logicalKvGB, wasteKvGB, wastePct,
      decodeAI, ridgePoint, decodeMfuPct, decodeBwUtilPct,
      kvTrafficSharePct, userTokPerSec, totalTokPerSec
    });

    return {
      gpu,
      model,
      wq,
      kvq,
      gpuCount,
      batchSize,
      promptLen,
      genLen,
      seqLen,
      effKvHeads,
      kvBytesPerToken,
      hardware: {
        totalVramGB,
        effBwGBs,
        effPeakTflops,
        ridgePoint,
        tpEff
      },
      memory: {
        weightsGB,
        logicalKvGB,
        allocatedKvGB,
        wasteKvGB,
        wastePct,
        activationGB,
        totalUsedGB,
        freeGB: Math.max(0, totalVramGB - totalUsedGB),
        vramUtilPct,
        isOOM
      },
      prefill: {
        ai: prefillAI,
        achievedTflops: prefillAchievedTflops,
        ttftMs,
        mfuPct: prefillMfuPct,
        bwUtilPct: prefillBwUtilPct,
        regime: prefillRegime
      },
      decode: {
        ai: decodeAI,
        achievedTflops: decodeAchievedTflops,
        tpotMs,
        userTokPerSec,
        totalTokPerSec,
        mfuPct: decodeMfuPct,
        bwUtilPct: decodeBwUtilPct,
        regime: decodeRegime,
        kvTrafficSharePct,
        weightTrafficSharePct,
        costPer1MTok
      },
      diagnosis
    };
  }

  function buildDiagnosis(d) {
    if (d.isOOM) {
      const deficit = (d.totalUsedGB - d.totalVramGB).toFixed(1);
      const tips = [];
      if (d.cfg.allocMode === "contiguous" && d.wasteKvGB > 4) {
        tips.push(`switch to PagedAttention (saves ${d.wasteKvGB.toFixed(1)} GB waste)`);
      }
      if (d.cfg.kvQuant === "fp16" && d.logicalKvGB > d.weightsGB * 0.35) {
        tips.push("enable FP8 KV-Cache (halves KV memory)");
      }
      if (d.cfg.weightQuant === "fp16") {
        tips.push("quantize weights to FP8 or INT4 AWQ");
      }
      if (d.cfg.attnMode === "mha") {
        tips.push("replace legacy MHA with GQA or MLA");
      }
      if (!tips.length) {
        tips.push(`scale to ${Math.min(8, d.gpuCount * 2)}× Tensor Parallel GPUs or reduce batch size`);
      }
      return {
        severity: "danger",
        title: `OUT OF MEMORY (+${deficit} GB Over ${d.totalVramGB} GB VRAM Limit)`,
        body: `Total footprint (${d.totalUsedGB.toFixed(1)} GB: ${d.weightsGB.toFixed(1)} GB weights + ${(d.logicalKvGB + d.wasteKvGB).toFixed(1)} GB KV) exceeds cluster capacity. Fix: ${tips.slice(0, 2).join(", or ")}.`
      };
    }

    if (d.decodeAI < d.ridgePoint * 0.15) {
      return {
        severity: "warn",
        title: `Severe Memory Wall — Tensor Cores Starving (${d.decodeMfuPct.toFixed(1)}% MFU)`,
        body: `Decode Arithmetic Intensity is only ${d.decodeAI.toFixed(1)} FLOPs/Byte vs. the ${d.gpu.shortName} ridge point of ${d.ridgePoint.toFixed(0)} FLOPs/Byte. HBM bandwidth is saturated (${d.decodeBwUtilPct.toFixed(0)}%) while ${(100 - d.decodeMfuPct).toFixed(1)}% of compute sits idle. Increase Batch Size or quantize weights to INT4/FP8 to amortize weight reads.`
      };
    }

    if (d.kvTrafficSharePct > 55) {
      return {
        severity: "warn",
        title: `KV-Cache Bandwidth Bottleneck (${d.kvTrafficSharePct.toFixed(0)}% of HBM Traffic)`,
        body: `At ${d.seqLen.toLocaleString()} context & Batch ${d.batchSize}, reading the KV-Cache dominates memory traffic over reading the model weights themselves. Enable FP8 KV-Cache or MLA compression to directly boost tokens/sec.`
      };
    }

    if (d.decodeAI >= d.ridgePoint * 0.75) {
      return {
        severity: "good",
        title: `Compute-Optimal Production Regime (${d.totalTokPerSec.toFixed(0)} tok/s)`,
        body: `Batch ${d.batchSize} pushes Arithmetic Intensity (${d.decodeAI.toFixed(1)} FLOPs/Byte) right up to the ${d.gpu.shortName} ridge point, achieving ${d.decodeMfuPct.toFixed(1)}% MFU with ${d.userTokPerSec.toFixed(1)} tok/s per user. Classic high-efficiency vLLM serving point.`
      };
    }

    return {
      severity: "info",
      title: `Balanced Memory-Bound Serving (${d.totalTokPerSec.toFixed(0)} tok/s total)`,
      body: `Each user receives ${d.userTokPerSec.toFixed(1)} tok/s (comfortably above human reading speed) while using ${((d.totalUsedGB / d.totalVramGB) * 100).toFixed(0)}% VRAM. Scaling Batch Size further increases throughput until hitting either VRAM capacity or the ${d.ridgePoint.toFixed(0)} FLOP/B ridge point.`
    };
  }

  function computeBatchSweep(cfg) {
    return BATCH_OPTIONS.map(b => {
      const m = computeMetrics({ ...cfg, batchSize: b });
      return {
        batchSize: b,
        isOOM: m.memory.isOOM,
        vramUtilPct: m.memory.vramUtilPct,
        decodeAI: m.decode.ai,
        decodeTflops: m.decode.achievedTflops,
        tpotMs: m.decode.tpotMs,
        userTokPerSec: m.decode.userTokPerSec,
        totalTokPerSec: m.decode.totalTokPerSec,
        costPer1MTok: m.decode.costPer1MTok
      };
    });
  }

  global.RooflineEngine = {
    GPUS,
    MODELS,
    WEIGHT_QUANTS,
    KV_QUANTS,
    BATCH_OPTIONS,
    PROMPT_OPTIONS,
    PRESETS,
    computeMetrics,
    computeBatchSweep
  };
})(window);