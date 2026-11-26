import { CLUSTERS } from '../core/config.js';

/** Deterministic Mulberry32 PRNG */
export function createRNG(seed = 20261126) {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller standard normal sample */
export function gaussianRandom(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/** Pure 3D vector algebra helpers */
export const vec3 = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  lenSq: (a) => a[0] * a[0] + a[1] * a[1] + a[2] * a[2],
  len: (a) => Math.hypot(a[0], a[1], a[2]),
  dist: (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]),
  normalize: (a) => {
    const l = Math.hypot(a[0], a[1], a[2]) || 1e-8;
    return [a[0] / l, a[1] / l, a[2] / l];
  },
  lerp: (a, b, t) => [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t
  ],
  cosineSimilarity: (a, b) => {
    const na = vec3.normalize(a);
    const nb = vec3.normalize(b);
    return Math.max(-1, Math.min(1, vec3.dot(na, nb)));
  },
  centroid: (points) => {
    if (!points.length) return [0, 0, 0];
    let x = 0, y = 0, z = 0;
    for (const p of points) {
      x += p[0]; y += p[1]; z += p[2];
    }
    const n = points.length;
    return [x / n, y / n, z / n];
  }
};

/**
 * Generates a realistic 3D projected latent embedding manifold (~3,200 vectors)
 * with anisotropic Gaussian sub-clusters and hierarchical HNSW layer assignments.
 */
export function generateEmbeddingSpace(seed = 20261126) {
  const rng = createRNG(seed);
  const nodes = [];
  const clusterCentroids = [];
  let idCounter = 0;

  for (const cluster of CLUSTERS) {
    const clusterPoints = [];
    // Generate 3 organic sub-lobes per domain for realistic UMAP manifold structure
    const subLobes = Array.from({ length: 3 }, () => [
      cluster.center[0] + gaussianRandom(rng) * cluster.spread[0] * 0.45,
      cluster.center[1] + gaussianRandom(rng) * cluster.spread[1] * 0.45,
      cluster.center[2] + gaussianRandom(rng) * cluster.spread[2] * 0.45
    ]);

    for (let i = 0; i < cluster.count; i++) {
      const lobe = subLobes[i % subLobes.length];
      const rx = gaussianRandom(rng) * cluster.spread[0] * 0.56;
      const ry = gaussianRandom(rng) * cluster.spread[1] * 0.56;
      const rz = gaussianRandom(rng) * cluster.spread[2] * 0.56;

      const x = lobe[0] + rx;
      const y = lobe[1] + ry;
      const z = lobe[2] + rz;
      const norm = vec3.normalize([x, y, z]);

      // Exponentially decaying HNSW level assignment: -ln(uniform) * mL
      const u = Math.max(1e-6, rng());
      const rawLevel = Math.floor(-Math.log(u) * 0.42);
      const layer = Math.min(2, rawLevel);

      const importance = layer === 2 ? 1.0 : layer === 1 ? 0.68 : 0.35 + rng() * 0.35;
      const topic = cluster.topics[Math.floor(rng() * cluster.topics.length)];
      const chunkNum = String(100 + Math.floor(rng() * 899));

      const node = {
        id: idCounter++,
        pos: [x, y, z],
        norm,
        clusterId: cluster.id,
        clusterName: cluster.shortName,
        hex: cluster.hex,
        color: cluster.color,
        layer,
        importance,
        tokenCount: 128 + Math.floor(rng() * 384),
        label: `${topic} #${chunkNum}`
      };

      nodes.push(node);
      clusterPoints.push(node.pos);
    }

    clusterCentroids.push({
      id: cluster.id,
      name: cluster.name,
      shortName: cluster.shortName,
      hex: cluster.hex,
      color: cluster.color,
      center: vec3.centroid(clusterPoints),
      radius: Math.max(...cluster.spread) * 1.38
    });
  }

  return { nodes, clusterCentroids };
}
