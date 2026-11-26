import { SAFETY_BOUNDARY_Z } from '../core/config.js';
import { createRNG, gaussianRandom, vec3 } from './vector-space.js';
import { searchHNSW } from './hnsw-graph.js';

/**
 * Evaluates a full RAG + RLHF Alignment + Guardrail pipeline in 3D latent space.
 * All scores (Faithfulness, Relevance, NDCG@K, Safety, KL-Div) are derived
 * directly from geometric vector positions and retrieval rankings.
 */
export function runEvaluation(space, graph, scenario, params) {
  const { topK, temperature, alignment, seedOffset = 0 } = params;
  const rng = createRNG(1337 + scenario.id.length * 997 + seedOffset * 7919);

  // 1. Anchor Query Vector inside target semantic domain
  const domain = space.clusterCentroids[scenario.queryCluster];
  const queryPos = vec3.add(domain.center, scenario.queryOffset);

  // 2. Real Multi-Layer HNSW Retrieval
  const retrieval = searchHNSW(graph, queryPos, topK);

  // 3. Grounding Envelope (Centroid + Radius of retrieved evidence)
  const retrievedPositions = retrieval.topNodes.map(t => t.node.pos);
  const groundingCentroid = vec3.centroid(retrievedPositions);
  let groundingRadius = 2.6;
  for (const p of retrievedPositions) {
    groundingRadius = Math.max(groundingRadius, vec3.dist(p, groundingCentroid) + 0.85);
  }

  // 4. Retrieval Metrics: Precision@K & NDCG@K
  let relevantHits = 0;
  let dcg = 0;
  let idcg = 0;
  retrieval.topNodes.forEach((item, i) => {
    const isRel = item.node.clusterId === scenario.queryCluster ? 1 : 0;
    const gradedRel = isRel ? Math.max(0.4, item.cosineSimilarity) : 0.12;
    if (isRel) relevantHits++;
    const denom = Math.log2(i + 2);
    dcg += (Math.pow(2, gradedRel) - 1) / denom;
    idcg += (Math.pow(2, 0.96) - 1) / denom;
  });
  const precisionAtK = relevantHits / Math.max(1, retrieval.topNodes.length);
  const ndcgAtK = Math.min(0.99, dcg / Math.max(1e-6, idcg));

  // 5. Generate 6 Candidate LLM Trajectories (Base policy vs RLHF-steered policy)
  const numCandidates = 6;
  const candidates = [];

  for (let i = 0; i < numCandidates; i++) {
    // Temperature controls isotropic sampling dispersion in latent space
    const noiseScale = 1.6 + temperature * 6.2;
    const rawNoise = [
      gaussianRandom(rng) * noiseScale,
      gaussianRandom(rng) * noiseScale * 0.85,
      gaussianRandom(rng) * noiseScale
    ];

    // Unaligned base LLM endpoint (subject to scenario drift + temperature variance)
    const driftFactor = i === 0 ? 0.72 : 0.92 + rng() * 0.35;
    const unalignedTarget = vec3.add(
      vec3.add(groundingCentroid, vec3.scale(scenario.driftVector, driftFactor)),
      rawNoise
    );

    // RLHF Alignment Steering pulls trajectory toward Grounding Centroid & away from Safety Plane
    const safeAnchor = [
      groundingCentroid[0],
      groundingCentroid[1],
      Math.min(groundingCentroid[2], SAFETY_BOUNDARY_Z - 7.5)
    ];

    // Candidate 0 gets slight greedy advantage; others explore Pareto trade-offs
    const effectiveAlign = Math.min(0.98, Math.max(0.0, alignment * (i === 0 ? 1.06 : 0.65 + rng() * 0.42)));
    let finalTarget = vec3.lerp(unalignedTarget, safeAnchor, effectiveAlign);

    // Guardrail Deflection check for adversarial boundary penetration
    let guardrailTriggered = false;
    if (finalTarget[2] > SAFETY_BOUNDARY_Z - 0.5) {
      if (alignment >= 0.45) {
        // Active guardrail reflects vector cleanly back into safe manifold
        guardrailTriggered = true;
        finalTarget = [
          finalTarget[0] * 0.88,
          finalTarget[1] - 1.2,
          SAFETY_BOUNDARY_Z - 3.8 - alignment * 3.0
        ];
      }
    }

    // Build smooth 4-point trajectory path (Query -> Context -> Mid-Decode -> Final)
    const mid1 = vec3.lerp(queryPos, groundingCentroid, 0.58);
    const mid2 = vec3.add(
      vec3.lerp(groundingCentroid, finalTarget, 0.55),
      [rawNoise[0] * 0.18, rawNoise[1] * 0.25 + 1.1, guardrailTriggered ? 4.5 : rawNoise[2] * 0.18]
    );

    // Per-Candidate Mathematical Judge Scoring
    const driftDist = vec3.dist(finalTarget, groundingCentroid);
    const normalizedDrift = Math.max(0, (driftDist - groundingRadius * 0.65) / 14.0);
    const faithfulness = Math.max(0.05, Math.min(0.99, Math.exp(-normalizedDrift * 1.85)));

    const queryDist = vec3.dist(finalTarget, queryPos);
    const relevance = Math.max(0.08, Math.min(0.99, Math.exp(-queryDist / 26.0) * (0.72 + 0.28 * ndcgAtK)));

    const safetyViolationDepth = Math.max(0, finalTarget[2] - SAFETY_BOUNDARY_Z);
    const safetyMargin = safetyViolationDepth > 0
      ? Math.max(0.04, 0.35 - safetyViolationDepth * 0.06)
      : Math.min(0.99, 0.55 + Math.min(1.0, (SAFETY_BOUNDARY_Z - finalTarget[2]) / 14.0) * 0.44);

    // KL Divergence proxy: ||v_aligned - v_unaligned||^2 scaled by alignment beta
    const steeringShift = vec3.dist(finalTarget, unalignedTarget);
    const klDivergence = Math.min(1.0, (steeringShift / 22.0) * (0.4 + alignment * 0.8));

    // Composite Reward Model score
    const rewardScore = (
      faithfulness * 0.36 +
      relevance * 0.24 +
      safetyMargin * 0.30 +
      ndcgAtK * 0.10 -
      klDivergence * 0.08
    );

    let status = 'pruned';
    if (safetyViolationDepth > 0) status = 'breach';
    else if (guardrailTriggered) status = 'deflected';
    else if (faithfulness < 0.52) status = 'hallucinated';

    candidates.push({
      index: i,
      label: `Beam #${i + 1}`,
      controlPoints: [queryPos, mid1, mid2, finalTarget],
      unalignedTarget,
      finalTarget,
      driftDist,
      faithfulness,
      relevance,
      safetyMargin,
      klDivergence,
      rewardScore,
      guardrailTriggered,
      safetyBreach: safetyViolationDepth > 0,
      status
    });
  }

  // 6. Select Winning Trajectory (Reward Model argmax, unless low-alignment hallucination/breach wins unsteered)
  candidates.sort((a, b) => b.rewardScore - a.rewardScore);
  const winner = candidates[0];
  if (winner.status === 'pruned') {
    winner.status = 'selected';
  }

  // 7. Compute Overall Verdict & Judge Trace Log
  const compositeScore = Math.max(0, Math.min(100, winner.rewardScore * 108));
  let verdict = { code: 'PASS', label: 'PASS · ALIGNED', cssClass: 'pass' };

  if (winner.safetyBreach) {
    verdict = { code: 'FAIL', label: 'BREACH · UNSAFE', cssClass: 'fail' };
  } else if (scenario.isAdversarial && winner.guardrailTriggered) {
    verdict = { code: 'BLOCKED', label: 'GUARDRAIL · DEFLECTED', cssClass: 'blocked' };
  } else if (winner.faithfulness < 0.55) {
    verdict = { code: 'FAIL', label: 'FAIL · HALLUCINATION', cssClass: 'fail' };
  } else if (winner.faithfulness < 0.76 || winner.klDivergence > 0.65) {
    verdict = { code: 'WARN', label: 'WARN · MODERATE DRIFT', cssClass: 'warn' };
  }

  const traceSteps = [
    {
      step: '01 · HNSW',
      metric: `${retrieval.traversalHops.length} hops`,
      status: 'ok',
      message: `Layer 2→0 greedy search explored ${retrieval.visitedCount} nodes; retrieved Top-${topK} (NDCG ${(ndcgAtK * 100).toFixed(1)}%).`
    },
    {
      step: '02 · GROUND',
      metric: `Δ ${winner.driftDist.toFixed(1)}u`,
      status: winner.faithfulness >= 0.72 ? 'ok' : winner.faithfulness >= 0.52 ? 'warn' : 'fail',
      message: winner.faithfulness >= 0.72
        ? `Winner sits inside retrieved evidence hull (Faithfulness ${(winner.faithfulness * 100).toFixed(0)}%).`
        : `Response drifted ${winner.driftDist.toFixed(1)}u outside evidence hull — ungrounded claim risk.`
    },
    {
      step: '03 · RLHF RM',
      metric: `KL ${winner.klDivergence.toFixed(2)}`,
      status: winner.klDivergence < 0.55 ? 'ok' : 'warn',
      message: `Scored ${numCandidates} candidate beams at τ=${temperature.toFixed(2)}; β=${alignment.toFixed(2)} steered optimal trajectory.`
    },
    {
      step: '04 · SAFETY',
      metric: `${(winner.safetyMargin * 100).toFixed(0)}%`,
      status: winner.safetyBreach ? 'fail' : winner.guardrailTriggered ? 'ok' : 'ok',
      message: winner.safetyBreach
        ? `CRITICAL: Trajectory pierced z=${SAFETY_BOUNDARY_Z} safety hyperplane. Raise β alignment.`
        : winner.guardrailTriggered
          ? `Adversarial vector intercepted at z=${SAFETY_BOUNDARY_Z} boundary and safely deflected.`
          : `Trajectory verified within safe policy manifold (z=${winner.finalTarget[2].toFixed(1)}).`
    }
  ];

  return {
    scenario,
    queryPos,
    retrieval,
    groundingCentroid,
    groundingRadius,
    candidates,
    winner,
    metrics: {
      compositeScore,
      faithfulness: winner.faithfulness,
      relevance: winner.relevance,
      ndcgAtK,
      precisionAtK,
      safetyMargin: winner.safetyMargin,
      klDivergence: winner.klDivergence,
      driftDist: winner.driftDist,
      hops: retrieval.traversalHops.length
    },
    verdict,
    traceSteps
  };
}
