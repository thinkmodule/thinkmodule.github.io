import * as THREE from 'three';
import { SCENARIOS } from './core/config.js';
import { store } from './core/state.js';
import { generateEmbeddingSpace, vec3 } from './math/vector-space.js';
import { buildHNSWGraph } from './math/hnsw-graph.js';
import { runEvaluation } from './math/eval-engine.js';
import { SceneManager } from './scene/scene-manager.js';
import { EmbeddingCloud } from './scene/embedding-cloud.js';
import { HNSWVisualizer } from './scene/hnsw-visualizer.js';
import { EvalVisualizer } from './scene/eval-visualizer.js';
import { EvalRadarChart } from './ui/radar-chart.js';
import { HUDController } from './ui/hud-controller.js';

function bootstrap() {
  const container = document.getElementById('canvas-container');
  const loadingOverlay = document.getElementById('loading-overlay');
  const radarCanvas = document.getElementById('eval-radar-canvas');

  // 1. Build 3,200-vector 3D semantic manifold & real 3-layer HNSW index
  const space = generateEmbeddingSpace(20261126);
  const graph = buildHNSWGraph(space.nodes);

  // 2. Initialize WebGL2 3D Scene, Post-Processing & Visual Subsystems
  const sceneManager = new SceneManager(container);
  const embeddingCloud = new EmbeddingCloud(sceneManager.scene, space);
  const hnswViz = new HNSWVisualizer(sceneManager.scene, graph);
  const evalViz = new EvalVisualizer(sceneManager.scene);
  const radarChart = new EvalRadarChart(radarCanvas);

  let currentEval = null;
  let currentTopKIds = new Set();

  const hud = new HUDController(store, {
    onCameraPreset: (preset) => {
      const customTarget = (preset === 'focus' && currentEval)
        ? currentEval.groundingCentroid
        : null;
      sceneManager.flyToPreset(preset, customTarget);
    },
    onReplay: () => {
      evalViz.replay();
    }
  });

  // 3. Recompute RAG + RLHF Evaluation Pipeline whenever hyperparameters change
  function recomputePipeline() {
    const state = store.get();
    const scenario = SCENARIOS[state.scenarioId] || SCENARIOS.rag_grounded;
    currentEval = runEvaluation(space, graph, scenario, state);
    currentTopKIds = new Set(currentEval.retrieval.topNodes.map(t => t.node.id));

    embeddingCloud.setHighlightedNodes(currentEval.retrieval.topNodes);
    hnswViz.updateEvaluation(currentEval);
    evalViz.updateEvaluation(currentEval, state);
    hud.updateEvaluation(currentEval);
    radarChart.setMetrics(currentEval.metrics, currentEval.verdict.cssClass);
  }

  // Subscribe to reactive state changes
  const evalTriggerKeys = new Set(['scenarioId', 'topK', 'temperature', 'alignment', 'seedOffset']);
  store.subscribe('*', (changedKeys, state) => {
    if (changedKeys.some(k => evalTriggerKeys.has(k))) {
      recomputePipeline();
    }
    if (changedKeys.includes('bloomStrength')) {
      sceneManager.setBloomStrength(state.bloomStrength);
    }
    if (changedKeys.includes('autoRotate')) {
      sceneManager.setAutoRotate(state.autoRotate);
    }
    if (changedKeys.includes('showHNSW')) {
      hnswViz.setVisible(state.showHNSW);
    }
    if (changedKeys.includes('showHalos')) {
      embeddingCloud.setHalosVisible(state.showHalos);
    }
    if (
      changedKeys.includes('showHull') ||
      changedKeys.includes('showRewardField') ||
      changedKeys.includes('showSafetyPlane')
    ) {
      evalViz.setLayerVisibility({
        hull: state.showHull,
        rewardField: state.showRewardField,
        safetyPlane: state.showSafetyPlane
      });
    }
  });

  // 4. Interactive GPU Raycast Inspector (Hover any of the 3,200 embedding vectors)
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let lastPickTime = 0;

  window.addEventListener('pointermove', (e) => {
    // Only inspect vectors when hovering unobstructed 3D canvas
    if (e.target !== sceneManager.renderer.domElement) {
      embeddingCloud.setHoveredNode(null);
      hud.showNodeTooltip(null);
      return;
    }

    const now = performance.now();
    if (now - lastPickTime < 28) return;
    lastPickTime = now;

    const rect = sceneManager.renderer.domElement.getBoundingClientRect();
    pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(pointer, sceneManager.camera);
    const picked = embeddingCloud.pickNode(raycaster, 0.95);

    if (picked && currentEval) {
      embeddingCloud.setHoveredNode(picked.id);
      const dist = vec3.dist(picked.pos, currentEval.queryPos);
      const cosSim = Math.max(0.05, Math.min(0.996, 1.0 - dist / 28.0 + 0.08 * vec3.cosineSimilarity(picked.pos, currentEval.queryPos)));
      hud.showNodeTooltip(picked, e.clientX, e.clientY, cosSim, currentTopKIds.has(picked.id));
    } else {
      embeddingCloud.setHoveredNode(null);
      hud.showNodeTooltip(null);
    }
  });

  // 5. Initial Sync & Launch Render Loop
  recomputePipeline();
  const initState = store.get();
  sceneManager.setBloomStrength(initState.bloomStrength);
  sceneManager.setAutoRotate(initState.autoRotate);

  let lastFrame = performance.now();
  let elapsed = 0;
  let lastFpsPush = 0;

  function animate(now) {
    requestAnimationFrame(animate);
    const dt = Math.min(0.05, (now - lastFrame) / 1000);
    lastFrame = now;
    elapsed += dt;

    const isPlaying = store.get('isPlaying');
    embeddingCloud.update(elapsed);
    evalViz.update(elapsed, dt, sceneManager.camera, isPlaying);
    hnswViz.update(elapsed, sceneManager.camera, evalViz.pipelineProgress);
    radarChart.render(dt);
    sceneManager.render(dt);

    if (now - lastFpsPush > 500) {
      hud.updateFPS(sceneManager.fps);
      lastFpsPush = now;
    }
  }

  requestAnimationFrame(animate);

  // Fade out loading screen once first GPU frame is ready
  requestAnimationFrame(() => {
    setTimeout(() => loadingOverlay.classList.add('fade-out'), 260);
  });
}

bootstrap();
