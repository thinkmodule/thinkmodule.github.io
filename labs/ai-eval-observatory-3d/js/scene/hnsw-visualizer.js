import * as THREE from 'three';

export class HNSWVisualizer {
  constructor(scene, graph) {
    this.scene = scene;
    this.graph = graph;

    this.rootGroup = new THREE.Group();
    this.backboneGroup = new THREE.Group();
    this.activeSearchGroup = new THREE.Group();

    this.rootGroup.add(this.backboneGroup);
    this.rootGroup.add(this.activeSearchGroup);
    this.scene.add(this.rootGroup);

    this._hopCurve = null;
    this._photonMesh = null;
    this._topKFilaments = null;
    this._topKRings = [];

    this._buildStaticBackbone();
  }

  _buildStaticBackbone() {
    const { nodes, edgesByLayer, layerNodes } = this.graph;

    const layerStyles = [
      { color: 0x334155, opacity: 0.14 }, // Layer 0: subtle semantic mesh
      { color: 0x4f46e5, opacity: 0.24 }, // Layer 1: regional bridges
      { color: 0x38bdf8, opacity: 0.42 }  // Layer 2: express highways
    ];

    for (let layer = 0; layer <= 2; layer++) {
      const edges = edgesByLayer[layer];
      if (!edges.length) continue;

      const positions = new Float32Array(edges.length * 6);
      for (let i = 0; i < edges.length; i++) {
        const a = nodes[edges[i].from].pos;
        const b = nodes[edges[i].to].pos;
        positions[i * 6]     = a[0];
        positions[i * 6 + 1] = a[1];
        positions[i * 6 + 2] = a[2];
        positions[i * 6 + 3] = b[0];
        positions[i * 6 + 4] = b[1];
        positions[i * 6 + 5] = b[2];
      }

      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const mat = new THREE.LineBasicMaterial({
        color: layerStyles[layer].color,
        transparent: true,
        opacity: layerStyles[layer].opacity,
        depthWrite: false,
        blending: THREE.AdditiveBlending
      });

      this.backboneGroup.add(new THREE.LineSegments(geo, mat));
    }

    // Subtle glowing octahedron markers on Layer 2 Express Hubs
    const hubNodes = layerNodes[2];
    if (hubNodes.length > 0) {
      const hubGeo = new THREE.OctahedronGeometry(0.28, 0);
      const hubMat = new THREE.MeshBasicMaterial({
        color: 0x7dd3fc,
        wireframe: true,
        transparent: true,
        opacity: 0.55
      });
      const instanced = new THREE.InstancedMesh(hubGeo, hubMat, hubNodes.length);
      const dummy = new THREE.Object3D();
      hubNodes.forEach((n, idx) => {
        dummy.position.set(...n.pos);
        dummy.updateMatrix();
        instanced.setMatrixAt(idx, dummy.matrix);
      });
      instanced.instanceMatrix.needsUpdate = true;
      this.backboneGroup.add(instanced);
    }
  }

  _clearGroup(group) {
    while (group.children.length > 0) {
      const obj = group.children[0];
      group.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
        else obj.material.dispose();
      }
    }
  }

  updateEvaluation(evalResult) {
    this._clearGroup(this.activeSearchGroup);
    this._topKRings = [];
    this._hopCurve = null;

    const { nodes } = this.graph;
    const { retrieval, queryPos } = evalResult;
    const queryVec = new THREE.Vector3(...queryPos);

    // 1. Build glowing HNSW traversal hop path (Entry -> Layer 2 -> Layer 1 -> Layer 0 -> Query)
    const pathPoints = [new THREE.Vector3(...nodes[retrieval.entryId].pos)];
    for (const hop of retrieval.traversalHops) {
      const pt = new THREE.Vector3(...nodes[hop.toId].pos);
      if (pt.distanceTo(pathPoints[pathPoints.length - 1]) > 0.15) {
        pathPoints.push(pt);
      }
    }
    pathPoints.push(queryVec);

    if (pathPoints.length >= 2) {
      this._hopCurve = new THREE.CatmullRomCurve3(pathPoints, false, 'centripetal', 0.25);
      const sampled = this._hopCurve.getPoints(Math.max(40, pathPoints.length * 10));
      const hopGeo = new THREE.BufferGeometry().setFromPoints(sampled);
      const hopMat = new THREE.LineBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.85,
        blending: THREE.AdditiveBlending
      });
      this.activeSearchGroup.add(new THREE.Line(hopGeo, hopMat));

      // Traveling HNSW search photon
      const photonGeo = new THREE.SphereGeometry(0.32, 16, 16);
      const photonMat = new THREE.MeshBasicMaterial({ color: 0xe0f2fe });
      this._photonMesh = new THREE.Mesh(photonGeo, photonMat);
      this.activeSearchGroup.add(this._photonMesh);
    }

    // 2. Crisp Top-K Retrieval Filaments & Target Rings
    const filPos = [];
    const ringGeo = new THREE.RingGeometry(0.42, 0.56, 24);

    for (const item of retrieval.topNodes) {
      const p = item.node.pos;
      filPos.push(queryPos[0], queryPos[1], queryPos[2], p[0], p[1], p[2]);

      const ringMat = new THREE.MeshBasicMaterial({
        color: item.node.hex,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.85,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });
      const ring = new THREE.Mesh(ringGeo.clone(), ringMat);
      ring.position.set(p[0], p[1], p[2]);
      ring.userData.phase = item.rank * 0.7;
      this.activeSearchGroup.add(ring);
      this._topKRings.push(ring);
    }
    ringGeo.dispose();

    const filGeo = new THREE.BufferGeometry();
    filGeo.setAttribute('position', new THREE.Float32BufferAttribute(filPos, 3));
    const filMat = new THREE.LineBasicMaterial({
      color: 0x7dd3fc,
      transparent: true,
      opacity: 0.62,
      blending: THREE.AdditiveBlending
    });
    this._topKFilaments = new THREE.LineSegments(filGeo, filMat);
    this.activeSearchGroup.add(this._topKFilaments);
  }

  setVisible(visible) {
    this.backboneGroup.visible = visible;
  }

  update(time, camera, pipelineProgress) {
    // Animate photon zipping along the HNSW hop trajectory
    if (this._hopCurve && this._photonMesh) {
      const t = (time * 0.42) % 1.0;
      const pt = this._hopCurve.getPointAt(t);
      this._photonMesh.position.copy(pt);
      const s = 0.85 + 0.35 * Math.sin(time * 8.0);
      this._photonMesh.scale.setScalar(s);
    }

    // Billboard Top-K rings toward camera with gentle pulse
    for (const ring of this._topKRings) {
      ring.quaternion.copy(camera.quaternion);
      const pulse = 0.9 + 0.25 * Math.sin(time * 3.4 + ring.userData.phase);
      ring.scale.setScalar(pulse * Math.min(1, pipelineProgress * 2.2));
    }
  }
}
