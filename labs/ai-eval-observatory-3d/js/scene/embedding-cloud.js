import * as THREE from 'three';
import {
  EMBEDDING_VERTEX_SHADER,
  EMBEDDING_FRAGMENT_SHADER
} from '../shaders/embedding-shaders.js';
import {
  HALO_VERTEX_SHADER,
  HALO_FRAGMENT_SHADER
} from '../shaders/field-shaders.js';

export class EmbeddingCloud {
  constructor(scene, space) {
    this.scene = scene;
    this.nodes = space.nodes;
    this.clusterCentroids = space.clusterCentroids;

    this.group = new THREE.Group();
    this.haloGroup = new THREE.Group();
    this.scene.add(this.group);
    this.scene.add(this.haloGroup);

    this._buildPointCloud();
    this._buildDomainHalos();

    // Reusable vector for fast raycast distance checks
    this._tmpVec = new THREE.Vector3();
  }

  _buildPointCloud() {
    const count = this.nodes.length;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const sizes = new Float32Array(count);
    const importance = new Float32Array(count);
    const highlight = new Float32Array(count);
    const nodeIds = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const n = this.nodes[i];
      positions[i * 3] = n.pos[0];
      positions[i * 3 + 1] = n.pos[1];
      positions[i * 3 + 2] = n.pos[2];

      colors[i * 3] = n.color[0];
      colors[i * 3 + 1] = n.color[1];
      colors[i * 3 + 2] = n.color[2];

      sizes[i] = n.layer === 2 ? 1.55 : n.layer === 1 ? 1.05 : 0.72;
      importance[i] = n.importance;
      highlight[i] = 0.0;
      nodeIds[i] = n.id;
    }

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.geometry.setAttribute('aColor', new THREE.BufferAttribute(colors, 3));
    this.geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
    this.geometry.setAttribute('aImportance', new THREE.BufferAttribute(importance, 1));
    this.highlightAttr = new THREE.BufferAttribute(highlight, 1);
    this.geometry.setAttribute('aHighlight', this.highlightAttr);
    this.geometry.setAttribute('aNodeId', new THREE.BufferAttribute(nodeIds, 1));

    this.material = new THREE.ShaderMaterial({
      vertexShader: EMBEDDING_VERTEX_SHADER,
      fragmentShader: EMBEDDING_FRAGMENT_SHADER,
      uniforms: {
        uTime: { value: 0 },
        uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
        uHoveredId: { value: -1 }
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    this.points = new THREE.Points(this.geometry, this.material);
    this.group.add(this.points);
  }

  _buildDomainHalos() {
    const sphereGeo = new THREE.IcosahedronGeometry(1, 3);
    for (const c of this.clusterCentroids) {
      const mat = new THREE.ShaderMaterial({
        vertexShader: HALO_VERTEX_SHADER,
        fragmentShader: HALO_FRAGMENT_SHADER,
        uniforms: {
          uColor: { value: new THREE.Color(...c.color) }
        },
        transparent: true,
        depthWrite: false,
        side: THREE.FrontSide,
        blending: THREE.AdditiveBlending
      });
      const mesh = new THREE.Mesh(sphereGeo, mat);
      mesh.position.set(...c.center);
      mesh.scale.setScalar(c.radius);
      this.haloGroup.add(mesh);
    }
  }

  setHalosVisible(visible) {
    this.haloGroup.visible = visible;
  }

  setHighlightedNodes(topNodes) {
    const arr = this.highlightAttr.array;
    arr.fill(0.0);
    for (const item of topNodes) {
      if (item.node.id < arr.length) {
        arr[item.node.id] = 1.0;
      }
    }
    this.highlightAttr.needsUpdate = true;
  }

  setHoveredNode(nodeId) {
    this.material.uniforms.uHoveredId.value = nodeId !== null ? nodeId : -1;
  }

  /**
   * Fast ray-to-point spatial picking for 3,200 vectors without Points raycast stutter.
   */
  pickNode(raycaster, maxRayDist = 0.85) {
    let bestNode = null;
    let bestScore = Infinity;
    const ray = raycaster.ray;
    const origin = ray.origin;

    for (let i = 0; i < this.nodes.length; i++) {
      const n = this.nodes[i];
      this._tmpVec.set(n.pos[0], n.pos[1], n.pos[2]);
      const distToRay = ray.distanceToPoint(this._tmpVec);
      if (distToRay < maxRayDist) {
        const camDist = origin.distanceTo(this._tmpVec);
        // Favor closer nodes to both ray center and camera
        const score = distToRay * 8.0 + camDist * 0.04;
        if (score < bestScore) {
          bestScore = score;
          bestNode = n;
        }
      }
    }
    return bestNode;
  }

  update(time) {
    this.material.uniforms.uTime.value = time;
  }
}
