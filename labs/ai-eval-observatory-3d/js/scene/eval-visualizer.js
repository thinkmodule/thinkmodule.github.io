import * as THREE from 'three';
import { SAFETY_BOUNDARY_Z } from '../core/config.js';
import {
  SAFETY_PLANE_VERTEX_SHADER,
  SAFETY_PLANE_FRAGMENT_SHADER,
  ENVELOPE_VERTEX_SHADER,
  ENVELOPE_FRAGMENT_SHADER
} from '../shaders/field-shaders.js';

export class EvalVisualizer {
  constructor(scene) {
    this.scene = scene;

    this.dynamicGroup = new THREE.Group();
    this.scene.add(this.dynamicGroup);

    this.pipelineProgress = 0;
    this._tokenPulses = [];
    this._winnerCurve = null;
    this._winnerOrb = null;
    this._queryBeacon = null;
    this._querySonar = null;

    this._buildSafetyHyperplane();
    this._buildGroundingEnvelope();
    this._buildRewardPotentialField();
  }

  _buildSafetyHyperplane() {
    const planeGeo = new THREE.PlaneGeometry(54, 34, 1, 1);
    this.safetyMaterial = new THREE.ShaderMaterial({
      vertexShader: SAFETY_PLANE_VERTEX_SHADER,
      fragmentShader: SAFETY_PLANE_FRAGMENT_SHADER,
      uniforms: {
        uTime: { value: 0 },
        uImpactPos: { value: new THREE.Vector3(10, -2, SAFETY_BOUNDARY_Z) },
        uImpactStrength: { value: 0.15 },
        uBreach: { value: 0.0 }
      },
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending
    });

    this.safetyGroup = new THREE.Group();
    this.safetyMesh = new THREE.Mesh(planeGeo, this.safetyMaterial);
    this.safetyMesh.position.set(0, 1.5, SAFETY_BOUNDARY_Z);
    this.safetyGroup.add(this.safetyMesh);

    // Crisp glowing border frame around the Safety Hyperplane
    const edges = new THREE.EdgesGeometry(planeGeo);
    const frameMat = new THREE.LineBasicMaterial({
      color: 0xf43f5e,
      transparent: true,
      opacity: 0.55
    });
    const frame = new THREE.LineSegments(edges, frameMat);
    frame.position.copy(this.safetyMesh.position);
    this.safetyGroup.add(frame);

    this.scene.add(this.safetyGroup);
  }

  _buildGroundingEnvelope() {
    this.envelopeGroup = new THREE.Group();

    const sphereGeo = new THREE.IcosahedronGeometry(1, 4);
    this.envelopeMaterial = new THREE.ShaderMaterial({
      vertexShader: ENVELOPE_VERTEX_SHADER,
      fragmentShader: ENVELOPE_FRAGMENT_SHADER,
      uniforms: {
        uColor: { value: new THREE.Color(0x10b981) },
        uTime: { value: 0 },
        uOpacity: { value: 0.92 }
      },
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending
    });
    this.envelopeMesh = new THREE.Mesh(sphereGeo, this.envelopeMaterial);
    this.envelopeGroup.add(this.envelopeMesh);

    // Architectural geodesic wireframe around the evidence hull
    const wireGeo = new THREE.IcosahedronGeometry(1.015, 2);
    this.envelopeWireMat = new THREE.MeshBasicMaterial({
      color: 0x34d399,
      wireframe: true,
      transparent: true,
      opacity: 0.16,
      depthWrite: false
    });
    this.envelopeWire = new THREE.Mesh(wireGeo, this.envelopeWireMat);
    this.envelopeGroup.add(this.envelopeWire);

    this.scene.add(this.envelopeGroup);
  }

  _buildRewardPotentialField() {
    // 3D deforming RLHF Reward landscape mesh beneath the latent cloud
    const segs = 44;
    this.fieldGeo = new THREE.PlaneGeometry(66, 56, segs, segs);
    this.fieldGeo.rotateX(-Math.PI / 2);

    const vertCount = this.fieldGeo.attributes.position.count;
    const colors = new Float32Array(vertCount * 3);
    this.fieldGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const fieldMat = new THREE.MeshBasicMaterial({
      vertexColors: true,
      wireframe: true,
      transparent: true,
      opacity: 0.24,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    this.rewardFieldMesh = new THREE.Mesh(this.fieldGeo, fieldMat);
    this.rewardFieldMesh.position.set(0, -16.2, -2);
    this.scene.add(this.rewardFieldMesh);
  }

  _updateRewardFieldTopology(groundingCentroid, alignment) {
    const pos = this.fieldGeo.attributes.position;
    const col = this.fieldGeo.attributes.color;
    const gx = groundingCentroid[0];
    const gz = groundingCentroid[2];

    const cLow = new THREE.Color(0xf43f5e);
    const cMid = new THREE.Color(0x1e3a8a);
    const cHigh = new THREE.Color(0x10b981);
    const tmp = new THREE.Color();

    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);
      const worldZ = z - 2.0;

      // Positive Gaussian reward peak around retrieved grounding manifold
      const d2 = (x - gx) * (x - gx) + (worldZ - gz) * (worldZ - gz);
      const rewardPeak = Math.exp(-d2 / 145.0) * (2.2 + alignment * 4.8);

      // Sharp penalty drop-off near and beyond Safety Hyperplane
      const safetyPenalty = Math.max(0, (worldZ - (SAFETY_BOUNDARY_Z - 7.0)) / 9.5);
      const penaltyDrop = -Math.pow(safetyPenalty, 1.8) * (1.5 + alignment * 3.2);

      const y = rewardPeak + penaltyDrop;
      pos.setY(i, y);

      if (y >= 0) {
        const t = Math.min(1, y / 5.5);
        tmp.copy(cMid).lerp(cHigh, t);
      } else {
        const t = Math.min(1, -y / 4.2);
        tmp.copy(cMid).lerp(cLow, t);
      }
      col.setXYZ(i, tmp.r, tmp.g, tmp.b);
    }

    pos.needsUpdate = true;
    col.needsUpdate = true;
  }

  _clearDynamic() {
    while (this.dynamicGroup.children.length > 0) {
      const obj = this.dynamicGroup.children[0];
      this.dynamicGroup.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
        else obj.material.dispose();
      }
    }
    this._tokenPulses = [];
    this._winnerCurve = null;
    this._winnerOrb = null;
    this._queryBeacon = null;
    this._querySonar = null;
  }

  updateEvaluation(evalResult, state) {
    this._clearDynamic();
    this.pipelineProgress = 0;

    const { queryPos, groundingCentroid, groundingRadius, candidates, winner } = evalResult;

    // 1. Update Grounding Envelope position, radius & status color
    this.envelopeGroup.position.set(...groundingCentroid);
    this.envelopeGroup.scale.setScalar(groundingRadius);

    let envHex = 0x10b981;
    if (winner.safetyBreach || winner.faithfulness < 0.55) envHex = 0xf43f5e;
    else if (winner.faithfulness < 0.76) envHex = 0xf59e0b;

    this.envelopeMaterial.uniforms.uColor.value.setHex(envHex);
    this.envelopeWireMat.color.setHex(envHex);

    // 2. Deform RLHF Reward Potential Field
    this._updateRewardFieldTopology(groundingCentroid, state.alignment);

    // 3. Configure Safety Hyperplane impact shockwave
    const hasBoundaryEvent = winner.safetyBreach || winner.guardrailTriggered || evalResult.scenario.isAdversarial;
    this.safetyMaterial.uniforms.uImpactPos.value.set(
      winner.finalTarget[0],
      winner.finalTarget[1],
      SAFETY_BOUNDARY_Z
    );
    this.safetyMaterial.uniforms.uImpactStrength.value = winner.safetyBreach ? 1.15 : hasBoundaryEvent ? 0.85 : 0.14;
    this.safetyMaterial.uniforms.uBreach.value = winner.safetyBreach ? 1.0 : 0.0;

    // 4. Query Vector Beacon (Cyan rotating crystal + sonar ring)
    const qGeo = new THREE.OctahedronGeometry(0.75, 0);
    const qMat = new THREE.MeshBasicMaterial({ color: 0xe0f2fe });
    this._queryBeacon = new THREE.Mesh(qGeo, qMat);
    this._queryBeacon.position.set(...queryPos);
    this.dynamicGroup.add(this._queryBeacon);

    const qWire = new THREE.Mesh(
      new THREE.OctahedronGeometry(1.12, 0),
      new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true, transparent: true, opacity: 0.85 })
    );
    this._queryBeacon.add(qWire);

    const sonarGeo = new THREE.RingGeometry(0.8, 0.96, 36);
    const sonarMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.7,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this._querySonar = new THREE.Mesh(sonarGeo, sonarMat);
    this._querySonar.position.set(...queryPos);
    this.dynamicGroup.add(this._querySonar);

    // 5. Render all 6 RLHF Candidate Trajectories (Winner = bold energy tube, others = ghost beams)
    for (const cand of candidates) {
      const pts = cand.controlPoints.map(p => new THREE.Vector3(...p));
      const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal', 0.45);
      const isWinner = cand === winner;

      let beamColor = 0x64748b;
      if (isWinner) {
        if (cand.safetyBreach || cand.faithfulness < 0.55) beamColor = 0xf43f5e;
        else if (cand.guardrailTriggered) beamColor = 0x38bdf8;
        else if (cand.faithfulness < 0.76) beamColor = 0xf59e0b;
        else beamColor = 0x10b981;
      } else if (cand.safetyBreach) {
        beamColor = 0xbe123c;
      } else {
        beamColor = 0x6366f1;
      }

      if (isWinner) {
        this._winnerCurve = curve;
        const tubeGeo = new THREE.TubeGeometry(curve, 64, 0.16, 12, false);
        const tubeMat = new THREE.MeshBasicMaterial({
          color: beamColor,
          transparent: true,
          opacity: 0.92,
          blending: THREE.AdditiveBlending
        });
        this.dynamicGroup.add(new THREE.Mesh(tubeGeo, tubeMat));

        // Subtle outer glow sheath around winner tube
        const haloTubeGeo = new THREE.TubeGeometry(curve, 64, 0.34, 12, false);
        const haloTubeMat = new THREE.MeshBasicMaterial({
          color: beamColor,
          transparent: true,
          opacity: 0.24,
          depthWrite: false,
          blending: THREE.AdditiveBlending
        });
        this.dynamicGroup.add(new THREE.Mesh(haloTubeGeo, haloTubeMat));

        // 4 Animated data-token pulses streaming along the winning trajectory
        const pulseGeo = new THREE.SphereGeometry(0.28, 12, 12);
        for (let k = 0; k < 4; k++) {
          const pMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
          const pulse = new THREE.Mesh(pulseGeo.clone(), pMat);
          pulse.userData.offset = k * 0.25;
          this.dynamicGroup.add(pulse);
          this._tokenPulses.push(pulse);
        }
        pulseGeo.dispose();

        // Winning Final Response Orb
        const orbGeo = new THREE.SphereGeometry(0.82, 24, 24);
        const orbMat = new THREE.MeshBasicMaterial({ color: beamColor });
        this._winnerOrb = new THREE.Mesh(orbGeo, orbMat);
        this._winnerOrb.position.set(...cand.finalTarget);
        this.dynamicGroup.add(this._winnerOrb);

        const outerOrb = new THREE.Mesh(
          new THREE.IcosahedronGeometry(1.28, 1),
          new THREE.MeshBasicMaterial({
            color: beamColor,
            wireframe: true,
            transparent: true,
            opacity: 0.65
          })
        );
        this._winnerOrb.add(outerOrb);
      } else {
        // Pruned candidate trajectory arc
        const linePts = curve.getPoints(42);
        const lineGeo = new THREE.BufferGeometry().setFromPoints(linePts);
        const lineMat = new THREE.LineBasicMaterial({
          color: beamColor,
          transparent: true,
          opacity: 0.32,
          blending: THREE.AdditiveBlending
        });
        this.dynamicGroup.add(new THREE.Line(lineGeo, lineMat));

        // Small endpoint node for pruned beam
        const endDot = new THREE.Mesh(
          new THREE.SphereGeometry(0.26, 10, 10),
          new THREE.MeshBasicMaterial({
            color: beamColor,
            transparent: true,
            opacity: 0.55
          })
        );
        endDot.position.set(...cand.finalTarget);
        this.dynamicGroup.add(endDot);
      }
    }

    // 6. Semantic Drift Diagnostic Vector (Ground Centroid -> Winning Answer)
    if (winner.driftDist > 1.8) {
      const driftPts = [
        new THREE.Vector3(...groundingCentroid),
        new THREE.Vector3(...winner.finalTarget)
      ];
      const driftGeo = new THREE.BufferGeometry().setFromPoints(driftPts);
      const driftColor = winner.faithfulness < 0.65 ? 0xf43f5e : 0xf59e0b;
      const driftMat = new THREE.LineDashedMaterial({
        color: driftColor,
        dashSize: 0.75,
        gapSize: 0.45,
        transparent: true,
        opacity: 0.9
      });
      const driftLine = new THREE.Line(driftGeo, driftMat);
      driftLine.computeLineDistances();
      this.dynamicGroup.add(driftLine);
    }
  }

  setLayerVisibility({ hull, rewardField, safetyPlane }) {
    if (hull !== undefined) this.envelopeGroup.visible = hull;
    if (rewardField !== undefined) this.rewardFieldMesh.visible = rewardField;
    if (safetyPlane !== undefined) this.safetyGroup.visible = safetyPlane;
  }

  replay() {
    this.pipelineProgress = 0;
  }

  update(time, dt, camera, isPlaying) {
    if (isPlaying) {
      this.pipelineProgress = Math.min(1, this.pipelineProgress + dt * 0.65);
    }

    // Shader clocks
    this.safetyMaterial.uniforms.uTime.value = time;
    this.envelopeMaterial.uniforms.uTime.value = time;
    this.envelopeWire.rotation.y = time * 0.14;
    this.envelopeWire.rotation.x = Math.sin(time * 0.1) * 0.12;

    // Query beacon rotation & expanding sonar wave
    if (this._queryBeacon) {
      this._queryBeacon.rotation.y = time * 1.4;
      this._queryBeacon.rotation.x = time * 0.7;
    }
    if (this._querySonar) {
      this._querySonar.quaternion.copy(camera.quaternion);
      const wave = (time * 0.75) % 1.0;
      this._querySonar.scale.setScalar(0.8 + wave * 2.8);
      this._querySonar.material.opacity = (1.0 - wave) * 0.65;
    }

    // Stream token pulses along the winning beam curve
    if (this._winnerCurve && this._tokenPulses.length) {
      for (const pulse of this._tokenPulses) {
        const t = ((time * 0.45) + pulse.userData.offset) % 1.0;
        const clampedT = Math.min(t, this.pipelineProgress);
        const pt = this._winnerCurve.getPointAt(clampedT);
        pulse.position.copy(pt);
        pulse.visible = this.pipelineProgress > 0.05;
      }
    }

    // Winner orb subtle heartbeat
    if (this._winnerOrb) {
      const s = (0.94 + 0.1 * Math.sin(time * 4.2)) * Math.min(1, this.pipelineProgress * 1.35);
      this._winnerOrb.scale.setScalar(Math.max(0.01, s));
      this._winnerOrb.rotation.y = -time * 0.9;
    }
  }
}
