import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { CAMERA_PRESETS } from '../core/config.js';

export class SceneManager {
  constructor(container) {
    this.container = container;
    this.width = container.clientWidth || window.innerWidth;
    this.height = container.clientHeight || window.innerHeight;

    // Scene & atmospheric depth fog
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x050811);
    this.scene.fog = new THREE.FogExp2(0x050811, 0.0082);

    // Camera
    this.camera = new THREE.PerspectiveCamera(46, this.width / this.height, 0.2, 300);
    const startPreset = CAMERA_PRESETS.overview;
    this.camera.position.set(...startPreset.pos);

    // WebGL2 Renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance',
      alpha: false
    });
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.12;
    this.container.appendChild(this.renderer.domElement);

    // OrbitControls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(...startPreset.target);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.055;
    this.controls.rotateSpeed = 0.72;
    this.controls.zoomSpeed = 0.85;
    this.controls.minDistance = 12;
    this.controls.maxDistance = 115;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.42;

    // Cancel camera tween if user manually interacts
    this._cameraTween = null;
    this.controls.addEventListener('start', () => {
      this._cameraTween = null;
    });

    // Post-processing pipeline (UnrealBloomPass)
    this.composer = new EffectComposer(this.renderer);
    this.renderPass = new RenderPass(this.scene, this.camera);
    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(this.width, this.height),
      0.42, // strength
      0.55, // radius
      0.32  // threshold
    );
    this.outputPass = new OutputPass();
    this.composer.addPass(this.renderPass);
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(this.outputPass);

    // Subtle studio lighting & spatial reference environment
    this._setupLighting();
    this._setupEnvironment();

    // FPS tracker
    this._fpsFrames = 0;
    this._fpsLast = performance.now();
    this.fps = 60;

    this._onResize = this._handleResize.bind(this);
    window.addEventListener('resize', this._onResize);
  }

  _setupLighting() {
    const ambient = new THREE.AmbientLight(0x94a3b8, 0.75);
    this.scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0x38bdf8, 1.1);
    keyLight.position.set(28, 42, 30);
    this.scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0xa78bfa, 0.65);
    rimLight.position.set(-34, -18, -28);
    this.scene.add(rimLight);
  }

  _setupEnvironment() {
    // Subtle dark architectural floor reference ring & axes
    const floorGroup = new THREE.Group();
    floorGroup.position.y = -17.5;

    const ringRadii = [12, 24, 36];
    const ringMat = new THREE.LineBasicMaterial({
      color: 0x1e293b,
      transparent: true,
      opacity: 0.45
    });

    for (const r of ringRadii) {
      const pts = [];
      const segs = 96;
      for (let i = 0; i <= segs; i++) {
        const a = (i / segs) * Math.PI * 2;
        pts.push(new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r));
      }
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      floorGroup.add(new THREE.Line(geo, ringMat));
    }

    // Crosshairs on floor
    const crossPts = [
      new THREE.Vector3(-38, 0, 0), new THREE.Vector3(38, 0, 0),
      new THREE.Vector3(0, 0, -38), new THREE.Vector3(0, 0, 38)
    ];
    const crossGeo = new THREE.BufferGeometry().setFromPoints(crossPts);
    floorGroup.add(new THREE.LineSegments(crossGeo, ringMat));
    this.scene.add(floorGroup);

    // Subtle deep-space ambient dust for parallax depth
    const starCount = 420;
    const starPos = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i++) {
      const r = 60 + Math.random() * 70;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      starPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      starPos[i * 3 + 1] = r * Math.cos(phi);
      starPos[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
    }
    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    const starMat = new THREE.PointsMaterial({
      color: 0x475569,
      size: 0.45,
      transparent: true,
      opacity: 0.45,
      depthWrite: false
    });
    this.starField = new THREE.Points(starGeo, starMat);
    this.scene.add(this.starField);
  }

  setBloomStrength(value) {
    this.bloomPass.strength = value;
  }

  setAutoRotate(enabled) {
    this.controls.autoRotate = enabled;
  }

  flyToPreset(presetName, customTarget = null) {
    const preset = CAMERA_PRESETS[presetName] || CAMERA_PRESETS.overview;
    const endPos = new THREE.Vector3(...preset.pos);
    const endTarget = customTarget
      ? new THREE.Vector3(...customTarget)
      : new THREE.Vector3(...preset.target);

    this._cameraTween = {
      startPos: this.camera.position.clone(),
      endPos,
      startTarget: this.controls.target.clone(),
      endTarget,
      elapsed: 0,
      duration: 0.95
    };
  }

  _handleResize() {
    this.width = this.container.clientWidth || window.innerWidth;
    this.height = this.container.clientHeight || window.innerHeight;
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.width, this.height);
    this.composer.setSize(this.width, this.height);
  }

  render(dt) {
    // Smooth exponential ease-out camera transition
    if (this._cameraTween) {
      const tw = this._cameraTween;
      tw.elapsed += dt;
      const t = Math.min(1, tw.elapsed / tw.duration);
      const ease = 1 - Math.pow(1 - t, 3);
      this.camera.position.lerpVectors(tw.startPos, tw.endPos, ease);
      this.controls.target.lerpVectors(tw.startTarget, tw.endTarget, ease);
      if (t >= 1) this._cameraTween = null;
    }

    if (this.starField) {
      this.starField.rotation.y += dt * 0.008;
    }

    this.controls.update();
    this.composer.render();

    // Rolling FPS counter
    this._fpsFrames++;
    const now = performance.now();
    if (now - this._fpsLast >= 500) {
      this.fps = Math.round((this._fpsFrames * 1000) / (now - this._fpsLast));
      this._fpsFrames = 0;
      this._fpsLast = now;
    }
  }
}
