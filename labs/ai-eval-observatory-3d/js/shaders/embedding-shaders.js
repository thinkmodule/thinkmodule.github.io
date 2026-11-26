export const EMBEDDING_VERTEX_SHADER = /* glsl */ `
  attribute float aSize;
  attribute float aImportance;
  attribute float aHighlight;
  attribute float aNodeId;
  attribute vec3 aColor;

  uniform float uTime;
  uniform float uPixelRatio;
  uniform float uHoveredId;

  varying vec3 vColor;
  varying float vHighlight;
  varying float vHovered;
  varying float vDepthFade;

  void main() {
    vColor = aColor;
    vHighlight = aHighlight;
    vHovered = abs(aNodeId - uHoveredId) < 0.5 ? 1.0 : 0.0;

    // Subtle organic breathing motion in latent space
    vec3 pos = position;
    float phase = aNodeId * 0.37 + uTime * 0.65;
    pos.x += sin(phase) * 0.08 * (1.0 - aHighlight * 0.6);
    pos.y += cos(phase * 0.85) * 0.09 * (1.0 - aHighlight * 0.6);
    pos.z += sin(phase * 0.55) * 0.08 * (1.0 - aHighlight * 0.6);

    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mvPosition;

    float dist = max(1.0, -mvPosition.z);
    vDepthFade = clamp(1.25 - (dist / 135.0), 0.25, 1.0);

    float pulse = 1.0 + 0.14 * sin(uTime * 3.2 + aNodeId) * aHighlight;
    float baseSize = aSize * (0.85 + 0.45 * aImportance);
    float boost = 1.0 + aHighlight * 1.45 + vHovered * 1.85;

    gl_PointSize = clamp((baseSize * boost * pulse * uPixelRatio * 240.0) / dist, 2.0, 58.0);
  }
`;

export const EMBEDDING_FRAGMENT_SHADER = /* glsl */ `
  varying vec3 vColor;
  varying float vHighlight;
  varying float vHovered;
  varying float vDepthFade;

  void main() {
    vec2 uv = gl_PointCoord - vec2(0.5);
    float d = length(uv);
    if (d > 0.5) discard;

    // Sharp glowing core + smooth radial falloff
    float core = smoothstep(0.24, 0.0, d);
    float halo = exp(-d * 5.6);

    vec3 highlightTint = vec3(0.92, 0.98, 1.0);
    vec3 finalColor = mix(vColor, highlightTint, clamp(core * 0.55 + vHighlight * 0.45 + vHovered * 0.65, 0.0, 1.0));

    // Subtle crisp ring on highlighted Top-K or hovered nodes
    float ring = smoothstep(0.46, 0.41, d) * smoothstep(0.32, 0.37, d) * max(vHighlight, vHovered);
    finalColor += vec3(0.4, 0.85, 1.0) * ring * 1.25;

    float alpha = (halo * 0.85 + core * 0.55 + ring * 0.7) * vDepthFade;
    alpha = clamp(alpha * (0.72 + vHighlight * 0.45 + vHovered * 0.5), 0.0, 1.0);

    gl_FragColor = vec4(finalColor * (1.0 + vHighlight * 0.55 + vHovered * 0.75), alpha);
  }
`;
