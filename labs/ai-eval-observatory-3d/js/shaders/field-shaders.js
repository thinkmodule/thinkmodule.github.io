export const SAFETY_PLANE_VERTEX_SHADER = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vWorldPos;

  void main() {
    vUv = uv;
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorldPos = wp.xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

export const SAFETY_PLANE_FRAGMENT_SHADER = /* glsl */ `
  uniform float uTime;
  uniform vec3 uImpactPos;
  uniform float uImpactStrength;
  uniform float uBreach;

  varying vec2 vUv;
  varying vec3 vWorldPos;

  void main() {
    // Crisp holographic architectural grid
    vec2 gridUv = vUv * vec2(26.0, 16.0);
    vec2 cell = abs(fract(gridUv - 0.5) - 0.5) / fwidth(gridUv);
    float line = 1.0 - min(min(cell.x, cell.y), 1.0);

    // Sweeping horizontal scanner beam
    float scan = exp(-pow((vUv.y - fract(uTime * 0.14)) * 12.0, 2.0)) * 0.45;

    // Soft vignette fade toward outer edges
    vec2 centered = (vUv - 0.5) * 2.0;
    float edgeFade = smoothstep(1.05, 0.25, length(centered * vec2(0.95, 1.05)));

    // Dynamic impact ripple when adversarial trajectory hits or breaches the wall
    float distToImpact = length(vWorldPos.xy - uImpactPos.xy);
    float rippleWave = sin(distToImpact * 1.8 - uTime * 6.5) * 0.5 + 0.5;
    float impactFalloff = exp(-distToImpact * 0.22) * uImpactStrength;
    float impactGlow = impactFalloff * (0.55 + 0.75 * rippleWave);

    vec3 baseColor = mix(vec3(0.96, 0.32, 0.42), vec3(1.0, 0.12, 0.26), uBreach);
    vec3 safeEdgeColor = vec3(0.22, 0.74, 0.97);
    vec3 col = mix(baseColor, safeEdgeColor, (1.0 - uBreach) * 0.22);

    float intensity = (0.09 + line * 0.34 + scan) * edgeFade + impactGlow * 1.25;
    col += vec3(1.0, 0.85, 0.75) * impactGlow * 0.75;

    float alpha = clamp(intensity * (0.48 + uImpactStrength * 0.35), 0.0, 0.88);
    gl_FragColor = vec4(col, alpha);
  }
`;

export const ENVELOPE_VERTEX_SHADER = /* glsl */ `
  varying vec3 vNormal;
  varying vec3 vViewDir;
  varying vec3 vLocalPos;

  void main() {
    vLocalPos = position;
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vViewDir = normalize(-mvPos.xyz);
    gl_Position = projectionMatrix * mvPos;
  }
`;

export const ENVELOPE_FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uColor;
  uniform float uTime;
  uniform float uOpacity;

  varying vec3 vNormal;
  varying vec3 vViewDir;
  varying vec3 vLocalPos;

  void main() {
    float ndotv = abs(dot(normalize(vNormal), normalize(vViewDir)));
    float fresnel = pow(1.0 - ndotv, 2.35);

    // Subtle animated latitude scan rings
    float bands = 0.5 + 0.5 * sin(vLocalPos.y * 8.0 - uTime * 2.4);
    float shimmer = smoothstep(0.72, 1.0, bands) * 0.22;

    vec3 col = uColor * (1.15 + fresnel * 0.85);
    float alpha = clamp((0.06 + fresnel * 0.58 + shimmer * fresnel) * uOpacity, 0.0, 0.78);
    gl_FragColor = vec4(col, alpha);
  }
`;

export const HALO_VERTEX_SHADER = /* glsl */ `
  varying vec3 vNormal;
  varying vec3 vViewDir;

  void main() {
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vViewDir = normalize(-mvPos.xyz);
    gl_Position = projectionMatrix * mvPos;
  }
`;

export const HALO_FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uColor;

  varying vec3 vNormal;
  varying vec3 vViewDir;

  void main() {
    float ndotv = abs(dot(normalize(vNormal), normalize(vViewDir)));
    float rim = pow(1.0 - ndotv, 3.0);
    float alpha = clamp(rim * 0.16 + 0.015, 0.0, 0.22);
    gl_FragColor = vec4(uColor, alpha);
  }
`;
