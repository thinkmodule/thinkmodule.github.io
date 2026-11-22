/**
 * ThinkModule Lab — LiDAR Time-of-Flight (ToF) & Optical Link Budget Physics Engine
 * Computes exact nanosecond round-trip time Δt = 2d/c, LiDAR range equation received power
 * (inverse-square 1/d² law, target Lambertian reflectivity ρ, 1550nm vs. 905nm Class 1 eye-safe power),
 * Signal-to-Noise Ratio (dB), and component engineering metadata.
 */

export const C_M_PER_NS = 0.299792458; // Speed of light in air/vacuum (m / ns)

export const WAVELENGTHS = {
  "1550": {
    id: "1550",
    label: "1550 nm",
    detector: "InGaAs APD",
    txPowerRel: 1.0, // Cornea absorbs 1550nm before retina -> allows ~40x higher eye-safe Class 1 peak power
    beamHex: 0x38bdf8,
    beamRgb: [0.28, 1.55, 2.3],
    echoRgb: [0.22, 1.95, 1.35],
    cssColor: "#38bdf8",
    note: "1550 nm is absorbed by the eye's cornea before reaching the retina, allowing much higher Class 1 eye-safe laser power to detect dark 8% targets beyond 100 m."
  },
  "905": {
    id: "905",
    label: "905 nm",
    detector: "Silicon APD",
    txPowerRel: 0.22, // Tighter IEC 60825-1 Class 1 retinal power limit
    beamHex: 0xf43f5e,
    beamRgb: [2.3, 0.28, 0.45],
    echoRgb: [2.2, 1.45, 0.28],
    cssColor: "#f43f5e",
    note: "905 nm uses lower-cost silicon detectors, but passes through the cornea to the retina — restricting eye-safe peak power and limiting range on dark low-reflectivity surfaces."
  }
};

export const TARGETS = {
  sign: {
    id: "sign",
    label: "Road Sign (85%)",
    reflectivity: 0.85,
    colorHex: 0xe6eef8,
    roughness: 0.18,
    metalness: 0.35
  },
  car: {
    id: "car",
    label: "Clearcoat Car (42%)",
    reflectivity: 0.42,
    colorHex: 0x3a6496,
    roughness: 0.22,
    metalness: 0.68
  },
  tire: {
    id: "tire",
    label: "Matte Tire (8%)",
    reflectivity: 0.08,
    colorHex: 0x161a21,
    roughness: 0.92,
    metalness: 0.04
  }
};

export const PART_INFO = {
  all: {
    tag: "FULL OPTICAL ASSEMBLY",
    title: "Bistatic Time-of-Flight LiDAR",
    desc: "A laser diode fires billionth-of-a-second infrared pulses off a 45° mirror onto an object. By timing the returning light echo in nanoseconds, the FPGA computes exact 3D coordinates.",
    specs: ["64 Channels", "10–20 Hz Spin", "±2 cm Precision", "IEC Class 1"]
  },
  window: {
    tag: "LAYER 05 • OPTICAL ENCLOSURE",
    title: "IR-Transparent Optical Dome & Cap",
    desc: "Specialized smoked polycarbonate blocks blinding visible sunlight while passing >94% of infrared laser light cleanly. Heated anti-fog coatings prevent snow and ice attenuation.",
    specs: ["IR Bandpass >94%", "IP69K Sealed", "CNC Titanium Cap"]
  },
  emitter: {
    tag: "LAYER 04A • TRANSMITTER (TX)",
    title: "Pulsed Laser Diode & Collimator",
    desc: "Emits ultra-sharp 3–5 nanosecond infrared light pulses (up to 1.3 million pulses/sec). Precision collimating glass keeps the laser beam narrow over 100+ meters.",
    specs: ["3 ns Pulse Width", "1.3M pts/sec", "Collimated Optics"]
  },
  mirror: {
    tag: "LAYER 04B • BEAM STEERING",
    title: "45° High-Reflectivity Deflection Mirror",
    desc: "Mounted directly on the motor shaft at a precision 45° angle, this dielectric mirror redirects vertical laser pulses 90° horizontally to sweep a continuous 360° horizon.",
    specs: ["45.00° Pitch", "99.8% Dielectric", "360° Horizontal FOV"]
  },
  receiver: {
    tag: "LAYER 04C • RECEIVER (RX)",
    title: "Aspheric Lens & Avalanche Photodiode",
    desc: "The wide glass lens collects faint reflected photons scattered back from the target and focuses them onto an APD chip, which multiplies single photons into a measurable electrical spike.",
    specs: ["Aspheric Glass", "100× APD Gain", "Optical Bandpass"]
  },
  motor: {
    tag: "LAYER 03 • ROTOR & ENCODER",
    title: "Brushless Motor & Optical Encoder",
    desc: "12 copper stator windings spin the optical head smoothly at 600–1200 RPM. A high-resolution slotted encoder disc measures the exact azimuth angle θ to 0.01° for every fired pulse.",
    specs: ["12-Pole BLDC", "0.01° Azimuth", "Inductive Slip-Ring"]
  },
  fpga: {
    tag: "LAYER 02 • TIMING & DSP",
    title: "FPGA & Picosecond TDC Circuitry",
    desc: "A Time-to-Digital Converter (TDC) clocks Δt with sub-50 picosecond resolution (±7 mm precision). The FPGA turns (Δt, θ, φ) polar readings into real-time (X, Y, Z) 3D point clouds.",
    specs: ["<50 ps TDC", "Polar → XYZ", "1 Gbps Automotive Eth"]
  }
};

export function evaluateLidarPhysics({ distanceM, targetId, wavelengthId }) {
  const wl = WAVELENGTHS[wavelengthId] || WAVELENGTHS["1550"];
  const tgt = TARGETS[targetId] || TARGETS.car;
  const d = Math.max(5, Math.min(120, distanceM));

  // Exact round-trip time of flight in nanoseconds: Δt = 2d / c
  const tofNs = (2 * d) / C_M_PER_NS;

  // LiDAR Range Equation: Pr ∝ (Pt * ρ) / d²
  const K = 22000;
  const noiseFloor = 0.05;
  const receivedPower = (K * wl.txPowerRel * tgt.reflectivity) / (d * d);
  const snrLinear = receivedPower / noiseFloor;
  const snrDb = 10 * Math.log10(Math.max(0.05, snrLinear));

  const thresholdDb = 6.0;
  const detected = snrDb >= thresholdDb;
  const weak = detected && snrDb < 10.5;

  // Visual return amplitude normalized [0.05..1.0] for 3D echo brightness & oscilloscope peak
  const echoNorm = Math.max(0.06, Math.min(1.0, (snrDb + 4) / 26));

  let statusText = `ECHO LOCKED • SNR ${snrDb.toFixed(1)} dB`;
  let statusSeverity = "optimal";
  let insight = wl.note;

  if (!detected) {
    statusText = `SIGNAL LOST • BELOW NOISE (${snrDb.toFixed(1)} dB)`;
    statusSeverity = "danger";
    insight = `At ${d} m on a ${(tgt.reflectivity * 100).toFixed(0)}% surface, ${wl.label} return power drops below the 6 dB APD threshold (1/d² law). Switch to 1550 nm or reduce target distance to regain lock.`;
  } else if (weak) {
    statusText = `MARGINAL ECHO • SNR ${snrDb.toFixed(1)} dB`;
    statusSeverity = "warn";
  }

  return {
    distanceM: d,
    tofNs,
    receivedPower,
    snrDb,
    echoNorm,
    detected,
    weak,
    wavelength: wl,
    target: tgt,
    statusText,
    statusSeverity,
    insight
  };
}