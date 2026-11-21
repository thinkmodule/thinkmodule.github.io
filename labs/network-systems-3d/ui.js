/**
 * ThinkModule Lab — 3D Network Systems, Routing & TCP Congestion Renderer (v1.0)
 * True 360-degree gimbal-lock-free 3x3 rotation matrix orbit, glowing 3D fiber backbone,
 * hop-by-hop live packet propagation with DNS/TCP/TLS/HTTP phases, real MAC/NAT/TTL inspector,
 * and live TCP CWND vs. BDP congestion control telemetry.
 */
(function () {
  "use strict";

  const Engine = window.NetEngine;
  if (!Engine) {
    console.error("NetEngine not loaded.");
    return;
  }

  const $ = id => document.getElementById(id);

  // ============================================================================
  // 3x3 ORTHONORMAL ROTATION MATRIX FOR TRUE UNLIMITED 360° 3D TUMBLE
  // ============================================================================

  function mat3Identity() {
    return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  }

  function mat3Mul(A, B) {
    const out = new Array(9);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        out[r * 3 + c] =
          A[r * 3 + 0] * B[0 * 3 + c] +
          A[r * 3 + 1] * B[1 * 3 + c] +
          A[r * 3 + 2] * B[2 * 3 + c];
      }
    }
    return out;
  }

  function mat3RotX(a) {
    const c = Math.cos(a), s = Math.sin(a);
    return [1, 0, 0, 0, c, -s, 0, s, c];
  }

  function mat3RotY(a) {
    const c = Math.cos(a), s = Math.sin(a);
    return [c, 0, s, 0, 1, 0, -s, 0, c];
  }

  function mat3Orthonormalize(m) {
    let x0 = m[0], x1 = m[1], x2 = m[2];
    let y0 = m[3], y1 = m[4], y2 = m[5];
    const xl = Math.hypot(x0, x1, x2) || 1;
    x0 /= xl; x1 /= xl; x2 /= xl;
    const dot = x0 * y0 + x1 * y1 + x2 * y2;
    y0 -= dot * x0; y1 -= dot * x1; y2 -= dot * x2;
    const yl = Math.hypot(y0, y1, y2) || 1;
    y0 /= yl; y1 /= yl; y2 /= yl;
    const z0 = x1 * y2 - x2 * y1;
    const z1 = x2 * y0 - x0 * y2;
    const z2 = x0 * y1 - x1 * y0;
    return [x0, x1, x2, y0, y1, y2, z0, z1, z2];
  }

  function defaultCamMatrix() {
    return mat3Orthonormalize(mat3Mul(mat3RotX(0.32), mat3RotY(-0.34)));
  }

  // ============================================================================
  // APPLICATION STATE
  // ============================================================================

  const LIFECYCLE_PHASES = [
    { id: "dns",  label: "01 • DNS LOOKUP", color: "#fbbf24", sub: "api.thinkmodule.io → 104.18.32.7", dir: 1 },
    { id: "syn",  label: "02 • TCP 3-WAY",  color: "#38bdf8", sub: "SYN → SYN-ACK → ACK",             dir: 1 },
    { id: "tls",  label: "03 • TLS 1.3",    color: "#a78bfa", sub: "X25519 KeyShare + AES-256-GCM",   dir: 1 },
    { id: "enc",  label: "04 • ENCAP L7→L2",color: "#f472b6", sub: "Wrap TCP/IP/ETH (1,500 B MTU)",   dir: 1 },
    { id: "data", label: "05 • LIVE STREAM",color: "#34d399", sub: "Encrypted HTTP/2 + TCP ACKs",     dir: 1 }
  ];

  const state = {
    activePreset: "healthy_fiber",
    cfg: { ...Engine.PRESETS.healthy_fiber },
    edges: [],
    pathInfo: null,
    tcp: Engine.createTcpState(),
    lastSnap: null,

    playing: true,
    speedMult: 0.65,
    autoOrbit: true,
    animTime: 0,
    lastTs: 0,
    tcpAccum: 0,

    // Protocol & Hero Packet Tracer
    phaseIdx: 0,        // 0..4 (transitions cleanly into 4 = Live Stream)
    heroHopIdx: 0,      // Current active hop index along Dijkstra path
    heroProgress: 0.0,  // 0..1 along current hop
    heroDir: 1,         // 1 = Client->Origin, -1 = Origin->Client (ACK/Reply)
    wirePackets: [],    // Ambient background packets flying along active route
    dropFlashes: [],    // Visual red drop bursts on fiber links

    cam: {
      rot: defaultCamMatrix(),
      zoom: 1.0,
      dragging: false,
      lastX: 0,
      lastY: 0
    }
  };

  const dom = {
    presetGrid: $("presetGrid"),
    algoSelect: $("algoSelect"),
    bwSlider: $("bwSlider"),
    bwVal: $("bwVal"),
    lossSlider: $("lossSlider"),
    lossVal: $("lossVal"),
    bufferSlider: $("bufferSlider"),
    bufferVal: $("bufferVal"),
    cutPrimaryBtn: $("cutPrimaryBtn"),
    cutPrimaryBadge: $("cutPrimaryBadge"),
    cutBackupBtn: $("cutBackupBtn"),
    cutBackupBadge: $("cutBackupBadge"),
    injectDropBtn: $("injectDropBtn"),

    heroBadges: $("heroBadges"),
    playPauseBtn: $("playPauseBtn"),
    stepHopBtn: $("stepHopBtn"),
    resetSimBtn: $("resetSimBtn"),
    speedCalmBtn: $("speedCalmBtn"),
    speedNormalBtn: $("speedNormalBtn"),
    speedFastBtn: $("speedFastBtn"),
    autoOrbitBtn: $("autoOrbitBtn"),
    resetCamBtn: $("resetCamBtn"),

    canvas3dNet: $("canvas3dNet"),
    canvasLifecycle: $("canvasLifecycle"),
    canvasCwnd: $("canvasCwnd"),

    kpiRtt: $("kpiRtt"),
    kpiRttSub: $("kpiRttSub"),
    kpiGoodput: $("kpiGoodput"),
    kpiGoodputSub: $("kpiGoodputSub"),
    kpiBdp: $("kpiBdp"),
    kpiBdpSub: $("kpiBdpSub"),
    kpiCwnd: $("kpiCwnd"),
    kpiCwndSub: $("kpiCwndSub"),
    kpiQueue: $("kpiQueue"),
    kpiQueueSub: $("kpiQueueSub"),
    kpiPathHops: $("kpiPathHops"),
    kpiPathSub: $("kpiPathSub"),

    diagnosisBox: $("diagnosisBox"),
    diagnosisTitle: $("diagnosisTitle"),
    diagnosisBody: $("diagnosisBody"),

    lifecycleStageBadge: $("lifecycleStageBadge"),
    hopInspectorMeta: $("hopInspectorMeta"),
    hdrL2: $("hdrL2"),
    hdrL3: $("hdrL3"),
    hdrL4: $("hdrL4"),
    hdrL7: $("hdrL7"),
    cwndMeta: $("cwndMeta"),
    cwndStateMeta: $("cwndStateMeta"),
    routingTableBody: $("routingTableBody")
  };

  // ============================================================================
  // HELPERS
  // ============================================================================

  function setupCanvas(canvas, cssHeight) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(Math.floor(rect.width), 280);
    const height = cssHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function roundRect(ctx, x, y, w, h, r) {
    const rad = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rad, y);
    ctx.arcTo(x + w, y, x + w, y + h, rad);
    ctx.arcTo(x + w, y + h, x, y + h, rad);
    ctx.arcTo(x, y + h, x, y, rad);
    ctx.arcTo(x, y, x + w, y, rad);
    ctx.closePath();
  }

  function initWirePackets() {
    state.wirePackets = [];
    for (let i = 0; i < 9; i++) {
      state.wirePackets.push({
        u: i / 9,
        dir: i % 3 === 0 ? -1 : 1,
        speed: 0.22 + (i % 3) * 0.04
      });
    }
  }

  // ============================================================================
  // ROUTING & TELEMETRY REFRESH
  // ============================================================================

  function recomputeRouting() {
    state.edges = Engine.buildEdges(state.cfg.cutPrimary, state.cfg.cutBackup);
    state.pathInfo = Engine.computeShortestPath(state.edges, "client", "origin");
    if (state.pathInfo.reachable) {
      state.heroHopIdx = Math.min(state.heroHopIdx, Math.max(0, state.pathInfo.edges.length - 1));
    } else {
      state.heroHopIdx = 0;
    }
    renderRoutingTable();
    updateHeaderInspector();
  }

  function resetHandshakeAndTcp() {
    state.tcp = Engine.createTcpState();
    state.phaseIdx = 0;
    state.heroHopIdx = 0;
    state.heroProgress = 0;
    state.heroDir = 1;
    state.dropFlashes = [];
    recomputeRouting();
    // Seed initial 8 RTT rounds so chart is immediately rich & readable
    for (let i = 0; i < 8; i++) {
      state.lastSnap = Engine.stepTcpRound(state.tcp, state.cfg, state.pathInfo, false);
    }
    updateReadouts();
    drawCwndChart();
  }

  function syncSidebar() {
    const c = state.cfg;
    dom.algoSelect.value = c.algo;
    dom.bwSlider.value = String(c.bwMbps);
    dom.bwVal.textContent = `${c.bwMbps} Mbps`;

    dom.lossSlider.value = String(c.lossPct);
    dom.lossVal.textContent = `${c.lossPct.toFixed(1)}%`;
    dom.lossVal.className = "field-val" + (c.lossPct >= 1.5 ? " danger" : c.lossPct >= 0.5 ? " warn" : "");

    dom.bufferSlider.value = String(c.bufferPkts);
    dom.bufferVal.textContent = `${c.bufferPkts} pkts`;
    dom.bufferVal.className = "field-val" + (c.bufferPkts >= 140 ? " warn" : "");

    dom.cutPrimaryBtn.classList.toggle("cut", c.cutPrimary);
    dom.cutPrimaryBadge.textContent = c.cutPrimary ? "CUT (DOWN)" : "UP";
    dom.cutBackupBtn.classList.toggle("cut", c.cutBackup);
    dom.cutBackupBadge.textContent = c.cutBackup ? "CUT (DOWN)" : "UP";

    dom.presetGrid.querySelectorAll(".preset-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.preset === state.activePreset);
    });
  }

  function updateReadouts() {
    const snap = state.lastSnap;
    const path = state.pathInfo;
    if (!snap || !path) return;

    // Hero Badges
    const badges = [];
    if (!path.reachable) {
      badges.push(`<span class="badge danger">⚠ NETWORK PARTITIONED (0 ROUTES)</span>`);
    } else {
      badges.push(`<span class="badge good">OSPF Converged: ${path.edges.length} Hops (${path.baseRttMs} ms)</span>`);
      const bloatCls = snap.queueDelayMs >= 65 ? "warn" : "good";
      badges.push(`<span class="badge ${bloatCls}">${state.cfg.algo.toUpperCase()}: ${snap.goodputMbps.toFixed(1)} Mbps</span>`);
    }
    badges.push(`<span class="badge">Drops: ${state.tcp.totalDrops}</span>`);
    dom.heroBadges.innerHTML = badges.join("");

    // KPIs
    dom.kpiRtt.textContent = path.reachable ? `${snap.totalRttMs.toFixed(0)} ms` : "TIMEOUT";
    dom.kpiRtt.className = "kpi-val " + (!path.reachable ? "red" : snap.queueDelayMs >= 65 ? "amber" : "cyan");
    dom.kpiRttSub.textContent = path.reachable
      ? `Base ${path.baseRttMs}ms + Q ${snap.queueDelayMs.toFixed(0)}ms`
      : "No L3 route";

    dom.kpiGoodput.textContent = path.reachable ? `${snap.goodputMbps.toFixed(1)} Mbps` : "0.0 Mbps";
    dom.kpiGoodput.className = "kpi-val " + (!path.reachable ? "red" : snap.linkUtilPct >= 70 ? "green" : "amber");
    dom.kpiGoodputSub.textContent = path.reachable ? `${snap.linkUtilPct.toFixed(0)}% of ${state.cfg.bwMbps} Mbps` : "Stalled";

    dom.kpiBdp.textContent = path.reachable ? `${snap.bdpSegs.toFixed(0)} pkts` : "--";
    dom.kpiBdpSub.textContent = path.reachable ? `${((snap.bdpSegs * Engine.MSS_BYTES) / 1024).toFixed(0)} KB in flight` : "--";

    dom.kpiCwnd.textContent = `${snap.cwnd.toFixed(1)} MSS`;
    dom.kpiCwnd.className = "kpi-val " + (snap.dropped ? "red" : "amber");
    dom.kpiCwndSub.textContent = snap.mode;

    dom.kpiQueue.textContent = `${snap.queueSegs.toFixed(0)} / ${state.cfg.bufferPkts}`;
    dom.kpiQueue.className = "kpi-val " + (snap.queueDelayMs >= 65 ? "red" : snap.queueDelayMs >= 20 ? "amber" : "green");
    dom.kpiQueueSub.textContent = `+${snap.queueDelayMs.toFixed(1)} ms bufferbloat`;

    dom.kpiPathHops.textContent = path.reachable ? `${path.edges.length} Hops` : "BLACKHOLE";
    dom.kpiPathHops.className = "kpi-val " + (path.reachable ? "cyan" : "red");
    dom.kpiPathSub.textContent = path.reachable
      ? path.nodeIds.map(id => Engine.getNode(id).short).slice(1, -1).join("→")
      : "All subsea cut";

    // Diagnosis
    const diag = Engine.buildDiagnosis(snap, state.cfg, path);
    dom.diagnosisBox.className = `diagnosis ${diag.severity}`;
    dom.diagnosisTitle.textContent = diag.title;
    dom.diagnosisBody.textContent = diag.body;

    // Lifecycle & CWND meta
    const ph = LIFECYCLE_PHASES[state.phaseIdx];
    dom.lifecycleStageBadge.textContent = `Active: ${ph.label} (${ph.sub})`;
    dom.cwndMeta.textContent = `Algo: ${state.cfg.algo.toUpperCase()} | Round ${state.tcp.round}`;
    dom.cwndStateMeta.textContent = `State: ${snap.mode}`;

    updateHeaderInspector();
    renderRoutingTable();
  }

  function updateHeaderInspector() {
    const path = state.pathInfo;
    if (!path || !path.reachable || path.nodeIds.length < 2) {
      dom.hopInspectorMeta.textContent = "Packet dropped — no active route";
      dom.hdrL2.textContent = "Frame Dropped (Link Down)\nNo Next-Hop ARP Entry";
      dom.hdrL3.textContent = "ICMP Type 3, Code 0\nNetwork Unreachable";
      dom.hdrL4.textContent = "TCP State: SYN_SENT / RTO\nRetransmit Timer Active";
      dom.hdrL7.textContent = "TLS 1.3 Tunnel: Disconnected";
      return;
    }

    const hops = path.edges.length;
    const idx = Math.max(0, Math.min(hops - 1, state.heroHopIdx));
    const fromId = state.heroDir === 1 ? path.nodeIds[idx] : path.nodeIds[idx + 1];
    const toId = state.heroDir === 1 ? path.nodeIds[idx + 1] : path.nodeIds[idx];
    const fromNode = Engine.getNode(fromId);
    const toNode = Engine.getNode(toId);

    const passedNat = state.heroDir === 1 ? idx >= 1 : idx < hops - 1;
    const srcIp = state.heroDir === 1
      ? (passedNat ? "84.112.40.1 (NAT)" : "192.168.1.42 (LAN)")
      : "104.18.32.7 (Origin)";
    const dstIp = state.heroDir === 1
      ? "104.18.32.7 (Origin)"
      : (passedNat ? "84.112.40.1 (NAT)" : "192.168.1.42 (LAN)");

    const hopsTraversed = state.heroDir === 1 ? idx : (hops - 1 - idx);
    const ttl = 64 - hopsTraversed;

    const ph = LIFECYCLE_PHASES[state.phaseIdx];
    const tcpFlags = state.phaseIdx === 1
      ? (state.heroDir === 1 ? "SYN" : "SYN, ACK")
      : (state.heroDir === 1 ? "PSH, ACK" : "ACK");

    dom.hopInspectorMeta.textContent =
      `Hop ${idx + 1}/${hops}: ${fromNode.short} → ${toNode.short} (${state.heroDir === 1 ? "Outbound Request" : "Inbound Reply/ACK"})`;

    dom.hdrL2.textContent =
`Src MAC: ${fromNode.mac}
Dst MAC: ${toNode.mac}
EtherType: 0x0800 (IPv4)
Hop: ${fromNode.short} ──▶ ${toNode.short}`;

    dom.hdrL3.textContent =
`Src IP: ${srcIp}
Dst IP: ${dstIp}
TTL:    ${ttl}  (64 − ${hopsTraversed} hops)
Proto:  6 (TCP) | DF=1`;

    dom.hdrL4.textContent =
`Port:  ${state.heroDir === 1 ? "51492 → 443" : "443 → 51492"}
Flags: [${tcpFlags}]
Seq:   ${state.tcp.seqNum.toLocaleString()}
CWND:  ${state.tcp.cwnd.toFixed(1)} MSS (${Math.round(state.tcp.cwnd * 1.46)} KB)`;

    const l7Body = state.phaseIdx === 0
      ? "DNS Query: A api.thinkmodule.io\nUDP/53 → Answer: 104.18.32.7"
      : state.phaseIdx === 1
        ? "Waiting for TCP Handshake\n(No TLS bytes sent yet)"
        : state.phaseIdx === 2
          ? "TLS 1.3 ClientHello / Finished\nCipher: TLS_AES_256_GCM_SHA384"
          : `Record: Application Data (0x17)\nCiphertext: 7f 9a 2c e1 4b 08...\nStatus: ${ph.label}`;
    dom.hdrL7.textContent = l7Body;
  }

  function renderRoutingTable() {
    const path = state.pathInfo;
    if (!path || !path.reachable) {
      dom.routingTableBody.innerHTML =
        `<tr><td colspan="5" style="color:#f87171;text-align:center;padding:1rem;">No valid route from CLIENT to ORIGIN (BGP Blackhole)</td></tr>`;
      return;
    }

    const rows = path.nodeIds.map((nid, i) => {
      const node = Engine.getNode(nid);
      const edge = i < path.edges.length ? path.edges[i] : null;
      const isCurrent = i === state.heroHopIdx;
      return `<tr class="${isCurrent ? "active-hop" : ""}">
        <td>#${i + 1}</td>
        <td><strong>${node.short}</strong> <span style="color:var(--dim);">${node.asn}</span></td>
        <td>${node.ip}</td>
        <td>${edge ? `${edge.latencyMs} ms` : "DEST"}</td>
        <td><span class="pill-mini pill-up">${isCurrent ? "ACTIVE" : "UP"}</span></td>
      </tr>`;
    });
    dom.routingTableBody.innerHTML = rows.join("");
  }

  // ============================================================================
  // CANVAS 1: FULL 360° 3D GLOBAL BACKBONE TOPOLOGY
  // ============================================================================

  function project3D(p, width, height) {
    const R = state.cam.rot;
    const rx = R[0] * p[0] + R[1] * p[1] + R[2] * p[2];
    const ry = R[3] * p[0] + R[4] * p[1] + R[5] * p[2];
    const rz = R[6] * p[0] + R[7] * p[1] + R[8] * p[2];

    const dist = 5.4;
    const depth = dist - rz;
    const fov = Math.min(width, height) * 0.96 * state.cam.zoom;
    const scale = fov / Math.max(depth, 0.7);
    return {
      sx: width * 0.5 + rx * scale,
      sy: height * 0.5 - ry * scale,
      depth,
      scale
    };
  }

  function draw3DGlobeWireframe(ctx, width, height) {
    // Subtle 3D orbital reference rings for unmistakable 3D spatial depth while spinning 360°
    const rad = 2.12;
    const segs = 48;
    ctx.strokeStyle = "rgba(38, 64, 112, 0.24)";
    ctx.lineWidth = 1;

    const rings = [
      a => [Math.cos(a) * rad, 0, Math.sin(a) * rad],
      a => [Math.cos(a) * rad, Math.sin(a) * rad, 0],
      a => [0, Math.cos(a) * rad, Math.sin(a) * rad]
    ];

    rings.forEach(fn => {
      ctx.beginPath();
      for (let i = 0; i <= segs; i++) {
        const a = (i / segs) * Math.PI * 2;
        const p = project3D(fn(a), width, height);
        if (i === 0) ctx.moveTo(p.sx, p.sy);
        else ctx.lineTo(p.sx, p.sy);
      }
      ctx.stroke();
    });
  }

  function draw3DTopology() {
    const { ctx, width, height } = setupCanvas(dom.canvas3dNet, 440);
    ctx.clearRect(0, 0, width, height);

    // Ambient center glow
    const g = ctx.createRadialGradient(width * 0.5, height * 0.5, 25, width * 0.5, height * 0.5, width * 0.58);
    g.addColorStop(0, "rgba(24, 52, 94, 0.34)");
    g.addColorStop(1, "rgba(4, 9, 20, 0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, width, height);

    draw3DGlobeWireframe(ctx, width, height);

    // Project all 8 nodes
    const projMap = {};
    Engine.NODES.forEach(n => {
      projMap[n.id] = project3D(n.pos, width, height);
    });

    const activeEdgeSet = new Set(
      state.pathInfo && state.pathInfo.reachable ? state.pathInfo.edges.map(e => e.id) : []
    );

    // 1. Draw all fiber links (back-to-front sorted by midpoint depth)
    const sortedEdges = state.edges.map(e => {
      const pa = projMap[e.a];
      const pb = projMap[e.b];
      return { e, pa, pb, depth: (pa.depth + pb.depth) * 0.5 };
    }).sort((x, y) => y.depth - x.depth);

    sortedEdges.forEach(({ e, pa, pb }) => {
      const isAct = activeEdgeSet.has(e.id);
      ctx.save();
      if (!e.up) {
        ctx.setLineDash([6, 5]);
        ctx.strokeStyle = "rgba(248, 113, 113, 0.72)";
        ctx.lineWidth = 2.0;
      } else if (isAct) {
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2.8;
        ctx.shadowColor = "rgba(56, 189, 248, 0.7)";
        ctx.shadowBlur = 10;
      } else {
        ctx.strokeStyle = "rgba(100, 116, 139, 0.36)";
        ctx.lineWidth = 1.3;
      }

      ctx.beginPath();
      ctx.moveTo(pa.sx, pa.sy);
      ctx.lineTo(pb.sx, pb.sy);
      ctx.stroke();
      ctx.restore();

      // Link latency / CUT badge at midpoint
      const mx = (pa.sx + pb.sx) * 0.5;
      const my = (pa.sy + pb.sy) * 0.5;
      ctx.font = "600 9.5px ui-monospace, Menlo, monospace";
      ctx.textAlign = "center";
      if (!e.up) {
        ctx.fillStyle = "#fca5a5";
        ctx.fillText("✕ CUT", mx, my - 5);
      } else if (isAct) {
        ctx.fillStyle = "#bae6fd";
        ctx.fillText(`${e.latencyMs}ms`, mx, my - 5);
      }
    });

    // 2. Draw flying 3D packets along active Dijkstra path
    const path = state.pathInfo;
    if (path && path.reachable && path.edges.length > 0) {
      const nHops = path.edges.length;

      // Helper: interpolate 3D position along full path parameter u in [0, 1]
      const posAlongPath = u => {
        const clamped = Math.max(0, Math.min(0.9999, u));
        const scaled = clamped * nHops;
        const hIdx = Math.floor(scaled);
        const f = scaled - hIdx;
        const aNode = Engine.getNode(path.nodeIds[hIdx]);
        const bNode = Engine.getNode(path.nodeIds[hIdx + 1]);
        return [
          aNode.pos[0] + (bNode.pos[0] - aNode.pos[0]) * f,
          aNode.pos[1] + (bNode.pos[1] - aNode.pos[1]) * f,
          aNode.pos[2] + (bNode.pos[2] - aNode.pos[2]) * f
        ];
      };

      // Ambient steady stream packets (in Live Stream phase)
      if (state.phaseIdx >= 4) {
        state.wirePackets.forEach(pkt => {
          const p3 = posAlongPath(pkt.u);
          const pr = project3D(p3, width, height);
          const col = pkt.dir === 1 ? "#34d399" : "#38bdf8";
          ctx.save();
          ctx.fillStyle = col;
          ctx.shadowColor = col;
          ctx.shadowBlur = 9;
          ctx.beginPath();
          ctx.arc(pr.sx, pr.sy, pkt.dir === 1 ? 4.2 : 3.2, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        });
      }

      // Hero Inspected Packet (large glowing beacon matching current protocol phase)
      const hFrom = state.heroDir === 1 ? path.nodeIds[state.heroHopIdx] : path.nodeIds[state.heroHopIdx + 1];
      const hTo = state.heroDir === 1 ? path.nodeIds[state.heroHopIdx + 1] : path.nodeIds[state.heroHopIdx];
      const nA = Engine.getNode(hFrom);
      const nB = Engine.getNode(hTo);
      const f = state.heroProgress;
      const hero3 = [
        nA.pos[0] + (nB.pos[0] - nA.pos[0]) * f,
        nA.pos[1] + (nB.pos[1] - nA.pos[1]) * f,
        nA.pos[2] + (nB.pos[2] - nA.pos[2]) * f
      ];
      const hp = project3D(hero3, width, height);
      const phColor = LIFECYCLE_PHASES[state.phaseIdx].color;

      ctx.save();
      ctx.fillStyle = phColor;
      ctx.shadowColor = phColor;
      ctx.shadowBlur = 16;
      ctx.beginPath();
      ctx.arc(hp.sx, hp.sy, 6.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.6;
      ctx.stroke();

      // Floating packet tag
      const pktTag = state.phaseIdx === 0 ? "DNS"
        : state.phaseIdx === 1 ? (state.heroDir === 1 ? "SYN" : "SYN-ACK")
        : state.phaseIdx === 2 ? "TLS 1.3"
        : state.heroDir === 1 ? "HTTP/2" : "TCP ACK";
      ctx.shadowBlur = 0;
      ctx.font = "700 9.5px ui-monospace, Menlo, monospace";
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "center";
      ctx.fillText(pktTag, hp.sx, hp.sy - 11);
      ctx.restore();
    }

    // 3. Draw packet drop flashes
    for (let i = state.dropFlashes.length - 1; i >= 0; i--) {
      const fl = state.dropFlashes[i];
      const pr = project3D(fl.pos, width, height);
      const alpha = Math.max(0, fl.life);
      const r = 6 + (1 - alpha) * 24;
      ctx.save();
      ctx.strokeStyle = `rgba(248, 113, 113, ${alpha.toFixed(2)})`;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(pr.sx, pr.sy, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = `rgba(254, 202, 202, ${alpha.toFixed(2)})`;
      ctx.font = "700 10px ui-monospace, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.fillText("PACKET DROP!", pr.sx, pr.sy - r - 4);
      ctx.restore();
    }

    // 4. Draw depth-sorted 3D Router / Server Nodes
    const activeNodeSet = new Set(path && path.reachable ? path.nodeIds : []);
    const sortedNodes = Engine.NODES.map(n => ({ n, p: projMap[n.id] }))
      .sort((a, b) => b.p.depth - a.p.depth);

    sortedNodes.forEach(({ n, p }) => {
      const isEndpoint = n.id === "client" || n.id === "origin";
      const isActive = activeNodeSet.has(n.id);
      const rad = (isEndpoint ? 11 : 8.5) * Math.max(0.75, Math.min(1.25, p.scale / 85));

      const baseCol = n.id === "client" ? "#34d399"
        : n.id === "origin" ? "#a78bfa"
        : isActive ? "#38bdf8" : "#64748b";

      // Outer pulse ring on active path nodes
      if (isActive) {
        ctx.strokeStyle = baseCol;
        ctx.globalAlpha = 0.35;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, rad + 4.5, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      // Glossy 3D sphere
      const sg = ctx.createRadialGradient(p.sx - rad * 0.3, p.sy - rad * 0.3, 1, p.sx, p.sy, rad);
      sg.addColorStop(0, "#ffffff");
      sg.addColorStop(0.38, baseCol);
      sg.addColorStop(1, "#091224");
      ctx.fillStyle = sg;
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, rad, 0, Math.PI * 2);
      ctx.fill();

      // Label pill
      const label = `${n.short}`;
      ctx.font = `${isActive ? "700" : "500"} 10px ui-monospace, Menlo, monospace`;
      const tw = ctx.measureText(label).width + 10;
      const lx = p.sx - tw / 2;
      const ly = p.sy + rad + 5;
      roundRect(ctx, lx, ly, tw, 15, 4);
      ctx.fillStyle = "rgba(4, 9, 20, 0.84)";
      ctx.fill();
      ctx.strokeStyle = isActive ? baseCol : "rgba(51, 78, 130, 0.65)";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = isActive ? "#f8fafc" : "#94a3b8";
      ctx.textAlign = "center";
      ctx.fillText(label, p.sx, ly + 11);
    });

    // Partition overlay alert if all routes are severed
    if (path && !path.reachable) {
      roundRect(ctx, width * 0.5 - 165, 18, 330, 34, 8);
      ctx.fillStyle = "rgba(69, 10, 10, 0.88)";
      ctx.fill();
      ctx.strokeStyle = "#f87171";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = "#fecaca";
      ctx.font = "700 11px ui-monospace, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.fillText("⚠ NETWORK PARTITION — NO ROUTE TO ORIGIN", width * 0.5, 39);
    }
  }

  // ============================================================================
  // CANVAS 2: LIVE 5-STAGE PROTOCOL & SIGNAL LIFECYCLE CIRCUIT
  // ============================================================================

  function drawLifecycleCircuit() {
    const { ctx, width, height } = setupCanvas(dom.canvasLifecycle, 205);
    ctx.clearRect(0, 0, width, height);

    const padX = 16;
    const gap = Math.max(18, Math.min(34, width * 0.028));
    const boxW = (width - padX * 2 - gap * 4) / 5;
    const boxY = 22;
    const boxH = 122;
    const midY = boxY + boxH / 2;

    // Connectors & animated pulses
    for (let i = 0; i < 4; i++) {
      const x1 = padX + i * (boxW + gap) + boxW;
      const x2 = x1 + gap;
      const isCompleted = i < state.phaseIdx;
      ctx.strokeStyle = isCompleted ? "rgba(52, 211, 153, 0.65)" : "rgba(56, 189, 248, 0.28)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x1, midY);
      ctx.lineTo(x2, midY);
      ctx.stroke();

      ctx.fillStyle = isCompleted ? "#34d399" : "#38bdf8";
      ctx.beginPath();
      ctx.moveTo(x2, midY);
      ctx.lineTo(x2 - 6, midY - 4);
      ctx.lineTo(x2 - 6, midY + 4);
      ctx.closePath();
      ctx.fill();

      if (i <= state.phaseIdx) {
        const pPhase = (state.animTime * 0.72 - i * 0.18) % 1;
        const t = pPhase < 0 ? pPhase + 1 : pPhase;
        const px = x1 + t * (gap - 6);
        ctx.fillStyle = LIFECYCLE_PHASES[i].color;
        ctx.shadowColor = LIFECYCLE_PHASES[i].color;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.arc(px, midY, 3.6, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    // Bottom ACK feedback line (Stage 5 -> Stage 4/5 loop)
    const retY = boxY + boxH + 28;
    const xRight = padX + 4 * (boxW + gap) + boxW * 0.5;
    const xLeft = padX + boxW * 0.5;
    ctx.strokeStyle = "rgba(56, 189, 248, 0.34)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(xRight, boxY + boxH);
    ctx.lineTo(xRight, retY);
    ctx.lineTo(xLeft, retY);
    ctx.lineTo(xLeft, boxY + boxH + 4);
    ctx.stroke();
    ctx.setLineDash([]);

    const ackT = (state.animTime * 0.28) % 1;
    const ackX = xRight - ackT * (xRight - xLeft);
    ctx.fillStyle = "#38bdf8";
    ctx.beginPath();
    ctx.arc(ackX, retY, 3.2, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#7f92b0";
    ctx.font = "600 10px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.fillText("TCP ACK Feedback Loop: Sliding Window advances CWND & triggers retransmit on 3-DupACK", width / 2, retY + 18);

    // Stage Cards
    const snap = state.lastSnap;
    const path = state.pathInfo;
    const details = [
      ["UDP/53 → Resolver", "A: 104.18.32.7", "TTL: 300 s"],
      ["SYN (Seq=0)", "SYN-ACK + MSS", "1 RTT Setup"],
      ["ECDHE X25519", "AES-256-GCM", "1-RTT Crypto"],
      ["L7→L4→L3→L2", "MTU: 1,500 B", "NAT + MAC Swap"],
      [
        snap ? `CWND: ${snap.cwnd.toFixed(1)} MSS` : "CWND: --",
        snap && path.reachable ? `Rate: ${snap.goodputMbps.toFixed(1)} Mbps` : "Rate: 0 Mbps",
        `Algo: ${state.cfg.algo.toUpperCase()}`
      ]
    ];

    LIFECYCLE_PHASES.forEach((ph, i) => {
      const bx = padX + i * (boxW + gap);
      const isActive = i === state.phaseIdx;
      const isDone = i < state.phaseIdx;

      roundRect(ctx, bx, boxY, boxW, boxH, 9);
      ctx.fillStyle = isActive ? "#101e38" : "#0a1324";
      ctx.fill();
      ctx.strokeStyle = isActive ? ph.color : isDone ? "rgba(52, 211, 153, 0.55)" : "#1d2f52";
      ctx.lineWidth = isActive ? 2.0 : 1.1;
      ctx.stroke();

      ctx.fillStyle = isActive ? ph.color : isDone ? "#34d399" : "#94a3b8";
      ctx.font = "700 10px ui-monospace, Menlo, monospace";
      ctx.textAlign = "left";
      ctx.fillText(ph.label, bx + 10, boxY + 21);

      ctx.strokeStyle = "rgba(29, 47, 82, 0.85)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(bx + 8, boxY + 30);
      ctx.lineTo(bx + boxW - 8, boxY + 30);
      ctx.stroke();

      ctx.font = "10.5px ui-monospace, Menlo, monospace";
      details[i].forEach((ln, li) => {
        ctx.fillStyle = li === 2 ? "#8ea2c4" : "#e6eefc";
        ctx.fillText(ln, bx + 10, boxY + 52 + li * 23);
      });
    });
  }

  // ============================================================================
  // CANVAS 3: LIVE TCP CONGESTION CONTROL (CWND VS. BDP) CHART
  // ============================================================================

  function drawCwndChart() {
    const hist = state.tcp.history;
    if (!hist.length) return;

    const { ctx, width, height } = setupCanvas(dom.canvasCwnd, 320);
    ctx.clearRect(0, 0, width, height);

    const pad = { top: 26, right: 20, bottom: 38, left: 56 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const latest = hist[hist.length - 1];
    const maxY = Math.max(
      60,
      latest.dropCapSegs * 1.15,
      ...hist.map(h => h.cwnd * 1.12)
    );

    const xOf = idx => pad.left + (idx / Math.max(hist.length - 1, 1)) * plotW;
    const yOf = v => pad.top + plotH * (1 - Math.max(0, Math.min(maxY, v)) / maxY);

    // Grid lines
    ctx.strokeStyle = "rgba(29, 47, 82, 0.65)";
    ctx.lineWidth = 1;
    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.fillStyle = "#7f92b0";

    for (let g = 0; g <= 4; g++) {
      const y = pad.top + (plotH * g) / 4;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(`${Math.round(maxY * (1 - g / 4))}`, pad.left - 8, y + 3);
    }

    // Optimal BDP reference band
    if (latest.bdpSegs > 0) {
      const yBdp = yOf(latest.bdpSegs);
      const yDrop = yOf(latest.dropCapSegs);

      // Bufferbloat zone shading between BDP and Tail-Drop
      ctx.fillStyle = "rgba(251, 191, 36, 0.065)";
      ctx.fillRect(pad.left, yDrop, plotW, Math.max(0, yBdp - yDrop));

      // BDP Dashed Cyan Line
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = "rgba(56, 189, 248, 0.8)";
      ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(pad.left, yBdp); ctx.lineTo(pad.left + plotW, yBdp); ctx.stroke();

      // Buffer Tail-Drop Dashed Red Line
      ctx.strokeStyle = "rgba(248, 113, 113, 0.78)";
      ctx.beginPath(); ctx.moveTo(pad.left, yDrop); ctx.lineTo(pad.left + plotW, yDrop); ctx.stroke();
      ctx.restore();

      ctx.fillStyle = "#38bdf8";
      ctx.textAlign = "right";
      ctx.fillText(`Optimal BDP (${latest.bdpSegs.toFixed(0)})`, pad.left + plotW - 6, yBdp - 5);

      ctx.fillStyle = "#f87171";
      ctx.fillText(`Buffer Overflow (${latest.dropCapSegs.toFixed(0)})`, pad.left + plotW - 6, Math.max(pad.top + 12, yDrop - 5));
    }

    // CWND filled area + crisp line
    const lineCol = state.cfg.algo === "bbr" ? "#34d399" : state.cfg.algo === "cubic" ? "#a78bfa" : "#fbbf24";
    ctx.beginPath();
    hist.forEach((h, i) => {
      const px = xOf(i), py = yOf(h.cwnd);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = lineCol;
    ctx.lineWidth = 2.4;
    ctx.stroke();

    // Mark packet drop events with glowing red dots
    hist.forEach((h, i) => {
      if (!h.dropped) return;
      const px = xOf(i), py = yOf(h.cwnd);
      ctx.fillStyle = "#f87171";
      ctx.beginPath();
      ctx.arc(px, py, 4.2, 0, Math.PI * 2);
      ctx.fill();
    });

    // Axis titles
    ctx.fillStyle = "#94a3b8";
    ctx.textAlign = "center";
    ctx.fillText("TCP RTT Rounds (Time →)", pad.left + plotW / 2, height - 8);

    ctx.save();
    ctx.translate(15, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("CWND (MSS Segments)", 0, 0);
    ctx.restore();
  }

  // ============================================================================
  // PACKET DROP FLASH HELPER
  // ============================================================================

  function spawnDropFlashOnPath() {
    const path = state.pathInfo;
    if (!path || !path.reachable || path.edges.length === 0) return;
    const midIdx = Math.floor(path.edges.length / 2);
    const nA = Engine.getNode(path.nodeIds[midIdx]);
    const nB = Engine.getNode(path.nodeIds[midIdx + 1]);
    state.dropFlashes.push({
      pos: [
        (nA.pos[0] + nB.pos[0]) * 0.5,
        (nA.pos[1] + nB.pos[1]) * 0.5,
        (nA.pos[2] + nB.pos[2]) * 0.5
      ],
      life: 1.0
    });
  }

  // ============================================================================
  // MAIN 60 FPS LOOP
  // ============================================================================

  function advanceHeroHop() {
    const path = state.pathInfo;
    if (!path || !path.reachable || path.edges.length === 0) return;
    const maxHop = path.edges.length - 1;

    if (state.heroDir === 1) {
      if (state.heroHopIdx < maxHop) {
        state.heroHopIdx += 1;
      } else {
        // Reached Origin -> bounce back as Reply/ACK
        state.heroDir = -1;
      }
    } else {
      if (state.heroHopIdx > 0) {
        state.heroHopIdx -= 1;
      } else {
        // Completed full RTT -> advance protocol phase until Live Stream (4)
        state.heroDir = 1;
        if (state.phaseIdx < 4) state.phaseIdx += 1;
      }
    }
    state.heroProgress = 0;
    updateReadouts();
  }

  function tick(ts) {
    if (!state.lastTs) state.lastTs = ts;
    const dt = Math.min((ts - state.lastTs) / 1000, 0.05);
    state.lastTs = ts;
    state.animTime += dt;

    // Smooth auto-spin around Y & gentle X sway
    if (state.autoOrbit && !state.cam.dragging) {
      state.cam.rot = mat3Orthonormalize(mat3Mul(mat3RotY(dt * 0.24), state.cam.rot));
    }

    // Fade drop flashes
    for (let i = state.dropFlashes.length - 1; i >= 0; i--) {
      state.dropFlashes[i].life -= dt * 0.9;
      if (state.dropFlashes[i].life <= 0) state.dropFlashes.splice(i, 1);
    }

    if (state.playing && state.pathInfo && state.pathInfo.reachable) {
      // Advance Hero Packet calmly hop-by-hop
      state.heroProgress += dt * 0.95 * state.speedMult;
      if (state.heroProgress >= 1) {
        advanceHeroHop();
      }

      // Advance ambient stream packets
      state.wirePackets.forEach(p => {
        p.u = (p.u + dt * p.speed * state.speedMult * p.dir) % 1;
        if (p.u < 0) p.u += 1;
      });

      // Advance TCP Congestion Control RTT rounds
      state.tcpAccum += dt * 1.45 * state.speedMult;
      if (state.tcpAccum >= 1) {
        state.tcpAccum -= 1;
        state.lastSnap = Engine.stepTcpRound(state.tcp, state.cfg, state.pathInfo, false);
        if (state.lastSnap.dropped) spawnDropFlashOnPath();
        updateReadouts();
        drawCwndChart();
      }
    }

    draw3DTopology();
    drawLifecycleCircuit();
    requestAnimationFrame(tick);
  }

  // ============================================================================
  // EVENT BINDINGS & INIT
  // ============================================================================

  function applyPreset(key) {
    const p = Engine.PRESETS[key];
    if (!p) return;
    state.activePreset = key;
    state.cfg = { ...p };
    state.playing = true;
    dom.playPauseBtn.innerHTML = "&#9646;&#9646; Pause";
    syncSidebar();
    resetHandshakeAndTcp();
  }

  function bindEvents() {
    dom.presetGrid.addEventListener("click", evt => {
      const btn = evt.target.closest(".preset-btn");
      if (btn) applyPreset(btn.dataset.preset);
    });

    dom.algoSelect.addEventListener("change", () => {
      state.activePreset = null;
      state.cfg.algo = dom.algoSelect.value;
      syncSidebar();
      resetHandshakeAndTcp();
    });

    dom.bwSlider.addEventListener("input", () => {
      state.activePreset = null;
      state.cfg.bwMbps = parseInt(dom.bwSlider.value, 10);
      syncSidebar();
      updateReadouts();
      drawCwndChart();
    });

    dom.lossSlider.addEventListener("input", () => {
      state.activePreset = null;
      state.cfg.lossPct = parseFloat(dom.lossSlider.value);
      syncSidebar();
      updateReadouts();
    });

    dom.bufferSlider.addEventListener("input", () => {
      state.activePreset = null;
      state.cfg.bufferPkts = parseInt(dom.bufferSlider.value, 10);
      syncSidebar();
      updateReadouts();
      drawCwndChart();
    });

    dom.cutPrimaryBtn.addEventListener("click", () => {
      state.activePreset = null;
      state.cfg.cutPrimary = !state.cfg.cutPrimary;
      syncSidebar();
      recomputeRouting();
      state.lastSnap = Engine.stepTcpRound(state.tcp, state.cfg, state.pathInfo, false);
      updateReadouts();
      drawCwndChart();
    });

    dom.cutBackupBtn.addEventListener("click", () => {
      state.activePreset = null;
      state.cfg.cutBackup = !state.cfg.cutBackup;
      syncSidebar();
      recomputeRouting();
      state.lastSnap = Engine.stepTcpRound(state.tcp, state.cfg, state.pathInfo, false);
      updateReadouts();
      drawCwndChart();
    });

    dom.injectDropBtn.addEventListener("click", () => {
      if (!state.pathInfo || !state.pathInfo.reachable) return;
      state.lastSnap = Engine.stepTcpRound(state.tcp, state.cfg, state.pathInfo, true);
      spawnDropFlashOnPath();
      updateReadouts();
      drawCwndChart();
    });

    // Playback Toolbar
    dom.playPauseBtn.addEventListener("click", () => {
      state.playing = !state.playing;
      dom.playPauseBtn.innerHTML = state.playing ? "&#9646;&#9646; Pause" : "&#9654; Play";
    });

    dom.stepHopBtn.addEventListener("click", () => {
      state.playing = false;
      dom.playPauseBtn.innerHTML = "&#9654; Play";
      advanceHeroHop();
    });

    dom.resetSimBtn.addEventListener("click", () => {
      resetHandshakeAndTcp();
    });

    [dom.speedCalmBtn, dom.speedNormalBtn, dom.speedFastBtn].forEach(btn => {
      btn.addEventListener("click", () => {
        state.speedMult = parseFloat(btn.dataset.speed);
        [dom.speedCalmBtn, dom.speedNormalBtn, dom.speedFastBtn].forEach(b => b.classList.toggle("active", b === btn));
      });
    });

    dom.autoOrbitBtn.addEventListener("click", () => {
      state.autoOrbit = !state.autoOrbit;
      dom.autoOrbitBtn.classList.toggle("active", state.autoOrbit);
    });

    dom.resetCamBtn.addEventListener("click", () => {
      state.cam.rot = defaultCamMatrix();
      state.cam.zoom = 1.0;
    });

    // Full 360° Gimbal-Lock-Free 3D Trackball Rotation
    dom.canvas3dNet.addEventListener("pointerdown", evt => {
      state.cam.dragging = true;
      state.cam.lastX = evt.clientX;
      state.cam.lastY = evt.clientY;
      dom.canvas3dNet.setPointerCapture(evt.pointerId);
    });

    dom.canvas3dNet.addEventListener("pointermove", evt => {
      if (!state.cam.dragging) return;
      const dx = evt.clientX - state.cam.lastX;
      const dy = evt.clientY - state.cam.lastY;
      state.cam.lastX = evt.clientX;
      state.cam.lastY = evt.clientY;
      const rotStep = mat3Mul(mat3RotX(dy * 0.0085), mat3RotY(dx * 0.0085));
      state.cam.rot = mat3Orthonormalize(mat3Mul(rotStep, state.cam.rot));
    });

    const endDrag = () => { state.cam.dragging = false; };
    dom.canvas3dNet.addEventListener("pointerup", endDrag);
    dom.canvas3dNet.addEventListener("pointercancel", endDrag);

    dom.canvas3dNet.addEventListener("wheel", evt => {
      evt.preventDefault();
      const factor = evt.deltaY > 0 ? 0.93 : 1.07;
      state.cam.zoom = Math.max(0.62, Math.min(1.9, state.cam.zoom * factor));
    }, { passive: false });

    let resizeTimer = null;
    window.addEventListener("resize", () => {
      if (resizeTimer) cancelAnimationFrame(resizeTimer);
      resizeTimer = requestAnimationFrame(() => {
        drawCwndChart();
      });
    });
  }

  function init() {
    initWirePackets();
    syncSidebar();
    bindEvents();
    resetHandshakeAndTcp();
    requestAnimationFrame(tick);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();