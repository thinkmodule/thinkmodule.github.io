/**
 * ThinkModule Lab — 3D Network Systems, Dijkstra Routing & TCP Congestion Engine (v1.0)
 * First-principles networking simulation:
 *   - 8-node 3D global backbone graph with realistic MAC, IP, NAT, ASN & fiber latencies
 *   - Exact Dijkstra shortest-path solver with live link-cut failover & partition handling
 *   - Authentic TCP NewReno, TCP CUBIC (Linux), and Google BBR v2 congestion state machines
 *   - Real Bandwidth-Delay Product (BDP), router queue bufferbloat delay & effective goodput
 */
(function (global) {
  "use strict";

  const MSS_BYTES = 1460;
  const MTU_BYTES = 1500;

  // 8 Global Backbone Nodes arranged aesthetically on a 3D orbital manifold
  const NODES = [
    { id: "client", name: "Client Laptop",   short: "CLIENT", asn: "LAN",    ip: "192.168.1.42",  mac: "AA:10:4F:00:01:01", role: "Endpoint", pos: [-1.85,  0.30,  0.75] },
    { id: "nat",    name: "Home NAT / ISP",  short: "NAT-GW", asn: "AS3320", ip: "84.112.40.1",   mac: "BB:20:8C:00:02:02", role: "L3 NAT",   pos: [-1.18,  0.65,  0.22] },
    { id: "fra",    name: "DE-CIX Frankfurt",short: "FRA-01", asn: "AS6695", ip: "10.10.1.1",     mac: "CC:30:1A:00:03:03", role: "Core IXP", pos: [-0.42,  0.92,  0.48] },
    { id: "lon",    name: "LINX London",     short: "LON-02", asn: "AS5459", ip: "10.10.2.1",     mac: "CC:31:2B:00:04:04", role: "Core IXP", pos: [-0.45, -0.15,  1.05] },
    { id: "ams",    name: "AMS-IX Amst.",    short: "AMS-03", asn: "AS1200", ip: "10.10.3.1",     mac: "CC:32:3C:00:05:05", role: "Core IXP", pos: [-0.30, -0.82, -0.45] },
    { id: "nyc",    name: "Equinix NYC",     short: "NYC-04", asn: "AS174",  ip: "10.20.4.1",     mac: "DD:40:4D:00:06:06", role: "TransAtl", pos: [ 0.62,  0.72,  0.32] },
    { id: "iad",    name: "Ashburn US-East", short: "IAD-05", asn: "AS16509",ip: "10.20.5.1",     mac: "DD:41:5E:00:07:07", role: "CloudEdge",pos: [ 0.72, -0.68, -0.30] },
    { id: "origin", name: "API Origin",      short: "ORIGIN", asn: "AS13335",ip: "104.18.32.7",   mac: "EE:99:FF:00:08:08", role: "TLS Srv",  pos: [ 1.82,  0.02, -0.58] }
  ];

  // Bidirectional fiber links (one-way propagation latency in ms)
  const BASE_EDGES = [
    { id: "e_cli_nat", a: "client", b: "nat",    latencyMs: 3,  label: "Wi-Fi 6 / Fiber Access" },
    { id: "e_nat_fra", a: "nat",    b: "fra",    latencyMs: 7,  label: "ISP Metro Fiber" },
    { id: "e_nat_lon", a: "nat",    b: "lon",    latencyMs: 14, label: "ISP Secondary Peering" },
    { id: "e_fra_lon", a: "fra",    b: "lon",    latencyMs: 11, label: "EuroCore FRA-LON" },
    { id: "e_fra_ams", a: "fra",    b: "ams",    latencyMs: 8,  label: "EuroCore FRA-AMS" },
    { id: "e_fra_nyc", a: "fra",    b: "nyc",    latencyMs: 42, label: "Subsea AC-2 (Primary)", primarySubsea: true },
    { id: "e_lon_nyc", a: "lon",    b: "nyc",    latencyMs: 36, label: "Subsea TAT-14 (Backup)", backupSubsea: true },
    { id: "e_ams_iad", a: "ams",    b: "iad",    latencyMs: 56, label: "Subsea MAREA-South" },
    { id: "e_nyc_iad", a: "nyc",    b: "iad",    latencyMs: 7,  label: "US-East Backbone" },
    { id: "e_nyc_ori", a: "nyc",    b: "origin", latencyMs: 6,  label: "Anycast Edge NYC" },
    { id: "e_iad_ori", a: "iad",    b: "origin", latencyMs: 5,  label: "Anycast Edge IAD" }
  ];

  const PRESETS = {
    healthy_fiber: {
      algo: "cubic",
      bwMbps: 100,
      lossPct: 0.2,
      bufferPkts: 64,
      cutPrimary: false,
      cutBackup: false
    },
    fiber_cut: {
      algo: "cubic",
      bwMbps: 120,
      lossPct: 0.3,
      bufferPkts: 64,
      cutPrimary: true,
      cutBackup: false
    },
    bufferbloat: {
      algo: "reno",
      bwMbps: 50,
      lossPct: 0.0,
      bufferPkts: 200,
      cutPrimary: false,
      cutBackup: false
    },
    bbr_rescue: {
      algo: "bbr",
      bwMbps: 150,
      lossPct: 1.8,
      bufferPkts: 184,
      cutPrimary: false,
      cutBackup: false
    }
  };

  function getNode(id) {
    return NODES.find(n => n.id === id) || NODES[0];
  }

  function buildEdges(cutPrimary, cutBackup) {
    return BASE_EDGES.map(e => ({
      ...e,
      up: !( (e.primarySubsea && cutPrimary) || (e.backupSubsea && cutBackup) )
    }));
  }

  // Exact Dijkstra Shortest Path (minimizing one-way propagation latency)
  function computeShortestPath(edges, srcId, dstId) {
    const dist = {};
    const prev = {};
    const prevEdge = {};
    const unvisited = new Set();

    NODES.forEach(n => {
      dist[n.id] = Infinity;
      prev[n.id] = null;
      prevEdge[n.id] = null;
      unvisited.add(n.id);
    });
    dist[srcId] = 0;

    while (unvisited.size > 0) {
      let u = null;
      let best = Infinity;
      unvisited.forEach(id => {
        if (dist[id] < best) {
          best = dist[id];
          u = id;
        }
      });
      if (u === null || best === Infinity) break;
      unvisited.delete(u);
      if (u === dstId) break;

      for (let i = 0; i < edges.length; i++) {
        const e = edges[i];
        if (!e.up) continue;
        let v = null;
        if (e.a === u && unvisited.has(e.b)) v = e.b;
        else if (e.b === u && unvisited.has(e.a)) v = e.a;
        if (!v) continue;

        const alt = dist[u] + e.latencyMs;
        if (alt < dist[v]) {
          dist[v] = alt;
          prev[v] = u;
          prevEdge[v] = e;
        }
      }
    }

    if (dist[dstId] === Infinity) {
      return { reachable: false, nodeIds: [], edges: [], oneWayMs: Infinity, baseRttMs: Infinity };
    }

    const nodeIds = [];
    const pathEdges = [];
    let cur = dstId;
    while (cur !== null) {
      nodeIds.unshift(cur);
      if (prevEdge[cur]) pathEdges.unshift(prevEdge[cur]);
      cur = prev[cur];
    }

    return {
      reachable: true,
      nodeIds,
      edges: pathEdges,
      oneWayMs: dist[dstId],
      baseRttMs: dist[dstId] * 2
    };
  }

  // TCP Congestion Control State Machine
  function createTcpState() {
    return {
      round: 0,
      cwnd: 10,          // IW10 (RFC 6928 Initial Window = 10 MSS)
      ssthresh: 64,
      wMax: 64,
      tSinceLoss: 0,
      bbrCycle: 0,
      mode: "Slow Start",
      seqNum: 1048576,
      ackNum: 2097152,
      totalDrops: 0,
      lastDropRound: -99,
      history: []
    };
  }

  function stepTcpRound(tcp, cfg, pathInfo, manualDrop) {
    if (!pathInfo.reachable) {
      tcp.round += 1;
      tcp.cwnd = 1;
      tcp.mode = "RTO / Partitioned";
      const snap = {
        round: tcp.round,
        cwnd: 1,
        bdpSegs: 0,
        dropCapSegs: 0,
        queueSegs: 0,
        queueDelayMs: 0,
        totalRttMs: Infinity,
        goodputMbps: 0,
        dropped: true,
        mode: tcp.mode
      };
      tcp.history.push(snap);
      if (tcp.history.length > 90) tcp.history.shift();
      return snap;
    }

    tcp.round += 1;
    const bwMbps = cfg.bwMbps;
    const baseRttMs = pathInfo.baseRttMs;
    const bufferPkts = cfg.bufferPkts;

    // Exact Bandwidth-Delay Product in 1460-byte MSS segments
    const bdpBytes = ((bwMbps * 1e6) / 8) * (baseRttMs / 1000);
    const bdpSegs = Math.max(4, bdpBytes / MSS_BYTES);
    const dropCapSegs = bdpSegs + bufferPkts;

    // Determine if packet loss happened this RTT (tail-drop overflow OR random loss OR manual chaos click)
    const overflowDrop = tcp.cwnd >= dropCapSegs;
    const randomDropProb = 1 - Math.pow(1 - cfg.lossPct / 100, Math.min(tcp.cwnd, 65));
    const randomDrop = cfg.lossPct > 0 && Math.random() < randomDropProb * 0.42;
    const dropped = !!(manualDrop || overflowDrop || randomDrop);

    if (cfg.algo === "reno") {
      if (dropped) {
        tcp.totalDrops += 1;
        tcp.lastDropRound = tcp.round;
        tcp.ssthresh = Math.max(4, tcp.cwnd * 0.5);
        tcp.cwnd = manualDrop ? Math.max(4, tcp.cwnd * 0.5) : tcp.ssthresh;
        tcp.mode = overflowDrop ? "Fast Recovery (Tail-Drop)" : "Fast Recovery (3-DupACK)";
      } else if (tcp.cwnd < tcp.ssthresh) {
        tcp.cwnd = Math.min(tcp.ssthresh, tcp.cwnd * 1.65);
        tcp.mode = "Slow Start (Exp)";
      } else {
        tcp.cwnd += 2.2;
        tcp.mode = "Congestion Avoidance (AIMD)";
      }
    } else if (cfg.algo === "cubic") {
      if (dropped) {
        tcp.totalDrops += 1;
        tcp.lastDropRound = tcp.round;
        tcp.wMax = tcp.cwnd;
        tcp.cwnd = Math.max(4, tcp.cwnd * 0.7); // CUBIC beta = 0.3 reduction
        tcp.ssthresh = tcp.cwnd;
        tcp.tSinceLoss = 0;
        tcp.mode = overflowDrop ? "CUBIC Backoff (Tail-Drop)" : "CUBIC Multiplicative (-30%)";
      } else if (tcp.cwnd < tcp.ssthresh) {
        tcp.cwnd = Math.min(tcp.ssthresh, tcp.cwnd * 1.7);
        tcp.mode = "HyStart Slow Start";
      } else {
        tcp.tSinceLoss += 0.38;
        const C = 0.4;
        const K = Math.cbrt((tcp.wMax * 0.3) / C);
        const wCubic = C * Math.pow(tcp.tSinceLoss - K, 3) + tcp.wMax;
        tcp.cwnd = Math.max(tcp.cwnd + 1.4, wCubic);
        tcp.mode = tcp.tSinceLoss < K ? "CUBIC Concave Recovery" : "CUBIC Convex Probing";
      }
    } else if (cfg.algo === "bbr") {
      // Google BBR paces near BDP, ignores isolated random loss, probes BW/RTT cyclically
      tcp.bbrCycle = (tcp.bbrCycle + 1) % 10;
      if (tcp.round <= 4 && tcp.cwnd < bdpSegs * 0.95) {
        tcp.cwnd = Math.min(bdpSegs * 1.25, tcp.cwnd * 1.85);
        tcp.mode = "BBR Startup (Pipe Fill)";
      } else if (tcp.bbrCycle === 0) {
        // ProbeBW gain = 1.25x
        tcp.cwnd = tcp.cwnd + (bdpSegs * 1.22 - tcp.cwnd) * 0.55;
        tcp.mode = "BBR ProbeBW (1.25×)";
      } else if (tcp.bbrCycle === 1) {
        // Drain excess queue gain = 0.75x
        tcp.cwnd = tcp.cwnd + (bdpSegs * 0.96 - tcp.cwnd) * 0.65;
        tcp.mode = "BBR Drain Queue";
      } else if (tcp.bbrCycle === 9) {
        tcp.cwnd = Math.max(4, bdpSegs * 0.85);
        tcp.mode = "BBR ProbeRTT (Min Latency)";
      } else {
        tcp.cwnd = tcp.cwnd + (bdpSegs * 1.04 - tcp.cwnd) * 0.45;
        tcp.mode = "BBR Cruise (Zero-Bloat)";
      }
      if (dropped) {
        tcp.totalDrops += 1;
        tcp.lastDropRound = tcp.round;
        if (manualDrop) tcp.cwnd = Math.max(4, tcp.cwnd * 0.88);
      }
    }

    // Clamp CWND realistically
    tcp.cwnd = Math.max(2, Math.min(dropCapSegs + 6, tcp.cwnd));

    // Queue buildup & Bufferbloat queueing delay
    const queueSegs = Math.max(0, Math.min(bufferPkts, tcp.cwnd - bdpSegs));
    const queueDelayMs = ((queueSegs * MTU_BYTES * 8) / (bwMbps * 1e6)) * 1000;
    const totalRttMs = baseRttMs + queueDelayMs;

    // Effective Goodput (Mbps)
    const rawRateMbps = ((tcp.cwnd * MSS_BYTES * 8) / (Math.max(totalRttMs, 1) / 1000)) / 1e6;
    const lossPenalty = dropped ? 0.86 : 1 - (cfg.lossPct / 100) * 1.5;
    const goodputMbps = Math.max(0.5, Math.min(bwMbps * 0.96, rawRateMbps * Math.max(0.3, lossPenalty)));
    const linkUtilPct = (goodputMbps / bwMbps) * 100;

    // Advance TCP sequence numbers realistically
    tcp.seqNum += Math.round(tcp.cwnd * MSS_BYTES);
    tcp.ackNum += Math.round(MSS_BYTES * 2);

    const snap = {
      round: tcp.round,
      cwnd: tcp.cwnd,
      bdpSegs,
      dropCapSegs,
      queueSegs,
      queueDelayMs,
      baseRttMs,
      totalRttMs,
      goodputMbps,
      linkUtilPct,
      dropped,
      mode: tcp.mode
    };

    tcp.history.push(snap);
    if (tcp.history.length > 90) tcp.history.shift();
    return snap;
  }

  function buildDiagnosis(snap, cfg, pathInfo) {
    if (!pathInfo.reachable) {
      return {
        severity: "danger",
        title: "BGP / OSPF NETWORK PARTITION — DESTINATION HOST UNREACHABLE",
        body: "All transoceanic fiber paths are severed. Core routers return ICMP Type 3 (Destination Unreachable) and TCP stalls in exponential RTO backoff. Restore at least one subsea cable to trigger Dijkstra convergence."
      };
    }

    if (snap.queueDelayMs >= 65) {
      return {
        severity: "warn",
        title: `Severe Bufferbloat (+${snap.queueDelayMs.toFixed(0)} ms Standing Router Queue)`,
        body: `${cfg.algo.toUpperCase()} pushed CWND (${snap.cwnd.toFixed(0)} pkts) far past the optimal BDP (${snap.bdpSegs.toFixed(0)} pkts), filling ${snap.queueSegs.toFixed(0)} router buffer slots before any drop occurred. Switch to Google BBR or shrink the router buffer to restore crisp latency.`
      };
    }

    if (cfg.cutPrimary && !cfg.cutBackup) {
      return {
        severity: "warn",
        title: `OSPF / BGP Failover Active — Rerouted via ${pathInfo.nodeIds.map(id => getNode(id).short).join(" → ")}`,
        body: `Primary FRA↔NYC subsea cable is down. Link-state Dijkstra automatically converged onto the surviving ${pathInfo.edges.length}-hop backup path (${pathInfo.baseRttMs} ms base RTT) with zero session teardown.`
      };
    }

    if (cfg.algo === "bbr") {
      return {
        severity: "good",
        title: `Google BBR Pacing Optimal (${snap.goodputMbps.toFixed(1)} Mbps @ +${snap.queueDelayMs.toFixed(1)} ms Queue)`,
        body: `Instead of filling router queues until tail-drop, BBR models bottleneck bandwidth and min-RTT directly — keeping CWND tight around the ${snap.bdpSegs.toFixed(0)}-segment BDP even under ${cfg.lossPct.toFixed(1)}% packet loss.`
      };
    }

    if (cfg.lossPct >= 1.5 && snap.linkUtilPct < 65) {
      return {
        severity: "warn",
        title: `Loss-Induced Throughput Collapse (${snap.linkUtilPct.toFixed(0)}% Link Utilization)`,
        body: `At ${cfg.lossPct.toFixed(1)}% packet loss on a ${pathInfo.baseRttMs} ms WAN path, loss-based ${cfg.algo.toUpperCase()} repeatedly slashes CWND before it can refill the ${snap.bdpSegs.toFixed(0)}-segment BDP pipe. Try switching to BBR to see the difference.`
      };
    }

    return {
      severity: "good",
      title: `Healthy Transatlantic Flow — ${snap.goodputMbps.toFixed(1)} Mbps (${snap.linkUtilPct.toFixed(0)}% Pipe Util)`,
      body: `Active ${pathInfo.edges.length}-hop Dijkstra path (${pathInfo.baseRttMs} ms base RTT). Watch the Live Header Inspector below: Ethernet MACs rewrite at every hop, NAT translates the client IP once, and TTL counts down 64 → ${64 - Math.max(0, pathInfo.edges.length - 1)}.`
    };
  }

  global.NetEngine = {
    MSS_BYTES,
    MTU_BYTES,
    NODES,
    PRESETS,
    getNode,
    buildEdges,
    computeShortestPath,
    createTcpState,
    stepTcpRound,
    buildDiagnosis
  };
})(window);